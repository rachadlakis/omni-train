"""Which strategy should this run use? `strategy: auto`, and the answer in plain words.

    python -m omni_train.advisor --config config.yaml                    # this machine's GPUs, the real dataset
    python -m omni_train.advisor --config config.yaml --gpu h100 --gpus 8    # some other machine
    python -m omni_train.advisor --config config.yaml --write run.yaml   # the same config with `auto` filled in

planner.py does the arithmetic; this file supplies what it needs to know:
  * detect_cluster(): the GPUs of this machine (the smallest one counts);
  * model_spec(): the architecture, from a preset or the model's config.json;
  * measure_data(): the number of rows and the real token lengths, with the trainer's own
    tokenisation. A sample is tokenised, plus the longest rows by text length.

`strategy: auto` means: keep the model, the data and the training method (full / LoRA / QLoRA)
and choose the rest, the simplest thing that fits. It may turn gradient checkpointing on or
lower the batch per GPU, and says so. `num_gpus: auto` = every visible GPU; a number = at most
that many. It is resolved before training: by scripts/launch.sh (which needs the answer to know
how many processes to start), by the console when a run is submitted, or by train.py itself.
Under torchrun the process count is already fixed, and rank 0 decides for every rank.

torch, transformers and datasets are imported only when needed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import threading

from omni_train import planner

SAMPLE = 500                     # rows tokenised
SCAN = 20_000                    # rows whose text length is read, to find the longest ones
LONGEST = 24                     # of those, how many of the longest are tokenised as well
STREAM = 2_000                   # rows read from a Hub dataset too large to download for this
FULL_DOWNLOAD_BYTES = 100 * 1024 ** 2
TEXT_TYPES = {"llm", "seq2seq", "encoder"}
MARGINAL = 1.05                  # an estimate this close to the real GPU memory is still tried


def is_auto(cfg: dict) -> bool:
    return str(cfg.get("strategy", "")).strip().lower() == "auto"


def _truthy(x) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes"}


def max_gpus(cfg: dict) -> int | None:
    """num_gpus as an upper limit; None = every visible GPU."""
    n = cfg.get("num_gpus", "auto")
    if n in (None, "") or str(n).strip().lower() == "auto":
        return None
    return max(1, int(n))


# ---------------------------------------------------------------------------
# Hardware and model
# ---------------------------------------------------------------------------

def detect_cluster(limit: int | None = None) -> planner.Cluster | None:
    """The GPUs this process can see, or None. Mixed GPUs count as the smallest one."""
    import torch
    if not torch.cuda.is_available():
        return None
    n = torch.cuda.device_count()
    props = min((torch.cuda.get_device_properties(i) for i in range(n)), key=lambda p: p.total_memory)
    return planner.Cluster(gpu_mem_gb=round(props.total_memory / planner.GB, 2), gpus_per_node=min(n, limit or n),
                           nodes=1, gpu_name=props.name)


_HF_CONFIGS: dict[str, dict | None] = {}


def hf_config(model_name: str, token: str | None = None) -> dict | None:
    """config.json of a Hub model or a local folder: cache first, then the Hub (unless offline)."""
    if not model_name or model_name in _HF_CONFIGS:
        return _HF_CONFIGS.get(model_name)
    result = None
    try:
        from transformers import AutoConfig
        token = token or os.environ.get("HF_TOKEN") or None
        for local_only in (True, False):
            if not local_only and os.environ.get("HF_HUB_OFFLINE") == "1":
                break
            try:
                result = AutoConfig.from_pretrained(model_name, token=token, local_files_only=local_only).to_dict()
                break
            except Exception:
                continue
    except Exception:
        result = None
    _HF_CONFIGS[model_name] = result
    return result


def model_spec(cfg: dict, token: str | None = None) -> planner.ModelSpec:
    model_type = str(cfg.get("model_type", "llm")).lower()
    name = str(cfg.get("model_name") or "")
    if model_type == "custom_transformer":
        return planner.spec_from_custom(cfg.get("custom_transformer_args") or {})
    preset = planner.find_preset(name)
    return preset or planner.resolve_spec(name, model_type, hf_config=hf_config(name, token))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _is_local(name: str) -> bool:
    from omni_train.data import LOCAL_FORMATS
    return os.path.splitext(str(name))[1].lower() in LOCAL_FORMATS


def _open_rows(name, subset, split, token):
    """(indexable or list of rows, total row count or None, every row is here)."""
    from omni_train.data import load_any_dataset
    if _is_local(name) or os.environ.get("HF_HUB_OFFLINE") == "1":
        ds = load_any_dataset(name, subset, split)
        return ds, len(ds), True
    from datasets import ReadInstruction, load_dataset, load_dataset_builder
    info = load_dataset_builder(name, subset or None, token=token).info
    sizes = {k: v.num_examples for k, v in (info.splits or {}).items()}
    small = info.download_size is not None and info.download_size <= FULL_DOWNLOAD_BYTES
    if small:                                   # training needs it anyway, and then it is cached
        ds = load_any_dataset(name, subset, split)
        return ds, len(ds), True
    total, base, skip, keep = None, split, 0, STREAM
    if sizes:
        pieces = ReadInstruction.from_spec(split).to_absolute(sizes)
        total = sum(p.to - p.from_ for p in pieces)
        first = pieces[0]
        base, skip, keep = first.splitname, first.from_, min(STREAM, first.to - first.from_)
    else:
        base = split.split("[")[0].split("+")[0]
    stream = load_dataset(name, subset or None, split=base, streaming=True, token=token)
    if skip:
        stream = stream.skip(skip)
    rows = list(stream.take(keep))
    return rows, total, total is not None and len(rows) >= total


def _chars(x) -> int:
    if isinstance(x, str):
        return len(x)
    if isinstance(x, dict):
        return sum(_chars(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(_chars(v) for v in x)
    return 0


def _percentile(values: list[int], q: float) -> int:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def _round_up(n: int, step: int = 64) -> int:
    return max(step, -(-int(n) // step) * step)


def _row_lengths(kind: str, cfg: dict, tokenizer):
    """row -> list of token lengths before max_length cuts them (one per sequence), [] = row is dropped."""
    from omni_train.post_training import data as ptd
    algo = cfg.get("algorithm") or {}
    if kind == "sft":
        return lambda row: [len(ptd.tokenize_sft(row, tokenizer, 10 ** 9)["input_ids"])]
    if kind in planner.PREFERENCE_ALGOS:
        max_prompt = int(algo.get("max_prompt_length") or 256)

        def pair(row):
            t = ptd.tokenize_preference(row, tokenizer, 10 ** 9, max_prompt)
            return [len(t["prompt_ids"]) + len(t[k]) for k in ("chosen_ids", "rejected_ids")]
        return pair
    if kind in planner.RL_ALGOS:            # grpo / ppo: prompts only, completions are sampled
        return lambda row: [len(ptd.tokenize_grpo(row, tokenizer, 10 ** 9, algo.get("system_prompt"))["prompt_ids"])]
    field = (cfg.get("dataset") or {}).get("text_field") or "text"

    def text(row):
        if field not in row:
            raise ValueError(f"the dataset has no '{field}' column (it has {sorted(row)}); set dataset.text_field")
        value = row[field] or ""
        return [len(tokenizer(value)["input_ids"])] if len(value.strip()) > 20 else []
    return text


def measure_data(cfg: dict, token: str | None = None, sample: int = SAMPLE, scan: int = SCAN,
                 tokenizer=None) -> dict | None:
    """Row count and token lengths of the run's dataset. None when there is nothing to measure
    (synthetic data, images). Raises ValueError when the rows do not match the goal.

    Lengths are before max_length cuts them, except `longest` and `batch_len`, which are what a
    batch can really hold."""
    model_type = str(cfg.get("model_type", "llm")).lower()
    ds_cfg = cfg.get("dataset") or {}
    name, split = str(ds_cfg.get("name") or ""), str(ds_cfg.get("split") or "train")
    kind = str((cfg.get("algorithm") or {}).get("type", "pretrain")).lower()
    if model_type not in TEXT_TYPES or not name or name == "synthetic":
        return None
    token = token or os.environ.get("HF_TOKEN") or None
    max_length = int((cfg.get("training") or {}).get("max_length", 512))
    batch = max(1, int((cfg.get("training") or {}).get("batch_size", 1)))

    rows, total, complete = _open_rows(name, ds_cfg.get("subset"), split, token)
    if len(rows) == 0:
        raise ValueError(f"the dataset {name} ({split}) is empty")
    rng = random.Random(0)
    n = len(rows)
    scanned = list(range(n)) if n <= scan else sorted(rng.sample(range(n), scan))
    by_chars = sorted(((_chars(rows[i]), i) for i in scanned), reverse=True)
    picked = list(range(n)) if n <= sample else rng.sample(scanned, min(sample, len(scanned)))
    tail = [i for _, i in by_chars[:LONGEST]]

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(cfg.get("model_name")), token=token)
    measure = _row_lengths(kind, cfg, tokenizer)
    lengths = {i: measure(rows[i]) for i in dict.fromkeys(picked + tail)}

    sampled = [lengths[i] for i in picked]          # the longest rows must not skew the averages
    used = [x for x in sampled if x]
    if not used:
        raise ValueError(f"none of the {len(sampled)} rows checked in {name} can be used for training")
    flat = [v for x in used for v in x]
    longest_seen = max(v for x in lengths.values() for v in x)
    share_used = len(used) / len(sampled)
    n_rows = None if total is None else max(1, round(total * share_used))
    exact = complete and len(scanned) == n          # every row's text length was read
    limit = max_length
    if kind in planner.RL_ALGOS:
        limit = int((cfg.get("algorithm") or {}).get("max_prompt_length") or max_length // 2)
    safe = longest_seen if exact else round(longest_seen * 1.25)

    # what a batch is padded to: its longest row, after the cut
    cut = [min(limit, max(x)) for x in used]
    draws = [max(rng.choice(cut) for _ in range(batch)) for _ in range(400)]
    over = sum(v > limit for v in flat) / len(flat)
    p99 = _percentile(flat, 0.99)
    suggested = _round_up(longest_seen if exact and longest_seen <= 1.5 * p99 else p99)
    mean = sum(flat) / len(flat)
    out = {
        "name": name, "kind": kind, "rows": n_rows, "rows_exact": total is not None and share_used == 1.0,
        "sampled": len(sampled), "scanned": len(scanned), "mean": round(mean, 1), "p50": _percentile(flat, 0.5),
        "p95": _percentile(flat, 0.95), "p99": p99, "longest_seen": longest_seen, "longest_exact": exact,
        "longest": min(limit, safe), "batch_len": round(sum(draws) / len(draws), 1), "limit": limit,
        "over_limit_pct": round(100 * over, 1), "suggested_max_length": suggested,
        "suggested_keeps_pct": round(100 * sum(v <= suggested for v in flat) / len(flat), 1),
        "total_tokens": None if n_rows is None else round(n_rows * mean),
    }
    out["tips"] = _tips(out, kind)
    return out


def _tips(d: dict, kind: str) -> list[str]:
    tips = []
    limit, name = d["limit"], "max_prompt_length" if kind in planner.RL_ALGOS else "max_length"
    if d["over_limit_pct"] >= 1:
        tips.append(f"{d['over_limit_pct']:g}% of your examples are longer than {name} ({limit:,}) and will be cut. "
                    f"{d['suggested_max_length']:,} keeps {d['suggested_keeps_pct']:g}% of them whole.")
    elif kind == "pretrain" and d["suggested_max_length"] <= 0.75 * limit and d["longest_exact"] \
            and d["longest_seen"] <= d["suggested_max_length"]:
        tips.append(f"No text is longer than {d['longest_seen']:,} tokens, but every row is padded to {limit:,}. "
                    f"max_length: {d['suggested_max_length']} trains on the same text with less memory and time.")
    return tips


def try_measure(cfg: dict, seconds: float = 120, **kw) -> tuple[dict | None, str]:
    """measure_data() that never stops a run: (data or None, why not). Gives up after `seconds`."""
    box: dict = {}

    def work():
        try:
            box["data"] = measure_data(cfg, **kw)
        except Exception as e:
            box["error"] = f"{e}" or type(e).__name__
    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        return None, f"reading the dataset took longer than {seconds:.0f} seconds"
    return box.get("data"), (box.get("error") or "").strip().split("\n")[0][:300]


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

def advise(cfg: dict, cluster: planner.Cluster, data: dict | None = None, token: str | None = None, **kw) -> dict:
    """planner.plan() for a training config."""
    return planner.plan(model_spec(cfg, token), planner.workload_from_config(cfg), cluster, data=data, **kw)


def apply_option(cfg: dict, option: dict) -> dict:
    """A copy of cfg that uses one option of a plan, with the changes it needs."""
    out = copy.deepcopy(cfg)
    out["strategy"], out["num_gpus"] = option["strategy"], int(option["gpus"])
    if option.get("parallelism"):
        out["parallelism"] = dict(option["parallelism"])
    changes = option.get("changes") or {}
    training = out.setdefault("training", {})
    if changes.get("grad_ckpt"):
        training["gradient_checkpointing"] = True
    if "batch_size" in changes:
        training["batch_size"] = int(changes["batch_size"])
    dist = out.get("dist_parameters") or {}
    if out["strategy"] == "fsdp" and _truthy(dist.get("mixed_precision", False)) and dist.get("param_dtype"):
        dist["output_dtype"] = dist["param_dtype"]      # FSDP needs one dtype (utils._check_fsdp_dtypes)
    return out


def resolve(cfg: dict, cluster: planner.Cluster | None, data: dict | None = None, processes: int | None = None,
            token: str | None = None) -> tuple[dict, dict | None]:
    """Fill in `strategy: auto`. Returns (config with a concrete strategy, the plan or None without GPUs).

    processes: the number of processes already started (torchrun); None = free choice.
    Raises ValueError, in plain words, when nothing can fit."""
    if cluster is None:                                  # no GPU: nothing to plan
        n = processes or 1
        needs_5d = planner.workload_from_config(cfg).needs_5d
        strategy = "5d" if needs_5d else ("ddp" if n > 1 else "solo")
        option = {"strategy": strategy, "gpus": n}
        if needs_5d:
            option["parallelism"] = {"dp": "auto", "tp": 1, "cp": 1, "pp": 1}
        return apply_option(cfg, option), None
    p = advise(cfg, cluster, data, token, all_gpus=bool(processes and processes > 1))
    rec = p["recommended"]
    if p["verdict"] == "infeasible":
        if rec["per_gpu_gb"] > cluster.gpu_mem_gb * MARGINAL:
            raise ValueError(planner.advice_text(p) + "\n\nTo try anyway, set `strategy` yourself instead of auto.")
        p["notes"].insert(0, "This is very close to the GPU's memory and may run out of it.")
    return apply_option(cfg, rec), p


def describe(p: dict | None, cfg: dict) -> str:
    """What `auto` chose, for the terminal and the run log."""
    n, unit = cfg["num_gpus"], "process" if p is None else "GPU"
    head = f"Automatic strategy: {cfg['strategy']} on {n} {unit}{'' if n == 1 else 'es' if p is None else 's'}"
    if _truthy((cfg.get("save_load") or {}).get("resume", False)):
        head += "\n   save_load.resume is on: the checkpoint must come from a run with this same strategy."
    if p is None:
        return head + "\n   No GPU was found, so this runs on the CPU."
    return head + "\n" + "\n".join("   " + line if line else "" for line in planner.advice_text(p).split("\n"))


def resolve_for_process(cfg: dict, log=lambda s: print(s, flush=True)) -> dict:
    """`strategy: auto` inside train.py. One process decides; under torchrun rank 0 decides for all,
    because ranks that chose differently would hang."""
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        data, why = try_measure(cfg)
        if why:
            log(f"Could not measure the dataset ({why}); using max_length.")
        visible = detect_cluster()
        try:
            out, p = resolve(cfg, detect_cluster(1), data, processes=1)
        except ValueError as e:
            if visible and visible.total_gpus > 1:
                raise ValueError(f"{e}\nThis machine has {visible.total_gpus} GPUs: `bash scripts/launch.sh` can use "
                                 "all of them.") from None
            raise
        log(describe(p, out))
        return out

    from datetime import timedelta

    import torch.distributed as dist
    store, _, _ = next(dist.rendezvous("env://", timeout=timedelta(minutes=10)))
    store = dist.PrefixStore(f"omni_auto/{os.environ.get('TORCHELASTIC_RESTART_COUNT', '0')}", store)
    mine = detect_cluster(1)
    if mine is not None:                                 # this rank's own GPU, not GPU 0
        import torch
        props = torch.cuda.get_device_properties(int(os.environ.get("LOCAL_RANK", "0")) % torch.cuda.device_count())
        mine.gpu_mem_gb, mine.gpu_name = round(props.total_memory / planner.GB, 2), props.name
    store.set(f"gpu/{rank}", json.dumps(None if mine is None else [mine.gpu_mem_gb, mine.gpu_name]))
    if rank == 0:
        try:
            gpus = [json.loads(store.get(f"gpu/{r}")) for r in range(world)]
            cluster = None
            if all(gpus):
                mem, name = min(gpus)
                local = int(os.environ.get("LOCAL_WORLD_SIZE", str(world)))
                cluster = planner.Cluster(gpu_mem_gb=mem, gpus_per_node=local, nodes=max(1, world // local), gpu_name=name)
            data, why = try_measure(cfg)
            if why:
                log(f"Could not measure the dataset ({why}); using max_length.")
            out, p = resolve(cfg, cluster, data, processes=world)
            log(describe(p, out))
            store.set("result", json.dumps({"cfg": out}))
        except Exception as e:
            store.set("result", json.dumps({"error": f"{e}"}))
    result = json.loads(store.get("result"))
    if "error" in result:
        raise ValueError(result["error"])
    return result["cfg"]


def preflight(cfg: dict, world: int = 1) -> str | None:
    """A warning when the strategy the user chose is estimated not to fit this machine's GPUs.
    Only a warning: it is an estimate, for sequences that reach max_length."""
    if str(cfg.get("model_type", "llm")).lower() not in {"llm", "custom_transformer"}:
        return None
    cluster = detect_cluster()
    if cluster is None:
        return None
    spec, work = model_spec(cfg), planner.workload_from_config(cfg)
    if spec.source == "estimate":
        return None
    cluster.nodes = max(1, -(-world // cluster.gpus_per_node))
    strategy = str(cfg.get("strategy")).lower()
    used = planner.evaluate_layout(spec, work, cluster, strategy, world, cfg.get("parallelism"))
    if used["per_gpu_gb"] <= cluster.gpu_mem_gb:
        return None
    p = planner.plan(spec, work, cluster)
    tip = p["advice"]
    better = f"{tip['title']}{' (' + tip['tech'] + ')' if tip['tech'] else ''}: {tip['detail']}" if tip["fits"] else \
        "; ".join(f"{o['title']}: {o['detail']}" for o in tip["others"][:2])
    return (f"This run may run out of GPU memory: about {planner.gb_text(used['per_gpu_gb'])} per GPU with "
            f"{used['label']}, and the GPU has {cluster.gpu_mem_gb:g} GB (estimated for sequences of "
            f"{work.seq_len:,} tokens)." + (f" Instead: {better}." if better else "")
            + " `strategy: auto` chooses for you.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Which strategy should this run use? Fills in `strategy: auto`.")
    ap.add_argument("--config", default=os.environ.get("CONFIG_PATH", "config.yaml"), help="training YAML")
    ap.add_argument("--write", help="write the config with `auto` filled in to this file")
    ap.add_argument("--strategy", help="use this strategy instead of choosing (with --write)")
    ap.add_argument("--gpus", type=int, help="GPUs per machine (default: the visible ones)")
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--gpu", help="plan for this GPU type instead of this machine's (h100, 4090, ...)")
    ap.add_argument("--gpu-mem", type=float, help="GB per GPU")
    ap.add_argument("--no-data", action="store_true", help="do not read the dataset; assume max_length")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    import yaml
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    try:
        with open(a.config) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Config file not found: {a.config}", file=sys.stderr)
        return 1

    try:
        if a.strategy and a.strategy.lower() != "auto":
            n = a.gpus or max_gpus(cfg)
            if n is None:
                found = detect_cluster()
                n = found.total_gpus if found else 1
            out, p = {**cfg, "strategy": a.strategy.lower(), "num_gpus": 1 if a.strategy.lower() == "solo" else n}, None
        else:
            if a.gpu or a.gpu_mem:
                cluster = planner.cluster_from_names(a.gpu, a.gpu_mem, a.gpus or max_gpus(cfg) or 1, a.nodes)
            else:
                cluster = detect_cluster(a.gpus or max_gpus(cfg))
                if cluster is None and not a.write:
                    print("No GPU found here. To plan for another machine: --gpu h100 --gpus 8", file=sys.stderr)
            data, why = (None, "") if a.no_data else try_measure(cfg)
            if why:
                print(f"Could not measure the dataset ({why}); using max_length.", file=sys.stderr)
            out, p = resolve(cfg, cluster, data)
            print(json.dumps(p, indent=2) if a.json else "\n" + describe(p, out) + "\n")
    except ValueError as e:
        print(f"\n{e}\n", file=sys.stderr)
        return 1
    if a.write:
        with open(a.write, "w") as f:
            yaml.safe_dump(out, f, sort_keys=False)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)      # a dataset download that was given up on must not keep the process alive

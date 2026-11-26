"""strategy: auto (advisor.py): what it measures, what it chooses and how it says so.

No GPU and no downloads: clusters are made up, data is a local .jsonl, the tokenizer is word-level.
"""
import copy
import json
import os
import subprocess
import sys

import pytest
import yaml

import advisor
import planner as P
from tests.post_training_helpers import PAIRS, make_tokenizer, tiny_llama
from utils import build_args

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA = "meta-llama/Llama-3.1-8B"
H100x8 = P.Cluster(gpu_mem_gb=80, gpus_per_node=8, gpu_name="h100")


def cfg(**over):
    with open(os.path.join(ROOT, "config.yaml")) as f:
        c = yaml.safe_load(f)
    c.update(model_name=LLAMA, dataset={"name": "some/dataset", "split": "train"}, strategy="auto", num_gpus="auto")
    c["training"].update(batch_size=1, max_length=4096)
    for key, value in over.items():
        node, *rest = key.split("__")
        if rest:
            c.setdefault(node, {})[rest[0]] = value
        else:
            c[node] = value
    return c


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


# ---- choosing ---------------------------------------------------------------

def test_auto_picks_fsdp_and_the_config_is_accepted():
    out, p = advisor.resolve(cfg(), H100x8)
    assert (out["strategy"], out["num_gpus"]) == ("fsdp", 8)
    args = build_args(out)
    assert args.strategy == "fsdp" and args.num_gpus == 8
    assert "FSDP" in advisor.describe(p, out) and "split the model" in advisor.describe(p, out)


def test_auto_never_changes_the_training_method():
    for method in ("full", "lora", "qlora"):
        c = cfg()
        c["peft"]["enabled"], c["quantization"]["enabled"] = method != "full", method == "qlora"
        c["peft"]["type"], c["quantization"]["bits"] = "qlora" if method == "qlora" else "lora", 4
        out, p = advisor.resolve(c, H100x8)
        assert out["peft"] == c["peft"] and out["quantization"] == c["quantization"]
        assert out["model_name"] == LLAMA and out["training"]["max_length"] == 4096
        build_args(out)


def test_auto_picks_a_5d_layout_the_trainer_accepts():
    out, p = advisor.resolve(cfg(training__max_length=32768), H100x8)
    assert out["strategy"] == "5d" and p["recommended"]["parallelism"]["cp"] > 1
    args = build_args(out)
    assert (args.par_dp or 1) * args.par_tp * args.par_cp * args.par_pp == 8


@pytest.mark.parametrize("change", [
    {"dataset": {"name": "./my.jsonl", "split": "train"}},                              # 5d reads Hub datasets only
    {"dataset": {"name": "a/b", "split": "train", "text_field": "body"}},
    {"save_load__export_hf": True},
    {"dist_parameters__param_dtype": "float16"},
])
def test_auto_avoids_5d_when_the_trainer_would_refuse_it(change):
    c = cfg(training__max_length=32768, **change)
    p = advisor.advise(c, H100x8)
    assert all(o["strategy"] != "5d" for o in p["options"])


def test_auto_turns_checkpointing_on_and_lowers_the_batch_before_5d():
    """A batch that is too large is not a reason for context parallelism."""
    c = cfg(model_name="HuggingFaceTB/SmolLM2-135M", training__max_length=32768, training__batch_size=4,
            training__gradient_checkpointing=False)
    out, p = advisor.resolve(c, H100x8)
    assert out["strategy"] in {"solo", "ddp"}
    assert out["training"]["gradient_checkpointing"] is True and out["training"]["batch_size"] < 4
    text = advisor.describe(p, out)
    assert "gradient checkpointing on" in text and "Lowers the batch" in text
    assert c["training"]["batch_size"] == 4                                             # the input is untouched


def test_auto_moe_uses_5d_even_on_one_gpu():
    with open(os.path.join(ROOT, "configs/custom_5d_moe_16gpu.yaml")) as f:
        c = yaml.safe_load(f)
    c.update(strategy="auto", num_gpus="auto")
    c.pop("parallelism")
    build_args(c)
    for gpus in (1, 8):
        out, p = advisor.resolve(c, P.Cluster(gpu_mem_gb=80, gpus_per_node=gpus, gpu_name="h100"))
        assert out["strategy"] == "5d" and out["num_gpus"] == gpus
        build_args(out)
    out, p = advisor.resolve(c, None)
    assert out["strategy"] == "5d" and p is None
    build_args(out)


def test_auto_without_a_gpu_runs_on_the_cpu():
    out, p = advisor.resolve(cfg(), None)
    assert (out["strategy"], out["num_gpus"], p) == ("solo", 1, None)
    assert "CPU" in advisor.describe(p, out)
    out, _ = advisor.resolve(cfg(), None, processes=2)
    assert (out["strategy"], out["num_gpus"]) == ("ddp", 2)


def test_auto_refuses_what_cannot_fit_and_says_what_would():
    with pytest.raises(ValueError) as e:
        advisor.resolve(cfg(), P.Cluster(gpu_mem_gb=24, gpus_per_node=1, gpu_name="4090"))
    text = str(e.value)
    assert "Does not fit" in text and "LoRA" in text and "set `strategy` yourself" in text


def test_auto_tries_a_run_that_is_only_just_too_big():
    """The estimate is not a guarantee: within the GPU's real memory the run is tried."""
    need = P.memory_per_gpu(P.PRESETS[LLAMA.lower()], P.Workload(seq_len=4096, batch_size=1), P.Layout("solo", 1))["total"]
    out, p = advisor.resolve(cfg(), P.Cluster(gpu_mem_gb=need * 0.98, gpus_per_node=1, gpu_name="h100"))
    assert p["verdict"] == "infeasible" and out["strategy"] == "solo" and "very close" in p["notes"][0]


def test_process_count_is_respected():
    """Under torchrun the processes already run: one GPU is not an option, even for a small model."""
    c = cfg(model_name="gpt2", training__max_length=128)
    out, _ = advisor.resolve(c, H100x8, data={"rows": 10}, processes=8)
    assert (out["strategy"], out["num_gpus"]) == ("ddp", 8)
    out, _ = advisor.resolve(c, H100x8, data={"rows": 10})
    assert out["strategy"] == "solo"


def test_auto_fixes_the_fsdp_dtype_pair():
    out, _ = advisor.resolve(cfg(dist_parameters__output_dtype="float32"), H100x8)
    assert out["strategy"] == "fsdp" and out["dist_parameters"]["output_dtype"] == "bfloat16"
    build_args(out)


def test_num_gpus_is_a_limit():
    assert advisor.max_gpus({"num_gpus": "auto"}) is None and advisor.max_gpus({}) is None
    assert advisor.max_gpus({"num_gpus": 4}) == 4 and advisor.max_gpus({"num_gpus": "2"}) == 2


def test_build_args_accepts_auto_but_not_auto_gpus_with_a_fixed_strategy():
    assert build_args(cfg()).strategy == "auto"
    assert build_args(cfg(algorithm={"type": "sft"})).algorithm == "sft"
    with pytest.raises(ValueError, match="num_gpus: auto needs strategy: auto"):
        build_args(cfg(strategy="fsdp"))
    assert build_args(cfg(strategy="solo")).num_gpus == 1


# ---- measuring --------------------------------------------------------------

def sft_rows(n=40, long=None):
    rows = [{"prompt": "what is two plus two", "completion": "four"} for _ in range(n)]
    if long:
        rows[n // 2]["completion"] = "four " * long
    return rows


def test_measure_sft_counts_the_trainers_tokens(tmp_path):
    from post_training.data import tokenize_sft
    tok = make_tokenizer()
    rows = sft_rows(long=300)
    c = cfg(dataset={"name": jsonl(tmp_path / "d.jsonl", rows), "split": "train"}, algorithm={"type": "sft"})
    d = advisor.measure_data(c, tokenizer=tok)
    lengths = [len(tokenize_sft(r, tok, 10 ** 9)["input_ids"]) for r in rows]
    assert d["rows"] == 40 and d["rows_exact"] and d["longest_exact"]
    assert d["longest_seen"] == max(lengths) and d["mean"] == pytest.approx(sum(lengths) / len(lengths), abs=0.1)
    assert d["over_limit_pct"] == 0 and d["longest"] == max(lengths)


def test_measure_finds_the_one_long_row_a_sample_would_miss(tmp_path):
    c = cfg(dataset={"name": jsonl(tmp_path / "d.jsonl", sft_rows(n=3000, long=400)), "split": "train"},
            algorithm={"type": "sft"})
    d = advisor.measure_data(c, sample=50, tokenizer=make_tokenizer())
    assert d["longest_seen"] > 400 and d["sampled"] == 50 and d["mean"] < 30


def test_measure_reports_rows_cut_by_max_length(tmp_path):
    rows = sft_rows(20) + [{"prompt": "hello", "completion": "good " * 100} for _ in range(20)]
    c = cfg(dataset={"name": jsonl(tmp_path / "d.jsonl", rows), "split": "train"}, algorithm={"type": "sft"},
            training__max_length=64)
    d = advisor.measure_data(c, tokenizer=make_tokenizer())
    assert d["over_limit_pct"] == 50 and d["longest"] == 64 and d["suggested_max_length"] == 128
    assert "50% of your examples are longer" in d["tips"][0]


def test_measure_preference_and_grpo_and_pretrain(tmp_path):
    tok = make_tokenizer()
    c = cfg(dataset={"name": jsonl(tmp_path / "p.jsonl", PAIRS * 5), "split": "train"}, algorithm={"type": "dpo"})
    assert advisor.measure_data(c, tokenizer=tok)["rows"] == 20
    c = cfg(dataset={"name": jsonl(tmp_path / "g.jsonl", [{"question": "what is two plus two", "answer": "four"}] * 5),
                     "split": "train"}, algorithm={"type": "grpo", "max_prompt_length": 16})
    d = advisor.measure_data(c, tokenizer=tok)
    assert d["kind"] == "grpo" and d["limit"] == 16
    texts = [{"text": "hello world what is the answer"}] * 30 + [{"text": "short"}] * 10      # short rows are dropped
    c = cfg(dataset={"name": jsonl(tmp_path / "t.jsonl", texts), "split": "train"})
    d = advisor.measure_data(c, tokenizer=tok)
    assert d["rows"] == 30 and "padded to 4,096" in d["tips"][0]


def test_wrong_columns_are_explained(tmp_path):
    c = cfg(dataset={"name": jsonl(tmp_path / "d.jsonl", [{"body": "hello world what is the answer"}]), "split": "train"})
    data, why = advisor.try_measure(c, tokenizer=make_tokenizer())
    assert data is None and "no 'text' column" in why and "body" in why
    data, why = advisor.try_measure(cfg(dataset={"name": "missing.jsonl", "split": "train"}))
    assert data is None and "not found" in why


def test_measuring_gives_up_in_time(monkeypatch):
    import time
    monkeypatch.setattr(advisor, "measure_data", lambda *a, **k: time.sleep(5))
    data, why = advisor.try_measure(cfg(), seconds=0.1)
    assert data is None and "longer than" in why


def test_nothing_to_measure():
    assert advisor.measure_data(cfg(model_type="custom_transformer")) is None
    assert advisor.measure_data(cfg(dataset={"name": "synthetic", "split": "train"})) is None


def test_short_examples_change_the_decision(tmp_path):
    """SFT pads to the longest row of the batch: data far below max_length needs less memory."""
    qwen = "Qwen/Qwen2.5-7B-Instruct"
    c = cfg(model_name=qwen, dataset={"name": jsonl(tmp_path / "d.jsonl", sft_rows()), "split": "train"},
            algorithm={"type": "sft"}, training__max_length=16384, training__batch_size=1, peft__enabled=True)
    c["peft"]["type"] = "lora"
    gpu = P.Cluster(gpu_mem_gb=24, gpus_per_node=1, gpu_name="4090")
    with pytest.raises(ValueError):
        advisor.resolve(c, gpu)
    out, p = advisor.resolve(c, gpu, advisor.measure_data(c, tokenizer=make_tokenizer()))
    assert out["strategy"] == "solo" and "Your data: 40 examples" in p["advice"]["data"]


def test_pretraining_memory_ignores_short_rows():
    w = P.apply_data(P.Workload(seq_len=2048), {"rows": 100, "longest": 50, "batch_len": 20, "total_tokens": 5000})
    assert w.seq_len == 2048 and w.avg_len is None and w.num_samples == 100


# ---- launching --------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("tiny"))
    tok = make_tokenizer()
    tiny_llama(len(tok)).save_pretrained(path)
    tok.save_pretrained(path)
    return path


def env(**extra):
    return dict(os.environ, HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="", **extra)


def test_cli_writes_a_concrete_config(tmp_path, tiny_model_dir):
    c = cfg(model_name=tiny_model_dir, dataset={"name": jsonl(tmp_path / "d.jsonl", sft_rows()), "split": "train"},
            algorithm={"type": "sft"})
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(c))
    r = subprocess.run([sys.executable, "advisor.py", "--config", str(tmp_path / "c.yaml"), "--gpu", "h100", "--gpus", "8",
                        "--write", str(tmp_path / "out.yaml")], cwd=ROOT, env=env(), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "Your data: 40 examples" in r.stdout and "Recommended: use one GPU" in r.stdout
    out = yaml.safe_load((tmp_path / "out.yaml").read_text())
    assert out["strategy"] == "solo" and out["dataset"] == c["dataset"]


def test_cli_refuses_an_unknown_gpu(tmp_path):
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg()))
    r = subprocess.run([sys.executable, "advisor.py", "--config", str(tmp_path / "c.yaml"), "--gpu", "rtx9999", "--no-data"],
                       cwd=ROOT, env=env(), capture_output=True, text=True, timeout=120)
    assert r.returncode == 1 and "--gpu-mem" in r.stderr


def launch(tmp_path, c, **extra):
    """scripts/launch.sh with a `python` / `torchrun` that record how training would be started."""
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(c))
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    log = tmp_path / "started.txt"
    save = f'cp "$CONFIG_PATH" "{tmp_path}/used.yaml"'
    (d / "python").write_text(f'#!/bin/sh\nif [ "$1" = train.py ]; then echo "python $@" >> {log}; {save}; '
                              f'else exec {sys.executable} "$@"; fi\n')
    (d / "torchrun").write_text(f'#!/bin/sh\necho "torchrun $@" >> {log}; {save}\n')
    for f in d.iterdir():
        f.chmod(0o755)
    e = env(CONFIG_PATH=str(path), **extra)
    e["PATH"] = f"{d}{os.pathsep}{e['PATH']}"
    r = subprocess.run(["bash", "scripts/launch.sh"], cwd=ROOT, env=e, capture_output=True, text=True, timeout=120)
    started = log.read_text() if log.exists() else ""
    used = yaml.safe_load((tmp_path / "used.yaml").read_text()) if (tmp_path / "used.yaml").exists() else None
    return r, started, used


def test_launch_sh_resolves_auto(tmp_path, tiny_model_dir):
    c = cfg(model_name=tiny_model_dir, dataset={"name": jsonl(tmp_path / "d.jsonl", sft_rows()), "split": "train"},
            algorithm={"type": "sft"})
    r, started, used = launch(tmp_path, c)
    assert r.returncode == 0, r.stderr
    assert started.startswith("python train.py") and used["strategy"] == "solo" and "Automatic strategy" in r.stdout


def test_launch_sh_writes_command_line_overrides_into_the_config(tmp_path):
    for c in (cfg(), cfg(strategy="fsdp", num_gpus=1)):
        r, started, used = launch(tmp_path, c, STRATEGY="ddp", NUM_GPUS="2")
        assert r.returncode == 0, r.stderr
        assert "torchrun --nproc_per_node=2" in started and (used["strategy"], used["num_gpus"]) == ("ddp", 2)


def test_launch_sh_runs_a_concrete_config_as_is(tmp_path):
    c = cfg(strategy="fsdp", num_gpus=3)
    r, started, used = launch(tmp_path, c)
    assert "torchrun --nproc_per_node=3" in started and used == c


def test_launch_sh_stops_when_the_config_is_missing(tmp_path):
    r = subprocess.run(["bash", "scripts/launch.sh"], cwd=ROOT, env=env(CONFIG_PATH=str(tmp_path / "no.yaml")),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "not found" in r.stderr


# ---- warning before a run ---------------------------------------------------

def test_preflight_warns_only_when_it_does_not_fit(monkeypatch):
    monkeypatch.setattr(advisor, "detect_cluster", lambda limit=None: copy.copy(H100x8))
    text = advisor.preflight(cfg(strategy="ddp", num_gpus=8), world=8)
    assert "may run out of GPU memory" in text and "FSDP" in text and "strategy: auto" in text
    assert advisor.preflight(cfg(strategy="fsdp", num_gpus=8), world=8) is None
    assert advisor.preflight(cfg(strategy="ddp", model_name="some/unknown-70b"), world=8) is None    # only a guess
    monkeypatch.setattr(advisor, "detect_cluster", lambda limit=None: None)
    assert advisor.preflight(cfg(strategy="ddp"), world=8) is None

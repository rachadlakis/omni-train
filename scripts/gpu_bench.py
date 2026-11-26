"""Measure real runs on this machine's GPUs and compare them with the planner.

    python scripts/gpu_bench.py                         # 1 GPU, then 2 and 4 if present: solo / DDP / FSDP
    python scripts/gpu_bench.py --gpus 1,2 --kinds full,lora
    python scripts/gpu_bench.py --dry-run               # the cases and the planner's predictions, no training

Each case is a short real train.py run (default: Qwen/Qwen2.5-0.5B-Instruct, every sequence at full
length, `--steps` optimizer steps). train.py's "Run summary" line gives the median seconds per step
(after the first 3) and the peak GPU memory (torch.cuda.max_memory_allocated, max over ranks); the
planner (omni_train.planner.evaluate_layout) gives its prediction for the same layout. The memory
comparison follows tests/test_planner.py: measured allocated peak vs planner total minus overhead.

Kinds (per-GPU batch fixed, so N GPUs train N times as many tokens per step):
    full   pretraining, full fine-tuning (fp32 weights + AdamW, bf16 autocast when the GPU has it)
    lora   pretraining with LoRA r=16 on every linear layer
    ppo    PPO with LoRA: sampling (generate) inside the step, then the update

Writes <out>/bench.md (table), bench.json, and one folder per case with its config and full log.
A failed or out-of-memory case is recorded and the next one runs. Never needs an HF token.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from omni_train import advisor, planner  # noqa: E402

SUMMARY = re.compile(r"Run summary \| steps: (\d+) \| sec/step: ([0-9.]+) \| train seconds: ([0-9.]+) "
                     r"\| peak alloc GB: ([0-9.]+) \| peak reserved GB: ([0-9.]+)")
WORDS = ("the model reads every token of this sentence and learns to predict the next one while the "
         "optimizer updates its weights after each batch of text that the data loader hands over").split()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def write_rows(path: str, kind: str, n: int) -> None:
    rng = random.Random(0)
    with open(path, "w") as f:
        for i in range(n):
            if kind == "ppo":
                a, b = rng.randint(1, 50), rng.randint(1, 50)
                row = {"prompt": f"What is {a} plus {b}? Answer with the number only.", "answer": str(a + b)}
            else:   # long enough that max_length cuts every row: full-length sequences, as the planner assumes
                row = {"text": " ".join(rng.choice(WORDS) for _ in range(900))}
            f.write(json.dumps(row) + "\n")


def make_config(kind: str, strategy: str, gpus: int, a, data: str, ckpt: str, bf16: bool) -> dict:
    lora = kind in {"lora", "ppo"}
    cfg = {
        "model_name": a.model, "model_type": "llm", "seed": 0,
        "dataset": {"name": data, "split": "train"},
        "strategy": strategy, "num_gpus": gpus, "checkpoint_dir": ckpt, "save": False,
        "training": {"epochs": 1, "batch_size": a.batch, "max_length": a.seq, "learning_rate": 1e-5,
                     "gradient_checkpointing": a.grad_ckpt, "warmup_steps": 0, "lr_scheduler": "constant",
                     "weight_decay": 0.0, "grad_clip": 1.0},
        "dist_parameters": {"mixed_precision": bf16, "param_dtype": "bfloat16" if bf16 else "float32",
                            "reduce_dtype": "float32", "output_dtype": "bfloat16" if bf16 else "float32",
                            "cast_forward_inputs": False, "distribute_api": "dcp_api"},
        "save_load": {"resume": False, "resume_path": "", "load_model_from_hf": True, "export_hf": False},
        "peft": {"enabled": lora, "type": "lora", "r": 16, "alpha": 32, "dropout": 0.0,
                 "target_modules": "all-linear", "bias": "none"},
        "quantization": {"enabled": False, "bits": 4, "quant_type": "nf4", "compute_dtype": "bfloat16",
                         "double_quant": True},
        "prefetch": {"explicit": False, "forward": 1, "backward": 1},
        "wandb": {"wandb_log_with_train": False, "wandb_entity": "", "wandb_project": "bench", "wandb_run_name": ""},
    }
    if kind == "ppo":
        cfg["algorithm"] = {"type": "ppo", "num_generations": 1, "max_prompt_length": 64,
                            "max_completion_length": a.completion, "rewards": ["accuracy"],
                            "ppo_epochs": 1, "num_minibatches": 1}
        cfg["training"]["max_length"] = 64 + a.completion
    return cfg


def predict(cfg: dict, strategy: str, gpus: int, cluster: planner.Cluster) -> dict:
    m = advisor.model_spec(cfg)
    w = planner.workload_from_config(cfg)
    o = planner.evaluate_layout(m, w, cluster, strategy, gpus)
    mem = o["memory_gb"]
    return {"gb": round(mem["total"] - mem["overhead"], 3), "total_gb": mem["total"], "fits": o["fits"],
            "sec_per_step": o["perf"].get("step_seconds"), "tokens_per_step": o["perf"].get("tokens_per_step")}


def run_case(cfg: dict, gpus: int, folder: str, timeout: int) -> dict:
    import yaml
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = visible.split(",") if visible else [str(i) for i in range(gpus)]
    env = dict(os.environ, CONFIG_PATH=path, PYTHONUNBUFFERED="1", HF_HUB_DISABLE_XET="1",
               CUDA_VISIBLE_DEVICES=",".join(ids[:gpus]) if visible != "" else "")
    if cfg["strategy"] == "solo":
        cmd = [sys.executable, "train.py"]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={gpus}",
               f"--master_port={free_port()}", "train.py"]
    start = time.time()
    try:
        res = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
        out, code = res.stdout + res.stderr, res.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        out, code = out + f"\n[timeout after {timeout} s]", -1
    out = out.replace("\r", "\n")
    with open(os.path.join(folder, "train.log"), "w") as f:
        f.write(out)
    hit = SUMMARY.search(out)
    status = "ok" if code == 0 and hit else ("OOM" if re.search(r"out of memory|OutOfMemoryError", out, re.I)
                                             else "timeout" if code == -1 else f"failed (exit {code})")
    result = {"status": status, "wall_seconds": round(time.time() - start, 1)}
    if hit:
        steps, sps, secs, alloc, reserved = hit.groups()
        result.update(steps=int(steps), sec_per_step=float(sps), train_seconds=float(secs),
                      peak_gb=float(alloc), reserved_gb=float(reserved))
    if status != "ok":      # the real error, not torchrun's "ChildFailedError" boilerplate
        lines = out.splitlines()
        errors = [l for l in lines if re.search(r"Error|❌|error:|Killed|timeout after", l)
                  and "elastic" not in l and "ChildFailedError" not in l]
        result["tail"] = "\n".join(dict.fromkeys(errors[-8:])) or "\n".join(lines[-15:])
    return result


def ratio(measured, predicted):
    return round(measured / predicted, 2) if measured and predicted else None


def write_report(rows: list[dict], out: str, meta: dict) -> str:
    solo = {(r["kind"]): r for r in rows if r["gpus"] == 1 and r.get("sec_per_step")}
    lines = [f"# GPU benchmark: {meta['gpu']} x {meta['gpus_present']}", "",
             f"{meta['date']} · model {meta['model']} · batch {meta['batch']} per GPU · seq {meta['seq']} · "
             f"bf16 {'on' if meta['bf16'] else 'off'} · grad ckpt {'on' if meta['grad_ckpt'] else 'off'} · "
             f"torch {meta['torch']}", "",
             "Measured = train.py's Run summary (median s/step after the first 3 steps; peak "
             "torch.cuda.max_memory_allocated, max over ranks). Predicted = planner.evaluate_layout "
             "(memory: total minus overhead, as in tests/test_planner.py). Ratio = measured / predicted.", "",
             "| case | status | s/step | predicted | ratio | tokens/s | vs 1 GPU | peak GB | predicted GB | ratio | planner says fits |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p, sps = r["predicted"], r.get("sec_per_step")
        tps = round(p["tokens_per_step"] / sps) if sps and p.get("tokens_per_step") and r["kind"] != "ppo" else None
        base = solo.get(r["kind"])
        speedup = None
        if sps and base and r["kind"] != "ppo":
            speedup = round((r["gpus"] * base["sec_per_step"]) / sps, 2)   # tokens per second relative to 1 GPU
        cell = lambda v: "" if v is None else f"{v:,}" if isinstance(v, int) else f"{v}"  # noqa: E731
        lines.append(f"| {r['case']} | {r['status']} | {cell(sps)} | {cell(p['sec_per_step'])} | "
                     f"{cell(ratio(sps, p['sec_per_step']))} | {cell(tps)} | {cell(speedup)}"
                     f"{'x' if speedup else ''} | {cell(r.get('peak_gb'))} | {cell(p['gb'])} | "
                     f"{cell(ratio(r.get('peak_gb'), p['gb']))} | {'yes' if p['fits'] else 'no'} |")
    lines += ["", "\"vs 1 GPU\" = tokens per second relative to the 1-GPU run of the same kind "
              "(N GPUs train N x the tokens per step; 2.0x on 2 GPUs is perfect scaling).", ""]
    failed = [r for r in rows if r["status"] not in {"ok", "dry run"}]
    for r in failed:
        lines += [f"## {r['case']}: {r['status']}", "", "```", r.get("tail", ""), "```", ""]
    path = os.path.join(out, "bench.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpus", default="", help="GPU counts to try, e.g. 1,2,4 (default: 1, 2, 4 up to what is present)")
    ap.add_argument("--kinds", default="full,lora,ppo", help="full, lora, ppo (comma-separated)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--batch", type=int, default=4, help="sequences per GPU per step (PPO: prompts)")
    ap.add_argument("--seq", type=int, default=512, help="tokens per sequence (full / lora)")
    ap.add_argument("--completion", type=int, default=128, help="PPO: tokens generated per answer")
    ap.add_argument("--steps", type=int, default=25, help="optimizer steps per case")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing on")
    ap.add_argument("--timeout", type=int, default=1200, help="seconds per case")
    ap.add_argument("--out", default="", help="output folder (default gpu_check_logs/bench_<time>)")
    ap.add_argument("--dry-run", action="store_true", help="print the cases and predictions, do not train")
    ap.add_argument("--cpu", action="store_true", help=argparse.SUPPRESS)   # plumbing test: CPU processes, gloo
    a = ap.parse_args(argv)
    a.grad_ckpt = bool(a.grad_ckpt)

    import torch
    present = torch.cuda.device_count()
    if a.cpu:
        present = max(int(x) for x in (a.gpus or "1").split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if not present and not a.dry_run:
        print("No CUDA GPU visible: nothing to measure (use --dry-run to see the cases).")
        return 1
    counts = [int(x) for x in a.gpus.split(",") if x] if a.gpus else [g for g in (1, 2, 4, 8) if g <= max(1, present)]
    skipped = [g for g in counts if present and g > present]
    counts = [g for g in counts if not present or g <= present]
    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]
    bf16 = bool(present) and not a.cpu and torch.cuda.is_bf16_supported()
    gpu_name = ("CPU (plumbing test)" if a.cpu else torch.cuda.get_device_name(0) if present
                else "no GPU (dry run: planner for an H100)")
    out = a.out or os.path.join(ROOT, "gpu_check_logs", "bench_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    meta = {"date": time.strftime("%Y-%m-%d %H:%M"), "gpu": gpu_name, "gpus_present": present, "model": a.model,
            "batch": a.batch, "seq": a.seq, "bf16": bf16, "grad_ckpt": a.grad_ckpt, "torch": torch.__version__}
    print(f"{gpu_name} x {present} | cases: GPUs {counts} x {kinds} | out: {out}", flush=True)
    if skipped:
        print(f"skipped GPU counts (not present): {skipped}")

    rows = []
    for g in counts:
        cluster = advisor.detect_cluster(limit=g) or planner.cluster_from_names("h100", None, g)
        for strategy in (["solo"] if g == 1 else ["ddp", "fsdp"]):
            for kind in kinds:
                case = f"{kind}_{strategy}_{g}gpu"
                folder = os.path.join(out, case)
                os.makedirs(folder, exist_ok=True)
                data = os.path.join(folder, "data.jsonl")
                write_rows(data, kind, a.steps * a.batch * g)
                cfg = make_config(kind, strategy, g, a, data, os.path.join(folder, "ckpt"), bf16)
                row = {"case": case, "kind": kind, "strategy": strategy, "gpus": g, "predicted": predict(cfg, strategy, g, cluster)}
                if a.dry_run:
                    row["status"] = "dry run"
                else:
                    print(f"\n=== {case}: predicted {row['predicted']['gb']} GB, {row['predicted']['sec_per_step']} s/step",
                          flush=True)
                    row.update(run_case(cfg, g, folder, a.timeout))
                    print(f"    {row['status']}: {row.get('sec_per_step')} s/step, peak {row.get('peak_gb')} GB "
                          f"({row['wall_seconds']} s)", flush=True)
                rows.append(row)
                with open(os.path.join(out, "bench.json"), "w") as f:
                    json.dump({"meta": meta, "rows": rows}, f, indent=2)
    report = write_report(rows, out, meta)
    print("\n" + open(report).read())
    print(f"report: {report}")
    return 0 if all(r["status"] in {"ok", "dry run"} for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())

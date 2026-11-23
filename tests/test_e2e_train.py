"""End-to-end: the real train.py, as a subprocess, on CPU, with no downloads.

A tiny random Llama + a word-level tokenizer are saved to a temp folder (so
`model_name` is a local path) and the data is a local .jsonl. Each test runs solo
(`python train.py`) or 2-process DDP over gloo (`torchrun`), then checks what a
user would: it exits cleanly, the loss goes down, DPO starts at ln 2, checkpoints
resume with their optimizer state, and the HF export loads and matches.

These caught-by-integration bugs are the point: unit tests passed while the
preference loss was noisy because dropout was on, and only a real run showed it.

Run with:  python -m pytest -m e2e      (~2 min on CPU; CI runs it)
FSDP is not covered here: FSDP2 has no CPU support in torch 2.6 (GPU smoke tests cover it).
"""

import glob
import json
import math
import os
import random
import re
import subprocess
import sys

import pytest
import torch
import yaml

from tests.post_training_helpers import PAIRS, WORDS, make_tokenizer, tiny_llama

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """Tiny Llama with dropout ON (attention_dropout), so the preference runs
    prove that train.py switches it off."""
    path = str(tmp_path_factory.mktemp("tiny_llama"))
    tok = make_tokenizer()
    model = tiny_llama(len(tok), seed=0)
    model.config.attention_dropout = 0.2
    model.config.pad_token_id = tok.pad_token_id
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return path


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return path


def _text_rows():
    rng = random.Random(0)
    phrases = [" ".join(WORDS[i:i + 8]) for i in range(0, len(WORDS) - 8, 2)]
    return [{"text": rng.choice(phrases) + " " + rng.choice(phrases)} for _ in range(64)]


def _cfg(model_dir, data_path, tmp, strategy="solo", nproc=1, **over):
    cfg = {
        "model_name": model_dir, "model_type": "llm",
        "dataset": {"name": str(data_path), "split": "train"},
        "strategy": strategy, "num_gpus": nproc, "checkpoint_dir": str(tmp / "ckpt"), "save": False,
        "training": {"epochs": 3, "batch_size": 4, "max_length": 32, "learning_rate": 5e-3,
                     "gradient_checkpointing": False, "warmup_steps": 0, "lr_scheduler": "constant",
                     "weight_decay": 0.0, "grad_clip": 1.0},
        "dist_parameters": {"mixed_precision": False, "param_dtype": "float32", "reduce_dtype": "float32",
                            "output_dtype": "float32", "cast_forward_inputs": False, "distribute_api": "dcp_api"},
        "save_load": {"resume": False, "resume_path": "", "load_model_from_hf": True,
                      "export_hf": False, "export_dtype": "float32"},
        "peft": {"enabled": False, "type": "lora", "r": 4, "alpha": 8, "dropout": 0.1,
                 "target_modules": "all-linear", "bias": "none"},
        "quantization": {"enabled": False, "bits": 4, "quant_type": "nf4", "compute_dtype": "bfloat16",
                         "double_quant": True},
        "prefetch": {"explicit": False, "forward": 1, "backward": 1},
        "wandb": {"wandb_log_with_train": False, "wandb_entity": "", "wandb_project": "t", "wandb_run_name": ""},
    }
    for key, value in over.items():
        node = cfg
        *parents, leaf = key.split("__")
        for p in parents:
            node = node[p]
        node[leaf] = value
    return cfg


def _train(cfg, tmp, tag):
    path = tmp / f"{tag}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    env = dict(os.environ, CONFIG_PATH=str(path), HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1", PYTHONPATH=ROOT)
    if cfg["strategy"] == "solo":
        cmd = [sys.executable, "train.py"]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(cfg["num_gpus"]),
               "--master_port", str(random.randint(20000, 29999)), "train.py"]
    res = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    out = res.stdout + res.stderr
    assert res.returncode == 0, f"{tag} failed:\n" + "\n".join(out.splitlines()[-60:])
    return out.replace("\r", "\n")


def _step_losses(out):
    return [float(m) for m in re.findall(r"step\s+\d+/\d+ \| loss: ([0-9.]+)", out)]


def _epoch_losses(out):
    return [float(m) for m in re.findall(r"Epoch \d+ complete \| avg loss: ([0-9.]+)", out)]


def _logits(model_dir, ids):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    with torch.no_grad():
        return m(ids).logits


IDS = torch.tensor([[1, 9, 10, 11, 12, 13, 14, 2]])


# ----------------------------------------------------------------------
def test_pretrain_solo_save_resume_export(tiny_model_dir, tmp_path):
    data = _write_jsonl(tmp_path / "text.jsonl", _text_rows())
    cfg = _cfg(tiny_model_dir, data, tmp_path, save=True, save_load__export_hf=True)
    out = _train(cfg, tmp_path, "pretrain")
    epochs = _epoch_losses(out)
    assert len(epochs) == 3 and epochs[-1] < epochs[0], epochs

    (ckpt,) = glob.glob(str(tmp_path / "ckpt" / "solo" / "*.pt"))
    saved = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert set(saved) == {"model_state_dict", "optimizer_state_dict"}
    assert saved["optimizer_state_dict"]["state"], "AdamW moments missing from the checkpoint"

    # The export is a plain HF model equal to the checkpointed weights.
    (export,) = glob.glob(str(tmp_path / "ckpt" / "hf" / "*"))
    assert json.load(open(os.path.join(export, "omni_train.json")))["export"] == "full"
    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32).eval()
    ref.load_state_dict(saved["model_state_dict"])
    with torch.no_grad():
        torch.testing.assert_close(_logits(export, IDS), ref(IDS).logits, atol=1e-5, rtol=1e-5)

    # Resume: optimizer moments come back, and training continues from the trained weights.
    resume = _cfg(tiny_model_dir, data, tmp_path, training__epochs=1,
                  save_load__resume=True, save_load__resume_path=ckpt)
    out2 = _train(resume, tmp_path, "resume")
    assert "Optimizer state restored" in out2
    assert _step_losses(out2)[0] < epochs[0]


def test_resume_old_weights_only_checkpoint(tiny_model_dir, tmp_path):
    """Checkpoints from before optimizer state was saved still resume (fresh AdamW)."""
    data = _write_jsonl(tmp_path / "text.jsonl", _text_rows())
    from transformers import AutoModelForCausalLM
    old = tmp_path / "old.pt"
    torch.save(AutoModelForCausalLM.from_pretrained(tiny_model_dir).state_dict(), old)
    cfg = _cfg(tiny_model_dir, data, tmp_path, training__epochs=1,
               save_load__resume=True, save_load__resume_path=str(old))
    assert "no optimizer state (older format)" in _train(cfg, tmp_path, "old")


def test_sft_lora_ddp_two_processes_export_merged(tiny_model_dir, tmp_path):
    rows = [{"prompt": p["prompt"], "completion": p["chosen"]} for p in PAIRS] * 8
    data = _write_jsonl(tmp_path / "sft.jsonl", rows)
    cfg = _cfg(tiny_model_dir, data, tmp_path, strategy="ddp", nproc=2, algorithm={"type": "sft"},
               peft__enabled=True, save_load__export_hf=True)
    out = _train(cfg, tmp_path, "sft_ddp")
    epochs = _epoch_losses(out)
    assert epochs[-1] < epochs[0], epochs
    (export,) = glob.glob(str(tmp_path / "ckpt" / "hf" / "*"))
    meta = json.load(open(os.path.join(export, "omni_train.json")))
    assert meta["export"] == "merged" and meta["algorithm"] == "sft"
    # Merged: no adapter files, loads as a plain model, and differs from the base (it learned).
    assert not glob.glob(os.path.join(export, "adapter*"))
    assert not torch.allclose(_logits(export, IDS), _logits(tiny_model_dir, IDS))


@pytest.mark.parametrize("strategy,nproc", [("solo", 1), ("ddp", 2)])
def test_dpo_starts_at_ln2_and_learns(tiny_model_dir, tmp_path, strategy, nproc):
    data = _write_jsonl(tmp_path / "pairs.jsonl", PAIRS * 8)
    cfg = _cfg(tiny_model_dir, data, tmp_path, strategy=strategy, nproc=nproc,
               algorithm={"type": "dpo", "max_prompt_length": 16}, peft__enabled=True,
               training__learning_rate=1e-2)
    out = _train(cfg, tmp_path, f"dpo_{strategy}")
    steps = _step_losses(out)
    assert steps[0] == pytest.approx(math.log(2), abs=1e-4), steps[:3]  # policy == reference
    assert "Dropout disabled" in out
    epochs = _epoch_losses(out)
    assert epochs[-1] < epochs[0] - 0.05, epochs

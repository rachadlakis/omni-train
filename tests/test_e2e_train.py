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


def _train(cfg, tmp, tag, torchrun=0):
    path = tmp / f"{tag}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    env = dict(os.environ, CONFIG_PATH=str(path), HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1", PYTHONPATH=ROOT)
    if cfg["strategy"] == "solo" or (cfg["strategy"] == "auto" and not torchrun):
        cmd = [sys.executable, "train.py"]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(torchrun or cfg["num_gpus"]),
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
@pytest.mark.parametrize("strategy,nproc", [("solo", 1), ("ddp", 2)])
def test_seed_decides_the_run(tiny_model_dir, tmp_path, strategy, nproc):
    """`seed` reaches the shuffle (solo DataLoader / DistributedSampler) and torch: the same seed gives
    the same run, another seed another one. It used to be ignored outside post-training and 5d."""
    data = _write_jsonl(tmp_path / "text.jsonl", _text_rows())
    runs = {}
    for tag, seed in (("a", 0), ("b", 0), ("c", 1)):
        cfg = _cfg(tiny_model_dir, data, tmp_path, strategy=strategy, nproc=nproc, training__epochs=1)
        cfg["seed"] = seed
        runs[tag] = _step_losses(_train(cfg, tmp_path, f"seed_{tag}"))
    assert runs["a"] == runs["b"], "same seed, different run"
    assert runs["a"] != runs["c"], "seed changed nothing"


@pytest.mark.parametrize("strategy,nproc", [("solo", 1), ("ddp", 2)])
def test_resume_keeps_tied_embeddings(tmp_path, strategy, nproc):
    """Most small LLMs (OPT, Qwen2.5-0.5B, Llama-3.2-1B) share the input embedding and the LM head.
    A resume must build the same tied model: untying it adds a parameter, so the saved AdamW state
    no longer matches (it was thrown away under DDP), and head and embedding then drift apart."""
    model_dir = str(tmp_path / "tied")
    tok = make_tokenizer()
    model = tiny_llama(len(tok), seed=0)
    model.config.tie_word_embeddings = True
    model.tie_weights()
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    data = _write_jsonl(tmp_path / "text.jsonl", _text_rows())
    out = _train(_cfg(model_dir, data, tmp_path, strategy=strategy, nproc=nproc, save=True), tmp_path, "tied")
    (ckpt,) = glob.glob(str(tmp_path / "ckpt" / strategy / "*.pt"))
    resume = _cfg(model_dir, data, tmp_path, strategy=strategy, nproc=nproc, training__epochs=1,
                  save_load__resume=True, save_load__resume_path=ckpt)
    out2 = _train(resume, tmp_path, "tied_resume")
    assert "Optimizer state restored" in out2, [l for l in out2.splitlines() if "ptimizer" in l]
    assert _step_losses(out2)[0] < _step_losses(out)[0]


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


def _auto_cfg(tiny_model_dir, tmp_path):
    rows = [{"prompt": p["prompt"], "completion": p["chosen"]} for p in PAIRS] * 8
    cfg = _cfg(tiny_model_dir, _write_jsonl(tmp_path / "sft.jsonl", rows), tmp_path, algorithm={"type": "sft"})
    cfg.update(strategy="auto", num_gpus="auto")
    return cfg


@pytest.mark.parametrize("nproc", [1, 2])
def test_strategy_auto_in_train_py(tiny_model_dir, tmp_path, nproc):
    """`python train.py` decides alone; under torchrun rank 0 decides for every rank."""
    out = _train(_auto_cfg(tiny_model_dir, tmp_path), tmp_path, f"auto{nproc}", torchrun=nproc if nproc > 1 else 0)
    chosen = "ddp on 2 processes" if nproc > 1 else "solo on 1 process"
    assert out.count(f"Automatic strategy: {chosen}") == 1, out[:2000]
    epochs = _epoch_losses(out)
    assert epochs[-1] < epochs[0], epochs


def test_strategy_auto_rejects_a_bad_config_on_every_rank(tiny_model_dir, tmp_path):
    cfg = _auto_cfg(tiny_model_dir, tmp_path)
    cfg["algorithm"] = {"type": "sft", "nonsense": 1}
    with pytest.raises(AssertionError, match="Configuration error"):
        _train(cfg, tmp_path, "auto_bad", torchrun=2)


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
def test_grpo_reward_goes_up(tiny_model_dir, tmp_path, strategy, nproc):
    rows = [{"prompt": p["prompt"], "answer": p["chosen"]} for p in PAIRS] * 8
    data = _write_jsonl(tmp_path / "grpo.jsonl", rows)
    cfg = _cfg(tiny_model_dir, data, tmp_path, strategy=strategy, nproc=nproc, training__epochs=8,
               training__batch_size=4 // nproc, training__learning_rate=1e-2,
               algorithm={"type": "grpo", "num_generations": 8, "max_prompt_length": 16,
                          "max_completion_length": 8, "rewards": ["contains"]})
    rewards = [float(r) for r in re.findall(r"\| reward: ([-0-9.]+)", _train(cfg, tmp_path, f"grpo_{strategy}"))]
    n = len(rewards) // 4
    first, last = sum(rewards[:n]) / n, sum(rewards[-n:]) / n
    assert last > first + 0.05, (first, last)


@pytest.mark.parametrize("strategy,nproc", [("solo", 1), ("ddp", 2)])
def test_ppo_reward_goes_up_resumes_and_exports(tiny_model_dir, tmp_path, strategy, nproc):
    rows = [{"prompt": p["prompt"], "answer": p["chosen"]} for p in PAIRS] * 8
    data = _write_jsonl(tmp_path / "ppo.jsonl", rows)
    algo = {"type": "ppo", "num_generations": 8, "max_prompt_length": 16, "max_completion_length": 8, "rewards": ["contains"],
            "ppo_epochs": 2, "num_minibatches": 2, "value_lr": 0.01}
    # lr 3e-3, not 5e-3: at 5e-3 some trajectories learn, then collapse by epoch 6 (CI's CPU hit one:
    # reward 0.08 -> 0.29 -> 0.02). Over 24 seed x thread-count runs, 3e-3 gains >= +0.115, none collapses.
    cfg = _cfg(tiny_model_dir, data, tmp_path, strategy=strategy, nproc=nproc, training__epochs=6, algorithm=algo,
               training__batch_size=4 // nproc, training__learning_rate=3e-3, save=True, save_load__export_hf=True)
    out = _train(cfg, tmp_path, f"ppo_{strategy}")
    rewards = [float(r) for r in re.findall(r"\| reward: ([-0-9.]+)", out)]
    assert len(rewards) == 6 * 8 * 2 * 2                        # epochs x batches x ppo_epochs x minibatches
    n = len(rewards) // 4
    first, last = sum(rewards[:n]) / n, sum(rewards[-n:]) / n
    assert last > first + 0.05, (first, last)

    (ckpt,) = glob.glob(str(tmp_path / "ckpt" / strategy / "*.pt"))
    assert os.path.exists(ckpt[:-3] + ".value_head")
    (export,) = glob.glob(str(tmp_path / "ckpt" / "hf" / "*"))   # a plain policy: no trace of the value head
    from safetensors import safe_open
    with safe_open(os.path.join(export, "model.safetensors"), "pt") as f:
        assert not any("value" in k for k in f.keys())
    assert torch.isfinite(_logits(export, IDS)).all()
    again = _cfg(tiny_model_dir, data, tmp_path, strategy=strategy, nproc=nproc, training__epochs=1, algorithm=algo,
                 training__batch_size=4 // nproc, training__learning_rate=1e-5,
                 save_load__resume=True, save_load__resume_path=ckpt)
    out = _train(again, tmp_path, f"ppo_resume_{strategy}")
    assert "Value head restored" in out and "Optimizer state restored" in out
    resumed = [float(r) for r in re.findall(r"\| reward: ([-0-9.]+)", out)]
    assert sum(resumed) / len(resumed) > first                   # it continues from the trained policy


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

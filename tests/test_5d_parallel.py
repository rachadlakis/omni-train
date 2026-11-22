"""strategy: 5d — config validation, HF parity, and layout equivalence.

Default suite (fast, in-process):
    * build_args accepts / rejects parallelism layouts
    * the 5D model loaded from tiny random Llama / Qwen2 / Mistral checkpoints gives
      the SAME logits as the Hugging Face model, and export_hf round-trips

`python -m pytest -m dist` (multi-process on CPU with gloo, a few minutes):
    * every layout (DP, ZeRO-1, TP, SP, CP, PP 1F1B/AFAB, EP, and mixes up to
      8 processes) reproduces the 1-process run: losses, global grad norm AND
      every parameter's gradient. Comparing gradients matters: AdamW cancels a
      per-tensor gradient scale, so a loss-only check misses scale bugs.
    * save -> resume (same layout) -> refused resume (other layout) -> HF export
"""

from __future__ import annotations

import copy
import glob
import json
import os
import random
import subprocess
import sys

import pytest
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(ROOT, "tests", "dist5d_worker.py")
sys.path.insert(0, ROOT)

from utils import build_args  # noqa: E402

STEPS = 3
BATCH = 2          # micro-batch size
MICRO_TOTAL = 4    # micro-batches x dp, so the global batch is 8 sequences
SEQ = 32


# ----------------------------------------------------------------- configs
def make_cfg(model_type="custom_transformer", model_name=None, par=None, moe=False,
             mixed_precision=False, gradient_checkpointing=False, nproc=1, **top):
    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    par = dict(par or {})
    model_par = par.get("tp", 1) * par.get("cp", 1) * par.get("pp", 1)
    dp = max(1, nproc // max(1, model_par))
    par.setdefault("micro_batches", MICRO_TOTAL // dp)
    cfg.update(model_type=model_type, model_name=model_name or "custom", strategy="5d",
               num_gpus=nproc, save=False, parallelism=par)
    cfg["dataset"] = {"name": "synthetic", "subset": None, "split": "train"}
    cfg["training"].update(epochs=1, batch_size=BATCH, max_length=SEQ, learning_rate=3e-3,
                           warmup_steps=0, lr_scheduler="constant", weight_decay=0.0, grad_clip=1.0,
                           gradient_checkpointing=gradient_checkpointing)
    cfg["dist_parameters"].update(mixed_precision=mixed_precision, param_dtype="bfloat16")
    cfg["custom_transformer_args"] = {"n_layers": 4, "vocab_size": 256, "max_seq_len": SEQ, "dim": 64,
                                      "n_heads": 4, "n_kv_heads": 2, "dropout_p": 0.0}
    if moe:
        cfg["custom_transformer_args"].update(num_experts=4, top_k=2)
    cfg.update(top)
    return cfg


# --------------------------------------------------------- build_args checks
class TestConfig:
    def test_defaults_and_auto_dp(self):
        args = build_args(make_cfg(par={"tp": 2, "pp": 2}, nproc=8))
        assert (args.par_dp, args.par_tp, args.par_pp, args.par_cp, args.par_ep) == (None, 2, 2, 1, 1)
        assert args.par_pp_schedule == "1f1b" and not args.par_sp and not args.par_zero1

    def test_full_5d_layout(self):
        cfg = make_cfg(par={"dp": 2, "tp": 2, "sp": True, "cp": 2, "pp": 2, "ep": 2, "zero1": True,
                            "micro_batches": 4}, moe=True, nproc=16)
        args = build_args(cfg)
        assert (args.par_dp, args.par_ep, args.par_sp, args.par_zero1) == (2, 2, True, True)

    @pytest.mark.parametrize("par,nproc,msg", [
        ({"tp": 3}, 4, "divisible by tp\\*cp\\*pp"),
        ({"dp": 3, "tp": 2}, 4, "must equal num_gpus"),
        ({"sp": True}, 1, "needs tp > 1"),
        ({"pp_schedule": "zigzag"}, 1, "1f1b or afab"),
        ({"tp": 0}, 1, "at least 1"),
        ({"tpp": 2}, 1, "Unknown parallelism option"),
        ({"cp": 3}, 3, "divisible by cp"),
        ({"ep": 2}, 2, "Mixture-of-Experts"),
    ])
    def test_rejects(self, par, nproc, msg):
        with pytest.raises(ValueError, match=msg):
            build_args(make_cfg(par=par, nproc=nproc))

    def test_ep_must_divide_dp(self):
        with pytest.raises(ValueError, match="must divide dp"):
            build_args(make_cfg(par={"ep": 4}, moe=True, nproc=2))

    def test_rejects_peft_and_other_model_types(self):
        cfg = make_cfg(model_type="llm", model_name="x")
        cfg["peft"]["enabled"] = True
        with pytest.raises(ValueError, match="PEFT"):
            build_args(cfg)
        with pytest.raises(ValueError, match="supports model_type"):
            build_args(make_cfg(model_type="vision", model_name="x"))

    def test_moe_needs_5d(self):
        cfg = make_cfg(moe=True)
        cfg["strategy"] = "ddp"
        with pytest.raises(ValueError, match="need strategy: 5d"):
            build_args(cfg)


# ------------------------------------------------------------ tiny HF models
def _save_tiny(path, kind):
    from transformers import AutoModelForCausalLM, LlamaConfig, MistralConfig, Qwen2Config
    common = dict(vocab_size=250, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
                  num_attention_heads=8, num_key_value_heads=4, max_position_embeddings=128)
    if kind == "llama":        # tied embeddings + GQA; vocab 250 needs padding for tp=4
        cfg = LlamaConfig(tie_word_embeddings=True, **common)
    elif kind == "llama3rope":  # llama3 rope scaling + attention biases (q/k/v/o)
        cfg = LlamaConfig(tie_word_embeddings=False, attention_bias=True, rope_parameters={
            "rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 32}, **common)
    elif kind == "qwen2":       # q/k/v biases, untied
        cfg = Qwen2Config(tie_word_embeddings=False, **common)
    else:                        # mistral, explicit head_dim != hidden/heads
        cfg = MistralConfig(tie_word_embeddings=False, sliding_window=None, head_dim=32, **common)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg)
    with torch.no_grad():  # random biases so the bias paths are really exercised
        for name, p in model.named_parameters():
            if name.endswith("bias"):
                p.normal_(0, 0.1)
    model.save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory):
    base = tmp_path_factory.mktemp("tiny_hf")
    return {k: _save_tiny(str(base / k), k) for k in ("llama", "llama3rope", "qwen2", "mistral")}


@pytest.fixture(scope="module")
def single_process_group():
    import torch.distributed as dist
    created = False
    if not dist.is_initialized():
        port = random.randint(30000, 39999)
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
        created = True
    yield
    if created:
        dist.destroy_process_group()


def _ours(model_dir):
    from parallelism import ParallelContext
    from parallelism.hf import config_from_hf, load_hf_weights
    from parallelism.model import PipelineStage
    stage = PipelineStage(config_from_hf(model_dir, SEQ), ParallelContext(), seeded=False)
    load_hf_weights(stage, model_dir)
    return stage


@pytest.mark.parametrize("kind", ["llama", "llama3rope", "qwen2", "mistral"])
def test_hf_parity(tiny_models, single_process_group, kind):
    from transformers import AutoModelForCausalLM
    ids = torch.randint(0, 250, (2, SEQ), generator=torch.Generator().manual_seed(1))
    hf = AutoModelForCausalLM.from_pretrained(tiny_models[kind], dtype=torch.float32).eval()
    with torch.no_grad():
        ref = hf(ids).logits
        got = _ours(tiny_models[kind]).eval()(ids)
    torch.testing.assert_close(got, ref, atol=2e-5, rtol=1e-4)


def test_hf_export_roundtrip(tiny_models, single_process_group, tmp_path):
    from transformers import AutoModelForCausalLM
    from parallelism.hf import export_hf
    src = tiny_models["qwen2"]
    stage = _ours(src)
    export_hf(stage, src, str(tmp_path / "hf"))
    a = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).state_dict()
    b = AutoModelForCausalLM.from_pretrained(str(tmp_path / "hf"), dtype=torch.float32).state_dict()
    assert a.keys() == b.keys()
    for k in a:  # export is bf16
        torch.testing.assert_close(b[k], a[k].to(torch.bfloat16).float(), atol=0, rtol=0)


# ------------------------------------------------------ multi-process layouts
def _run(cfg, nproc, tmp, tag, steps=STEPS, check=True):
    path = os.path.join(tmp, f"{tag}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    out = os.path.join(tmp, tag)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(nproc),
           "--master_port", str(random.randint(20000, 29999)), WORKER, path, out, str(steps)]
    env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=ROOT)
    res = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    if check and res.returncode != 0:
        raise AssertionError(f"{tag} failed:\n{res.stdout[-2000:]}\n{res.stderr[-4000:]}")
    if not check:
        return res
    with open(f"{out}.metrics.json") as f:
        metrics = json.load(f)
    grads = {}
    for g in glob.glob(f"{out}.grads.*.json"):
        with open(g) as f:
            grads.update(json.load(f))
    return metrics, grads


def _compare(ref, got, tol):
    (rm, rg), (gm, gg) = ref, got
    for key in ("losses", "grad_norms"):
        for a, b in zip(rm[key], gm[key]):
            assert abs(a - b) <= tol * max(1.0, abs(a)), f"{key}: ref {rm[key]} got {gm[key]}"
    assert rg.keys() == gg.keys()
    bad = {k: (gg[k] / rg[k]) ** 0.5 for k in rg if rg[k] and abs((gg[k] / rg[k]) ** 0.5 - 1) > 1e-3}
    assert not bad, f"gradient norm ratio (got/ref) off for: {bad}"


DENSE = [  # (name, nproc, parallelism)
    ("dp2", 2, {}),
    ("dp2_zero1", 2, {"zero1": True}),
    ("tp2", 2, {"tp": 2}),
    ("tp2_sp", 2, {"tp": 2, "sp": True}),
    ("cp2", 2, {"cp": 2}),
    ("pp2_1f1b", 2, {"pp": 2}),
    ("pp2_afab", 2, {"pp": 2, "pp_schedule": "afab"}),
    ("tp2sp_pp2", 4, {"tp": 2, "sp": True, "pp": 2}),
    ("cp2_tp2sp", 4, {"cp": 2, "tp": 2, "sp": True}),
    ("cp2_pp2", 4, {"cp": 2, "pp": 2}),
    ("dp2_pp2_zero1", 4, {"pp": 2, "zero1": True}),
    ("dp2_tp2sp_pp2", 8, {"tp": 2, "sp": True, "pp": 2}),
]
MOE = [
    ("dp2", 2, {}),
    ("ep2", 2, {"ep": 2}),
    ("ep2_tp2sp", 4, {"ep": 2, "tp": 2, "sp": True}),
    ("dp4_ep2", 4, {"ep": 2}),
    ("cp2_ep2", 4, {"cp": 2, "ep": 2}),
    ("ep2_pp2", 4, {"ep": 2, "pp": 2}),
    ("dp2_ep2_zero1", 2, {"ep": 2, "zero1": True}),
]
HF = [
    ("llama", "tp2_sp", 2, {"tp": 2, "sp": True}),
    ("llama", "pp2_tied", 2, {"pp": 2}),
    ("llama", "cp2", 2, {"cp": 2}),
    ("llama", "tp4_vocab_pad", 4, {"tp": 4}),
    ("llama", "dp2_pp2", 4, {"pp": 2}),
    ("llama", "tp2sp_pp2", 4, {"tp": 2, "sp": True, "pp": 2}),
    ("qwen2", "tp2_sp", 2, {"tp": 2, "sp": True}),
    ("qwen2", "tp2_pp2", 4, {"tp": 2, "pp": 2}),
    ("llama3rope", "cp2_tp2", 4, {"cp": 2, "tp": 2}),
]


@pytest.fixture(scope="module")
def refs(tmp_path_factory):
    """1-process reference runs, computed once per model."""
    tmp = str(tmp_path_factory.mktemp("refs"))
    cache = {}

    def get(key, cfg):
        if key not in cache:
            cache[key] = _run(cfg, 1, tmp, f"ref_{key}")
        return cache[key]
    return get


@pytest.mark.dist
@pytest.mark.parametrize("name,nproc,par", DENSE, ids=[c[0] for c in DENSE])
def test_dense_layout(refs, tmp_path, name, nproc, par):
    ref = refs("dense", make_cfg())
    _compare(ref, _run(make_cfg(par=par, nproc=nproc), nproc, str(tmp_path), name), tol=1e-4)


@pytest.mark.dist
@pytest.mark.parametrize("name,nproc,par", MOE, ids=[c[0] for c in MOE])
def test_moe_layout(refs, tmp_path, name, nproc, par):
    ref = refs("moe", make_cfg(moe=True))
    _compare(ref, _run(make_cfg(par=par, moe=True, nproc=nproc), nproc, str(tmp_path), name), tol=1e-4)


@pytest.mark.dist
@pytest.mark.parametrize("kind,name,nproc,par", HF, ids=[f"{c[0]}_{c[1]}" for c in HF])
def test_hf_layout(refs, tiny_models, tmp_path, kind, name, nproc, par):
    base = dict(model_type="llm", model_name=tiny_models[kind])
    ref = refs(kind, make_cfg(**base))
    _compare(ref, _run(make_cfg(par=par, nproc=nproc, **base), nproc, str(tmp_path), name), tol=1e-4)


@pytest.mark.dist
def test_gradient_checkpointing_matches(refs, tmp_path):
    ref = refs("dense", make_cfg())
    cfg = make_cfg(par={"cp": 2, "tp": 2, "sp": True}, nproc=4, gradient_checkpointing=True)
    _compare(ref, _run(cfg, 4, str(tmp_path), "ckpt"), tol=1e-4)


@pytest.mark.dist
def test_bf16_autocast_close(refs, tmp_path):
    ref = refs("dense_bf16", make_cfg(mixed_precision=True))
    m, _ = _run(make_cfg(par={"tp": 2, "sp": True, "pp": 2}, nproc=4, mixed_precision=True), 4,
                str(tmp_path), "bf16")
    for a, b in zip(ref[0]["losses"], m["losses"]):
        assert abs(a - b) < 2e-2 * a


@pytest.mark.dist
def test_save_resume_export(tiny_models, tmp_path):
    """train.py end to end: save (pp2 x tp2) -> exported HF weights equal the input
    weights (lr=0) -> resume with the same layout -> other layout refused."""
    from transformers import AutoModelForCausalLM
    src = tiny_models["llama"]
    ck = str(tmp_path / "ck")
    cfg = make_cfg(model_type="llm", model_name=src, par={"tp": 2, "pp": 2}, nproc=4,
                   save=True, checkpoint_dir=ck)
    cfg["training"]["learning_rate"] = 0.0

    def train(c, nproc, tag):
        path = str(tmp_path / f"{tag}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(c, f)
        env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", CONFIG_PATH=path)
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(nproc),
               "--master_port", str(random.randint(20000, 29999)), "train.py"]
        return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)

    res = train(cfg, 4, "save")
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    (run_dir,) = glob.glob(os.path.join(ck, "5d", "*"))
    assert len(glob.glob(os.path.join(run_dir, "rank_*.pt"))) == 4
    a = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32).state_dict()
    b = AutoModelForCausalLM.from_pretrained(os.path.join(run_dir, "hf"), dtype=torch.float32).state_dict()
    for k in a:
        torch.testing.assert_close(b[k], a[k].to(torch.bfloat16).float(), atol=0, rtol=0)

    resume = copy.deepcopy(cfg)
    resume["save"] = False
    resume["save_load"].update(resume=True, resume_path=run_dir)
    res = train(resume, 4, "resume")
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    assert "Resuming from" in res.stdout

    other = copy.deepcopy(resume)
    other["parallelism"] = {"tp": 4, "micro_batches": 4}
    res = train(other, 4, "resume_other")
    assert res.returncode != 0 and "same layout" in res.stdout + res.stderr

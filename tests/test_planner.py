"""Capacity planner (planner.py): parameter counts, memory model, strategy choice.

Pure Python, no GPU, no downloads.
"""
import math

import pytest
import yaml

import planner as P

LLAMA_8B = P.PRESETS["meta-llama/llama-3.1-8b"]
H100x8 = P.Cluster(gpu_mem_gb=80, gpus_per_node=8, nodes=1, gpu_name="h100")


@pytest.mark.parametrize("name, published", [
    ("meta-llama/Llama-3.1-8B", 8.03e9),
    ("meta-llama/Llama-3.1-70B-Instruct", 70.6e9),
    ("Qwen/Qwen2.5-0.5B-Instruct", 0.494e9),
    ("Qwen/Qwen2.5-7B-Instruct", 7.62e9),
    ("mistralai/Mistral-7B-v0.3", 7.25e9),
    ("mistralai/Mixtral-8x7B-v0.1", 46.7e9),
    ("facebook/opt-125m", 125e6),
    ("gpt2", 124e6),
])
def test_preset_param_counts_match_published(name, published):
    assert P.PRESETS[name.lower()].params == pytest.approx(published, rel=0.02)


def test_moe_active_params():
    mix = P.PRESETS["mistralai/mixtral-8x7b-v0.1"]
    assert mix.active_params == pytest.approx(12.9e9, rel=0.03)


def test_hf_config_matches_preset():
    cfg = {"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
           "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 128256,
           "tie_word_embeddings": False, "hidden_act": "silu"}
    spec = P.spec_from_hf_config(cfg, "x")
    assert spec.params == LLAMA_8B.params
    assert spec.five_d_ready and spec.gated_mlp


def test_vlm_text_config_is_used():
    cfg = {"model_type": "idefics3", "text_config": {"model_type": "llama", "hidden_size": 576, "num_hidden_layers": 30,
           "num_attention_heads": 9, "num_key_value_heads": 3, "intermediate_size": 1536, "vocab_size": 49280}}
    spec = P.spec_from_hf_config(cfg)
    assert spec.layers == 30 and spec.arch == "idefics3" and not spec.five_d_ready


@pytest.mark.parametrize("name, expected", [
    ("org/Some-7B-Instruct", 7e9), ("opt-350m", 350e6), ("Foo-1.5b", 1.5e9), ("no-size-here", None),
])
def test_params_from_name(name, expected):
    assert P.params_from_name(name) == (int(expected) if expected else None)


def test_custom_transformer_spec():
    spec = P.spec_from_custom({"n_layers": 4, "dim": 128, "n_heads": 4, "vocab_size": 256, "max_seq_len": 64})
    assert spec.arch == "custom" and spec.five_d_ready and spec.layers == 4
    moe = P.spec_from_custom({"n_layers": 8, "dim": 256, "n_heads": 8, "n_kv_heads": 4, "vocab_size": 1024,
                              "num_experts": 8, "top_k": 2})
    assert moe.num_experts == 8 and moe.active_params < moe.params


# ---------------------------------------------------------------------------
# Memory model
# ---------------------------------------------------------------------------

def test_full_finetune_is_16_bytes_per_param():
    mem = P.memory_per_gpu(LLAMA_8B, P.Workload(batch_size=1, seq_len=128), P.Layout("solo", 1))
    states = mem["weights"] + mem["gradients"] + mem["optimizer"]
    assert states == pytest.approx(16 * LLAMA_8B.params / P.GB, rel=1e-3)


def test_fsdp_shards_states():
    w = P.Workload(batch_size=1, seq_len=128)
    solo = P.memory_per_gpu(LLAMA_8B, w, P.Layout("solo", 1))
    fsdp = P.memory_per_gpu(LLAMA_8B, w, P.Layout("fsdp", 8, dp=8))
    assert fsdp["optimizer"] == pytest.approx(solo["optimizer"] / 8, rel=1e-3)
    assert fsdp["total"] < solo["total"] / 4


def test_lora_base_is_bf16_outside_fsdp_and_fp32_under_it():
    w = P.Workload(method="lora", batch_size=1, seq_len=128)
    solo = P.memory_per_gpu(LLAMA_8B, w, P.Layout("solo", 1))
    assert solo["weights"] == pytest.approx(2 * LLAMA_8B.params / P.GB, rel=0.02)
    fsdp2 = P.memory_per_gpu(LLAMA_8B, w, P.Layout("fsdp", 2, dp=2))
    assert fsdp2["weights"] > solo["weights"]          # fp32 / 2 shards > bf16 unsharded


def test_qlora_shrinks_weights():
    w = P.Workload(method="qlora", quant_bits=4, batch_size=1, seq_len=128)
    mem = P.memory_per_gpu(LLAMA_8B, w, P.Layout("solo", 1))
    assert mem["weights"] < 0.4 * 2 * LLAMA_8B.params / P.GB


def test_preference_doubles_rows_and_checkpointing_saves_memory():
    sft = P.Workload(algorithm="sft", batch_size=2, seq_len=1024)
    dpo = P.Workload(algorithm="dpo", batch_size=2, seq_len=1024)
    assert dpo.rows == 4 and sft.rows == 2
    lay = P.Layout("solo", 1)
    on = P.memory_per_gpu(LLAMA_8B, P.Workload(seq_len=4096, grad_ckpt=True), lay)
    off = P.memory_per_gpu(LLAMA_8B, P.Workload(seq_len=4096, grad_ckpt=False), lay)
    assert on["activations"] < off["activations"] / 5


def test_5d_splits_model_and_sequence():
    w = P.Workload(seq_len=8192)
    base = P.memory_per_gpu(LLAMA_8B, w, P.Layout("5d", 8, dp=8))
    tp = P.memory_per_gpu(LLAMA_8B, w, P.Layout("5d", 8, dp=4, tp=2, sp=True))
    cp = P.memory_per_gpu(LLAMA_8B, w, P.Layout("5d", 8, dp=4, cp=2))
    zero = P.memory_per_gpu(LLAMA_8B, w, P.Layout("5d", 8, dp=8, zero1=True))
    assert tp["weights"] < base["weights"] * 0.6
    assert cp["logits"] == pytest.approx(base["logits"] / 2, rel=1e-3)
    assert zero["optimizer"] == pytest.approx(base["optimizer"] / 8, rel=1e-3)


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def test_small_model_single_gpu():
    r = P.plan(P.PRESETS["qwen/qwen2.5-0.5b-instruct"], P.Workload(method="lora", batch_size=2),
               P.Cluster(gpu_mem_gb=24, gpus_per_node=1))
    assert r["verdict"] == "single_gpu" and not r["distributed_needed"]
    assert r["recommended"]["strategy"] == "solo"


def test_fits_one_gpu_but_more_available_suggests_ddp():
    r = P.plan(P.PRESETS["qwen/qwen2.5-0.5b-instruct"], P.Workload(method="lora"), H100x8)
    assert r["verdict"] == "distributed_optional" and r["recommended"]["strategy"] == "ddp"
    assert not r["distributed_needed"]


def test_8b_full_finetune_needs_fsdp():
    r = P.plan(LLAMA_8B, P.Workload(seq_len=4096), H100x8)
    assert r["verdict"] == "distributed_required" and r["distributed_needed"]
    assert r["recommended"]["strategy"] == "fsdp" and r["recommended"]["fits"]
    assert r["min_gpus"]["fsdp"] == 2
    assert P.needs_distributed(LLAMA_8B, P.Workload(seq_len=4096), H100x8)


def test_long_context_falls_through_to_5d():
    r = P.plan(LLAMA_8B, P.Workload(seq_len=32768), H100x8)
    rec = r["recommended"]
    assert rec["strategy"] == "5d" and rec["fits"]
    par = rec["parallelism"]
    assert par["dp"] * par["tp"] * par["pp"] * par["cp"] == 8 and par["cp"] > 1


def test_70b_on_one_node_is_infeasible_with_fixes():
    r = P.plan(P.PRESETS["meta-llama/llama-3.1-70b-instruct"], P.Workload(seq_len=8192), H100x8)
    assert r["verdict"] == "infeasible"
    assert r["min_gpus"]["fsdp"] > 8 and r["min_gpus"]["5d"] > 8
    changes = [s["change"] for s in r["suggestions"]]
    assert any("LoRA" in c for c in changes)


def test_single_small_gpu_gets_a_combined_recipe():
    r = P.plan(P.PRESETS["qwen/qwen2.5-7b-instruct"], P.Workload(batch_size=4, seq_len=2048),
               P.Cluster(gpu_mem_gb=24, gpus_per_node=1, gpu_name="4090"))
    assert r["verdict"] == "infeasible" and r["distributed_needed"]
    assert r["headline"].startswith("Needs distributed training: at least")
    assert any("+" in s["change"] for s in r["suggestions"])


def test_qlora_never_recommends_fsdp():
    r = P.plan(LLAMA_8B, P.Workload(method="qlora"), H100x8)
    assert all(o["strategy"] != "fsdp" for o in r["options"])
    assert r["min_gpus"]["fsdp"] is None


@pytest.mark.parametrize("algo", ["sft", "dpo", "simpo", "orpo"])
def test_post_training_is_never_5d(algo):
    r = P.plan(LLAMA_8B, P.Workload(algorithm=algo, seq_len=32768), H100x8)
    assert all(o["strategy"] != "5d" for o in r["options"])
    assert r["min_gpus"]["5d"] is None


def test_non_llama_architecture_excluded_from_5d():
    r = P.plan(P.PRESETS["facebook/opt-1.3b"], P.Workload(seq_len=2048), H100x8)
    assert all(o["strategy"] != "5d" for o in r["options"])


def test_five_d_layouts_satisfy_trainer_constraints():
    w = P.Workload(seq_len=4096)
    layouts = P.five_d_layouts(LLAMA_8B, w, 16, gpus_per_node=8)
    assert layouts
    for lay in layouts:
        assert lay.dp * lay.tp * lay.pp * lay.cp == 16
        assert lay.tp <= 8 and LLAMA_8B.kv_heads % lay.tp == 0
        assert w.seq_len % lay.cp == 0 and (w.seq_len // lay.cp) % lay.tp == 0
        assert not lay.sp or lay.tp > 1
        assert lay.dp % lay.ep == 0
        assert lay.pp == 1 or lay.micro_batches >= lay.pp


def test_moe_layout_uses_expert_parallelism():
    moe = P.spec_from_custom({"n_layers": 8, "dim": 256, "n_heads": 8, "n_kv_heads": 4, "vocab_size": 1024,
                              "max_seq_len": 512, "num_experts": 8, "top_k": 2})
    w = P.Workload(model_type="custom_transformer", seq_len=512)
    assert any(l.ep > 1 for l in P.five_d_layouts(moe, w, 16, 8))


def test_throughput_scales_with_gpus():
    w = P.Workload(seq_len=2048, num_samples=10_000)
    one = P.throughput(LLAMA_8B, w, P.Layout("fsdp", 8, dp=8), H100x8)
    two = P.throughput(LLAMA_8B, w, P.Layout("fsdp", 16, dp=16), P.Cluster(gpu_mem_gb=80, gpus_per_node=8, nodes=2))
    assert two["tokens_per_second"] > 1.5 * one["tokens_per_second"]
    assert one["total_steps"] == math.ceil(10_000 / 8)


@pytest.mark.parametrize("path", ["configs/llm_dpo_lora_single_gpu.yaml", "configs/llm_sft_lora_single_gpu.yaml",
                                  "configs/llm_5d_tp_pp_dp_8gpu.yaml", "configs/custom_5d_moe_16gpu.yaml"])
def test_workload_from_repo_configs(path):
    cfg = yaml.safe_load(open(path))
    w = P.workload_from_config(cfg)
    spec = P.resolve_spec(cfg.get("model_name", ""), w.model_type, cfg.get("custom_transformer_args"))
    r = P.plan(spec, w, P.Cluster(gpu_mem_gb=80, gpus_per_node=8, nodes=2))
    assert r["verdict"] in {"single_gpu", "distributed_optional", "distributed_required", "infeasible"}
    sel = P.evaluate_layout(spec, w, P.Cluster(gpu_mem_gb=80, gpus_per_node=8, nodes=2),
                            cfg["strategy"], cfg["num_gpus"], cfg.get("parallelism"))
    assert sel["per_gpu_gb"] > 0
    if "dpo" in path:
        assert w.algorithm == "dpo" and w.method == "lora" and w.rows == 2 * w.batch_size


def test_match_gpu():
    assert P.match_gpu("NVIDIA H100 80GB HBM3")[1] == 80
    assert P.match_gpu("NVIDIA A100-SXM4-40GB")[1] == 40
    assert P.match_gpu("NVIDIA A100-SXM4-80GB")[1] == 80
    assert P.match_gpu("NVIDIA GeForce RTX 4090")[1] == 24
    assert P.match_gpu("mystery") is None


def test_cli_runs(capsys):
    P.main(["--model", "meta-llama/Llama-3.1-8B", "--gpus", "8", "--gpu", "h100", "--seq", "4096"])
    out = capsys.readouterr().out
    assert "FSDP x8" in out and "Recommended" in out


def test_grpo_rows_and_sampling_phase():
    qwen = P.PRESETS["qwen/qwen2.5-0.5b-instruct"]
    w = P.Workload(algorithm="grpo", method="lora", batch_size=2, num_generations=8, seq_len=512, completion_len=320)
    assert w.rows == 16
    lay = P.Layout("solo", 1)
    small = P.memory_per_gpu(qwen, P.Workload(algorithm="grpo", method="lora", batch_size=2, num_generations=2,
                                              seq_len=512, completion_len=320), lay)
    assert P.memory_per_gpu(qwen, w, lay)["activations"] > small["activations"]
    c = P.Cluster(gpu_mem_gb=24, gpus_per_node=1, gpu_name="4090")
    perf = P.throughput(qwen, w, lay, c)
    assert perf["generation_seconds"] > 0
    assert perf["step_seconds"] > perf["generation_seconds"]


def test_ppo_workload_from_config_and_steps():
    cfg = yaml.safe_load(open("configs/llm_ppo_lora_single_gpu.yaml"))
    w = P.workload_from_config(cfg)
    assert (w.algorithm, w.num_generations, w.sampled_rows, w.rows, w.updates_per_rollout) == ("ppo", 1, 8, 4, 8)
    c, lay, spec = P.Cluster(gpu_mem_gb=6, gpus_per_node=1, gpu_name="4050"), P.Layout("solo", 1), P.resolve_spec(cfg["model_name"])
    w.num_samples = 80
    assert P.throughput(spec, w, lay, c)["total_steps"] == 10 * 8                  # 10 rollouts x 8 updates
    t = P.throughput(spec, w, lay, c)
    assert t["generation_seconds"] < t["step_seconds"]                             # its share of one gradient step
    r = P.plan(spec, w, c)
    assert r["verdict"] == "single_gpu" and any("PPO" in n for n in r["notes"])
    assert all(o["strategy"] != "5d" for o in r["options"])
    cfg["algorithm"]["reward_model"] = {"name": "Skywork/Skywork-Reward-V2-Qwen3-0.6B", "weight": 1.0}
    wr = P.workload_from_config(cfg)
    extra = P.memory_per_gpu(spec, wr, lay)["weights"] - P.memory_per_gpu(spec, w, lay)["weights"]
    assert extra == pytest.approx(2 * 0.6e9 / 2**30, rel=0.05)                     # frozen, bf16


def test_grpo_fsdp_sampling_is_not_faster_than_ddp():
    w = P.Workload(algorithm="grpo", method="lora", batch_size=2, seq_len=1024, completion_len=512)
    c = P.Cluster(gpu_mem_gb=80, gpus_per_node=8, gpu_name="h100")
    ddp = P.throughput(LLAMA_8B, w, P.Layout("ddp", 8, dp=8), c)
    fsdp = P.throughput(LLAMA_8B, w, P.Layout("fsdp", 8, dp=8), c)
    assert fsdp["generation_seconds"] >= ddp["generation_seconds"]


def test_grpo_workload_from_config():
    cfg = yaml.safe_load(open("configs/llm_grpo_lora_single_gpu.yaml"))
    w = P.workload_from_config(cfg)
    assert (w.algorithm, w.method, w.num_generations, w.completion_len) == ("grpo", "lora", 4, 320)
    assert w.seq_len == 192 + 320 and w.rows == 8
    r = P.plan(P.resolve_spec(cfg["model_name"]), w, P.Cluster(gpu_mem_gb=6, gpus_per_node=1, gpu_name="4050"))
    assert r["verdict"] == "single_gpu" and any("GRPO" in n for n in r["notes"])
    assert all(o["strategy"] != "5d" for o in r["options"])


# Peak allocated memory (torch.cuda.max_memory_allocated, GiB) of real train.py runs on an RTX 4050,
# every sequence at full length. The planner must stay close; DDP/FSDP/5d are not measured.
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
MEASURED = [
    ("facebook/opt-125m", dict(method="full", batch_size=4, seq_len=256), 2.46),
    (QWEN, dict(method="lora", algorithm="sft", batch_size=2, seq_len=512), 2.83),
    (QWEN, dict(method="lora", algorithm="sft", batch_size=2, seq_len=512, grad_ckpt=False), 4.49),
    (QWEN, dict(method="lora", algorithm="dpo", batch_size=2, seq_len=512), 3.45),
    (QWEN, dict(method="lora", algorithm="grpo", batch_size=2, seq_len=512, num_generations=4, completion_len=320), 3.82),
    (QWEN, dict(method="lora", algorithm="ppo", batch_size=8, seq_len=512, num_generations=1, completion_len=320,
                num_minibatches=2), 2.59),
]


@pytest.mark.parametrize("model, work, measured_gb", MEASURED)
def test_planner_matches_measured_peaks(model, work, measured_gb):
    mem = P.memory_per_gpu(P.PRESETS[model.lower()], P.Workload(**work), P.Layout("solo", 1))
    assert mem["total"] - mem["overhead"] == pytest.approx(measured_gb, rel=0.15)

# Omni-Train — Presenter Script
**Target: 5–10 minutes · 9 slides + live demo**

---

## Slide 0 — Title (0:00–0:30)

> "Hi everyone. Today I'm presenting **Omni-Train** — a modular framework I built to eliminate the pain of training large neural networks across multiple GPUs.
>
> The core idea is simple: you write one YAML config file, and the framework handles everything underneath — process spawning, gradient synchronization, memory sharding, checkpointing. You never touch a line of `torchrun` boilerplate.
>
> Let me start by explaining *why* this problem even needs a framework."

---

## Slide 1 — Why Distributed Training is Hard (0:30–1:30)

> "Distributed training fails in ways that are almost uniquely frustrating — no error messages, silent hangs, gradients that look fine but are quietly corrupted.
>
> Here are six traps I ran into personally while building this.
>
> **First** — process setup. `torchrun` injects `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` as environment variables. One wrong value and every process freezes with zero output. No stack trace. Nothing.
>
> **Second** — collective operations like `all_reduce` must be called by every GPU rank in the exact same order. If rank 0 calls `all_reduce` inside an `if rank == 0:` block, all other ranks wait forever. Global deadlock.
>
> **Third** — with FSDP, you must shard individual transformer layers *before* sharding the root model. Reverse the order and gradients are silently corrupted — training proceeds, loss looks plausible, but the model diverges.
>
> **Fourth** — only rank 0 should download checkpoints or initialize W&B. If all ranks try simultaneously, you get filesystem races and corrupt files.
>
> **Fifth** — naive `torch.save` on an FSDP model only captures rank 0's local shard. The checkpoint is unrestorable on any other configuration. You need DCP or DTensor APIs.
>
> **Sixth** — the mixed precision policy has three independent dtype fields. Mismatching `reduce_dtype` and `param_dtype` produces NaN gradients, with no error, just training divergence.
>
> These aren't hypothetical. Each one cost me real debugging time."

---

## Slide 2 — Why Distributed Training is Necessary (1:30–2:15)

> "So why bother at all? Because the math forces it.
>
> Take LLaMA-2 7B in a full fine-tune. You need 14 GB for weights, 14 GB for gradients, 56 GB for AdamW optimizer states, plus activation memory — roughly **96 GB total**. A single A100 80 GB cannot hold that.
>
> Scale up to LLaMA-2 70B and you need at minimum 8 A100s just for the weights.
>
> The table on the right shows how techniques stack up. A consumer RTX 4090 with 24 GB has literally one option for a 7B model — QLoRA at 4-bit quantization, which gets it down to about 3.5 GB for weights.
>
> FSDP is the key insight: it divides every weight, every gradient, and every optimizer state shard across N GPUs. Memory per GPU shrinks by exactly 1/N. It's not a trick — it's just division."

---

## Slide 3 — DDP (2:15–3:00)

> "Let's talk about the two main parallelism strategies, starting with DDP — Distributed Data Parallel.
>
> Each GPU gets a **full copy** of the model. The dataset is split across GPUs using a `DistributedSampler`, so each rank sees a disjoint batch. Forward and backward passes run independently in parallel, then at the end of each backward, an **all-reduce** averages the gradients across all ranks. Every rank then takes an identical optimizer step, so all model copies stay in sync.
>
> DDP is the simpler strategy. Checkpointing is easy — rank 0 just saves `model.module.state_dict()`, one portable file.
>
> The catch: memory does not shrink with more GPUs. Each GPU still holds the full model. You scale throughput, not capacity. Use DDP when your model fits on a single card."

---

## Slide 4 — FSDP2 (3:00–3:50)

> "FSDP — Fully Sharded Data Parallel — is different. The model is **sharded across every GPU**. Weights, gradients, and optimizer states are all split.
>
> The implementation follows a specific sequence. You first build the model on a `meta` device — no real memory allocated. You then call `fully_shard` on each transformer layer individually, and finally `fully_shard` on the root model. That order is critical, as I mentioned on slide 1.
>
> During the forward pass, each layer's full weights are **all-gathered** on demand, used for computation, and then discarded. During backward, a **reduce-scatter** keeps only the local gradient shard on each GPU.
>
> The mixed precision policy sets three independent dtypes: `bfloat16` for params and output, `float32` for gradient reduction — because stable all-reduce needs the extra precision.
>
> The result: a 7B model across 4 GPUs goes from 14 GB per card down to 3.5 GB. Memory shrinks linearly."

---

## Slide 5 — What This Project Does (3:50–4:40)

> "So what does Omni-Train actually give you?
>
> **Launching** is zero-boilerplate. `launch.sh` reads `strategy` and `num_gpus` from `config.yaml` and constructs the full `torchrun` command automatically. You switch from solo to DDP to FSDP by changing one YAML field.
>
> **Model coverage** is broad. The same framework handles LLMs, vision CNNs, object detection, vision-language models, and embedding models — over 20 architectures, all driven by `model_type` in config.
>
> **PEFT** is built in. Set `peft.enabled: true`, pick a LoRA rank, and the framework wraps the model with adapters — no code changes. QLoRA works the same way with `quantization.enabled: true`.
>
> **The web UI** at `localhost:8787` gives you a visual YAML editor, one-click launch, and a live log stream in the browser — no CLI knowledge needed.
>
> **Checkpointing** uses both DCP and DTensor backends, both safe for FSDP2, and both recoverable from any rank configuration."

---

## Slide 6 — The Hard Things (4:40–5:15)

> "This slide summarizes what the framework absorbs so you don't have to think about it.
>
> NCCL collective routing is handled automatically — the right backend, the right device mapping, collectives always called on all ranks.
>
> Layer sharding order for FSDP is determined by `get_model_layers()`, which probes the architecture and wraps in the correct sequence.
>
> Prefetching — overlapping the all-gather of the next layer's weights with the current layer's compute — is configured through a single YAML field: `prefetch.forward: 2`.
>
> Rank-0 guards around HuggingFace downloads and W&B initialization are in place with barriers. And the mixed precision policy is validated before training starts.
>
> One more thing worth calling out — **ZeRO stages**. ZeRO, Zero Redundancy Optimizer, is the framework behind why memory savings are even possible in the first place. Stage 1 shards only the optimizer states. Stage 2 also shards gradients. Stage 3 shards weights too — which is exactly what FSDP implements. You need to understand which stage is active because it determines your memory budget, your communication volume, and why a config that looks right might still OOM.
>
> The entire goal is: you configure what you want to train, not how distributed training works internally."

---

## Slide 7 — Roadmap (5:15–5:45)

> "Two things on the roadmap.
>
> **Multi-dimensional parallelism.** Currently the framework supports 1D sharding — FSDP or DDP. I'm working on 2D: Tensor Parallelism combined with FSDP, where individual weight matrices are sharded across one axis and the data across another. This unlocks models in the hundreds-of-billions range. The plan goes all the way to 6D — adding context, expert, and pipeline parallelism for mixture-of-experts architectures.
>
> **SLURM multi-node.** `launch_slurm.py` will auto-generate job scripts from config, injecting `MASTER_ADDR`, `MASTER_PORT`, and `WORLD_SIZE` automatically. It will also handle checkpoint-on-signal and auto-requeue on preemption."

---

## Slide 8 — Live Demo (5:45–8:30)

> "Let me show it running."

**[Demo steps — do these live]**

1. Open `config.yaml` in the editor and briefly show the key fields:
   ```
   model_name: facebook/opt-125m
   strategy:   fsdp
   num_gpus:   2
   ```
   > "This is the only file you touch. Model, strategy, GPU count — that's it."

2. Launch from the terminal:
   ```bash
   CONFIG_PATH=configs/llm_lora_ddp.yaml bash launch.sh
   ```
   > "One command. The framework reads the config, constructs `torchrun --nproc_per_node=2`, and starts two processes."

3. Point out the terminal output:
   > "You can see rank 0 and rank 1 initializing. Rank 0 downloads the model, hits the barrier, then both ranks load their shards. After that, training begins and you get per-rank loss output."

4. *(Optional — if time allows)* Switch to the Web UI:
   ```bash
   bash ui/launch_ui.sh
   # open http://127.0.0.1:8787
   ```
   > "For anyone who prefers not to use the terminal — open the UI, pick your model and strategy from dropdowns, toggle LoRA on, and hit Launch. Same result."

---

## Slide 9 — Thank You (8:30–10:00)

> "To wrap up — Omni-Train is one config file, any model, any parallelism strategy.
>
> The distributed complexity — NCCL collectives, FSDP sharding order, mixed precision policies, DCP checkpointing — is handled once, correctly, inside the framework. You focus on the experiment, not the infrastructure.
>
> The code is on GitHub. There's a beginner guide and a full technical reference in the docs. Happy to answer questions."

---

## Timing Reference

| Slide | Topic | Time |
|-------|-------|------|
| 0 | Title | 0:00–0:30 |
| 1 | Why Hard | 0:30–1:30 |
| 2 | Why Necessary | 1:30–2:15 |
| 3 | DDP | 2:15–3:00 |
| 4 | FSDP2 | 3:00–3:50 |
| 5 | What It Does | 3:50–4:40 |
| 6 | Hard Things Saved | 4:40–5:15 |
| 7 | Roadmap | 5:15–5:45 |
| 8 | Demo | 5:45–8:30 |
| 9 | Thank You + Q&A | 8:30–10:00 |

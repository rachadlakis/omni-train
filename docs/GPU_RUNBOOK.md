# GPU runbook: checking OMNI-Train on 2 or 4 rented GPUs

Everything that GitHub's CPU runners cannot check runs on a rented machine with one command,
`bash scripts/gpu_check.sh`. This page covers renting the machine, getting the code onto it, running the
check, and bringing the results back. Plan on **about 1.5 to 2.5 hours of machine time** for 2 GPUs.
4 GPUs take a little longer, because the 4-GPU cases also run.

## What it checks

| Step | What | Why it needs real GPUs |
|---|---|---|
| 1. environment | GPU names, memory, NVLink / PCIe topology, driver, torch | for the record |
| 2. NCCL | all-reduce, all-gather, reduce-scatter, broadcast: correctness and bandwidth | NCCL only runs on GPUs |
| 3. unit tests | the normal `pytest` suite | checks the install |
| 4. `-m dist` on NCCL | 5D layouts and SFT / DPO / GRPO / **PPO** under DDP **and FSDP** give the same loss and every parameter's gradient as one process | FSDP2 cannot run on CPU (torch 2.6), so the FSDP cases have **never run** |
| 5. smoke | real `train.py` runs: DDP / FSDP on 2 and 4 GPUs, save + resume, QLoRA, LoRA, bf16, HF export, GRPO and PPO on DDP and FSDP, 5D (TP, PP, CP, ZeRO-1, MoE), vision / detection / VLM | multi-GPU training has never run on real hardware |
| 6. `strategy: auto` | what the advisor picks for these GPUs, then `launch.sh` runs it | it reads the real GPUs |
| 7. benchmark | seconds per step and peak memory of solo / DDP / FSDP on 1, 2, 4 GPUs (full, LoRA, PPO), next to the planner's prediction | the planner's speed numbers have never been measured beyond one RTX 4050 |

A case that needs more GPUs than the machine has is **skipped**, not failed.

## 1. Rent the machine

- **GPUs:** 2 or 4 of the **same** card, with **at least 24 GB each**. Good choices: RTX 4090, RTX A6000,
  L4, A10G, L40S, A100 or H100. 16 GB cards (T4, V100-16GB) work, but the full-fine-tune benchmark
  cases will run out of memory. That is recorded and is not a failure.
- **One machine** with all its GPUs (not several machines).
- **Image:** any Ubuntu image with an NVIDIA driver for **CUDA 12.4 or newer**. A "PyTorch" template
  (RunPod, Lambda, vast.ai) is fine: the script installs its own torch 2.6.0 in a `.venv`.
- **Disk:** 40 GB or more. **Shared memory (`/dev/shm`):** 2 GB or more. On RunPod and vast.ai Docker
  machines, NCCL fails with a small `/dev/shm`. Check it with `df -h /dev/shm` once you are logged in.
- Note the **SSH command** the provider shows, for example `ssh root@1.2.3.4 -p 22022 -i ~/.ssh/id_ed25519`.
  Below, `HOST`, `PORT` and `KEY` stand for its parts.

## 2. Get the code onto the machine (from WSL, on your laptop)

The repository is private, so copy it over instead of putting a GitHub token on the rented machine:

```bash
cd ~/projects/omni-train
git archive --format=tar.gz -o /tmp/omni-train.tgz main      # what is committed on main
scp -P PORT -i KEY /tmp/omni-train.tgz root@HOST:~/
ssh -p PORT -i KEY root@HOST
```

On the rented machine:

```bash
mkdir -p ~/omni-train && tar -xzf ~/omni-train.tgz -C ~/omni-train && cd ~/omni-train
nvidia-smi                 # every GPU listed; "CUDA Version" 12.4 or higher
df -h /dev/shm             # 2 GB or more
```

No `.env` is needed. The default cases only download public models (facebook/opt-125m,
Qwen/Qwen2.5-0.5B-Instruct and a few small vision models) and small public datasets.

## 3. Run the check (on the rented machine)

Run it inside `tmux`, so it keeps going if your SSH connection drops:

```bash
tmux new -s check
bash scripts/gpu_check.sh --setup 2>&1 | tee ~/gpu_check_console.txt
```

- `--setup` creates `.venv` and installs torch 2.6.0 (cu124) plus `requirements.txt`. It takes about
  5 to 10 minutes and is needed only the first time. To run the check again later:
  `bash scripts/gpu_check.sh`.
- To leave it running, press **Ctrl-b**, then **d**. To come back: `ssh ...`, then `tmux attach -t check`.
- It prints one line per step as it goes, and a summary at the end:

```
================ GPU check summary (2 GPU(s), quick) ================
  nccl                               PASS             12s
  unit tests                         PASS             61s
  dist on GPU (NCCL)                 PASS             410s
  smoke (curated)                    PASS             2350s
  strategy: auto (advisor)           PASS             20s
  strategy: auto (launch.sh)         PASS             95s
  benchmark (vs planner)             PASS             1100s
logs: gpu_check_logs/20261127_101500
one file with everything: gpu_check_logs/20261127_101500.tar.gz
```

**If a step fails, let it finish.** Every step runs even when an earlier one failed, and its log has
the details. A FAIL is exactly what we are paying to find.

Options:

| Command | When |
|---|---|
| `bash scripts/gpu_check.sh --no-bench` | to save about 20 minutes: no benchmark |
| `bash scripts/gpu_check.sh --full` | if you have hours to spare: adds the whole 1-GPU smoke grid (~300 cases) |
| `python scripts/gpu_bench.py --kinds lora,ppo` | the benchmark alone, without full fine-tuning (16 GB cards) |

## 4. Bring the results back (from WSL, on your laptop)

```bash
mkdir -p ~/projects/omni-train/gpu_check_logs
scp -P PORT -i KEY "root@HOST:~/omni-train/gpu_check_logs/*.tar.gz" ~/projects/omni-train/gpu_check_logs/
scp -P PORT -i KEY root@HOST:~/gpu_check_console.txt ~/projects/omni-train/gpu_check_logs/
```

Then **stop or delete the machine** in the provider's console. Most providers keep billing a
stopped machine for its disk.

Tell Claude "the GPU logs are in gpu_check_logs". It will read them, fix what failed, and use the
benchmark to correct the planner and the README's claims. `gpu_check_logs/` is git-ignored.

What is inside the `.tar.gz`:

```
<timestamp>/
├── summary.txt                 the table above
├── environment.txt             GPUs, topology, driver, versions
├── nccl.log, unit_tests.log, dist_on_GPU_NCCL.log, smoke_curated.log, strategy_auto_*.log
└── bench/
    ├── bench.md                measured vs predicted, one row per case
    ├── bench.json              the same, for scripts
    └── <case>/train.log        full log of each benchmark run
```

## 5. If something goes wrong

| Symptom | What to do |
|---|---|
| `nvidia-smi` shows "CUDA Version" below 12.4 | pick another machine or image (the driver is too old for torch's cu124 wheels) |
| The NCCL step hangs, or "NCCL error ... unhandled system error" | `export NCCL_P2P_DISABLE=1`, then run again (common on consumer GPUs and on vast.ai). If `/dev/shm` is small: `export NCCL_SHM_DISABLE=1` |
| "Address already in use" | a previous run is still alive: `pkill -f torch.distributed.run; pkill -f train.py` |
| Out of memory in the benchmark | expected for full fine-tuning on 16 GB cards: it is recorded, and the next case runs |
| A Hub download is very slow or rate-limited | put `HF_TOKEN=hf_...` in `~/omni-train/.env` (a read token is enough), then run again |
| SSH dropped | `ssh ...`, then `tmux attach -t check`: it kept running |

## 6. Running pieces by hand (optional)

Activate the environment first: `source .venv/bin/activate`. Replace `2` with your GPU count.

```bash
torchrun --nproc_per_node=2 scripts/nccl_check.py                          # NCCL
OMNI_DIST_DEVICE=cuda python -m pytest -m dist -q -rs                      # gradients vs 1 process
OMNI_DIST_DEVICE=cuda python -m pytest -m dist -q -rs -k "fsdp and ppo"    # just FSDP + PPO
python -m pytest -m smoke -q -rs -k "2gpu"                                 # every 2-GPU train.py run
python -m pytest -m smoke -q -rs -k "post_ppo or post_grpo"                # RL on every strategy
python -m pytest -m smoke -q -rs -k "5d_"                                  # 5D through train.py
python -m omni_train.advisor --config scripts/gpu_check_auto.yaml          # what auto would pick here
python scripts/gpu_bench.py --gpus 1,2                                     # benchmark vs planner
python scripts/gpu_bench.py --dry-run                                      # the cases, no training
```

A real run on all GPUs, for example SFT with LoRA, with FSDP across 2 GPUs:

```bash
CONFIG_PATH=configs/llm_sft_lora_single_gpu.yaml STRATEGY=fsdp NUM_GPUS=2 bash scripts/launch.sh
```

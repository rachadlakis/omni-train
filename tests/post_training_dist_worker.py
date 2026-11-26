"""Worker for tests/test_post_training_dist.py. Launched on CPU/gloo (ddp) or CUDA/nccl (fsdp):

    python -m torch.distributed.run --nproc_per_node N tests/post_training_dist_worker.py \
        STRATEGY(ddp|fsdp) LORA(0|1) ALGO(dpo|simpo|orpo|grpo|grpo_dapo|grpo_dr|grpo_kl|ppo|ppo_kl) OUT

Runs the real objective (reference precompute included) on a tiny Llama wrapped
the way train.py wraps it, and writes OUT.pt on rank 0 with:
(GRPO: sampling is random, so the rollout is built from FIXED completions and rewards
with the objective's own build_rollout / attach_reference / loss; G = 2 per prompt.)
  grads: full (unsharded) gradient of every trainable parameter after step 1
  losses: global mean loss of steps 1 and 2 (step 2 runs after one SGD update,
          so it uses the cached reference with policy != reference)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from datasets import Dataset  # noqa: E402

from omni_train.post_training.data import collate_preference, make_loader, tokenize_preference  # noqa: E402
from omni_train.post_training.grpo import GRPOObjective, build_rollout, completion_mask_from  # noqa: E402
from omni_train.post_training.objectives import PreferenceObjective  # noqa: E402
from omni_train.post_training.ppo import PPOObjective  # noqa: E402
from tests.post_training_helpers import PAIRS, make_tokenizer, tiny_llama  # noqa: E402


class Args:
    algo_beta, algo_dpo_loss, algo_label_smoothing, algo_simpo_gamma = 0.1, "sigmoid", 0.0, 0.5


def main():
    strategy, lora, algo, out = sys.argv[1], sys.argv[2] == "1", sys.argv[3], sys.argv[4]
    # FSDP2 needs CUDA (torch 2.6 has no CPU stream support); DDP runs on CPU/gloo.
    if strategy == "fsdp" or os.environ.get("OMNI_DIST_DEVICE", "cpu").lower() == "cuda":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        device = torch.device("cuda", local_rank)
    else:
        dist.init_process_group("gloo")
        device = torch.device("cpu")
    rank, world = dist.get_rank(), dist.get_world_size()

    tok = make_tokenizer()
    model = tiny_llama(len(tok), seed=0).to(device)
    if lora:
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(1)
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules="all-linear"))
        # B starts at 0; make it non-zero so every LoRA weight gets a gradient.
        with torch.no_grad():
            gen = torch.Generator().manual_seed(2)  # same on every rank
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    p.copy_(torch.randn(p.shape, generator=gen) * 0.02)

    if strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
        named = lambda: model.module.named_parameters()  # noqa: E731
    else:
        from torch.distributed._composable.fsdp import fully_shard
        from omni_train.strategies.common import get_model_layers
        for stack, _ in get_model_layers(model):  # same per-layer sharding as apply_fsdp
            for layer in stack:
                fully_shard(layer)
        fully_shard(model)
        named = model.named_parameters

    if algo.startswith(("grpo", "ppo")):
        return run_rl(model, named, tok, algo, rank, world, device, out)

    args = Args()
    args.algorithm = algo
    obj = PreferenceObjective(args)
    rows = [tokenize_preference(p, tok, 64, 32) | {"idx": i} for i, p in enumerate(PAIRS)]
    ds = Dataset.from_list(rows)
    collate = lambda r: collate_preference(r, tok.pad_token_id)  # noqa: E731
    # before the no-grad reference pass, as in train.py (see run_rl)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.5)
    if obj.needs_reference:
        obj.precompute_reference(model, ds, collate, rank, world, batch_size=1, device=device)

    loader = make_loader(ds, collate, rank, world, batch_size=len(PAIRS) // world, shuffle=False)
    batch = next(iter(loader))
    model.train()
    losses, grads = [], {}
    for step in range(2):
        opt.zero_grad()
        loss, _ = obj(model, batch, device)
        loss.backward()
        if step == 0:
            for n, p in named():
                if p.requires_grad:
                    g = p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad
                    grads[n] = g.detach().cpu().clone()
        opt.step()
        global_loss = loss.detach().clone()
        dist.all_reduce(global_loss)
        losses.append(global_loss.item() / world)
    if rank == 0:
        torch.save({"grads": grads, "losses": losses}, out + ".pt")
    dist.barrier()
    dist.destroy_process_group()


def lora_run(named):
    return any("lora_" in n for n, _ in named())


def run_rl(model, named, tok, algo, rank, world, device, out):
    import types
    loss_type = {"grpo_dapo": "dapo", "grpo_dr": "dr_grpo"}.get(algo, "grpo")
    args = types.SimpleNamespace(
        algo_num_generations=2, algo_max_completion_length=8, algo_temperature=0.7, algo_top_p=1.0,
        algo_top_k=0, algo_beta=0.1 if algo == "grpo_kl" else 0.0, algo_epsilon=0.2, algo_epsilon_high=0.28,
        algo_loss_type=loss_type, algo_scale_rewards=True, algo_rewards=["contains"], strategy="ddp",
        algo_reward_model=None, algo_ppo_epochs=1, algo_num_minibatches=1, algo_gamma=0.99, algo_gae_lambda=0.95,
        algo_vf_coef=0.5, algo_value_clip=0.2, algo_value_lr=1e-2, algo_eos_penalty=0.0)
    ppo = algo.startswith("ppo")
    if ppo:
        args.algo_beta = 0.1 if algo == "ppo_kl" else 0.0
    obj = (PPOObjective if ppo else GRPOObjective)(args, tok)
    obj.prepare(model)
    # Before any forward, as in train.py: after a no-grad forward FSDP's root still holds full copies
    # of its weights, and an optimizer built then would update those instead of the real shards.
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.5)
    scaler = torch.amp.GradScaler(device.type, enabled=False)
    if ppo:   # a non-trivial value head, so that values, GAE and its gradient all matter
        with torch.no_grad():
            g = torch.Generator().manual_seed(5)
            obj.value_head.weight.copy_(torch.randn(obj.value_head.weight.shape, generator=g) * 0.3)
    # global batch: every PAIRS prompt with 2 completions (chosen, rejected) of DIFFERENT
    # lengths (so the token normalisations differ), rewards 1 / 0.2 / ...
    prompts, comps, rewards = [], [], []
    eos = tok.eos_token_id
    for i, p in enumerate(PAIRS):
        pid = tok(tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False,
                                          add_generation_prompt=True), add_special_tokens=False)["input_ids"]
        for j, ans in enumerate((p["chosen"], p["rejected"])):
            prompts.append(pid)
            comps.append(tok(ans, add_special_tokens=False)["input_ids"] + [eos])
            rewards.append(1.0 if j == 0 else 0.2 * i)
    per_rank = len(prompts) // world
    sl = slice(rank * per_rank, (rank + 1) * per_rank)   # whole groups (G = 2) per rank
    prompts, comps, rewards = prompts[sl], comps[sl], rewards[sl]
    P = max(len(x) for x in prompts)
    C = max(len(x) for x in comps)
    pid = torch.tensor([[tok.pad_token_id] * (P - len(x)) + x for x in prompts], device=device)
    pmask = torch.tensor([[0] * (P - len(x)) + [1] * len(x) for x in prompts], device=device)
    cid = torch.tensor([x + [eos] * (C - len(x)) for x in comps], device=device)
    cmask = completion_mask_from(cid, [eos])
    if ppo:
        roll = obj.build(model, pid, pmask, cid, cmask, torch.tensor(rewards))
    else:
        roll = build_rollout(pid, pmask, cid, cmask, torch.tensor(rewards, device=device), 2)
    if args.algo_beta > 0:
        if not ppo:
            obj.attach_reference(model, roll)
        with torch.no_grad():  # move the policy away from the reference so KL matters
            for n, p in named():
                if "lora_A" in n:
                    p.mul_(1.5)
    model.train()
    losses, grads = [], {}
    for step in range(2):
        opt.zero_grad()
        if ppo:
            obj.value_opt.zero_grad()
        loss, _ = obj.loss(model, roll)
        loss.backward()
        if ppo:
            obj.step_extra(scaler, 0.0)    # the real update: averages the value head's gradient, then steps
        if step == 0 and ppo:
            for n, p in obj.value_head.named_parameters():
                grads[f"value_head.{n}" + ("_lora_" if lora_run(named) else "")] = p.grad.detach().cpu().clone()
        if step == 0:
            for n, p in named():
                if p.requires_grad:
                    g = p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad
                    grads[n] = g.detach().cpu().clone()
        opt.step()
        # the global loss of DAPO is the SUM of per-rank losses (each divides by global/world)
        global_loss = loss.detach().clone()
        dist.all_reduce(global_loss)
        losses.append(global_loss.item() / world)
    if rank == 0:
        torch.save({"grads": grads, "losses": losses}, out + ".pt")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

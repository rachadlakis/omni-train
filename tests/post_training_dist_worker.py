"""Worker for tests/test_post_training_dist.py. Launched on CPU/gloo (ddp) or CUDA/nccl (fsdp):

    python -m torch.distributed.run --nproc_per_node N tests/post_training_dist_worker.py \
        STRATEGY(ddp|fsdp) LORA(0|1) ALGO(dpo|simpo|orpo) OUT

Runs the real objective (reference precompute included) on a tiny Llama wrapped
the way train.py wraps it, and writes OUT.pt on rank 0 with:
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

from post_training.data import collate_preference, make_loader, tokenize_preference  # noqa: E402
from post_training.objectives import PreferenceObjective  # noqa: E402
from tests.post_training_helpers import PAIRS, make_tokenizer, tiny_llama  # noqa: E402


class Args:
    algo_beta, algo_dpo_loss, algo_label_smoothing, algo_simpo_gamma = 0.1, "sigmoid", 0.0, 0.5


def main():
    strategy, lora, algo, out = sys.argv[1], sys.argv[2] == "1", sys.argv[3], sys.argv[4]
    # FSDP2 needs CUDA (torch 2.6 has no CPU stream support); DDP runs on CPU/gloo.
    if strategy == "fsdp":
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
        model = DDP(model)
        named = lambda: model.module.named_parameters()  # noqa: E731
    else:
        from torch.distributed._composable.fsdp import fully_shard
        from distributed_utils import get_model_layers
        for stack, _ in get_model_layers(model):  # same per-layer sharding as apply_fsdp
            for layer in stack:
                fully_shard(layer)
        fully_shard(model)
        named = model.named_parameters

    args = Args()
    args.algorithm = algo
    obj = PreferenceObjective(args)
    rows = [tokenize_preference(p, tok, 64, 32) | {"idx": i} for i, p in enumerate(PAIRS)]
    ds = Dataset.from_list(rows)
    collate = lambda r: collate_preference(r, tok.pad_token_id)  # noqa: E731
    if obj.needs_reference:
        obj.precompute_reference(model, ds, collate, rank, world, batch_size=1, device=device)

    loader = make_loader(ds, collate, rank, world, batch_size=len(PAIRS) // world, shuffle=False)
    batch = next(iter(loader))
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.5)
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


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

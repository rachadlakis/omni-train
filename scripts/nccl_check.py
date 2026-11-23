"""Check that every GPU can talk to every other one over NCCL, and how fast.

    torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) scripts/nccl_check.py

Checks the four collectives training uses (all-reduce: DDP gradients; all-gather /
reduce-scatter: FSDP; broadcast: checkpoints, ZeRO-1) for correct results, then
measures all-reduce bus bandwidth and rank 0 <-> rank 1 point-to-point bandwidth
(what pipeline parallelism uses). Exits 1 if any result is wrong.

Rough bus-bandwidth expectations: NVLink 100-400+ GB/s, PCIe 4.0 ~10-25 GB/s,
PCIe without P2P (through host memory) a few GB/s.
"""

import os
import sys
import time

import torch
import torch.distributed as dist


def timed(fn, iters=10):
    fn()  # warm-up
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", local_rank)
    ok = True

    def check(name, cond):
        nonlocal ok
        flag = torch.tensor([1.0 if cond else 0.0], device=dev)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if rank == 0:
            print(f"  {'OK  ' if flag.item() == 1 else 'FAIL'} {name}", flush=True)
        ok = ok and flag.item() == 1

    if rank == 0:
        print(f"NCCL {'.'.join(map(str, torch.cuda.nccl.version()))} | world size {world}", flush=True)
    names = [None] * world
    dist.all_gather_object(names, f"rank {rank}: {torch.cuda.get_device_name(dev)} ({socket_host()})")
    if rank == 0:
        print("\n".join("  " + n for n in names), flush=True)

    # Correctness
    x = torch.full((1024,), float(rank + 1), device=dev)
    dist.all_reduce(x)
    check("all_reduce", torch.allclose(x, torch.full_like(x, world * (world + 1) / 2)))
    parts = [torch.empty(4, device=dev) for _ in range(world)]
    dist.all_gather(parts, torch.full((4,), float(rank), device=dev))
    check("all_gather", all(torch.all(p == r) for r, p in enumerate(parts)))
    out = torch.empty(4, device=dev)
    dist.reduce_scatter_tensor(out, torch.arange(4 * world, device=dev, dtype=torch.float32))
    check("reduce_scatter", torch.allclose(out, world * torch.arange(4 * rank, 4 * rank + 4, device=dev,
                                                                     dtype=torch.float32)))
    b = torch.full((8,), float(rank), device=dev)
    dist.broadcast(b, src=world - 1)
    check("broadcast", torch.all(b == world - 1).item())

    # Bandwidth
    size_mb = int(os.environ.get("NCCL_CHECK_MB", "256"))
    big = torch.ones(size_mb * 2**20 // 4, device=dev)
    t = timed(lambda: dist.all_reduce(big))
    bus = 2 * (world - 1) / world * big.numel() * 4 / t / 1e9 if world > 1 else 0.0
    if rank == 0 and world > 1:
        print(f"  all_reduce {size_mb} MB: {t * 1e3:.1f} ms | bus bandwidth {bus:.1f} GB/s", flush=True)
    elif rank == 0:
        print("  bandwidth: n/a with 1 GPU", flush=True)
    if world > 1:
        def p2p():
            if rank == 0:
                dist.send(big, 1)
            elif rank == 1:
                dist.recv(big, 0)
        t = timed(p2p)
        if rank == 0:
            print(f"  send rank 0 -> 1 {size_mb} MB: {big.numel() * 4 / t / 1e9:.1f} GB/s", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("NCCL check passed" if ok else "NCCL check FAILED", flush=True)
    sys.exit(0 if ok else 1)


def socket_host():
    import socket
    return socket.gethostname()


if __name__ == "__main__":
    main()

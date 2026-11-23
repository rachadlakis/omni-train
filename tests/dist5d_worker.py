"""Worker for tests/test_5d_parallel.py. Launched with torchrun on CPU (gloo):

    python -m torch.distributed.run --nproc_per_node N tests/dist5d_worker.py CONFIG OUT STEPS

Runs STEPS optimizer steps of the real strategy: 5d code path and writes:
  OUT.metrics.json   losses + global grad norms (from the logging rank)
  OUT.grads.<r>.json squared grad norm of every parameter after step 0's sync,
                     keyed by layout-independent name (TP shards summed over TP)
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import yaml  # noqa: E402

from parallelism.trainer import FiveDRun  # noqa: E402
from utils import build_args  # noqa: E402


def main():
    config, out, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(config) as f:
        args = build_args(yaml.safe_load(f))
    run = FiveDRun(args).setup()
    pc = run.pc
    losses, norms = [], []
    for step in range(steps):
        loss = run.forward_backward(0, step)
        if step == 0:
            grads = {}
            for name, p in run.stage.named_parameters():
                if getattr(p, "tied_copy", False):
                    continue
                sq = p.grad.double().pow(2).sum()
                if getattr(p, "tp_sharded", False):
                    dist.all_reduce(sq, group=pc.tp_group)
                grads[run.stage.global_name(name)] = sq.item()
            if pc.tp_rank == 0 and pc.cp_rank == 0 and pc.dp_rank < pc.ep_size:
                with open(f"{out}.grads.{pc.rank}.json", "w") as f:
                    json.dump(grads, f)
        norms.append(run.clip_and_step(args.learning_rate))
        losses.append(loss)
    if pc.is_logging_rank:
        with open(f"{out}.metrics.json", "w") as f:
            json.dump({"losses": losses, "grad_norms": norms}, f)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

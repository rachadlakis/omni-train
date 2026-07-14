## dist_kit/data.py ##
##
## Text dataloader (llm / seq2seq / encoder) extracted from OMNI-Train's
## data.py. The vision / detection / VLM pipelines stayed in the app — they
## are model-family specific; bring your own dataloader for those and the
## kit's wrapping/checkpointing still applies.

import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from .dist_setup import print_on_rank_0, print_on_all_ranks


def get_text_dataloader(args, tokenizer, rank, world_size):
    """HF text dataset → tokenized DataLoader with DistributedSampler.

    Expects the dataset to expose a `text` column (wikitext / alpaca-style).
    Uses args.dataset_name / dataset_subset / dataset_split / batch_size /
    max_length from DistConfig.
    """
    if args.model_type not in ("llm", "seq2seq", "encoder"):
        raise ValueError(
            f"dist_kit's dataloader only covers text model types (llm/seq2seq/encoder); "
            f"got model_type='{args.model_type}'. Supply your own DataLoader for "
            "vision/vlm/detection — see OMNI-Train's data.py for reference collates."
        )

    from datasets import load_dataset

    subset = args.dataset_subset or None
    print_on_rank_0(rank, f"Loading dataset: {args.dataset_name} / {subset} | split: {args.dataset_split}", "📂")
    # Pass HF_TOKEN so gated/rate-limited datasets resolve without hanging on
    # unauthenticated Hub requests.
    dataset = load_dataset(
        args.dataset_name, subset, split=args.dataset_split, streaming=False,
        token=os.getenv("HF_TOKEN"),
    )  # type: ignore
    print_on_rank_0(rank, f"Raw dataset size: {len(dataset)} samples")

    max_length = args.max_length

    def tokenize(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 20)
    print_on_rank_0(rank, f"After filtering short texts: {len(dataset)} samples")

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format(type="torch")
    print_on_rank_0(rank, f"Tokenization done. Max length: {max_length} tokens")

    use_distributed_sampler = dist.is_available() and dist.is_initialized()
    sampler = None
    shuffle = True
    if use_distributed_sampler:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,  # drop last to ensure even batches across ranks
        )
        shuffle = False

    # Worker processes fork with a copy of the parent's address space; on
    # memory-tight machines (e.g. WSL2) set DIST_KIT_NUM_WORKERS=0 to avoid
    # the OOM killer taking down ranks during multi-process CPU runs.
    num_workers = int(os.environ.get(
        "DIST_KIT_NUM_WORKERS",
        max(1, min(4, (os.cpu_count() or 1) // world_size)),
    ))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,  # ensure all batches are the same size across ranks
    )
    print_on_rank_0(rank, f"DataLoader ready | batches per rank: {len(dataloader)} | batch size: {args.batch_size}")
    if use_distributed_sampler:
        print_on_all_ranks(rank, f"Distributed Sampler configured | replicas={world_size} | sampler_rank={rank} | num_workers={num_workers}", "🧺")
    else:
        print_on_rank_0(rank, f"Distributed Sampler NOT configured | single-process shuffle=True | num_workers={num_workers}", "🧺")
    return dataloader

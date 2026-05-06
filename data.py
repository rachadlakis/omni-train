## data.py ##

import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from distributed_utils import print_on_rank_0, print_on_all_ranks

def get_dataloader(dataset, dataset_full_name, dataset_split, tokenizer, rank, world_size, batch_size, max_length, model_type="llm", **kwargs):
    try:
        # ---------------------------------------------------------------
        # Synthetic dataloader for the custom transformer (no HF dataset)
        # ---------------------------------------------------------------
        if model_type == "custom_transformer":
            vocab_size = int(kwargs.get("vocab_size") or 8)
            seq_len    = max(2, int(max_length))
            n_samples  = 2000  # fixed pool of synthetic sequences
            print_on_rank_0(rank, f"Generating synthetic data | samples={n_samples} seq_len={seq_len} vocab_size={vocab_size}", "🎲")
            data = torch.randint(0, vocab_size, (n_samples, seq_len))
            from torch.utils.data import TensorDataset
            tensor_dataset = TensorDataset(data)

            use_distributed_sampler = dist.is_available() and dist.is_initialized()
            sampler = None
            shuffle = True
            if use_distributed_sampler:
                sampler = DistributedSampler(
                    tensor_dataset, 
                    num_replicas=world_size, 
                    rank=rank, 
                    shuffle=True,
                    drop_last=True,  # drop last to ensure even batches across ranks
                )
                shuffle = False

            num_workers = 0  # TensorDataset doesn't need workers
            dataloader = DataLoader(
                tensor_dataset,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,  # ensure all batches are the same size across ranks
            )
            print_on_rank_0(rank, f"Synthetic DataLoader ready | batches per rank: {len(dataloader)} | batch size: {batch_size}")
            return dataloader

        print_on_rank_0(rank, f"Loading dataset: {dataset} / {dataset_full_name} | split: {dataset_split}", "📂")
        dataset = load_dataset(dataset, dataset_full_name, split=dataset_split, streaming=False) # type: ignore
        print_on_rank_0(rank, f"Raw dataset size: {len(dataset)} samples")

        if tokenizer is not None and model_type in {"vision", "yolo"}:
            img_col = "img" if "img" in dataset.column_names else "image"
            label_col = "label" if "label" in dataset.column_names else "labels"

            def process_images(example):
                inputs = tokenizer(images=example[img_col], return_tensors="pt")
                return {
                    "pixel_values": inputs["pixel_values"].squeeze(0),
                    "labels": example[label_col],
                }

            orig_cols = dataset.column_names
            num_proc = min(4, os.cpu_count() or 1)
            dataset = dataset.map(process_images, remove_columns=orig_cols, num_proc=num_proc)
            dataset.set_format(type="torch")
            print_on_rank_0(rank, f"Image processing done | img_col={img_col} | label_col={label_col}")

        elif tokenizer is not None:
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
        else:
            print_on_rank_0(rank, "Tokenizer not provided — skipping tokenization. Ensure the dataset has the required input fields.", "⚠️")
            dataset.set_format(type="torch")
        
        use_distributed_sampler = dist.is_available() and dist.is_initialized() # and world_size > 1
        sampler = None
        shuffle = True
        if use_distributed_sampler:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
            shuffle = False

        # num_workers = min(4, os.cpu_count() // world_size) if os.cpu_count() else 2
        num_workers = max(1, min(4, (os.cpu_count() or 1) // world_size))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
            )
        print_on_rank_0(rank, f"DataLoader ready | batches per rank: {len(dataloader)} | batch size: {batch_size}")
        if use_distributed_sampler:
            print_on_all_ranks(rank, f"Distributed Sampler configured | replicas={world_size} | sampler_rank={rank} | num_workers={num_workers}", "🧺")
        else:
            print_on_rank_0(rank, f"Distributed Sampler NOT configured | single-process shuffle=True | num_workers={num_workers}", "🧺")
        return dataloader
    except Exception as e:
        print_on_rank_0(rank, f"❌ Failed to create dataloader: {e}", "❌")
        raise

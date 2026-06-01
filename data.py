## data.py ##

import os
import functools
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from distributed_utils import print_on_rank_0, print_on_all_ranks


# ----------------------------------------------------------------------
# Object-detection collate
# ----------------------------------------------------------------------
# Detection datasets (cppe-5, COCO-style HF datasets) store one PIL image
# plus an `objects` column holding parallel lists of bbox / category / area.
# `AutoModelForObjectDetection` (DETR, YOLOS, …) expects `labels` as a list of
# per-image dicts with `class_labels` and normalized `boxes`. The HF image
# processor produces exactly that when given COCO-format `annotations`, so we
# build the annotation dicts here and let the processor do the geometry.
# This runs as a DataLoader collate_fn (not dataset.map) because the per-image
# label dicts are variable-length and cannot be default-collated into a tensor.
def _detection_collate(examples, processor):
    images, annotations = [], []
    for ex in examples:
        img = ex["image"]
        if hasattr(img, "convert") and getattr(img, "mode", "RGB") != "RGB":
            img = img.convert("RGB")
        images.append(img)

        objs = ex["objects"]
        image_id = int(ex.get("image_id", 0))
        bboxes = objs["bbox"]
        cats = objs["category"]
        areas = objs.get("area")
        ids = objs.get("id")
        crowds = objs.get("iscrowd")
        anns = [
            {
                "image_id": image_id,
                "category_id": int(cats[i]),
                "bbox": [float(x) for x in bboxes[i]],  # COCO [x, y, w, h]
                "area": float(areas[i]) if areas is not None else 0.0,
                "iscrowd": int(crowds[i]) if crowds is not None else 0,
                "id": int(ids[i]) if ids is not None else i,
            }
            for i in range(len(bboxes))
        ]
        annotations.append({"image_id": image_id, "annotations": anns})

    enc = processor(images=images, annotations=annotations, return_tensors="pt")
    batch = {"pixel_values": enc["pixel_values"], "labels": enc["labels"]}
    if "pixel_mask" in enc:  # DETR-family processors pad and emit a mask; YOLOS does not
        batch["pixel_mask"] = enc["pixel_mask"]
    return batch


# ----------------------------------------------------------------------
# Vision-language (image + text) collate
# ----------------------------------------------------------------------
# VLM datasets pair one image with a caption. We frame each row as a short
# instruction ("Describe the image." → caption) using the processor's chat
# template, then let the processor expand the <image> placeholder into the
# exact number of image tokens the vision tower will produce. Labels are the
# input ids with pad and image-placeholder tokens masked to -100 so the loss is
# computed only over the assistant's text. Done as a collate_fn (not map) so the
# processor handles image-token expansion and padding per batch.
def _vlm_image_token_id(processor):
    tok = processor.tokenizer
    image_token = getattr(processor, "image_token", None)
    if isinstance(image_token, str):
        return tok.convert_tokens_to_ids(image_token)
    if image_token is not None and hasattr(image_token, "content"):
        return tok.convert_tokens_to_ids(image_token.content)
    return None


def _vlm_collate(examples, processor, image_col, text_col):
    texts, images = [], []
    for ex in examples:
        img = ex[image_col]
        if hasattr(img, "convert") and getattr(img, "mode", "RGB") != "RGB":
            img = img.convert("RGB")
        caption = str(ex[text_col])
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the image."},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": caption}]},
        ]
        texts.append(processor.apply_chat_template(messages, tokenize=False))
        images.append([img])  # nested list: one list of images per text sample

    enc = processor(text=texts, images=images, return_tensors="pt", padding=True)

    labels = enc["input_ids"].clone()
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100
    image_token_id = _vlm_image_token_id(processor)
    if image_token_id is not None:
        labels[labels == image_token_id] = -100
    enc["labels"] = labels
    return dict(enc)

def get_dataloader(dataset, dataset_full_name, dataset_split, tokenizer, rank, world_size, batch_size, max_length, model_type="llm", **kwargs):
    try:
        # ---------------------------------------------------------------
        # Synthetic dataloader for the custom transformer (no HF dataset)
        # ---------------------------------------------------------------
        if model_type == "custom_transformer":
            vocab_size = int(kwargs.get("vocab_size") or 8)
            # Toy model has no tokenizer; couple seq_len to the model's own
            # max_seq_len so it can't exceed the positional embedding table.
            custom_max_seq_len = kwargs.get("custom_max_seq_len")
            seq_len_src = custom_max_seq_len if custom_max_seq_len else max_length
            seq_len    = max(2, int(seq_len_src))
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
        # Pass HF_TOKEN so gated/rate-limited datasets resolve without hanging on
        # unauthenticated Hub requests (the token is loaded by train.py via dotenv).
        dataset = load_dataset(
            dataset, dataset_full_name, split=dataset_split, streaming=False,
            token=os.getenv("HF_TOKEN"),
        )  # type: ignore
        print_on_rank_0(rank, f"Raw dataset size: {len(dataset)} samples")

        # collate_fn stays None for the map-based paths (vision classification,
        # text) and is set for the paths whose batches can't be default-collated
        # (detection label dicts, multimodal VLM batches).
        collate_fn = None

        if model_type == "vision":
            if tokenizer is None:
                raise ValueError("Image classification requires an image processor (got None).")
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

        elif model_type == "yolo":
            if tokenizer is None:
                raise ValueError("Object detection requires an image processor (got None).")
            # LIMITATION: this path assumes the common HF detection schema — an
            # `objects` column holding parallel `bbox` (COCO [x,y,w,h]) and
            # `category` lists (cppe-5 / COCO-HF style). Validate up front and
            # fail fast with guidance instead of a cryptic KeyError inside a
            # DataLoader worker for datasets shaped differently.
            cols = dataset.column_names
            sample_objs = dataset[0]["objects"] if "objects" in cols else None
            if not (isinstance(sample_objs, dict) and "bbox" in sample_objs and "category" in sample_objs):
                raise ValueError(
                    "Object detection expects an 'objects' column that is a dict with 'bbox' "
                    "([x,y,w,h]) and 'category' lists (cppe-5 / COCO-HF schema); "
                    f"got columns={cols}. Adapt _detection_collate in data.py for other layouts. "
                    "See CLAUDE.md 'Data Pipeline by Model Type'."
                )
            print_on_rank_0(rank, "Detection assumes COCO-HF schema: objects{bbox [x,y,w,h], category} → other layouts need a custom collate", "⚠️")
            # Keep the dataset raw (PIL images + objects column); the collate_fn
            # converts each batch into pixel_values + COCO-style label dicts.
            collate_fn = functools.partial(_detection_collate, processor=tokenizer)
            print_on_rank_0(rank, "Detection collate configured | objects → boxes + class_labels", "🎯")

        elif model_type == "vlm":
            if tokenizer is None:
                raise ValueError("VLM training requires a processor (got None).")
            cols = dataset.column_names
            image_col = next((c for c in ("image", "img") if c in cols), None)
            text_col = next((c for c in ("text", "caption", "captions", "sentence") if c in cols), None)
            if image_col is None or text_col is None:
                raise ValueError(
                    f"VLM dataset must expose an image column and a text/caption column; got columns={cols}"
                )
            # Keep the dataset raw; the collate_fn builds the prompt, expands
            # image tokens and pads each batch via the processor.
            collate_fn = functools.partial(
                _vlm_collate, processor=tokenizer, image_col=image_col, text_col=text_col
            )
            print_on_rank_0(rank, f"VLM collate configured | image_col={image_col} | text_col={text_col}", "🖼️")
            # LIMITATION: heads-up so users understand the assumptions baked into
            # the VLM path before a long run.
            print_on_rank_0(rank, "VLM limitations: fixed instruction prompt ('Describe the image.'), captions are NOT truncated (long text → long sequences), and the collate targets chat-template processors with nested image lists (SmolVLM/Idefics3/LLaVA-interleave families)", "⚠️")

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
        # Detection/VLM batches carry non-tensor structures (lists of label
        # dicts); skip pin_memory there since it only benefits plain tensors.
        pin_memory = torch.cuda.is_available() and collate_fn is None
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            collate_fn=collate_fn,
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

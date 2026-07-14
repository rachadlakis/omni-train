## dist_kit/tests/test_data.py ##
##
## Dataloader validation (unit) + a cached-wikitext round trip (integration).

import pytest
import torch

from dist_kit.config import DistConfig
from dist_kit.data import get_text_dataloader


def test_non_text_model_type_rejected():
    args = DistConfig(model_type="vision", dataset_name="beans")
    with pytest.raises(ValueError, match="text model types"):
        get_text_dataloader(args, tokenizer=None, rank=0, world_size=1)


def test_yolo_model_type_rejected():
    args = DistConfig(model_type="yolo", dataset_name="cppe-5")
    with pytest.raises(ValueError, match="text model types"):
        get_text_dataloader(args, tokenizer=None, rank=0, world_size=1)


@pytest.mark.integration
def test_wikitext_dataloader_round_trip(monkeypatch):
    """Tokenize cached wikitext and check batch shapes. Skips when the
    tokenizer/dataset are not in the local HF cache (offline-only test)."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("DIST_KIT_NUM_WORKERS", "0")

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    except Exception as e:  # not cached on this machine
        pytest.skip(f"facebook/opt-125m tokenizer not cached: {e}")

    args = DistConfig(
        strategy="solo",
        dataset_name="wikitext",
        dataset_subset="wikitext-2-raw-v1",
        dataset_split="train[:60]",
        batch_size=2,
        max_length=32,
    )
    try:
        dataloader = get_text_dataloader(args, tokenizer, rank=0, world_size=1)
    except Exception as e:
        pytest.skip(f"wikitext not cached: {e}")

    batch = next(iter(dataloader))
    assert set(batch) >= {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape == (2, 32)
    assert torch.equal(batch["input_ids"], batch["labels"])
    assert len(dataloader) > 0

"""Label masking in data.py (no GPU, no network)."""
import pytest
import torch
from datasets import Dataset

import data
from data import _vlm_collate, get_dataloader


# ------------------------------------------------------------------
# Text (llm / seq2seq / encoder): padding must not count toward the loss
# ------------------------------------------------------------------

class _FakeTokenizer:
    """Whitespace tokenizer that right-pads to max_length with pad id 0."""

    def __call__(self, texts, truncation, max_length, padding):
        input_ids, attention_mask = [], []
        for text in texts:
            ids = [len(word) + 1 for word in text.split()][:max_length]
            pad = max_length - len(ids)
            input_ids.append(ids + [0] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_text_labels_mask_padding(monkeypatch):
    rows = ["this sentence is long enough to pass the filter", "short but still more than twenty chars"]
    monkeypatch.setattr(data, "load_dataset", lambda *a, **k: Dataset.from_dict({"text": rows}))

    loader = get_dataloader("fake", None, "train", _FakeTokenizer(), rank=0, world_size=1,
                            batch_size=2, max_length=16, model_type="llm")
    loader.num_workers = 0
    batch = next(iter(loader))

    mask = batch["attention_mask"].bool()
    assert (batch["labels"][~mask] == -100).all()
    assert torch.equal(batch["labels"][mask], batch["input_ids"][mask])


# ------------------------------------------------------------------
# VLM: only the assistant's reply is trained on
# ------------------------------------------------------------------

EOS = PAD = 1  # pad == eos, as with many chat tokenizers
IMAGE_TOKENS_PER_IMAGE = 3


class _FakeVLMTokenizer:
    pad_token_id = PAD

    def __init__(self, vocab):
        self.vocab = vocab

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]


class _FakeVLMProcessor:
    image_token = "<image>"

    def __init__(self, padding_side):
        self.vocab = {"<pad/eos>": EOS, "<image>": 2}
        self.tokenizer = _FakeVLMTokenizer(self.vocab)
        self.padding_side = padding_side

    def _id(self, word):
        return self.vocab.setdefault(word, len(self.vocab) + 1)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        out = []
        for m in messages:
            if m["role"] == "user":
                out.append("USER: <image> " + m["content"][1]["text"])
            else:
                out.append("ASSISTANT: " + m["content"][0]["text"] + " <pad/eos>")
        if add_generation_prompt:
            out.append("ASSISTANT:")
        return " ".join(out)

    def __call__(self, text, images, return_tensors, padding):
        seqs = []
        for t in text:
            ids = []
            for word in t.split():
                ids += [self._id(word)] * (IMAGE_TOKENS_PER_IMAGE if word == "<image>" else 1)
            seqs.append(ids)
        width = max(len(s) for s in seqs)
        input_ids, attention_mask = [], []
        for ids in seqs:
            pad = [PAD] * (width - len(ids))
            real = [1] * len(ids)
            if self.padding_side == "left":
                input_ids.append(pad + ids); attention_mask.append([0] * len(pad) + real)
            else:
                input_ids.append(ids + pad); attention_mask.append(real + [0] * len(pad))
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "pixel_values": torch.zeros(len(text), 3, 4, 4),
        }


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_vlm_labels_cover_only_the_reply(padding_side):
    proc = _FakeVLMProcessor(padding_side)
    examples = [
        {"image": object(), "text": "a cat"},
        {"image": object(), "text": "a dog on a red sofa"},  # longer → row 0 gets padded
    ]
    batch = _vlm_collate(examples, proc, image_col="image", text_col="text")

    for row, ex in enumerate(examples):
        kept = batch["labels"][row][batch["labels"][row] != -100].tolist()
        expected = [proc.vocab[w] for w in ex["text"].split()] + [EOS]
        # Reply tokens AND the closing eos survive, even though pad == eos.
        assert kept == expected

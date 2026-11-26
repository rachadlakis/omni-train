"""Score completions with a Hugging Face reward model (GRPO and PPO).

    algorithm:
      reward_model: Skywork/Skywork-Reward-V2-Qwen3-0.6B      # weight 1
      reward_model: {name: ..., weight: 0.5, max_length: 1024, batch_size: 8}

Any AutoModelForSequenceClassification with ONE output. It is frozen, loaded whole on every
GPU (bf16 where supported) and added to the reward functions as `reward_model`. With a chat
template in the reward model's tokenizer the conversation is rendered with it; otherwise it
scores prompt + completion as plain text.
"""

from __future__ import annotations

import os

import torch

KEYS = {"name", "weight", "max_length", "batch_size"}


def parse_reward_model(spec) -> dict | None:
    if not spec:
        return None
    if isinstance(spec, str):
        spec = {"name": spec}
    if not isinstance(spec, dict) or not spec.get("name"):
        raise ValueError("algorithm.reward_model must be a model name or {name: ..., weight: ...}")
    unknown = set(spec) - KEYS
    if unknown:
        raise ValueError(f"Unknown reward_model keys: {sorted(unknown)}. Known: {sorted(KEYS)}")
    out = {"name": str(spec["name"]), "weight": float(spec.get("weight", 1.0)),
           "max_length": int(spec.get("max_length", 1024)), "batch_size": int(spec.get("batch_size", 8))}
    if out["max_length"] < 1 or out["batch_size"] < 1:
        raise ValueError("reward_model max_length and batch_size must be >= 1")
    return out


class RewardModel:
    """Callable with the reward-function signature (see rewards.py)."""

    def __init__(self, name: str, device, max_length: int = 1024, batch_size: int = 8):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        token = os.getenv("HF_TOKEN")
        self.tok = AutoTokenizer.from_pretrained(name, token=token)
        bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
        model, info = AutoModelForSequenceClassification.from_pretrained(
            name, token=token, dtype=torch.bfloat16 if bf16 else torch.float32, output_loading_info=True)
        missing = [k for k in info.get("missing_keys", ()) if "score" in k or "classifier" in k]
        if missing:
            raise ValueError(f"'{name}' is not a reward model: its scoring head {missing} is not in the "
                             "checkpoint, so the scores would be random")
        if model.config.num_labels != 1:
            raise ValueError(f"'{name}' has {model.config.num_labels} outputs; a reward model has one")
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        # the model reads the score at the last non-padding token, which it finds by this id
        model.config.pad_token_id = self.tok.pad_token_id
        self.model = model.to(device).eval().requires_grad_(False)
        self.device, self.max_length, self.batch_size = device, max_length, batch_size

    def text(self, prompt: str, completion: str, conversation=None) -> tuple[str, bool]:
        """(text, whether the tokenizer should still add its special tokens)."""
        if conversation and getattr(self.tok, "chat_template", None):
            msgs = list(conversation) + [{"role": "assistant", "content": completion}]
            return self.tok.apply_chat_template(msgs, tokenize=False), False
        return prompt + completion, True

    @torch.no_grad()
    def __call__(self, prompts, completions, answers=None, conversations=None, **info) -> list[float]:
        conversations = conversations or [None] * len(prompts)
        pairs = [self.text(p, c, conv) for p, c, conv in zip(prompts, completions, conversations)]
        scores: list[float] = []
        for i in range(0, len(pairs), self.batch_size):
            rows = []
            for text, special in pairs[i:i + self.batch_size]:
                ids = self.tok(text, add_special_tokens=special)["input_ids"]
                rows.append(ids[-self.max_length:] or [self.tok.pad_token_id])   # keep the end: the answer
            width = max(len(r) for r in rows)
            ids = torch.full((len(rows), width), self.tok.pad_token_id, device=self.device)
            mask = torch.zeros_like(ids)
            for j, r in enumerate(rows):
                ids[j, :len(r)] = torch.tensor(r, device=self.device)
                mask[j, :len(r)] = 1
            # the weights already have their dtype; train.py's autocast must not re-cast them
            with torch.autocast(self.device.type, enabled=False):
                scores += self.model(input_ids=ids, attention_mask=mask).logits[:, 0].float().tolist()
        return scores

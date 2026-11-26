"""Datasets for SFT, preference training (DPO / SimPO / ORPO) and GRPO.

Accepted dataset shapes
-----------------------
SFT (one of):
  * ``messages``: list of {"role", "content"} (chat). Loss on assistant turns.
  * ``prompt`` + ``completion``: strings or message lists.
  * ``instruction`` (+ ``input``) + ``output``: Alpaca style.
  * ``text``: plain text, loss on every token (``completion_only`` ignored).
Preference (one of):
  * ``prompt`` + ``chosen`` + ``rejected``: strings, or a message-list prompt
    and assistant message(s) as completions.
  * ``chosen`` + ``rejected`` only, as full conversations whose last message
    is the answer (e.g. trl-lib/ultrafeedback_binarized), or as strings that
    share a prompt ending in "\\n\\nAssistant:" (Anthropic/hh-rlhf).
GRPO / PPO (prompts only; the model writes the answers):
  * a prompt from ``prompt`` (string or messages) | ``question`` | ``problem`` |
    ``messages`` (trailing assistant turns dropped), and
  * optionally a reference answer for the rewards from ``answer`` | ``solution`` |
    ``ground_truth`` | ``target`` | ``final_answer`` (e.g. openai/gsm8k ``main``).

With a chat template (instruct models) prompts are framed by the template;
without one the text is concatenated as is (BOS + prompt + answer + EOS).

How the answer tokens are found: the prompt (rendered with
``add_generation_prompt=True``) must be a string prefix of the full rendered
conversation. Prompt and answer are tokenised SEPARATELY and concatenated, so
the mask boundary is exact even when a tokenizer would merge characters across
it.
"""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from .losses import IGNORE_INDEX

HH_SPLIT = "\n\nAssistant:"
ALPACA_PROMPT = "### Instruction:\n{instruction}\n\n### Response:\n"
ALPACA_PROMPT_INPUT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"


# ----------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------

def has_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def _render(tokenizer, messages, add_generation_prompt=False) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=add_generation_prompt)


def _ids(tokenizer, text: str, special: bool = False) -> list[int]:
    return tokenizer(text, add_special_tokens=special)["input_ids"]


def _as_messages(x, role: str):
    """A string becomes one message of `role`; a message list passes through."""
    if isinstance(x, str):
        return [{"role": role, "content": x}]
    if isinstance(x, dict):
        return [x]
    return list(x)


def _prompt_and_answer_ids(tokenizer, prompt, answer) -> tuple[list[int], list[int]]:
    """Token ids of a prompt and of one answer to it.

    prompt: string or message list; answer: string, message, or message list.
    """
    if has_chat_template(tokenizer):
        p_msgs = _as_messages(prompt, "user")
        full_msgs = p_msgs + _as_messages(answer, "assistant")
        p_text = _render(tokenizer, p_msgs, add_generation_prompt=True)
        full = _render(tokenizer, full_msgs)
        if not full.startswith(p_text):
            raise ValueError("chat template: the rendered prompt is not a prefix of the rendered "
                             "conversation, so the answer tokens cannot be located")
        return _ids(tokenizer, p_text), _ids(tokenizer, full[len(p_text):])
    if not (isinstance(prompt, str) and isinstance(answer, str)):
        raise ValueError("conversational (message-list) data needs a tokenizer with a chat template")
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    return _ids(tokenizer, prompt, special=True), _ids(tokenizer, answer) + eos


# ----------------------------------------------------------------------
# SFT
# ----------------------------------------------------------------------

def _sft_segments(tokenizer, row) -> list[tuple[list[int], bool]]:
    """Split one row into (token ids, trained?) segments."""
    if "messages" in row and row["messages"] is not None:
        msgs = list(row["messages"])
        if not has_chat_template(tokenizer):
            raise ValueError("`messages` data needs a tokenizer with a chat template")
        full = _render(tokenizer, msgs)
        segments, done = [], 0
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            prefix = _render(tokenizer, msgs[:i], add_generation_prompt=True) if i else ""
            upto = _render(tokenizer, msgs[:i + 1])
            start, end = len(prefix), len(upto)
            if not (done <= start <= end and full.startswith(prefix) and full.startswith(upto)):
                raise ValueError("chat template is not prefix-consistent; cannot mask assistant turns")
            segments += [(_ids(tokenizer, full[done:start]), False), (_ids(tokenizer, full[start:end]), True)]
            done = end
        segments.append((_ids(tokenizer, full[done:]), False))
        return segments
    if "prompt" in row and "completion" in row:
        p, a = _prompt_and_answer_ids(tokenizer, row["prompt"], row["completion"])
        return [(p, False), (a, True)]
    if "instruction" in row and "output" in row:
        inp = (row.get("input") or "").strip()
        if has_chat_template(tokenizer):
            prompt = row["instruction"] + (f"\n\n{inp}" if inp else "")
        else:
            tmpl = ALPACA_PROMPT_INPUT if inp else ALPACA_PROMPT
            prompt = tmpl.format(instruction=row["instruction"], input=inp)
        p, a = _prompt_and_answer_ids(tokenizer, prompt, row["output"])
        return [(p, False), (a, True)]
    if "text" in row:
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        return [(_ids(tokenizer, row["text"], special=True) + eos, True)]
    raise ValueError(f"SFT row has none of: messages | prompt+completion | instruction+output | text "
                     f"(got {sorted(row)})")


def tokenize_sft(row, tokenizer, max_length: int, completion_only: bool = True) -> dict:
    input_ids, labels = [], []
    for ids, trained in _sft_segments(tokenizer, row):
        input_ids += ids
        labels += ids if (trained or not completion_only) else [IGNORE_INDEX] * len(ids)
    input_ids, labels = input_ids[:max_length], labels[:max_length]
    return {"input_ids": input_ids, "labels": labels}


def collate_sft(rows, pad_id: int) -> dict:
    width = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        n = len(r["input_ids"])
        ids[i, :n] = torch.tensor(r["input_ids"])
        labels[i, :n] = torch.tensor(r["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


# ----------------------------------------------------------------------
# Preference pairs
# ----------------------------------------------------------------------

def split_preference_row(row):
    """Return (prompt, chosen_answer, rejected_answer) for any accepted shape."""
    chosen, rejected = row["chosen"], row["rejected"]
    if row.get("prompt") is not None:
        return row["prompt"], chosen, rejected
    if isinstance(chosen, str):
        k = chosen.rfind(HH_SPLIT)
        if k < 0:
            raise ValueError("preference strings without a `prompt` column must contain "
                             f"{HH_SPLIT!r} (Anthropic/hh-rlhf format)")
        k += len(HH_SPLIT)
        prompt = chosen[:k]
        if not rejected.startswith(prompt):
            raise ValueError("chosen and rejected do not share the same prompt")
        return prompt, chosen[k:], rejected[k:]
    chosen, rejected = list(chosen), list(rejected)
    if chosen[:-1] != rejected[:-1] or chosen[-1]["role"] != "assistant":
        raise ValueError("conversational chosen/rejected must share all turns but the last "
                         "(an assistant message)")
    return chosen[:-1], chosen[-1], rejected[-1]


def tokenize_preference(row, tokenizer, max_length: int, max_prompt_length: int) -> dict:
    prompt, chosen, rejected = split_preference_row(row)
    p_ids, c_ids = _prompt_and_answer_ids(tokenizer, prompt, chosen)
    p_ids2, r_ids = _prompt_and_answer_ids(tokenizer, prompt, rejected)
    if p_ids != p_ids2:
        raise ValueError("prompt tokenises differently for chosen and rejected")
    p_ids = p_ids[-max_prompt_length:]            # keep the end of a long prompt
    room = max_length - len(p_ids)
    return {"prompt_ids": p_ids, "chosen_ids": c_ids[:room], "rejected_ids": r_ids[:room]}


def collate_preference(rows, pad_id: int) -> dict:
    """Chosen sequences first, then rejected: [2B, T], so ONE forward covers both
    (with FSDP every rank must run the same number of forwards)."""
    seqs = [(r["prompt_ids"], r["chosen_ids"]) for r in rows] + \
           [(r["prompt_ids"], r["rejected_ids"]) for r in rows]
    width = max(len(p) + len(a) for p, a in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    labels = torch.full((len(seqs), width), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, (p, a) in enumerate(seqs):
        n = len(p) + len(a)
        ids[i, :n] = torch.tensor(p + a)
        labels[i, len(p):n] = torch.tensor(a, dtype=torch.long)
        mask[i, :n] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels,
            "idx": torch.tensor([r["idx"] for r in rows], dtype=torch.long)}


# ----------------------------------------------------------------------
# GRPO prompts
# ----------------------------------------------------------------------

PROMPT_COLUMNS = ("prompt", "question", "problem")
ANSWER_COLUMNS = ("answer", "solution", "ground_truth", "target", "final_answer")


def split_grpo_row(row):
    """(prompt as a string or message list, reference answer or None)."""
    prompt = next((row[c] for c in PROMPT_COLUMNS if row.get(c) is not None), None)
    if prompt is None and row.get("messages") is not None:
        msgs = list(row["messages"])
        while msgs and msgs[-1]["role"] == "assistant":
            msgs.pop()
        prompt = msgs or None
    if prompt is None:
        raise ValueError(f"GRPO row has no prompt: needs one of {PROMPT_COLUMNS} or messages (got {sorted(row)})")
    answer = next((row[c] for c in ANSWER_COLUMNS if row.get(c) is not None), None)
    return prompt, (None if answer is None else str(answer))


def tokenize_grpo(row, tokenizer, max_prompt_length: int, system_prompt: str | None = None) -> dict:
    prompt, answer = split_grpo_row(row)
    if has_chat_template(tokenizer):
        msgs = _as_messages(prompt, "user")
        if system_prompt and not any(m["role"] == "system" for m in msgs):
            msgs = [{"role": "system", "content": system_prompt}] + msgs
        ids = _ids(tokenizer, _render(tokenizer, msgs, add_generation_prompt=True))
    else:
        if not isinstance(prompt, str):
            raise ValueError("conversational (message-list) prompts need a tokenizer with a chat template")
        text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        ids = _ids(tokenizer, text, special=True)
    # the messages as JSON (a fixed schema for the dataset): reward models render their own template
    conversation = json.dumps(msgs) if has_chat_template(tokenizer) else None
    return {"prompt_ids": ids[-max_prompt_length:], "answer": answer, "conversation": conversation}


def collate_grpo(rows, pad_id: int) -> dict:
    """Prompts LEFT-padded (generation appends on the right)."""
    width = max(len(r["prompt_ids"]) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        n = len(r["prompt_ids"])
        ids[i, width - n:] = torch.tensor(r["prompt_ids"])
        mask[i, width - n:] = 1
    return {"input_ids": ids, "attention_mask": mask, "answers": [r["answer"] for r in rows],
            "conversations": [json.loads(r["conversation"]) if r.get("conversation") else None for r in rows],
            "idx": torch.tensor([r["idx"] for r in rows], dtype=torch.long)}


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_raw(name, subset, split):
    from data import load_any_dataset  # Hub dataset or local file
    return load_any_dataset(name, subset, split)


def build_dataset(raw, tokenizer, args):
    """Tokenise `raw` for args.algorithm. Drops rows with nothing to learn."""
    if args.algorithm == "sft":
        ds = raw.map(lambda r: tokenize_sft(r, tokenizer, args.max_length, args.algo_completion_only),
                     remove_columns=raw.column_names)
        return ds.filter(lambda r: any(t != IGNORE_INDEX for t in r["labels"][1:]))
    if args.algorithm in ("grpo", "ppo"):
        ds = raw.map(lambda r: tokenize_grpo(r, tokenizer, args.algo_max_prompt_length, args.algo_system_prompt),
                     remove_columns=raw.column_names)
        ds = ds.filter(lambda r: len(r["prompt_ids"]) > 0)
        return ds.map(lambda r, i: {"idx": i}, with_indices=True)
    ds = raw.map(lambda r: tokenize_preference(r, tokenizer, args.max_length, args.algo_max_prompt_length),
                 remove_columns=raw.column_names)
    ds = ds.filter(lambda r: len(r["chosen_ids"]) > 0 and len(r["rejected_ids"]) > 0)
    return ds.map(lambda r, i: {"idx": i}, with_indices=True)


def collate_fn_for(args, tokenizer):
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if args.algorithm == "sft":
        return lambda rows: collate_sft(rows, pad)
    if args.algorithm in ("grpo", "ppo"):
        return lambda rows: collate_grpo(rows, pad)
    return lambda rows: collate_preference(rows, pad)


def make_loader(dataset, collate, rank, world_size, batch_size, shuffle=True, drop_last=True, seed=0):
    sampler = None
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=shuffle, drop_last=drop_last, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=shuffle and sampler is None, drop_last=drop_last,
                      collate_fn=collate, num_workers=0)


def get_post_training_data(args, tokenizer, rank, world_size):
    """Returns (dataset, collate_fn, training DataLoader)."""
    from data import _rank0_first
    from distributed_utils import print_on_rank_0
    with _rank0_first():  # rank 0 fills the HF cache first
        raw = load_raw(args.dataset, args.dataset_full_name, args.dataset_split)
        print_on_rank_0(rank, f"Raw dataset: {len(raw)} rows | columns={raw.column_names}", "📂")
        ds = build_dataset(raw, tokenizer, args)
    print_on_rank_0(rank, f"{args.algorithm.upper()} dataset: {len(ds)} examples after tokenising "
                          f"(max_length={args.max_length})")
    if len(ds) < batch_needed(args, world_size):
        raise ValueError(f"{len(ds)} examples is fewer than one global batch "
                         f"({args.batch_size} x {world_size} ranks)")
    collate = collate_fn_for(args, tokenizer)
    loader = make_loader(ds, collate, rank, world_size, args.batch_size, seed=args.seed)
    print_on_rank_0(rank, f"DataLoader ready | batches per rank: {len(loader)} | batch size: {args.batch_size}")
    return ds, collate, loader


def batch_needed(args, world_size) -> int:
    return args.batch_size * world_size

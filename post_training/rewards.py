"""Reward functions for GRPO.

A reward function scores a group of completions:

    fn(prompts: list[str], completions: list[str], answers: list[str | None], **info) -> list[float]

``info`` carries ``completion_lengths`` (tokens) and ``max_completion_length``. Pick them in
YAML by name, optionally with weights; the total reward is the weighted sum:

    algorithm:
      type: grpo
      rewards: [accuracy, format]                 # weight 1 each
      rewards: {accuracy: 1.0, format: 0.2}       # explicit weights
      rewards: {my_pkg.rewards:is_polite: 0.5}    # any importable "module:function"

Built-in rewards
----------------
accuracy     1 if the completion's final answer equals the reference answer, else 0.
             The final answer is read from <answer>...</answer>, \\boxed{...}, "#### x",
             "the answer is x", or else the last number. References get the same
             treatment, so GSM8K solutions ("... #### 72") work as they are.
             Numbers compare numerically (1,000 == 1000.0); text compares case- and
             whitespace-insensitively.
format       1 if the completion is exactly <think>...</think> <answer>...</answer>, else 0.
exact_match  1 if the whole completion equals the reference (normalised), else 0.
contains     1 if the normalised reference appears in the completion, else 0.
length       0 up to (max_completion_length - L/4) tokens, then a linear penalty down to
             -1 at max_completion_length (DAPO's soft overlong punishment).
"""

from __future__ import annotations

import importlib
import re
from typing import Callable

RewardFn = Callable[..., list]

_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_HASHES = re.compile(r"####\s*(.+)")
_ANSWER_IS = re.compile(r"answer\s*(?:is|:)\s*\**\s*([^\n]+)", re.IGNORECASE)
_FORMAT = re.compile(r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL)


def _last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...}, with nested braces."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
    return None


def extract_answer(text: str | None) -> str | None:
    """The final answer stated in a completion or a reference solution."""
    if text is None:
        return None
    text = str(text)
    tags = _ANSWER_TAG.findall(text)
    if tags:
        return tags[-1].strip()
    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    m = _HASHES.search(text)
    if m:
        return m.group(1).strip()
    m = _ANSWER_IS.findall(text)
    if m:
        return m[-1].strip().rstrip(".")
    nums = re.findall(_NUMBER, text)
    if nums:
        return nums[-1]
    return text.strip()


def _as_number(s: str) -> float | None:
    s = s.strip().replace("$", "").replace("%", "").rstrip(".")
    m = re.fullmatch(_NUMBER, s)
    if not m:
        nums = re.findall(_NUMBER, s)
        if len(nums) != 1 or re.sub(_NUMBER, "", s).strip(" .*\\{}()") not in {"", "dollars", "units"}:
            return None
        s = nums[0]
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def normalize(s: str | None) -> str:
    s = "" if s is None else str(s)
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = s.replace("$", "").strip().strip(".").lower()
    return re.sub(r"\s+", " ", s)


def answers_match(predicted: str | None, reference: str | None) -> bool:
    if predicted is None or reference is None:
        return False
    a, b = _as_number(predicted), _as_number(reference)
    if a is not None and b is not None:
        return abs(a - b) <= 1e-6 * max(1.0, abs(b))
    return normalize(predicted) == normalize(reference)


def _need_answers(answers, name):
    if any(a is None for a in answers):
        raise ValueError(f"reward '{name}' needs a reference answer column "
                         "(answer | solution | ground_truth | target | final_answer)")


def accuracy(prompts, completions, answers, **info):
    _need_answers(answers, "accuracy")
    return [float(answers_match(extract_answer(c), extract_answer(a))) for c, a in zip(completions, answers)]


def format_reward(prompts, completions, answers, **info):
    return [float(bool(_FORMAT.match(c))) for c in completions]


def exact_match(prompts, completions, answers, **info):
    _need_answers(answers, "exact_match")
    return [float(normalize(c) == normalize(a)) for c, a in zip(completions, answers)]


def contains(prompts, completions, answers, **info):
    _need_answers(answers, "contains")
    return [float(normalize(a) in normalize(c)) for c, a in zip(completions, answers)]


def length(prompts, completions, answers, completion_lengths=None, max_completion_length=None, **info):
    if completion_lengths is None or not max_completion_length:
        return [0.0] * len(completions)
    cache = max(1, max_completion_length // 4)
    start = max_completion_length - cache
    return [0.0 if n <= start else -min(1.0, (n - start) / cache) for n in completion_lengths]


REWARDS: dict[str, RewardFn] = {
    "accuracy": accuracy,
    "format": format_reward,
    "exact_match": exact_match,
    "contains": contains,
    "length": length,
}


def parse_reward_spec(spec) -> dict[str, float]:
    """`[a, b]`, `"a"` or `{a: 1.0, b: 0.2}` -> {name: weight}. Checks names exist."""
    if spec is None:
        spec = ["accuracy"]
    if isinstance(spec, str):
        spec = [spec]
    if isinstance(spec, (list, tuple)):
        spec = {str(n): 1.0 for n in spec}
    if not isinstance(spec, dict) or not spec:
        raise ValueError("algorithm.rewards must be a reward name, a list of names, or {name: weight}")
    out = {}
    for name, weight in spec.items():
        name = str(name)
        if name not in REWARDS and ":" not in name:
            raise ValueError(f"Unknown reward '{name}'. Built-in: {', '.join(REWARDS)}; "
                             "or give an importable 'module:function'")
        out[name] = float(weight)
    return out


def load_reward(name: str) -> RewardFn:
    if name in REWARDS:
        return REWARDS[name]
    module, _, attr = name.partition(":")
    try:
        fn = getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Cannot import reward function '{name}': {e}") from e
    if not callable(fn):
        raise ValueError(f"Reward '{name}' is not callable")
    return fn


def build_rewards(spec) -> list[tuple[str, RewardFn, float]]:
    return [(name, load_reward(name), w) for name, w in parse_reward_spec(spec).items()]

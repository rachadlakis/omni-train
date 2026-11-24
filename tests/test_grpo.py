"""GRPO: rewards, loss math, masks, data, config rules and the objective on a tiny
random Llama. All offline, CPU. Multi-process gradient checks: test_post_training_dist.py."""

import copy
import types

import pytest
import torch

from post_training import losses as L
from post_training import rewards as R
from post_training.data import collate_grpo, split_grpo_row, tokenize_grpo
from post_training.grpo import GRPOObjective, build_rollout, completion_mask_from
from tests.post_training_helpers import make_tokenizer, tiny_llama
from utils import build_args


# ----------------------------------------------------------------------
# Rewards
# ----------------------------------------------------------------------

@pytest.mark.parametrize("text, want", [
    ("<think>2+2</think> <answer>4</answer>", "4"),
    ("so the result is \\boxed{\\frac{1}{2}}.", "\\frac{1}{2}"),
    ("Natalia sold 48/2 = 24 clips.\n#### 72", "72"),
    ("The answer is 1,250.", "1,250"),
    ("first 3 then 7 apples", "7"),
    ("no digits here", "no digits here"),
    (None, None),
])
def test_extract_answer(text, want):
    assert R.extract_answer(text) == want


@pytest.mark.parametrize("pred, ref, ok", [
    ("72", "72", True), ("1,000", "1000.0", True), ("$5", "5", True), ("72", "73", False),
    ("Paris", " paris. ", True), ("Paris", "London", False), (None, "1", False),
])
def test_answers_match(pred, ref, ok):
    assert R.answers_match(pred, ref) is ok


def test_accuracy_reads_gsm8k_references():
    gsm8k = "She has 3 + 4 = 7 apples.\n#### 7"
    got = R.accuracy(["q"] * 3, ["I think 7", "<answer>8</answer>", "\\boxed{7}"], [gsm8k] * 3)
    assert got == [1.0, 0.0, 1.0]


@pytest.mark.parametrize("completion, reference", [
    ("The answer is 72 apples", "#### 72"), ("The answer is 72 apples.", "#### 72"),
    ("So the answer is: **72**", "#### 72"), ("**The answer is 72.**", "#### 72"),
    ("The answer is $72.", "#### 72"), ("The answer is -$5", "#### -5"), ("The answer is \\$1,250.50", "#### 1250.5"),
    ("The answer is 72 because 70 + 2 = 72", "#### 72"), ("The answer is 72. Let me know!", "#### 72"),
    ("\\boxed{72 \\text{ apples}}", "#### 72"), ("\\boxed{x = 72}", "#### 72"), ("\\boxed{1\\,000}", "#### 1000"),
    ("<answer>72 apples</answer>", "#### 72"), ("#### 72 apples", "#### 72"),
    ("The answer is 72 km/h", "#### 72"), ("\\boxed{72 \\text{ cm}^{2}}", "#### 72"),
    ("The answer is 3.5 hours", "#### 3.5"), ("The answer is -5 degrees", "#### -5"), ("The answer is 50%", "#### 50"),
    ("The answer is 72", "72 apples"), ("The answer is 72 apples", "72 apples"),
    ("The answer is 2/3", "2/3"), ("I get 2/3", "2/3"), ("\\boxed{\\frac{2}{3}}", "2/3"), ("The answer is 0.5", "1/2"),
    ("The answer is Paris", "Paris"), ("<answer>route 66</answer>", "Route 66"), ("Route 66", "Route 66"),
    ("**x+1**", "x+1"), ("so 3 + 4 gives \\frac{1}{2}", "\\frac{1}{2}"), ("She has \\frac{1}{2} left.", "0.5"),
])
def test_accuracy_accepts(completion, reference):
    assert R.accuracy(["q"], [completion], [reference]) == [1.0]


@pytest.mark.parametrize("completion, reference", [
    # hedging: more than one value, or a value that is taken back
    ("The answer is 71 or 72", "#### 72"), ("The answer is 72 or 71", "#### 72"), ("The answer is 72, 73 or 74", "#### 72"),
    ("<answer>72 73</answer>", "#### 72"), ("The answer is not 72", "#### 72"), ("The answer is 72 or so", "#### 72"),
    ("The answer is between 70 and 72", "#### 72"), ("The answer is 72 if x else 5", "#### 72"), ("The answer is 72-ish", "#### 72"),
    # words and symbols that change the value
    ("The answer is 72 thousand", "#### 72"), ("The answer is 72 million dollars", "#### 72"), ("The answer is 72k", "#### 72"),
    ("The answer is 2 dozen eggs", "#### 2"), ("The answer is 72 and a half", "#### 72"), ("The answer is 3 quarters", "#### 3"),
    ("The answer is 5x", "#### 5"), ("The answer is 12 - 5", "#### 12"), ("The answer is 5*3", "#### 53"),
    ("The answer is 72 less than that", "#### 72"), ("The answer is 10:30", "#### 10"), ("The answer is 5^2", "#### 5"),
    ("first 3 then 2/3", "\\frac{1}{3}"), ("so 3 + 4 gives \\frac{1}{2}", "#### 2"),
    ("The answer is 2/3", "#### 2"), ("I get 2/3", "#### 3"), ("The answer is 5/0", "#### 5"),
    # different units, text that merely contains the number
    ("The answer is 72 oranges", "72 apples"), ("I saw 66", "Route 66"), ("The answer is 66", "Route 66"),
    ("The answer is Paris, the capital", "London"), ("", "#### 72"), ("The answer is apples", "#### 72"),
])
def test_accuracy_rejects(completion, reference):
    assert R.accuracy(["q"], [completion], [reference]) == [0.0]


def test_accuracy_needs_answers():
    with pytest.raises(ValueError, match="reference answer"):
        R.accuracy(["q"], ["1"], [None])


def test_format_contains_exact_length():
    good, bad = "<think>a</think>\n<answer>b</answer>", "<answer>b</answer>"
    assert R.format_reward(None, [good, bad], None) == [1.0, 0.0]
    assert R.contains(None, ["the answer: Four"], ["four"]) == [1.0]
    assert R.exact_match(None, [" Yes. ", "yes sir"], ["yes", "yes"]) == [1.0, 0.0]
    assert R.length(None, ["a"] * 3, None, completion_lengths=[10, 14, 16], max_completion_length=16) == [0.0, -0.5, -1.0]


def test_reward_spec_and_custom_import():
    assert R.parse_reward_spec(None) == {"accuracy": 1.0}
    assert R.parse_reward_spec(["accuracy", "format"]) == {"accuracy": 1.0, "format": 1.0}
    assert R.parse_reward_spec({"accuracy": 1, "format": 0.2}) == {"accuracy": 1.0, "format": 0.2}
    with pytest.raises(ValueError, match="Unknown reward"):
        R.parse_reward_spec(["acc"])
    fns = R.build_rewards({"post_training.rewards:contains": 0.5})
    assert fns[0][1] is R.contains and fns[0][2] == 0.5
    with pytest.raises(ValueError, match="Cannot import"):
        R.build_rewards(["nope.module:fn"])


# ----------------------------------------------------------------------
# Loss math
# ----------------------------------------------------------------------

def test_group_advantages():
    r = torch.tensor([1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0])
    adv = L.group_advantages(r, 4)
    assert adv[:4].mean().abs() < 1e-6 and adv[0] > 0 > adv[1]
    assert torch.allclose(adv[4:], torch.zeros(4))             # all equal: nothing to learn
    assert adv[0] == pytest.approx(0.75 / (0.5 + 1e-4), rel=1e-6)
    assert L.group_advantages(r, 4, scale=False)[0] == pytest.approx(0.75)


def test_on_policy_gradient_is_advantage_weighted_logprob():
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 7, requires_grad=True)
    ids = torch.randint(0, 7, (3, 5))
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]])
    adv = torch.tensor([1.0, -2.0, 0.5])
    logps = L.selective_logps(logits, ids)
    loss, stats = L.grpo_loss(logps, logps.detach(), adv, mask)
    loss.backward()
    g1 = logits.grad.clone()
    logits.grad = None
    # ratio = 1 on-policy: grad = -mean_i( A_i * mean_t grad log p )
    manual = -((L.selective_logps(logits, ids) * mask).sum(1) / mask.sum(1) * adv).mean()
    manual.backward()
    torch.testing.assert_close(g1, logits.grad)
    assert stats["clip_frac"] == 0


def test_clipping_and_loss_types():
    logps = torch.zeros(2, 4)
    old = torch.full((2, 4), -1.0)                   # ratio = e > 1.28
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    adv = torch.tensor([1.0, -1.0])
    loss, stats = L.grpo_loss(logps, old, adv, mask, epsilon=0.2, epsilon_high=0.28)
    # A>0 clipped at 1.28; A<0 uses the unclipped ratio e (min picks the worse case)
    assert loss.item() == pytest.approx((-1.28 + torch.e) / 2, rel=1e-5)
    assert stats["clip_frac"].item() == pytest.approx(4 / 6)
    on = torch.zeros(2, 4)
    dapo, _ = L.grpo_loss(on, on, adv, mask, loss_type="dapo")
    assert dapo.item() == pytest.approx((-4 + 2) / 6)
    dr, _ = L.grpo_loss(on, on, adv, mask, loss_type="dr_grpo", max_completion_length=8)
    assert dr.item() == pytest.approx((-4 + 2) / (2 * 8))
    with pytest.raises(ValueError):
        L.grpo_loss(on, on, adv, mask, loss_type="ppo")


def test_kl_k3():
    logps = torch.tensor([[-1.0, -2.0]])
    mask = torch.ones(1, 2)
    _, same = L.grpo_loss(logps, logps, torch.zeros(1), mask, ref_logps=logps, beta=0.1)
    assert same["kl"].item() == pytest.approx(0.0, abs=1e-7)
    _, diff = L.grpo_loss(logps, logps, torch.zeros(1), mask, ref_logps=logps - 0.5, beta=0.1)
    assert diff["kl"].item() > 0


def test_completion_mask_stops_after_first_eos():
    ids = torch.tensor([[5, 6, 2, 2, 2], [5, 2, 7, 2, 0], [5, 6, 7, 8, 9]])
    assert completion_mask_from(ids, [2]).tolist() == [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]]
    assert completion_mask_from(torch.tensor([[5, 3, 9]]), [2, 3]).tolist() == [[1, 1, 0]]


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def test_split_grpo_rows():
    assert split_grpo_row({"question": "q?", "answer": "#### 3"}) == ("q?", "#### 3")
    assert split_grpo_row({"problem": "p", "solution": "s"}) == ("p", "s")
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
    assert split_grpo_row({"messages": msgs}) == ([msgs[0]], None)
    with pytest.raises(ValueError, match="no prompt"):
        split_grpo_row({"text": "x"})


def test_tokenize_and_left_pad():
    tok = make_tokenizer()
    a = tokenize_grpo({"prompt": "hello", "answer": "yes"}, tok, 32, system_prompt="ok sure")
    assert tok.decode(a["prompt_ids"]).startswith("<|system|> ok sure")
    assert tok.decode(a["prompt_ids"]).rstrip().endswith("<|assistant|>")
    b = tokenize_grpo({"prompt": "what is two plus two"}, tok, 3)
    assert len(b["prompt_ids"]) == 3 and b["answer"] is None              # cut from the left
    batch = collate_grpo([a | {"idx": 0}, b | {"idx": 1}], tok.pad_token_id)
    assert batch["attention_mask"][1].tolist()[:len(a["prompt_ids"]) - 3] == [0] * (len(a["prompt_ids"]) - 3)
    assert batch["answers"] == ["yes", None] and batch["input_ids"][1, -1] == b["prompt_ids"][-1]


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

def _grpo(cfg, **algo):
    cfg = copy.deepcopy(cfg)
    cfg["algorithm"] = {"type": "grpo", **algo}
    return cfg


class TestConfig:
    def test_defaults(self, cfg):
        a = build_args(_grpo(cfg))
        assert (a.algorithm, a.algo_beta, a.algo_num_generations, a.algo_loss_type) == ("grpo", 0.0, 4, "grpo")
        assert a.algo_max_prompt_length + a.algo_max_completion_length == a.max_length

    def test_beta_needs_lora(self, cfg):
        with pytest.raises(ValueError, match="needs LoRA"):
            build_args(_grpo(cfg, beta=0.04))
        lora = _grpo(cfg, beta=0.04)
        lora["peft"]["enabled"] = True
        assert build_args(lora).algo_beta == 0.04

    @pytest.mark.parametrize("bad, match", [
        ({"num_generations": 1}, "num_generations"),
        ({"temperature": 0}, "temperature"),
        ({"loss_type": "ppo"}, "loss_type"),
        ({"rewards": ["vibes"]}, "Unknown reward"),
        ({"max_prompt_length": 20, "max_completion_length": 20}, "must be <="),
        ({"top_p": 0}, "top_p"),
    ])
    def test_rejects(self, cfg, bad, match):
        with pytest.raises(ValueError, match=match):
            build_args(_grpo(cfg, **bad))

    def test_grpo_keys_on_other_algorithms(self, cfg):
        c = copy.deepcopy(cfg)
        c["algorithm"] = {"type": "dpo", "num_generations": 4}
        with pytest.raises(ValueError, match="only apply to type: grpo"):
            build_args(c)

    def test_no_5d(self, cfg):
        c = _grpo(cfg)
        c["strategy"] = "5d"
        with pytest.raises(ValueError):
            build_args(c)


# ----------------------------------------------------------------------
# Objective on a tiny model
# ----------------------------------------------------------------------

def _args(**kw):
    base = dict(algo_num_generations=4, algo_max_completion_length=6, algo_temperature=1.0, algo_top_p=1.0,
                algo_top_k=0, algo_beta=0.0, algo_epsilon=0.2, algo_epsilon_high=0.2, algo_loss_type="grpo",
                algo_scale_rewards=True, algo_rewards=["contains"], strategy="solo")
    return types.SimpleNamespace(**(base | kw))


def _batch(tok):
    rows = [tokenize_grpo({"prompt": "is the answer good", "answer": "yes"}, tok, 32) | {"idx": 0},
            tokenize_grpo({"question": "what is two plus two", "answer": "four"}, tok, 32) | {"idx": 1}]
    return collate_grpo(rows, tok.pad_token_id)


def test_rollout_shapes_and_masks():
    tok = make_tokenizer()
    model = tiny_llama(len(tok))
    obj = GRPOObjective(_args(), tok)
    obj.prepare(model)
    torch.manual_seed(0)
    roll = obj.rollout(model, _batch(tok), torch.device("cpu"))
    P = _batch(tok)["input_ids"].shape[1]
    assert roll.input_ids.shape[0] == 8 and roll.input_ids.shape[1] == P + roll.completion_mask.shape[1]
    assert roll.completion_mask.shape[1] <= 6
    assert torch.equal(roll.attention_mask[:, P:], roll.completion_mask)
    assert roll.advantages.view(2, 4).mean(1).abs().max() < 1e-5
    assert {"reward", "reward_std", "len"} <= set(roll.metrics)
    loss, metrics = obj.loss(model, roll)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def _tiny_gpt2(vocab_size):
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(vocab_size=vocab_size, n_embd=32, n_layer=2, n_head=4, n_positions=64))


@pytest.mark.parametrize("arch", ["llama", "gpt2"])
def test_loss_logprobs_match_what_generation_sampled(arch):
    """The training forward must reproduce the distribution the completions were sampled
    from: same logits slice, same temperature, and for left-padded prompts the same
    positions. RoPE (llama) is shift-invariant; learned absolute positions (gpt2, OPT) are
    not, which is why position_ids are passed."""
    from transformers import GenerationConfig
    tok = make_tokenizer()
    model = (tiny_llama(len(tok)) if arch == "llama" else _tiny_gpt2(len(tok))).eval()
    obj = GRPOObjective(_args(algo_num_generations=2), tok)
    obj.prepare(model)
    batch = _batch(tok)
    ids = batch["input_ids"].repeat_interleave(2, 0)
    mask = batch["attention_mask"].repeat_interleave(2, 0)
    torch.manual_seed(1)
    out = model.generate(input_ids=ids, attention_mask=mask, output_logits=True, return_dict_in_generate=True,
                         generation_config=GenerationConfig(do_sample=True, max_new_tokens=5, top_k=None,
                                                            eos_token_id=-1, pad_token_id=tok.pad_token_id))
    comp = out.sequences[:, ids.shape[1]:]
    gen = torch.stack([torch.log_softmax(l.float(), -1).gather(-1, comp[:, i:i + 1]).squeeze(-1)
                       for i, l in enumerate(out.logits)], 1)
    with torch.no_grad():
        ours = obj.completion_logps(model, out.sequences, torch.cat([mask, torch.ones_like(comp)], 1), comp.shape[1])
    torch.testing.assert_close(ours, gen, rtol=1e-4, atol=1e-4)


def _has_yes(prompts, completions, answers, **info):
    return [float("yes" in c.split()) for c in completions]


def test_grpo_learns_a_simple_reward():
    tok = make_tokenizer()
    torch.manual_seed(0)
    model = tiny_llama(len(tok))
    obj = GRPOObjective(_args(algo_num_generations=8, algo_max_completion_length=4,
                              algo_rewards={"tests.test_grpo:_has_yes": 1.0}), tok)
    obj.prepare(model)
    opt = torch.optim.Adam(model.parameters(), lr=3e-2)
    batch, rewards = _batch(tok), []
    for _ in range(40):
        opt.zero_grad()
        loss, metrics = obj(model, batch, torch.device("cpu"))
        loss.backward()
        opt.step()
        rewards.append(metrics["reward"])
    first, last = sum(rewards[:8]) / 8, sum(rewards[-8:]) / 8
    assert last > first + 0.3 and last > 0.6, rewards


def test_kl_reference_is_the_model_without_adapters():
    from peft import LoraConfig, get_peft_model
    tok = make_tokenizer()
    model = get_peft_model(tiny_llama(len(tok)), LoraConfig(r=4, lora_alpha=8, target_modules="all-linear"))
    obj = GRPOObjective(_args(algo_beta=0.1), tok)
    obj.prepare(model)
    torch.manual_seed(0)
    roll = obj.rollout(model, _batch(tok), torch.device("cpu"))
    assert roll.ref_logps is not None
    _, metrics = obj.loss(model, roll)
    assert metrics["kl"] == pytest.approx(0.0, abs=1e-6)         # LoRA B = 0: policy == reference
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.add_(0.05)
    loss, metrics = obj.loss(model, roll)
    assert metrics["kl"] > 0
    loss.backward()
    assert all(p.grad is None for n, p in model.named_parameters() if "lora_" not in n)


def test_build_rollout_groups():
    prompt = torch.tensor([[0, 5], [0, 5], [6, 7], [6, 7]])
    pmask = torch.tensor([[0, 1], [0, 1], [1, 1], [1, 1]])
    comp = torch.tensor([[8, 2], [9, 2], [8, 2], [9, 2]])
    roll = build_rollout(prompt, pmask, comp, torch.ones_like(comp), torch.tensor([1.0, 0.0, 1.0, 1.0]), 2)
    assert roll.advantages[0] > 0 > roll.advantages[1] and torch.all(roll.advantages[2:] == 0)
    assert torch.equal(roll.completion_ids, comp)


def test_works_with_gradient_checkpointing():
    """The training forward must not use the KV cache: with gradient checkpointing a
    growing cache changes the recomputed shapes (CheckpointError)."""
    tok = make_tokenizer()
    model = tiny_llama(len(tok))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = True              # what a real HF checkpoint ships with
    obj = GRPOObjective(_args(), tok)
    obj.prepare(model)
    model.train()
    torch.manual_seed(0)
    loss, _ = obj(model, _batch(tok), torch.device("cpu"))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize("ckpt", [False, True])
def test_lora_gets_gradients_under_autocast(ckpt):
    """train.py runs the objective inside torch.autocast. Sampling (no grad) must not leave
    bf16 copies of the LoRA weights in autocast's cache: the training forward would reuse
    them, detached from the real weights (no LoRA gradient; CheckpointError with ckpt)."""
    from peft import LoraConfig, get_peft_model
    tok = make_tokenizer()
    model = get_peft_model(tiny_llama(len(tok)), LoraConfig(r=4, lora_alpha=8, target_modules="all-linear"))
    if ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    obj = GRPOObjective(_args(algo_beta=0.04), tok)
    obj.prepare(model)
    model.train()
    torch.manual_seed(0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = obj(model, _batch(tok), torch.device("cpu"))
    loss.backward()
    lora = [p for n, p in model.named_parameters() if "lora_B" in n]   # A gets 0 while B = 0
    assert lora and all(p.grad is not None and p.grad.abs().sum() > 0 for p in lora)

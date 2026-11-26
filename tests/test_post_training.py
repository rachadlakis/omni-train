"""Post-training (algorithm: sft | dpo | simpo | orpo; GRPO: test_grpo.py): config rules, loss
formulas, data masking, and objectives on a tiny random Llama. All offline, CPU.
The multi-process gradient checks live in test_post_training_dist.py (-m dist)."""

import copy
import math

import pytest
import torch
import torch.nn.functional as F

from post_training import losses as L
from post_training.data import (
    IGNORE_INDEX, collate_preference, collate_sft, split_preference_row,
    tokenize_preference, tokenize_sft,
)
from post_training.objectives import PreferenceObjective, SFTObjective
from tests.post_training_helpers import PAIRS, make_tokenizer, tiny_llama
from utils import build_args


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

def _with_algo(cfg, **algo):
    cfg = copy.deepcopy(cfg)
    cfg["algorithm"] = algo
    return cfg


class TestConfig:
    def test_default_is_pretrain(self, cfg):
        assert build_args(cfg).algorithm == "pretrain"

    @pytest.mark.parametrize("algo,beta", [("dpo", 0.1), ("simpo", 2.0), ("orpo", 0.1)])
    def test_default_beta(self, cfg, algo, beta):
        assert build_args(_with_algo(cfg, type=algo)).algo_beta == beta

    def test_default_max_prompt_length(self, cfg):
        assert build_args(_with_algo(cfg, type="dpo")).algo_max_prompt_length == 16  # max_length 32

    def test_unknown_algorithm(self, cfg):
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            build_args(_with_algo(cfg, type="rlhf"))

    def test_unknown_key(self, cfg):
        with pytest.raises(ValueError, match="Unknown algorithm keys"):
            build_args(_with_algo(cfg, type="dpo", betta=0.1))

    def test_needs_llm(self, cfg):
        cfg["model_type"] = "vision"
        with pytest.raises(ValueError, match="model_type: llm"):
            build_args(_with_algo(cfg, type="sft"))

    def test_no_5d(self, cfg):
        cfg["strategy"], cfg["model_type"], cfg["num_gpus"] = "5d", "llm", 1
        cfg["model_name"] = "meta-llama/Llama-3.2-1B"
        with pytest.raises(ValueError, match="solo, ddp or fsdp"):
            build_args(_with_algo(cfg, type="dpo"))

    @pytest.mark.parametrize("strategy", ["solo", "ddp", "fsdp"])
    def test_strategies_ok(self, cfg, strategy):
        cfg["strategy"] = strategy
        assert build_args(_with_algo(cfg, type="dpo")).algorithm == "dpo"

    def test_bad_dpo_loss(self, cfg):
        with pytest.raises(ValueError, match="dpo_loss"):
            build_args(_with_algo(cfg, type="dpo", dpo_loss="kto"))

    def test_label_smoothing_only_sigmoid(self, cfg):
        with pytest.raises(ValueError, match="label_smoothing"):
            build_args(_with_algo(cfg, type="dpo", dpo_loss="ipo", label_smoothing=0.1))

    def test_prompt_length_below_max_length(self, cfg):
        with pytest.raises(ValueError, match="max_prompt_length"):
            build_args(_with_algo(cfg, type="dpo", max_prompt_length=32))

    def test_beta_positive(self, cfg):
        with pytest.raises(ValueError, match="beta"):
            build_args(_with_algo(cfg, type="dpo", beta=0))

    def test_subset_optional(self, cfg):
        del cfg["dataset"]["subset"]
        assert build_args(cfg).dataset_full_name is None

    def test_lora_with_dpo_ok(self, cfg):
        cfg["strategy"] = "ddp"
        cfg["peft"]["enabled"] = True
        args = build_args(_with_algo(cfg, type="dpo"))
        assert args.peft_enabled and args.algorithm == "dpo"


# ----------------------------------------------------------------------
# Loss formulas (checked against hand-written math)
# ----------------------------------------------------------------------

class TestLosses:
    def test_sequence_logps_matches_naive(self):
        torch.manual_seed(0)
        logits = torch.randn(3, 6, 11)
        labels = torch.randint(0, 11, (3, 6))
        labels[0, :3] = IGNORE_INDEX
        labels[2, 4:] = IGNORE_INDEX
        sums, counts = L.sequence_logps(logits, labels)
        logp = F.log_softmax(logits, -1)
        for b in range(3):
            want, n = 0.0, 0
            for t in range(1, 6):
                if labels[b, t] != IGNORE_INDEX:
                    want += logp[b, t - 1, labels[b, t]].item()
                    n += 1
            assert sums[b].item() == pytest.approx(want, abs=1e-5)
            assert counts[b].item() == n

    def test_sequence_logps_bf16_input_is_fp32(self):
        sums, _ = L.sequence_logps(torch.randn(1, 4, 5).bfloat16(), torch.tensor([[1, 2, 3, 4]]))
        assert sums.dtype == torch.float32

    def _pair(self):
        return (torch.tensor([-3.0, -5.0]), torch.tensor([-4.0, -4.5]),
                torch.tensor([-3.5, -5.0]), torch.tensor([-3.8, -4.0]))

    def test_dpo_sigmoid(self):
        pc, pr, rc, rr = self._pair()
        losses, cr, rj = L.dpo_loss(pc, pr, rc, rr, beta=0.5)
        for i in range(2):
            h = (pc[i] - rc[i]) - (pr[i] - rr[i])
            assert losses[i].item() == pytest.approx(-math.log(1 / (1 + math.exp(-0.5 * h))), abs=1e-6)
            assert cr[i].item() == pytest.approx(0.5 * (pc[i] - rc[i]).item())
            assert rj[i].item() == pytest.approx(0.5 * (pr[i] - rr[i]).item())

    def test_dpo_at_reference_is_log2(self):
        x = torch.tensor([-2.0, -7.0])
        losses, _, _ = L.dpo_loss(x, x + 1, x, x + 1, beta=0.1)
        assert torch.allclose(losses, torch.full((2,), math.log(2)))

    def test_dpo_label_smoothing(self):
        pc, pr, rc, rr = self._pair()
        h = (pc - rc) - (pr - rr)
        losses, _, _ = L.dpo_loss(pc, pr, rc, rr, beta=0.3, label_smoothing=0.2)
        want = -0.8 * F.logsigmoid(0.3 * h) - 0.2 * F.logsigmoid(-0.3 * h)
        assert torch.allclose(losses, want)

    def test_ipo(self):
        pc, pr, rc, rr = self._pair()
        h = (pc - rc) - (pr - rr)
        losses, _, _ = L.dpo_loss(pc, pr, rc, rr, beta=0.25, kind="ipo")
        assert torch.allclose(losses, (h - 2.0) ** 2)

    def test_hinge(self):
        pc, pr, rc, rr = self._pair()
        h = (pc - rc) - (pr - rr)
        losses, _, _ = L.dpo_loss(pc, pr, rc, rr, beta=2.0, kind="hinge")
        assert torch.allclose(losses, torch.clamp(1 - 2.0 * h, min=0))

    def test_simpo(self):
        c, r = torch.tensor([-0.5, -1.0]), torch.tensor([-0.7, -0.6])
        losses, _, _ = L.simpo_loss(c, r, beta=2.0, gamma=0.5)
        want = [-math.log(1 / (1 + math.exp(-(2.0 * (ci - ri) - 0.5)))) for ci, ri in zip(c.tolist(), r.tolist())]
        assert losses.tolist() == pytest.approx(want, abs=1e-6)

    def test_orpo(self):
        c, r = torch.tensor([-0.5, -2.0]), torch.tensor([-0.9, -0.1])
        losses, _, _ = L.orpo_loss(c, r, lam=0.1)
        for i in range(2):
            pc_, pr_ = math.exp(c[i]), math.exp(r[i])
            log_odds = math.log(pc_ / (1 - pc_)) - math.log(pr_ / (1 - pr_))
            want = -c[i].item() - 0.1 * math.log(1 / (1 + math.exp(-log_odds)))
            assert losses[i].item() == pytest.approx(want, abs=1e-5)

    def test_orpo_finite_near_zero_logp(self):
        losses, _, _ = L.orpo_loss(torch.tensor([0.0, -1e-9]), torch.tensor([-50.0, -30.0]), lam=0.1)
        assert torch.isfinite(losses).all()

    def test_dpo_gradient_pushes_chosen_up(self):
        pc = torch.tensor([-3.0], requires_grad=True)
        pr = torch.tensor([-3.0], requires_grad=True)
        L.dpo_loss(pc, pr, torch.tensor([-3.0]), torch.tensor([-3.0]), beta=0.1)[0].sum().backward()
        assert pc.grad.item() < 0 < pr.grad.item()  # descent raises chosen, lowers rejected
        assert pc.grad.item() == pytest.approx(-0.05)  # -beta * sigmoid(0)


# ----------------------------------------------------------------------
# Data: masking and pair construction
# ----------------------------------------------------------------------

def _decode_trained(tok, ids, labels):
    return tok.decode([i for i, l in zip(ids, labels) if l != IGNORE_INDEX])


class TestSFTData:
    def test_messages_mask_only_assistant_turns(self):
        tok = make_tokenizer()
        row = {"messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hello world"},
            {"role": "user", "content": "tell me a joke"},
            {"role": "assistant", "content": "sure why because"},
        ]}
        out = tokenize_sft(row, tok, max_length=100)
        trained = _decode_trained(tok, out["input_ids"], out["labels"])
        assert trained.split() == "hello world <|end|> sure why because <|end|>".split()
        full = tok.decode(out["input_ids"])
        assert full.split()[:2] == ["<|user|>", "hello"]

    def test_prompt_completion_with_template(self):
        tok = make_tokenizer()
        out = tokenize_sft({"prompt": "what is two plus two", "completion": "four"}, tok, 100)
        assert _decode_trained(tok, out["input_ids"], out["labels"]).split() == ["four", "<|end|>"]
        assert tok.decode(out["input_ids"]).split() == \
            "<|user|> what is two plus two <|end|> <|assistant|> four <|end|>".split()

    def test_prompt_completion_without_template(self):
        tok = make_tokenizer(chat=False)
        out = tokenize_sft({"prompt": "what is two plus two", "completion": "four"}, tok, 100)
        assert _decode_trained(tok, out["input_ids"], out["labels"]).split() == ["four", "<eos>"]

    def test_completion_only_false_trains_everything(self):
        tok = make_tokenizer()
        out = tokenize_sft({"prompt": "hello", "completion": "world"}, tok, 100, completion_only=False)
        assert out["labels"] == out["input_ids"]

    def test_alpaca(self):
        tok = make_tokenizer()
        out = tokenize_sft({"instruction": "tell me a joke", "input": "", "output": "ok"}, tok, 100)
        assert _decode_trained(tok, out["input_ids"], out["labels"]).split() == ["ok", "<|end|>"]

    def test_plain_text_trains_all(self):
        tok = make_tokenizer()
        out = tokenize_sft({"text": "hello world"}, tok, 100)
        assert out["labels"] == out["input_ids"]

    def test_truncation(self):
        tok = make_tokenizer()
        out = tokenize_sft({"prompt": "hello", "completion": "a b c d e f"}, tok, 5)
        assert len(out["input_ids"]) == len(out["labels"]) == 5

    def test_collate_pads_with_ignore(self):
        batch = collate_sft([{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]},
                             {"input_ids": [5], "labels": [5]}], pad_id=0)
        assert batch["input_ids"].tolist() == [[5, 6, 7], [5, 0, 0]]
        assert batch["labels"].tolist() == [[-100, 6, 7], [5, -100, -100]]
        assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


class TestPreferenceData:
    def test_explicit_prompt(self):
        assert split_preference_row({"prompt": "p", "chosen": "a", "rejected": "b"}) == ("p", "a", "b")

    def test_hh_rlhf_strings(self):
        row = {"chosen": "\n\nHuman: hi\n\nAssistant: hello there",
               "rejected": "\n\nHuman: hi\n\nAssistant: go away"}
        p, c, r = split_preference_row(row)
        assert p == "\n\nHuman: hi\n\nAssistant:" and c == " hello there" and r == " go away"

    def test_conversational_without_prompt(self):
        u = {"role": "user", "content": "hello"}
        row = {"chosen": [u, {"role": "assistant", "content": "good"}],
               "rejected": [u, {"role": "assistant", "content": "bad"}]}
        p, c, r = split_preference_row(row)
        assert p == [u] and c["content"] == "good" and r["content"] == "bad"

    def test_mismatched_prompts_rejected(self):
        row = {"chosen": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "x"}],
               "rejected": [{"role": "user", "content": "b"}, {"role": "assistant", "content": "y"}]}
        with pytest.raises(ValueError, match="share"):
            split_preference_row(row)

    def test_tokenize_and_collate(self):
        tok = make_tokenizer()
        rows = []
        for i, pair in enumerate(PAIRS[:2]):
            r = tokenize_preference(pair, tok, max_length=64, max_prompt_length=32)
            r["idx"] = i
            rows.append(r)
        batch = collate_preference(rows, pad_id=tok.pad_token_id)
        assert batch["input_ids"].shape[0] == 4
        assert batch["idx"].tolist() == [0, 1]
        # chosen rows first, rejected after; only the answers are trained
        got = [_decode_trained(tok, ids.tolist(), lab.tolist()).split()
               for ids, lab in zip(batch["input_ids"], batch["labels"])]
        assert got == [["four", "<|end|>"], ["sure", "why", "because", "<|end|>"],
                       ["five", "<|end|>"], ["no", "<|end|>"]]

    def test_prompt_cut_from_the_left(self):
        tok = make_tokenizer()
        r = tokenize_preference({"prompt": "the answer is two plus two", "chosen": "four",
                                 "rejected": "five"}, tok, max_length=10, max_prompt_length=3)
        assert len(r["prompt_ids"]) == 3
        assert tok.decode(r["prompt_ids"]).split()[-1] == "<|assistant|>"  # the end is kept
        assert len(r["prompt_ids"]) + len(r["chosen_ids"]) <= 10


# ----------------------------------------------------------------------
# Objectives on a tiny model
# ----------------------------------------------------------------------

class _Args:
    algorithm, algo_beta, algo_dpo_loss, algo_label_smoothing, algo_simpo_gamma = "dpo", 0.1, "sigmoid", 0.0, 0.5


def _pref_dataset(tok):
    from datasets import Dataset
    rows = [tokenize_preference(p, tok, 64, 32) | {"idx": i} for i, p in enumerate(PAIRS)]
    return Dataset.from_list(rows)


class TestObjectives:
    def test_sft_objective_matches_manual_ce(self):
        tok = make_tokenizer()
        model = tiny_llama(len(tok))
        rows = [tokenize_sft(p | {"completion": p["chosen"]}, tok, 64) for p in PAIRS]
        batch = collate_sft(rows, tok.pad_token_id)
        loss, _ = SFTObjective()(model, batch, torch.device("cpu"))
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        want = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               batch["labels"][:, 1:].reshape(-1), ignore_index=IGNORE_INDEX)
        assert loss.item() == pytest.approx(want.item(), abs=1e-5)

    def test_dpo_starts_at_log2_and_learns(self):
        tok = make_tokenizer()
        model = tiny_llama(len(tok))
        ds = _pref_dataset(tok)
        collate = lambda rows: collate_preference(rows, tok.pad_token_id)  # noqa: E731
        obj = PreferenceObjective(_Args())
        obj.precompute_reference(model, ds, collate, 0, 1, batch_size=3, device=torch.device("cpu"))
        assert obj.ref.shape == (4, 2) and torch.isfinite(obj.ref).all()

        batch = collate([ds[i] for i in range(4)])
        model.train()
        loss, metrics = obj(model, batch, torch.device("cpu"))
        assert loss.item() == pytest.approx(math.log(2), abs=1e-5)  # policy == reference
        assert metrics["margin"] == pytest.approx(0.0, abs=1e-5)

        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        for _ in range(15):
            opt.zero_grad()
            loss, metrics = obj(model, batch, torch.device("cpu"))
            loss.backward()
            opt.step()
        assert loss.item() < 0.5 and metrics["acc"] == 1.0

    def test_reference_matches_direct_logps(self):
        tok = make_tokenizer()
        model = tiny_llama(len(tok)).eval()
        ds = _pref_dataset(tok)
        collate = lambda rows: collate_preference(rows, tok.pad_token_id)  # noqa: E731
        obj = PreferenceObjective(_Args())
        obj.precompute_reference(model, ds, collate, 0, 1, batch_size=4, device=torch.device("cpu"))
        with torch.no_grad():
            sums, _ = obj.logps(model, collate([ds[i] for i in range(4)]))
        assert torch.allclose(obj.ref[:, 0], sums[:4], atol=1e-5)
        assert torch.allclose(obj.ref[:, 1], sums[4:], atol=1e-5)

    @pytest.mark.parametrize("algo", ["simpo", "orpo"])
    def test_reference_free_objectives_learn(self, algo):
        tok = make_tokenizer()
        model = tiny_llama(len(tok)).train()
        ds = _pref_dataset(tok)
        args = _Args()
        args.algorithm, args.algo_beta = algo, (2.0 if algo == "simpo" else 0.1)
        obj = PreferenceObjective(args)
        assert not obj.needs_reference
        batch = collate_preference([ds[i] for i in range(4)], tok.pad_token_id)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        first = None
        for _ in range(15):
            opt.zero_grad()
            loss, metrics = obj(model, batch, torch.device("cpu"))
            first = first if first is not None else loss.item()
            loss.backward()
            opt.step()
        assert loss.item() < first and metrics["acc"] == 1.0

    def test_disable_dropout(self):
        from peft import LoraConfig, get_peft_model
        from post_training.objectives import disable_dropout
        from transformers import OPTConfig, OPTForCausalLM
        model = OPTForCausalLM(OPTConfig(vocab_size=50, hidden_size=16, ffn_dim=32, num_hidden_layers=2,
                                         num_attention_heads=2, max_position_embeddings=64,
                                         word_embed_proj_dim=16, dropout=0.3, attention_dropout=0.2))
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.5, target_modules="all-linear"))
        assert disable_dropout(model) > 0
        model.train()
        x = torch.randint(0, 50, (2, 10))
        with torch.no_grad():
            assert torch.equal(model(input_ids=x).logits, model(input_ids=x).logits)
        assert disable_dropout(model) == 0  # nothing left

    def test_preference_step0_equals_reference_in_train_mode(self):
        """DPO starts at ln 2 in TRAIN mode once dropout is off."""
        tok = make_tokenizer()
        model = tiny_llama(len(tok))
        model.config.attention_dropout = 0.3
        for layer in model.model.layers:
            layer.self_attn.attention_dropout = 0.3
        obj = PreferenceObjective(_Args())
        assert obj.prepare(model) > 0
        ds = _pref_dataset(tok)
        collate = lambda rows: collate_preference(rows, tok.pad_token_id)  # noqa: E731
        obj.precompute_reference(model, ds, collate, 0, 1, batch_size=2, device=torch.device("cpu"))
        model.train()
        loss, _ = obj(model, collate([ds[i] for i in range(4)]), torch.device("cpu"))
        assert loss.item() == pytest.approx(math.log(2), abs=1e-5)

    def test_dpo_with_lora(self):
        from peft import LoraConfig, get_peft_model
        tok = make_tokenizer()
        model = get_peft_model(tiny_llama(len(tok)), LoraConfig(r=4, lora_alpha=8, target_modules="all-linear"))
        ds = _pref_dataset(tok)
        collate = lambda rows: collate_preference(rows, tok.pad_token_id)  # noqa: E731
        obj = PreferenceObjective(_Args())
        obj.precompute_reference(model, ds, collate, 0, 1, batch_size=4, device=torch.device("cpu"))
        # LoRA B starts at zero, so the reference equals the base model
        with torch.no_grad(), model.disable_adapter():
            base, _ = obj.logps(model, collate([ds[i] for i in range(4)]))
        assert torch.allclose(obj.ref[:, 0], base[:4], atol=1e-5)
        model.train()
        loss, _ = obj(model, collate([ds[i] for i in range(4)]), torch.device("cpu"))
        loss.backward()
        assert all(p.grad is None for n, p in model.named_parameters() if "lora_" not in n)
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for n, p in model.named_parameters() if "lora_B" in n)

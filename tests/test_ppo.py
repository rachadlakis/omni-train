"""PPO: GAE, losses, config rules, reward models and the objective on a tiny random Llama.
All offline, CPU. Multi-process gradient checks: test_post_training_dist.py."""

import copy
import math
import types

import pytest
import torch

from omni_train.post_training import losses as L
from omni_train.post_training.data import collate_grpo, tokenize_grpo
from omni_train.post_training.grpo import completion_mask_from
from omni_train.post_training.ppo import PPOObjective, normalize_advantages
from omni_train.post_training.reward_model import RewardModel, parse_reward_model
from tests.post_training_helpers import PAIRS, make_tokenizer, tiny_llama
from omni_train.utils import build_args

CPU = torch.device("cpu")


# ----------------------------------------------------------------------
# Math
# ----------------------------------------------------------------------

@pytest.mark.parametrize("gamma,lam", [(1.0, 0.95), (0.9, 0.8), (1.0, 1.0), (0.99, 0.0)])
def test_gae_matches_the_definition(gamma, lam):
    """A_t = sum_k (gamma lam)^k delta_{t+k}, episodes of different lengths."""
    torch.manual_seed(0)
    lens, C = [5, 2, 1, 4], 5
    r, v = torch.randn(4, C), torch.randn(4, C)
    mask = (torch.arange(C)[None] < torch.tensor(lens)[:, None]).long()
    adv, ret = L.gae(r * mask, v, mask, gamma, lam)      # v has garbage past the end on purpose
    for i, n in enumerate(lens):
        for t in range(C):
            want = 0.0
            for k in range(t, n):
                nxt = v[i, k + 1].item() if k + 1 < n else 0.0
                want += (gamma * lam) ** (k - t) * (r[i, k].item() + gamma * nxt - v[i, k].item())
            assert adv[i, t].item() == pytest.approx(want if t < n else 0.0, abs=1e-5)
            assert ret[i, t].item() == pytest.approx(want + v[i, t].item() if t < n else 0.0, abs=1e-5)


def test_gae_lambda_one_is_the_discounted_return():
    r, v, mask = torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([[0.3, -0.2, 0.5]]), torch.ones(1, 3)
    _, ret = L.gae(r, v, mask, gamma=0.5, lam=1.0)
    assert ret.tolist()[0] == pytest.approx([0.25, 0.5, 1.0])


def test_ppo_loss_by_hand():
    lp = torch.log(torch.tensor([[1.5, 0.5, 1.0]]))            # ratios (old log-prob 0)
    adv, mask = torch.tensor([[1.0, 1.0, -2.0]]), torch.tensor([[1, 1, 0]])
    v, v0, ret = torch.tensor([[1.0, 0.0, 9.0]]), torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([[2.0, 1.0, 0.0]])
    loss, st = L.ppo_loss(lp, torch.zeros(1, 3), adv, mask, v, v0, ret, epsilon=0.2, value_clip=0.2, vf_coef=0.5)
    pg = (-min(1.5, 1.2) - min(0.5, 0.8)) / 2
    vf = 0.5 * (max((1 - 2) ** 2, (0.2 - 2) ** 2) + max(1.0, 1.0)) / 2     # clipped V: 0 + 0.2
    assert st["pg"].item() == pytest.approx(pg) and st["vf"].item() == pytest.approx(vf)
    assert loss.item() == pytest.approx(pg + 0.5 * vf) and st["clip_frac"].item() == 1.0
    _, off = L.ppo_loss(lp, torch.zeros(1, 3), adv, mask, v, v0, ret, value_clip=0.0)
    assert off["vf"].item() == pytest.approx(0.5 * (1 + 1) / 2)


def test_on_policy_gradient_is_the_advantage():
    lp = torch.tensor([[-1.0, -2.0], [-0.5, -3.0]], requires_grad=True)
    adv, mask = torch.tensor([[1.0, -2.0], [0.5, 3.0]]), torch.tensor([[1, 1], [1, 0]])
    z = torch.zeros(2, 2)
    L.ppo_loss(lp, lp.detach(), adv, mask, z, z, z)[0].backward()
    torch.testing.assert_close(lp.grad, -adv * mask / 3)


def test_advantages_are_normalised_over_real_tokens_only():
    adv, mask = torch.tensor([[1.0, 3.0, 99.0], [5.0, 99.0, 99.0]]), torch.tensor([[1, 1, 0], [1, 0, 0]])
    n = normalize_advantages(adv, mask)
    real = n[mask.bool()]
    assert real.mean().item() == pytest.approx(0, abs=1e-6) and real.pow(2).mean().item() == pytest.approx(1, abs=1e-5)
    assert (n[~mask.bool()] == 0).all()
    assert torch.isfinite(normalize_advantages(torch.ones(1, 2), torch.ones(1, 2))).all()     # zero variance


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

def _ppo(cfg, **algo):
    cfg = copy.deepcopy(cfg)
    cfg["algorithm"] = {"type": "ppo", **algo}
    return cfg


class TestConfig:
    def test_defaults(self, cfg):
        a = build_args(_ppo(cfg))
        assert (a.algorithm, a.algo_num_generations, a.algo_ppo_epochs, a.algo_beta) == ("ppo", 1, 4, 0.0)
        assert a.algo_rewards == ["accuracy"] and a.algo_reward_model is None

    def test_reward_model_replaces_the_default_reward(self, cfg):
        a = build_args(_ppo(cfg, reward_model="org/rm"))
        assert a.algo_rewards is None and a.algo_reward_model["name"] == "org/rm" and a.algo_reward_model["weight"] == 1.0
        both = build_args(_ppo(cfg, reward_model={"name": "org/rm", "weight": 0.5}, rewards=["length"]))
        assert both.algo_rewards == ["length"] and both.algo_reward_model["weight"] == 0.5

    def test_beta_needs_lora(self, cfg):
        with pytest.raises(ValueError, match="ppo with beta > 0 needs LoRA"):
            build_args(_ppo(cfg, beta=0.05))
        cfg["peft"]["enabled"] = True
        assert build_args(_ppo(cfg, beta=0.05)).algo_beta == 0.05

    @pytest.mark.parametrize("bad, match", [
        ({"num_generations": 0}, "num_generations"), ({"ppo_epochs": 0}, "ppo_epochs"),
        ({"num_minibatches": 3}, "must divide"), ({"gamma": 1.5}, "gamma"), ({"gae_lambda": -0.1}, "gae_lambda"),
        ({"vf_coef": 0}, "vf_coef"), ({"value_lr": 0}, "value_lr"), ({"value_clip": -1}, "value_clip"),
        ({"eos_penalty": -1}, "eos_penalty"), ({"rewards": ["vibes"]}, "Unknown reward"), ({"rewards": {}}, "rewards must be"),
        ({"reward_model": {"weight": 1}}, "reward_model"), ({"reward_model": {"name": "x", "wieght": 1}}, "Unknown reward_model"),
        ({"loss_type": "dapo"}, "only apply to type: grpo"), ({"temperature": 0}, "temperature"),
    ])
    def test_rejects(self, cfg, bad, match):
        with pytest.raises(ValueError, match=match):
            build_args(_ppo(cfg, **bad))

    @pytest.mark.parametrize("algo", ["grpo", "dpo"])
    def test_ppo_keys_on_other_algorithms(self, cfg, algo):
        cfg["algorithm"] = {"type": algo, "gae_lambda": 0.9}
        with pytest.raises(ValueError, match="only apply to type: ppo"):
            build_args(cfg)

    def test_no_5d(self, cfg):
        c = _ppo(cfg)
        c["strategy"] = "5d"
        with pytest.raises(ValueError):
            build_args(c)


# ----------------------------------------------------------------------
# Objective
# ----------------------------------------------------------------------

def _args(**kw):
    base = dict(algo_num_generations=2, algo_max_completion_length=6, algo_temperature=1.0, algo_top_p=1.0, algo_top_k=0,
                algo_beta=0.0, algo_epsilon=0.2, algo_epsilon_high=0.2, algo_loss_type="grpo", algo_scale_rewards=True,
                algo_rewards=["contains"], algo_reward_model=None, strategy="solo", algo_ppo_epochs=2, algo_num_minibatches=2,
                algo_gamma=1.0, algo_gae_lambda=0.95, algo_vf_coef=0.1, algo_value_clip=0.2, algo_value_lr=1e-2,
                algo_eos_penalty=0.0)
    return types.SimpleNamespace(**(base | kw))


def _batch(tok):
    rows = [tokenize_grpo({"prompt": p["prompt"], "answer": p["chosen"]}, tok, 32) | {"idx": i} for i, p in enumerate(PAIRS)]
    return collate_grpo(rows, tok.pad_token_id)


def _lora(model):
    from peft import LoraConfig, get_peft_model
    return get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules="all-linear"))


def _setup(lora=False, **kw):
    tok = make_tokenizer()
    model = tiny_llama(len(tok))
    model = _lora(model) if lora else model
    obj = PPOObjective(_args(**kw), tok)
    obj.prepare(model)
    return tok, model, obj


def test_values_read_the_state_before_each_token():
    tok, model, obj = _setup()
    torch.nn.init.normal_(obj.value_head.weight)
    ids = torch.randint(8, len(tok), (2, 9))
    _, values = obj.logps_and_values(model, ids, torch.ones_like(ids), 4)
    hidden = model.model(input_ids=ids).last_hidden_state           # what lm_head sees
    torch.testing.assert_close(values, obj.value_head(hidden[:, 4:8]).squeeze(-1), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("ckpt", [False, True])
@pytest.mark.parametrize("lora", [False, True])
def test_first_update_starts_on_policy(lora, ckpt):
    """The training forward reproduces the rollout's log-probs and values: ratio 1, V = V_old."""
    tok, model, obj = _setup(lora)
    if ckpt:
        model.gradient_checkpointing_enable()
        if lora:
            model.enable_input_require_grads()
    torch.nn.init.normal_(obj.value_head.weight, std=0.1)
    roll = obj.rollout(model, _batch(tok), CPU)
    model.train()
    lp, v = obj.logps_and_values(model, roll.input_ids, roll.attention_mask, roll.completion_mask.shape[1])
    m = roll.completion_mask.bool()
    torch.testing.assert_close(lp[m], roll.old_logps[m], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(v[m], roll.old_values[m], atol=1e-4, rtol=1e-4)
    loss, metrics = obj(model, roll, CPU)
    loss.backward()
    assert metrics["clip"] == 0.0
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert any(g.abs().sum() > 0 for g in grads) and obj.value_head.weight.grad.abs().sum() > 0


def test_score_is_on_the_last_token_and_cut_answers_are_penalised():
    tok, model, obj = _setup(algo_gae_lambda=1.0, algo_eos_penalty=0.5)
    eos = tok.eos_token_id
    pid = torch.full((2, 3), 10)
    cid = torch.tensor([[11, eos, eos], [11, 12, 13]])
    mask = completion_mask_from(cid, [eos])
    roll = obj.build(model, pid, torch.ones_like(pid), cid, mask, torch.tensor([1.0, 1.0]), torch.tensor([True, False]))
    # V = 0 (zero-initialised head), gamma = lambda = 1: every real token's return is the score
    assert roll.returns.tolist() == [[1.0, 1.0, 0.0], [0.5, 0.5, 0.5]]


def test_kl_penalty_lowers_the_returns():
    tok, model, obj = _setup(lora=True, algo_beta=0.5, algo_gae_lambda=1.0)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.2)
    pid, cid = torch.full((1, 3), 10), torch.tensor([[11, 12, 13]])
    ids = torch.cat([pid, cid], 1)
    roll = obj.build(model, pid, torch.ones_like(pid), cid, torch.ones_like(cid), torch.tensor([1.0]))
    with torch.no_grad():
        lp = obj.completion_logps(model, ids, torch.ones_like(ids), 3)
        with model.disable_adapter():
            ref = obj.completion_logps(model, ids, torch.ones_like(ids), 3)
    kl = lp - ref
    assert kl.abs().sum() > 1e-3
    assert roll.returns[0, 0].item() == pytest.approx(1.0 - 0.5 * kl.sum().item(), abs=1e-4)
    assert roll.metrics["kl"] == pytest.approx(kl.sum().item(), abs=1e-4)


def test_loader_yields_every_row_once_per_epoch():
    tok, model, obj = _setup(algo_ppo_epochs=3, algo_num_minibatches=4)
    loader = obj.wrap_loader([_batch(tok), _batch(tok)], model, CPU)
    mbs = list(loader)
    assert len(mbs) == len(loader) == 2 * 3 * 4 and all(len(mb) == 2 for mb in mbs)   # 4 prompts x 2 / 4
    epoch = torch.cat([mb.input_ids for mb in mbs[:4]])
    again = torch.cat([mb.input_ids for mb in mbs[4:8]])
    assert sorted(map(tuple, epoch.tolist())) == sorted(map(tuple, again.tolist()))


def _train(model, obj, tok, rollouts, lr=1e-2):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    hist = []
    for mb in obj.wrap_loader([_batch(tok)] * rollouts, model, CPU):
        opt.zero_grad()
        loss, m = obj(model, mb, CPU)
        loss.backward()
        opt.step()
        obj.step_extra(scaler, 1.0)
        hist.append(m)
    return hist


def test_learns_a_simple_reward_and_its_value():
    torch.manual_seed(0)
    tok, model, obj = _setup(algo_num_generations=8, algo_ppo_epochs=4, algo_num_minibatches=2)
    obj.rewards = [("yes", lambda p, c, a, **k: [float(x.split().count("yes")) / 6 for x in c], 1.0)]
    h = _train(model, obj, tok, 25)
    per = [x["reward"] for x in h[::8]]
    assert sum(per[-3:]) / 3 > sum(per[:3]) / 3 + 0.3, per
    assert h[-1]["v_err"] < 0.1                        # V follows the returns, which went from ~0 to ~1


def test_value_head_fits_returns_of_a_frozen_policy():
    torch.manual_seed(0)
    tok, model, obj = _setup(algo_num_generations=8, algo_value_lr=3e-2, algo_value_clip=0.0)
    obj.rewards = [("len", lambda p, c, a, completion_lengths=None, **k: [n / 6 for n in completion_lengths], 1.0)]
    roll, scaler, losses = obj.rollout(model, _batch(tok), CPU), torch.amp.GradScaler("cpu", enabled=False), []
    for _ in range(300):
        loss, m = obj(model, roll, CPU)
        loss.backward()
        obj.step_extra(scaler, 0.0)
        losses.append(m["vf"])
    assert losses[-1] < losses[0] / 20


def test_value_head_survives_save_and_resume(tmp_path):
    tok, model, obj = _setup()
    _train(model, obj, tok, 2)
    for ck in (tmp_path / "123.pt", tmp_path / "456"):
        if ck.suffix != ".pt":
            ck.mkdir()
        obj.save_extra(str(ck))
        _, _, fresh = _setup(algo_value_lr=3e-4)
        assert fresh.load_extra(str(ck))
        assert torch.equal(fresh.value_head.weight, obj.value_head.weight) and fresh.value_head.weight.abs().sum() > 0
        assert fresh.value_opt.state_dict()["state"] and fresh.value_opt.param_groups[0]["lr"] == 3e-4
    assert not fresh.load_extra(str(tmp_path / "missing.pt"))
    assert not any("value" in n for n in model.state_dict())        # never part of the policy


def test_model_without_linear_output_layer_is_refused():
    tok = make_tokenizer()
    model = tiny_llama(len(tok))
    model.get_output_embeddings = lambda: None
    with pytest.raises(ValueError, match="linear output layer"):
        PPOObjective(_args(), tok).prepare(model)


# ----------------------------------------------------------------------
# Reward models
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def rm_dir(tmp_path_factory):
    from transformers import LlamaConfig, LlamaForSequenceClassification
    path, tok = str(tmp_path_factory.mktemp("rm")), make_tokenizer()
    torch.manual_seed(1)
    LlamaForSequenceClassification(LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                               num_attention_heads=4, num_labels=1, pad_token_id=tok.pad_token_id)).save_pretrained(path)
    tok.save_pretrained(path)
    return path


def test_reward_model_scores_match_the_model_one_by_one(rm_dir):
    from transformers import AutoModelForSequenceClassification
    rm = RewardModel(rm_dir, CPU, batch_size=2)
    prompts, comps = [p["prompt"] for p in PAIRS], [p["chosen"] for p in PAIRS]
    convs = [[{"role": "user", "content": p}] for p in prompts]
    got = rm(prompts, comps, None, conversations=convs)
    ref = AutoModelForSequenceClassification.from_pretrained(rm_dir, dtype=torch.float32).eval()
    tok = make_tokenizer()
    for g, conv, c in zip(got, convs, comps):
        text = tok.apply_chat_template(conv + [{"role": "assistant", "content": c}], tokenize=False)
        ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]])
        assert g == pytest.approx(ref(ids).logits.item(), abs=1e-4)      # padding in the batch changes nothing
    plain = rm(prompts, comps)                                            # no conversation: prompt + completion text
    assert plain != got and all(math.isfinite(x) for x in plain)
    assert not any(p.requires_grad for p in rm.model.parameters())


def test_not_a_reward_model_is_refused(tmp_path, rm_dir):
    tok = make_tokenizer()
    tiny_llama(len(tok)).save_pretrained(tmp_path / "lm")
    tok.save_pretrained(tmp_path / "lm")
    with pytest.raises(ValueError, match="not a reward model|outputs"):
        RewardModel(str(tmp_path / "lm"), CPU)


def test_parse_reward_model():
    assert parse_reward_model(None) is None
    assert parse_reward_model("a/b") == {"name": "a/b", "weight": 1.0, "max_length": 1024, "batch_size": 8}


@pytest.mark.parametrize("algo", ["ppo", "grpo"])
def test_rollout_with_a_reward_model(rm_dir, algo):
    from omni_train.post_training.grpo import GRPOObjective
    tok = make_tokenizer()
    model = tiny_llama(len(tok))
    spec = parse_reward_model({"name": rm_dir, "weight": 2.0})
    obj = (PPOObjective if algo == "ppo" else GRPOObjective)(_args(algo_rewards=None, algo_reward_model=spec), tok)
    obj.prepare(model)
    s = obj.sample(model, _batch(tok), CPU)
    direct = RewardModel(rm_dir, CPU)
    texts = [tok.decode(r[:n], skip_special_tokens=True) for r, n in zip(s["completion_ids"].tolist(), s["completion_mask"].sum(1).tolist())]
    convs = [[{"role": "user", "content": p["prompt"]}] for p in PAIRS for _ in range(2)]
    want = direct([""] * 8, texts, None, conversations=convs)
    assert s["rewards"].tolist() == pytest.approx([2 * w for w in want], abs=1e-4)

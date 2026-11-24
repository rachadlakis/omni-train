# Post-training: SFT, DPO, IPO, SimPO, ORPO, GRPO

Pick the algorithm with an `algorithm:` block in the config. Everything else (model loading,
solo / DDP / FSDP, LoRA / QLoRA, mixed precision, LR schedule, gradient clipping,
checkpoints, `launch.sh`, SLURM) works the same as for normal fine-tuning. Only the data
pipeline and the per-step loss change.

```bash
CONFIG_PATH=configs/llm_sft_lora_single_gpu.yaml bash scripts/launch.sh
CONFIG_PATH=configs/llm_dpo_lora_single_gpu.yaml bash scripts/launch.sh
CONFIG_PATH=configs/llm_grpo_lora_single_gpu.yaml bash scripts/launch.sh   # online RL, see "GRPO" below
```

Typical recipe: SFT an instruction model on your chat data, then DPO it on preference pairs,
or GRPO it on prompts whose answers can be checked (math, code, formats).
To DPO or GRPO a model you just fine-tuned, set `save_load.export_hf: true` on the SFT run and
point the next run's `model_name` at the exported folder (`checkpoints/hf/<ts>.../`, LoRA already
merged). The reference is whatever model training starts from.

## The `algorithm:` block

```yaml
algorithm:
  type: dpo            # pretrain (default) | sft | dpo | simpo | orpo | grpo
  beta: 0.1            # dpo: KL strength | simpo: reward scale (default 2.0) | orpo: lambda (0.1)
  dpo_loss: sigmoid    # dpo only: sigmoid | ipo | hinge
  label_smoothing: 0.0 # dpo sigmoid only, in [0, 0.5)
  simpo_gamma: 0.5     # simpo only: target reward margin
  max_prompt_length: 256   # preference data: longer prompts are cut from the LEFT (default max_length // 2)
  completion_only: true    # sft: loss on the answer tokens only
```

Unknown keys are errors, and GRPO-only keys on another type are errors. With no block (or
`type: pretrain`) training is unchanged. GRPO's own keys are listed under [GRPO](#grpo).

| `type` | Loss | Reference model | Data |
|---|---|---|---|
| `sft` | cross-entropy on the answer tokens | none | chat / prompt-completion / Alpaca / text |
| `dpo` + `sigmoid` | `-log σ(β·h)` (DPO) | yes (precomputed) | preference pairs |
| `dpo` + `ipo` | `(h − 1/2β)²`, on per-token average log-probs (IPO) | yes | preference pairs |
| `dpo` + `hinge` | `max(0, 1 − β·h)` (SLiC) | yes | preference pairs |
| `simpo` | `-log σ(β·(avg_w − avg_l) − γ)` | **none** | preference pairs |
| `orpo` | `−avg_w − λ·log σ(log odds_w − log odds_l)` | **none** | preference pairs |
| `grpo` | clipped policy gradient on sampled answers, group-relative advantages (+ β·KL) | optional (LoRA: adapters off) | prompts (+ reference answers) |

with `h = (log π(y_w) − log ref(y_w)) − (log π(y_l) − log ref(y_l))` and `avg` = log-prob
divided by the number of answer tokens.

## Support matrix

| | solo | ddp | fsdp | 5d |
|---|---|---|---|---|
| sft / dpo / simpo / orpo / grpo, full fine-tune | ✓ | ✓ | ✓ | ✗ |
| + LoRA | ✓ | ✓ | ✓ | ✗ |
| + QLoRA / 8-bit | ✓ | ✓ | ✗ (as for all FSDP runs) | ✗ |

Only `model_type: llm` (decoder-only causal LMs). `ppo` is accepted as a name but rejected with
"planned, not implemented yet". GRPO with `beta > 0` needs LoRA (see below).

## Dataset formats

**SFT**, the first that matches:
- `messages`: a list of `{role, content}`. Every assistant turn is trained, everything else is masked.
- `prompt` + `completion`: strings or message lists.
- `instruction` (+ `input`) + `output`: Alpaca style.
- `text`: plain text, every token trained.

**Preference:**
- `prompt` + `chosen` + `rejected`: strings, or a message-list prompt with assistant answers.
- `chosen` + `rejected` as full conversations sharing all turns but the last
  (e.g. `trl-lib/ultrafeedback_binarized`).
- `chosen` + `rejected` strings sharing a prompt that ends in `"\n\nAssistant:"` (`Anthropic/hh-rlhf`).

`dataset.name` can also be a local `.json` / `.jsonl` / `.csv` / `.parquet` file. `subset` is optional.

If the tokenizer has a chat template (instruct models), prompts and answers are rendered with
it. Without one, text is joined as is: BOS + prompt + answer + EOS. Message lists then need a
chat template.

**How the answer is located:** the prompt is rendered with `add_generation_prompt=True` and must
be a string prefix of the rendered conversation. Prompt and answer are tokenised separately and
concatenated, so the mask boundary is exact. Batches are padded to their longest sequence, not
to `max_length`.

## The DPO reference model, without a second model

DPO needs `log ref(y)`, the log-probs of the model as it was before training. At step 0 the
policy *is* that model, so before training one no-grad pass scores every pair
(`REFERENCE LOG-PROBS` in the log) and caches two floats per pair. After that, training needs
no second model in memory, with any strategy and with or without LoRA. The cost is one extra
forward pass over the data. The pass runs in eval mode (no dropout); every rank scores its
share, and the scores are all-gathered.

## Distributed correctness

Preference losses are a mean over pairs. With equal per-rank batches the DDP / FSDP gradient
average equals the single-process gradient over the global batch.
`tests/test_post_training_dist.py` (`-m dist`) checks the loss **and every parameter's
gradient** against a 1-process run, including after one update when the policy no longer equals
the reference. DDP cases run on CPU. The FSDP cases need ≥ 2 GPUs, because FSDP2 has no CPU
support in torch 2.6, and are skipped otherwise.

SFT uses the usual per-rank token mean (as in HF Trainer and TRL), so a rank with fewer answer
tokens weighs each token a bit more. This convention is standard and is not world-size
invariant.

## GRPO

Group Relative Policy Optimization (DeepSeekMath, 2024) learns from **rewards on the model's own
answers**: no preference pairs, no reward model, no value model. Every training step:

1. **Sample.** For each prompt of the local batch, generate `num_generations` (G) answers with the
   current policy (eval mode, no grad, KV cache).
2. **Score.** Decode them and apply the reward functions. The reward is the weighted sum.
3. **Advantage.** `A_i = (r_i − mean(group)) / (std(group) + ε)` within each prompt's G answers.
   A group where every answer scored the same has A = 0 and teaches nothing.
4. **Update.** One forward with gradients over prompt + answer and the clipped policy-gradient
   loss on the answer tokens: `−min(ρ·A, clip(ρ, 1−ε, 1+ε_high)·A) + β·KL`, with ρ the
   importance ratio (1 here: one gradient step per rollout) and KL the k3 estimator against the
   start model.

```yaml
algorithm:
  type: grpo
  num_generations: 4         # G, >= 2
  max_prompt_length: 192     # prompts are cut from the left
  max_completion_length: 320 # tokens generated per answer (default: max_length - max_prompt_length)
  temperature: 1.0           # also divides the logits in the loss, so log-probs match sampling
  top_p: 1.0
  top_k: 0                   # 0 = off
  beta: 0.04                 # KL penalty; > 0 needs LoRA. 0 = no KL (DAPO, Dr. GRPO)
  epsilon: 0.2               # clip range, lower side
  epsilon_high: 0.28         # upper side; 0.28 = DAPO "clip-higher"
  loss_type: dapo            # grpo | dapo | dr_grpo
  scale_rewards: true        # false = Dr. GRPO advantages (no division by the group std)
  rewards: {accuracy: 1.0}   # names, {name: weight}, or "module:function"
  system_prompt: Please reason step by step, and put your final answer within \boxed{}.
```

`training.batch_size` counts **prompts** per GPU, so a step trains on `batch_size × G` answers.
`max_prompt_length + max_completion_length` must fit in `training.max_length`.

**Rewards** (`post_training/rewards.py`):

| name | value |
|---|---|
| `accuracy` | 1 if the final answer equals the reference: read from `<answer>…</answer>`, `\boxed{…}`, `#### x`, "the answer is x" or the last number. The reference is read the same way (GSM8K works as is), except that a short reference without a marker (`2/3`, `Route 66`) is used whole. Units are allowed (`72 apples` = 72); hedges and multiples are not (`71 or 72`, `72 thousand`) |
| `format` | 1 for exactly `<think>…</think> <answer>…</answer>`. Only useful for models trained on that format. |
| `exact_match` / `contains` | the whole answer equals / contains the reference (normalised) |
| `length` | 0, then a linear penalty to −1 over the last quarter of `max_completion_length` (DAPO's soft overlong penalty) |
| `my_pkg.mod:fn` | any importable `fn(prompts, completions, answers, **info) -> list[float]`; `info` has `completion_lengths` and `max_completion_length` |

**Loss normalisation** (`loss_type`): `grpo` averages over each answer's tokens, then over answers
(long answers weigh less per token). `dapo` divides the sum over all tokens by the number of answer
tokens in the **global** batch (all-reduced). `dr_grpo` divides by `answers × max_completion_length`.
All three give the same gradients on N processes as on one.

**KL reference without a second model.** With LoRA, the reference is the same model with its
adapters disabled (`peft`'s `disable_adapter()`), scored once per rollout with no grad. A full
fine-tune would need a frozen copy of the weights, so it requires `beta: 0` (as DAPO and Dr. GRPO
recommend anyway).

**Data.** A prompt from `prompt` (string or messages) | `question` | `problem` | `messages`
(trailing assistant turns dropped), and an optional reference answer from `answer` | `solution` |
`ground_truth` | `target` | `final_answer`. `openai/gsm8k` (subset `main`) works as is.

**Details that matter for correctness:**
- Dropout is off (the log-probs of the loss must be those the answers were sampled from).
- Positions follow the left padding (`position_ids` from the attention mask). RoPE models do not
  care, but learned absolute positions (GPT-2, OPT) would otherwise score the wrong distribution.
- `train.py` runs the whole step under `torch.autocast`, whose cache keeps the bf16 copy of each
  trainable weight it casts. Copies made during sampling are detached, so the cache is cleared
  after the no-grad work. Without that, LoRA gets no gradient (or gradient checkpointing raises
  `CheckpointError`). `tests/test_grpo.py` covers both.
- FSDP: generation runs with `synced_gpus`, so a rank whose answers finished early keeps joining
  the all-gathers until every rank is done.
- Groups never span ranks: each rank samples all G answers of its own prompts.

**Reading the progress line.** The GRPO loss stays near 0 by construction (advantages average to
zero in each group). Watch `reward` (and `r_<name>` per reward when there are several),
`reward_std` (0 means every group agreed, so nothing to learn: raise the temperature or G, or use
harder or easier prompts), `len` (average answer tokens) and `kl`. The console charts `reward`.

**Memory and speed.** Sampling holds a KV cache for `batch_size × G` sequences, and decoding is
memory-bandwidth bound, so a GRPO step costs much more than an SFT step on the same tokens.
`python planner.py --config <yaml>` includes both phases.

## Logged metrics

Each step's progress line shows `loss`, and for preference training:
- `acc`: fraction of pairs where the chosen reward beats the rejected one.
- `margin`: mean (chosen − rejected) reward, with reward = β·log-ratio (DPO) or the scaled
  average log-prob (SimPO / ORPO).

At step 0 DPO's loss is exactly `ln 2 ≈ 0.693` and the margin is 0.

## Not supported yet

- PPO (reward model + value head): next. GRPO reuses one rollout for one gradient step
  (`num_iterations > 1` and vLLM-backed generation are not implemented).
- KTO, reward-model training, sequence packing, `strategy: 5d`.
- A frozen reference copy for full-fine-tune GRPO with `beta > 0` (use LoRA, or `beta: 0`).

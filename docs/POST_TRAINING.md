# Post-training: SFT, DPO, IPO, SimPO, ORPO

Pick the algorithm with an `algorithm:` block in the config. Everything else (model loading,
solo / DDP / FSDP, LoRA / QLoRA, mixed precision, LR schedule, gradient clipping,
checkpoints, `launch.sh`, SLURM) works the same as for normal fine-tuning. Only the data
pipeline and the per-step loss change.

```bash
CONFIG_PATH=configs/llm_sft_lora_single_gpu.yaml bash scripts/launch.sh
CONFIG_PATH=configs/llm_dpo_lora_single_gpu.yaml bash scripts/launch.sh
```

Typical recipe: SFT an instruction model on your chat data, then DPO it on preference pairs.
To DPO a model you just fine-tuned, point `model_name` at the HF-format export (or merge the
LoRA adapters first). The reference is whatever model training starts from.

## The `algorithm:` block

```yaml
algorithm:
  type: dpo            # pretrain (default) | sft | dpo | simpo | orpo
  beta: 0.1            # dpo: KL strength | simpo: reward scale (default 2.0) | orpo: lambda (0.1)
  dpo_loss: sigmoid    # dpo only: sigmoid | ipo | hinge
  label_smoothing: 0.0 # dpo sigmoid only, in [0, 0.5)
  simpo_gamma: 0.5     # simpo only: target reward margin
  max_prompt_length: 256   # preference data: longer prompts are cut from the LEFT (default max_length // 2)
  completion_only: true    # sft: loss on the answer tokens only
```

Unknown keys are errors. With no block (or `type: pretrain`) training is unchanged.

| `type` | Loss | Reference model | Data |
|---|---|---|---|
| `sft` | cross-entropy on the answer tokens | none | chat / prompt-completion / Alpaca / text |
| `dpo` + `sigmoid` | `-log σ(β·h)` (DPO) | yes (precomputed) | preference pairs |
| `dpo` + `ipo` | `(h − 1/2β)²`, on per-token average log-probs (IPO) | yes | preference pairs |
| `dpo` + `hinge` | `max(0, 1 − β·h)` (SLiC) | yes | preference pairs |
| `simpo` | `-log σ(β·(avg_w − avg_l) − γ)` | **none** | preference pairs |
| `orpo` | `−avg_w − λ·log σ(log odds_w − log odds_l)` | **none** | preference pairs |

with `h = (log π(y_w) − log ref(y_w)) − (log π(y_l) − log ref(y_l))` and `avg` = log-prob
divided by the number of answer tokens.

## Support matrix

| | solo | ddp | fsdp | 5d |
|---|---|---|---|---|
| sft / dpo / simpo / orpo, full fine-tune | ✓ | ✓ | ✓ | ✗ |
| + LoRA | ✓ | ✓ | ✓ | ✗ |
| + QLoRA / 8-bit | ✓ | ✓ | ✗ (as for all FSDP runs) | ✗ |

Only `model_type: llm` (decoder-only causal LMs). `grpo` and `ppo` are accepted as names but
rejected with "planned, not implemented yet".

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

## Logged metrics

Each step's progress line shows `loss`, and for preference training:
- `acc`: fraction of pairs where the chosen reward beats the rejected one.
- `margin`: mean (chosen − rejected) reward, with reward = β·log-ratio (DPO) or the scaled
  average log-prob (SimPO / ORPO).

At step 0 DPO's loss is exactly `ln 2 ≈ 0.693` and the margin is 0.

## Not supported yet

- GRPO and PPO (online RL with generation and reward functions): next.
- KTO, reward-model training, sequence packing, the web UI, `strategy: 5d`.
- An on-the-fly reference model. It is needed for online methods and will come with GRPO;
  with LoRA it will be the same model with adapters disabled.

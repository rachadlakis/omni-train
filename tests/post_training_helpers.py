"""Offline fixtures for the post-training tests: a word-level tokenizer with a
chat template, and a tiny random Llama."""

import torch

WORDS = ("hello world what is two plus two four five yes no the answer good bad "
         "tell me a joke ok sure why because").split()
SPECIAL = ["<pad>", "<bos>", "<eos>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
    "{% if add_generation_prompt %}<|assistant|> {% endif %}"
)


def make_tokenizer(chat: bool = True):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {t: i for i, t in enumerate(dict.fromkeys(SPECIAL + ["[UNK]"] + WORDS))}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", bos_token="<bos>",
                                   eos_token="<eos>", unk_token="[UNK]")
    fast.add_special_tokens({"additional_special_tokens": SPECIAL[3:]})
    if chat:
        fast.chat_template = CHAT_TEMPLATE
    return fast


def tiny_llama(vocab_size: int, seed: int = 0):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=vocab_size, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128, tie_word_embeddings=False)
    return LlamaForCausalLM(cfg)


PAIRS = [
    {"prompt": "what is two plus two", "chosen": "four", "rejected": "five"},
    {"prompt": "tell me a joke", "chosen": "sure why because", "rejected": "no"},
    {"prompt": "hello", "chosen": "hello world", "rejected": "bad bad bad"},
    {"prompt": "is the answer good", "chosen": "yes good", "rejected": "no bad"},
]

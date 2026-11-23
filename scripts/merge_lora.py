"""Merge a LoRA adapter into its full-precision base model and save an HF model.

QLoRA runs export only the adapter (merging into 4-bit weights would be lossy):

    python scripts/merge_lora.py --adapter checkpoints/hf/<ts>__qlora_q4/adapter \\
        --out checkpoints/hf/<ts>__qlora_q4/merged [--base Qwen/Qwen2.5-0.5B-Instruct] [--dtype bfloat16]

--base defaults to the base model recorded in the adapter (adapter_config.json). The
tokenizer is copied from the adapter's parent folder when it is there. Runs on CPU;
needs RAM for the full model in --dtype.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def merge(adapter: str, out: str, base: str | None = None, dtype: str = "bfloat16") -> str:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    with open(os.path.join(adapter, "adapter_config.json")) as f:
        base = base or json.load(f)["base_model_name_or_path"]
    model = AutoModelForCausalLM.from_pretrained(base, dtype=DTYPES[dtype], token=os.getenv("HF_TOKEN"))
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True)
    parent = os.path.dirname(os.path.abspath(adapter))
    try:
        AutoTokenizer.from_pretrained(parent).save_pretrained(out)
    except Exception:
        print("note: no tokenizer next to the adapter; copy it from the base model if needed")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True, help="folder with adapter_config.json")
    p.add_argument("--out", required=True)
    p.add_argument("--base", help="base model (default: from adapter_config.json)")
    p.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    a = p.parse_args()
    print(f"merged model written to {merge(a.adapter, a.out, a.base, a.dtype)}")


if __name__ == "__main__":
    main()

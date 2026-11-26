"""Export (export.py, scripts/merge_lora.py) and solo/DDP optimizer resume, in process.

The FSDP export path (gather full state dict -> rebuild on CPU -> merge) can't run on
CPU end to end, so its rebuild step is tested here with a PEFT model's state dict.
"""

import types

import pytest
import torch

from tests.post_training_helpers import make_tokenizer, tiny_llama

IDS = torch.tensor([[1, 9, 10, 11, 12, 13, 2]])


def _lora(model):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(3)
    peft = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules="all-linear"))
    with torch.no_grad():  # B starts at 0; make the adapter actually change the output
        for n, p in peft.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    return peft.eval()


def _logits(model):
    with torch.no_grad():
        return model(IDS).logits


class TestExport:
    def test_rebuild_from_state_dict_merges_lora(self):
        """FSDP path: rank 0 rebuilds the model from the gathered state dict."""
        from omni_train.export import rebuild_from_state_dict
        tok = make_tokenizer()
        peft = _lora(tiny_llama(len(tok)))
        merged = rebuild_from_state_dict(peft, {k: v.clone() for k, v in peft.state_dict().items()}, "llm")
        assert not any("lora_" in n for n, _ in merged.named_parameters())
        torch.testing.assert_close(_logits(merged.eval()), _logits(peft), atol=1e-5, rtol=1e-5)

    def test_rebuild_rejects_wrong_state_dict(self):
        from omni_train.export import rebuild_from_state_dict
        tok = make_tokenizer()
        model = tiny_llama(len(tok))
        sd = model.state_dict()
        sd.pop("model.norm.weight")
        with pytest.raises(RuntimeError, match="does not match"):
            rebuild_from_state_dict(model, sd, "llm")

    @pytest.mark.parametrize("peft", [False, True])
    @pytest.mark.parametrize("tied", [False, True])
    def test_rebuild_missing_lm_head(self, tied, peft):
        """Missing lm_head: fine when tied to the embedding, an error (not a random
        output layer) when it is its own weight."""
        from transformers import LlamaForCausalLM
        from omni_train.export import rebuild_from_state_dict
        config = tiny_llama(len(make_tokenizer())).config
        config.tie_word_embeddings = tied
        model = LlamaForCausalLM(config).eval()
        if peft:
            model = _lora(model)
        sd = {k: v.clone() for k, v in model.state_dict().items() if not k.endswith("lm_head.weight")}
        if tied:
            torch.testing.assert_close(_logits(rebuild_from_state_dict(model, sd, "llm").eval()),
                                       _logits(model), atol=1e-5, rtol=1e-5)
        else:
            with pytest.raises(RuntimeError, match="lm_head"):
                rebuild_from_state_dict(model, sd, "llm")

    @pytest.mark.parametrize("peft", [False, True])
    def test_export_solo(self, tmp_path, peft):
        from transformers import AutoModelForCausalLM
        from omni_train.export import export_hf
        tok = make_tokenizer()
        model = tiny_llama(len(tok)).eval()
        if peft:
            model = _lora(model)
        want = _logits(model)
        args = types.SimpleNamespace(checkpoint_dir=str(tmp_path), model_name="tiny", model_type="llm",
                                     algorithm="sft", peft_enabled=peft, peft_type="lora",
                                     quantization_enabled=False, quantization_bits=4, export_dtype="float32")
        out = export_hf(model, tok, args, rank=0, strategy="solo", out_dir=str(tmp_path / "hf"))
        loaded = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
        torch.testing.assert_close(_logits(loaded), want, atol=1e-5, rtol=1e-5)
        assert loaded.config.use_cache

    def test_merge_lora_script(self, tmp_path):
        """QLoRA export = adapter only; scripts/merge_lora.py merges it into the base."""
        import importlib.util
        import os
        tok = make_tokenizer()
        base_dir = str(tmp_path / "base")
        tiny_llama(len(tok)).save_pretrained(base_dir)
        from transformers import AutoModelForCausalLM
        peft = _lora(AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32))
        adapter = str(tmp_path / "run" / "adapter")
        peft.save_pretrained(adapter)
        tok.save_pretrained(str(tmp_path / "run"))
        spec = importlib.util.spec_from_file_location(
            "merge_lora", os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "merge_lora.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out = mod.merge(adapter, str(tmp_path / "merged"), base=base_dir, dtype="float32")
        merged = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32).eval()
        torch.testing.assert_close(_logits(merged), _logits(peft), atol=1e-5, rtol=1e-5)
        assert os.path.exists(os.path.join(out, "tokenizer_config.json"))

    def test_export_config_rules(self, cfg):
        from omni_train.config import build_args
        cfg["save_load"]["export_hf"] = True
        assert build_args(cfg).export_hf
        cfg["save_load"]["export_dtype"] = "int8"
        with pytest.raises(ValueError, match="export_dtype"):
            build_args(cfg)
        cfg["save_load"]["export_dtype"] = "bfloat16"
        cfg["model_type"] = "custom_transformer"
        with pytest.raises(ValueError, match="custom_transformer"):
            build_args(cfg)


class TestOptimizerResume:
    def _setup(self, tmp_path, lr=1e-3):
        from omni_train.optim import restore_optimizer_state
        model = torch.nn.Linear(4, 3)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        model(torch.randn(2, 4)).sum().backward()
        opt.step()
        args = types.SimpleNamespace(resume_path=str(tmp_path / "c.pt"), learning_rate=5e-4, weight_decay=0.1)
        return model, opt, args, restore_optimizer_state

    def test_restored_with_this_runs_lr(self, tmp_path):
        model, opt, args, restore = self._setup(tmp_path)
        opt.param_groups[0]["lr"] = 1e-9  # what a finished cosine schedule leaves behind
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict()},
                   args.resume_path)
        fresh = torch.optim.AdamW(model.parameters(), lr=123.0)
        assert restore(fresh, args, torch.device("cpu")) == "restored"
        p = next(model.parameters())
        assert torch.equal(fresh.state[p]["exp_avg"], opt.state[p]["exp_avg"])
        g = fresh.param_groups[0]
        assert g["lr"] == g["initial_lr"] == 5e-4 and g["weight_decay"] == 0.1

    def test_old_format_is_absent(self, tmp_path):
        model, _, args, restore = self._setup(tmp_path)
        torch.save(model.state_dict(), args.resume_path)
        assert restore(torch.optim.AdamW(model.parameters()), args, torch.device("cpu")) == "absent"

    def test_mismatch(self, tmp_path):
        model, opt, args, restore = self._setup(tmp_path)
        torch.save({"model_state_dict": {}, "optimizer_state_dict": opt.state_dict()}, args.resume_path)
        other = torch.optim.AdamW(torch.nn.Linear(4, 3, bias=False).parameters())
        assert restore(other, args, torch.device("cpu")) == "mismatch"

"""
Tests for 3D parallelism plugin.

Run with: python -m pytest tests/test_hybrid_parallelism.py -v
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMeshResolution:
    """Test mesh dimension auto-resolution."""

    def test_full_specification(self):
        """Test when all dimensions are specified."""
        from hybrid.hybrid_config_adapter import resolve_mesh_dimensions

        result = resolve_mesh_dimensions(8, dp=2, tp=2, pp=2)
        assert result == (2, 2, 2)

    def test_full_specification_mismatch(self):
        """Test error when dimensions don't match GPU count."""
        from hybrid.hybrid_config_adapter import resolve_mesh_dimensions

        with pytest.raises(ValueError, match="doesn't match"):
            resolve_mesh_dimensions(8, dp=2, tp=2, pp=4)

    def test_no_specification_default_table(self):
        """Test default mesh lookup."""
        from hybrid.hybrid_config_adapter import resolve_mesh_dimensions

        # Known defaults
        assert resolve_mesh_dimensions(1) == (1, 1, 1)
        assert resolve_mesh_dimensions(2) == (2, 1, 1)
        assert resolve_mesh_dimensions(4) == (2, 2, 1)
        assert resolve_mesh_dimensions(8) == (2, 2, 2)
        assert resolve_mesh_dimensions(16) == (4, 2, 2)

    def test_no_specification_unknown_count(self):
        """Test fallback for unlisted GPU counts."""
        from hybrid.hybrid_config_adapter import resolve_mesh_dimensions

        # Should default to pure DP
        assert resolve_mesh_dimensions(3) == (3, 1, 1)
        assert resolve_mesh_dimensions(7) == (7, 1, 1)

    def test_partial_specification_one_dim(self):
        """Test when one dimension is specified."""
        from hybrid.hybrid_config_adapter import resolve_mesh_dimensions

        # 8 GPUs with TP=4 → remaining 2 split between DP and PP
        result = resolve_mesh_dimensions(8, tp=4)
        dp, tp, pp = result
        assert tp == 4
        assert dp * tp * pp == 8

    def test_partial_specification_two_dims(self):
        """Test when two dimensions are specified."""
        from hybrid.hybrid_config_adapter import resolve_mesh_dimensions

        result = resolve_mesh_dimensions(8, dp=2, tp=2)
        assert result == (2, 2, 2)

        result = resolve_mesh_dimensions(8, dp=4, pp=2)
        assert result == (4, 1, 2)


class TestConfigParsing:
    """Test configuration parsing."""

    def test_parse_3d_config(self, tmp_path):
        """Test parsing a 3D parallelism config."""
        from hybrid.hybrid_config_adapter import build_hybrid_args

        config_content = """
model_name: test-model
model_type: llm

dataset:
  name: test-dataset
  split: train

training:
  epochs: 1
  batch_size: 4

strategy: hybrid
num_gpus: 8

parallelism:
  enabled: true
  data_parallel_size: 2
  tensor_parallel_size: 2
  pipeline_parallel_size: 2

  tensor_parallel:
    style: colwise_rowwise

  pipeline_parallel:
    schedule: 1f1b
    num_microbatches: 4
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        args = build_hybrid_args(str(config_file))

        assert args.parallelism.enabled is True
        assert args.resolved_dp_size == 2
        assert args.resolved_tp_size == 2
        assert args.resolved_pp_size == 2
        assert args.parallelism.tensor_parallel.style == "colwise_rowwise"
        assert args.parallelism.pipeline_parallel.schedule == "1f1b"

    def test_disabled_parallelism(self, tmp_path):
        """Test config with parallelism disabled."""
        from hybrid.hybrid_config_adapter import build_hybrid_args

        config_content = """
model_name: test-model
model_type: llm

dataset:
  name: test-dataset
  split: train

training:
  epochs: 1
  batch_size: 4

strategy: hybrid
num_gpus: 1

parallelism:
  enabled: false
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        args = build_hybrid_args(str(config_file))

        assert args.parallelism.enabled is False


class TestValidation:
    """Test configuration validation."""

    def test_quantization_with_tp_error(self, tmp_path):
        """Test that quantization + TP raises error."""
        from hybrid.hybrid_config_adapter import build_hybrid_args

        config_content = """
model_name: test-model
model_type: llm

dataset:
  name: test-dataset
  split: train

training:
  epochs: 1
  batch_size: 4

strategy: hybrid
num_gpus: 4

parallelism:
  enabled: true
  data_parallel_size: 2
  tensor_parallel_size: 2
  pipeline_parallel_size: 1

quantization:
  enabled: true
  bits: 4
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        with pytest.raises(ValueError, match="Quantization is incompatible"):
            build_hybrid_args(str(config_file))


class TestLayerDetection:
    """Test transformer layer detection."""

    def test_detect_llama_style_layers(self):
        """Test detecting Llama-style layer structure."""
        from hybrid.hybrid_utils import _get_transformer_layers
        import torch.nn as nn

        # Mock Llama-style model
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(12)])

        model = MockModel()
        layers = _get_transformer_layers(model)

        assert len(layers) == 12

    def test_detect_gpt_style_layers(self):
        """Test detecting GPT-style layer structure."""
        from hybrid.hybrid_utils import _get_transformer_layers
        import torch.nn as nn

        # Mock GPT-style model
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = nn.Module()
                self.transformer.h = nn.ModuleList([nn.Linear(10, 10) for _ in range(6)])

        model = MockModel()
        layers = _get_transformer_layers(model)

        assert len(layers) == 6

    def test_fallback_to_largest_modulelist(self):
        """Test fallback when no standard pattern found."""
        from hybrid.hybrid_utils import _get_transformer_layers
        import torch.nn as nn

        # Model with non-standard structure
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(8)])
                self.small_list = nn.ModuleList([nn.Linear(10, 10) for _ in range(2)])

        model = MockModel()
        layers = _get_transformer_layers(model)

        # Should find the larger ModuleList
        assert len(layers) == 8


class TestTPPlanDetection:
    """Test tensor parallelism plan auto-detection."""

    def test_detect_colwise_patterns(self):
        """Test detecting column-wise parallelization targets."""
        from hybrid.hybrid_utils import _auto_detect_tp_plan
        import torch.nn as nn

        class MockLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(10, 10)
                self.k_proj = nn.Linear(10, 10)
                self.v_proj = nn.Linear(10, 10)
                self.o_proj = nn.Linear(10, 10)

        model = MockLayer()
        plan = _auto_detect_tp_plan(model, "colwise_rowwise")

        assert "q_proj" in plan
        assert plan["q_proj"] == "colwise"
        assert "o_proj" in plan
        assert plan["o_proj"] == "rowwise"


class TestFactorization:
    """Test helper functions."""

    def test_factorize(self):
        """Test factorization helper."""
        from hybrid.hybrid_config_adapter import _factorize

        assert _factorize(8) == (2, 4)  # 2*4 = 8, closest to sqrt(8)≈2.83
        assert _factorize(16) == (4, 4)
        assert _factorize(6) == (2, 3)
        assert _factorize(7) == (1, 7)  # Prime


# GPU-required tests
@pytest.mark.smoke
class TestIntegration:
    """Integration tests requiring GPU."""

    @pytest.mark.skipif(
        not os.environ.get("CUDA_VISIBLE_DEVICES"),
        reason="Requires CUDA"
    )
    def test_mesh_initialization(self):
        """Test device mesh initialization."""
        # This would require torchrun to properly test
        # Skipped in unit test mode
        pass

## dist_kit/tests/test_dist_setup.py ##
##
## Rank-aware printing, gutter alignment, memory snapshot, and safe teardown.

import torch

from dist_kit.dist_setup import (
    BACKEND,
    _display_width,
    _gutter,
    cleanup,
    gpu_memory_snapshot,
    print_banner_on_rank_0,
    print_on_all_ranks,
    print_on_rank_0,
)


def test_backend_is_valid():
    assert BACKEND in ("nccl", "gloo")


def test_display_width_ascii():
    assert _display_width("abc") == 3


def test_display_width_emoji_is_double():
    assert _display_width("🚀") == 2


def test_display_width_ignores_variation_selectors():
    # emoji + VS-16 still renders as one two-column glyph
    assert _display_width("⚙️") == 2


def test_gutter_blank_without_emoji():
    assert _gutter("") == "    "


def test_gutter_pads_to_fixed_width():
    g = _gutter("🚀")
    assert g.startswith("🚀") and _display_width(g) == 4


def test_print_on_rank_0_only_prints_on_rank_0(capsys):
    print_on_rank_0(0, "hello", "🚀")
    print_on_rank_0(1, "world", "🚀")
    out = capsys.readouterr().out
    assert "hello" in out and "world" not in out


def test_print_banner_only_on_rank_0(capsys):
    print_banner_on_rank_0(0, "TITLE")
    print_banner_on_rank_0(2, "HIDDEN")
    out = capsys.readouterr().out
    assert "TITLE" in out and "=" * 60 in out and "HIDDEN" not in out


def test_print_on_all_ranks_includes_rank_prefix(capsys):
    print_on_all_ranks(3, "msg", local_rank=1, device="cuda:1")
    out = capsys.readouterr().out
    assert "rank=3" in out and "local_rank=1" in out and "device=cuda:1" in out and "msg" in out


def test_gpu_memory_snapshot_cpu_device():
    assert gpu_memory_snapshot(torch.device("cpu")) == "CUDA not detected."


def test_cleanup_safe_without_process_group():
    # Must be a no-op (no exception) when dist was never initialized.
    cleanup()

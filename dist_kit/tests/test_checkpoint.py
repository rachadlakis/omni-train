## dist_kit/tests/test_checkpoint.py ##
##
## Checkpoint folder discovery and Checkpointer bookkeeping (no GPU needed).

import os

from dist_kit.checkpoint import Checkpointer, get_latest_checkpoint_folder


def test_missing_path_returns_none(tmp_path):
    assert get_latest_checkpoint_folder(str(tmp_path / "nope")) is None


def test_empty_dir_returns_none(tmp_path):
    assert get_latest_checkpoint_folder(str(tmp_path)) is None


def test_picks_numerically_largest_timestamp(tmp_path):
    for name in ("1000", "900", "20000"):
        (tmp_path / name).mkdir()
    assert get_latest_checkpoint_folder(str(tmp_path)) == "20000"


def test_handles_run_tag_suffixes(tmp_path):
    (tmp_path / "1000").mkdir()
    (tmp_path / "2000__lora_q4").mkdir()
    assert get_latest_checkpoint_folder(str(tmp_path)) == "2000__lora_q4"


def test_ignores_non_numeric_folders_and_files(tmp_path):
    (tmp_path / "not_a_checkpoint").mkdir()
    (tmp_path / "3000").write_text("a file, not a folder")
    (tmp_path / "1500").mkdir()
    assert get_latest_checkpoint_folder(str(tmp_path)) == "1500"


def test_checkpointer_empty_folder(tmp_path):
    ckpt = Checkpointer(folder=str(tmp_path), dcp_api=True)
    assert ckpt.is_empty()
    assert ckpt.last_training_time is None


def test_checkpointer_finds_latest_run(tmp_path):
    api_dir = tmp_path / "fsdp" / "dcp_api"
    os.makedirs(api_dir / "111")
    os.makedirs(api_dir / "222__lora")
    ckpt = Checkpointer(folder=str(tmp_path), dcp_api=True)
    assert not ckpt.is_empty()
    assert ckpt.last_training_time == "222__lora"


def test_checkpointer_apis_use_separate_folders(tmp_path):
    os.makedirs(tmp_path / "fsdp" / "dcp_api" / "111")
    dtensor_ckpt = Checkpointer(folder=str(tmp_path), dcp_api=False)
    assert dtensor_ckpt.is_empty()  # dtensor_api/ has no runs

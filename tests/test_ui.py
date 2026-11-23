"""UI server guards: Host-header check (DNS rebinding) and torchrun port selection."""
import socket

import pytest

from ui.app import _allowed_host
from ui.queue import find_free_port


@pytest.mark.parametrize("host", [
    "localhost", "localhost:8787", "127.0.0.1:8787", "[::1]:8787",
    "10.0.0.5:8787",                       # remote access by IP keeps working
    socket.gethostname(),
])
def test_allowed_hosts(host):
    assert _allowed_host(host)


@pytest.mark.parametrize("host", ["attacker.example", "attacker.example:8787", "", "evil.localhost.example"])
def test_rebinding_hosts_rejected(host):
    assert not _allowed_host(host)


def test_extra_hosts_from_env(monkeypatch):
    monkeypatch.setenv("OMNI_UI_ALLOWED_HOSTS", "gpu-box.example.org, other.lan")
    assert _allowed_host("gpu-box.example.org:8787")
    assert _allowed_host("other.lan")
    assert not _allowed_host("attacker.example")


def test_find_free_port_is_bindable():
    port = find_free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", port))


# ---------------------------------------------------------------------------
# Console API: planner, catalog, YAML round trip, queue guards, log metrics
# ---------------------------------------------------------------------------

import yaml
from fastapi.testclient import TestClient

import ui.app as ui_app
import ui.queue as ui_queue
from ui.queue import QueueManager, parse_metrics


@pytest.fixture
def client():
    # no `with`: skips the startup hook, so the real ui/queue.db is never touched
    return TestClient(ui_app.app, base_url="http://localhost")


@pytest.fixture
def tmp_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_queue, "RUNS_DIR", tmp_path / "runs")
    qm = QueueManager(db_path=str(tmp_path / "q.db"))
    monkeypatch.setattr(ui_app, "queue_manager", qm)
    return qm


LOCAL_8xH100 = {"gpu_mem_gb": 80, "gpus_per_node": 8, "nodes": 1, "gpu_name": "h100"}


def test_plan_endpoint_recommends_fsdp_for_8b(client):
    cfg = {"model_name": "meta-llama/Llama-3.1-8B", "model_type": "llm", "strategy": "solo", "num_gpus": 1,
           "training": {"batch_size": 1, "max_length": 4096, "gradient_checkpointing": True},
           "peft": {"enabled": False}, "quantization": {"enabled": False}, "algorithm": {"type": "pretrain"}}
    r = client.post("/api/plan", json={"config": cfg, "cluster": LOCAL_8xH100})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["verdict"] == "distributed_required" and j["recommended"]["strategy"] == "fsdp"
    assert j["selected"]["strategy"] == "solo" and not j["selected"]["fits"]


def test_plan_endpoint_evaluates_selected_5d_layout(client):
    cfg = yaml.safe_load(open("configs/custom_5d_moe_16gpu.yaml"))
    j = client.post("/api/plan", json={"config": cfg, "cluster": {**LOCAL_8xH100, "nodes": 2}}).json()
    assert j["model"]["source"] == "custom" and j["model"]["num_experts"] == 8
    assert j["selected"]["strategy"] == "5d" and "EP2" in j["selected"]["label"]


def test_catalog_lists_algorithms_grpo_ready_ppo_planned(client):
    j = client.get("/api/catalog").json()
    status = {a["id"]: a["status"] for a in j["algorithms"]}
    assert status["sft"] == status["dpo"] == status["grpo"] == "ready" and status["ppo"] == "planned"
    assert any(m["five_d_ready"] for m in j["models"]) and j["gpus"]


def test_inspect_preset_model_offline(client):
    j = client.get("/api/models/inspect", params={"name": "Qwen/Qwen2.5-0.5B-Instruct"}).json()
    assert j["source"] == "preset" and j["arch"] == "qwen2" and j["five_d_ready"]


def test_render_and_parse_round_trip(client):
    cfg = yaml.safe_load(open("configs/llm_dpo_lora_single_gpu.yaml"))
    rendered = client.post("/api/config/render", json={"config": cfg}).json()
    assert "launch_mode" not in rendered["config"]
    parsed = client.post("/api/config/parse", json={"yaml": rendered["yaml"]}).json()["config"]
    assert parsed["algorithm"]["type"] == "dpo" and parsed["peft"]["enabled"] is True
    bad = client.post("/api/config/parse", json={"yaml": "- just\n- a list"})
    assert bad.status_code == 400


def test_env_status_never_returns_values(client, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    j = client.get("/api/env/status").json()
    assert j["keys"]["HF_TOKEN"] is True
    assert "hf_secret_value" not in str(j)


def test_submit_rejects_more_gpus_than_the_machine_has(client, tmp_queue):
    cfg = yaml.safe_load(open("configs/custom_5d_moe_16gpu.yaml"))
    r = client.post("/api/queue/submit", json={"config": cfg})
    assert r.status_code == 400 and "16" in r.json()["detail"]
    assert tmp_queue.list_jobs() == []


def test_submit_rejects_invalid_config(client, tmp_queue):
    cfg = yaml.safe_load(open("configs/llm_dpo_lora_single_gpu.yaml"))
    cfg["strategy"] = "5d"
    r = client.post("/api/queue/submit", json={"config": cfg})
    assert r.status_code == 400 and "Invalid training config" in r.json()["detail"]


def test_job_names_and_log_files_survive_a_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(ui_queue, "RUNS_DIR", tmp_path / "runs")
    db = str(tmp_path / "q.db")
    qm = QueueManager(db_path=db)
    monkeypatch.setattr(qm, "allocate_gpus", lambda job_id, n: None)    # stay pending, never launch
    job = qm.submit_job(config={"strategy": "solo"}, gpu_count=1, name="my-run")
    qm._append_job_log(job.id, "   Epoch 1/1 [██░] 50.0% | step 1/2 | loss: 2.0000 | avg: 2.0000")
    fresh = QueueManager(db_path=db)                                      # "server restart"
    assert fresh.get_job(job.id).name == "my-run"
    assert fresh.get_job_logs(job.id) and fresh.get_job_metrics(job.id)["progress"]["loss"] == 2.0
    assert fresh.delete_job(job.id) is False                              # pending jobs are kept
    fresh.cancel_job(job.id)
    assert fresh.delete_job(job.id) and not (tmp_path / "runs" / f"{job.id}.log").exists()


def test_parse_metrics_reads_both_trainers():
    lines = [
        "   Epoch 1/2 [████░░] 50.0% | step 1/2 | loss: 3.5000 | avg: 3.5000",
        "   Epoch 1/2 [██████] 100.0% | step 2/2 | loss: 3.0000 | avg: 3.2500 | rewards/margin: 0.120",
        "   ✅ Epoch 1 complete | avg loss: 3.2500",
        "   Epoch 2/2 [███░░░] 50.0% | step 1/2 | loss: 2.5000 | avg: 2.5000 | grad norm: 0.812 | 12,345 tok/s",
        "   Epoch 2/2 [███░░░] 50.0% | step 2/2 | loss: nan | avg: nan",
        "random line",
    ]
    m = parse_metrics(lines)
    assert [p["global_step"] for p in m["points"]] == [1, 2, 3]
    assert m["points"][1]["rewards/margin"] == pytest.approx(0.12)
    assert m["points"][2]["tokens_per_s"] == 12345 and m["points"][2]["grad_norm"] == pytest.approx(0.812)
    assert m["epochs"] == [{"epoch": 1, "loss": 3.25}]
    assert m["progress"]["fraction"] == pytest.approx(0.75) and m["progress"]["total_steps"] == 4


def test_parse_metrics_downsamples():
    lines = [f"Epoch 1/1 [x] 1% | step {i}/5000 | loss: {5 - i / 5000:.4f} | avg: 1.0" for i in range(1, 5001)]
    m = parse_metrics(lines, max_points=500)
    assert len(m["points"]) <= 501 and m["points"][-1]["step"] == 5000


def test_grpo_example_config_plans_and_validates(client):
    cfg = yaml.safe_load(open("configs/llm_grpo_lora_single_gpu.yaml"))
    assert client.post("/api/config/validate", json={"config": cfg}).json()["valid"]
    j = client.post("/api/plan", json={"config": cfg, "cluster": {"gpu_mem_gb": 6, "gpus_per_node": 1, "gpu_name": "4050"}}).json()
    assert j["workload"]["algorithm"] == "grpo" and j["workload"]["num_generations"] == 4
    assert j["selected"]["fits"] and j["selected"]["perf"]["generation_seconds"] > 0

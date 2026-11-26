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


def test_catalog_lists_every_algorithm_as_ready(client):
    j = client.get("/api/catalog").json()
    status = {a["id"]: a["status"] for a in j["algorithms"]}
    assert status["sft"] == status["dpo"] == status["grpo"] == status["ppo"] == "ready"
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


# ---------------------------------------------------------------------------
# Console: templates from the configs, run summaries, full log, offline assets
# ---------------------------------------------------------------------------

from pathlib import Path

from ui.queue import summarize_metrics

STATIC = Path(ui_app.STATIC_DIR)


def test_templates_are_described_from_the_config_itself(client):
    t = {x["name"]: x for x in client.get("/api/configs").json()["templates"]}
    g = t["llm_grpo_lora_single_gpu"]
    assert (g["algorithm"], g["method"], g["strategy"], g["num_gpus"]) == ("grpo", "lora", "solo", 1)
    assert t["llm_5d_tp_pp_dp_8gpu"]["strategy"] == "5d" and t["custom_5d_moe_16gpu"]["moe"]
    assert not [x for x in t.values() if "error" in x]


def test_a_broken_template_does_not_hide_the_others(client, tmp_path, monkeypatch):
    (tmp_path / "ok.yaml").write_text("model_type: llm\nmodel_name: gpt2\nstrategy: ddp\nnum_gpus: 2\n")
    (tmp_path / "bad.yaml").write_text("a: [1, 2\n")
    monkeypatch.setattr(ui_app, "CONFIGS_DIR", tmp_path)
    t = {x["name"]: x for x in client.get("/api/configs").json()["templates"]}
    assert t["ok"]["num_gpus"] == 2 and "error" in t["bad"]


def test_summary_uses_reward_for_rl_and_loss_otherwise():
    rl = parse_metrics([f"Epoch 1/1 [x] 1% | step {i}/20 | loss: 0.0 | avg: 0.0 | reward: {i / 20:.2f}" for i in range(1, 21)])
    s = summarize_metrics(rl)
    assert s["key"] == "reward" and s["value"] == pytest.approx(sum(range(11, 21)) / 200)   # last 10 steps
    sft = parse_metrics(["Epoch 1/1 [x] 50% | step 1/2 | loss: 3.0 | avg: 3.0"])
    assert summarize_metrics(sft) == {"key": "loss", "value": 3.0, "steps": 1}
    assert summarize_metrics(parse_metrics(["no steps yet"])) is None


def test_job_list_has_summaries_and_the_full_log_downloads(client, tmp_queue, monkeypatch):
    monkeypatch.setattr(tmp_queue, "allocate_gpus", lambda job_id, n: None)
    job = tmp_queue.submit_job(config={"strategy": "solo"}, gpu_count=1, name="r")
    for i in range(1, 6001):                                   # more than the 5000-line live buffer
        tmp_queue._append_job_log(job.id, f"Epoch 1/1 [x] 1% | step {i}/6000 | loss: 1.5 | avg: 1.5")
    tmp_queue.cancel_job(job.id)
    row = client.get("/api/queue/jobs").json()["jobs"][0]
    assert row["summary"]["value"] == 1.5 and row["summary"]["steps"] == 6000
    r = client.get(f"/api/queue/jobs/{job.id}/log")
    assert r.headers["content-type"].startswith("text/plain") and len(r.text.strip().split("\n")) == 6000
    assert client.get("/api/queue/jobs/..%2F..%2Fsecret/log").status_code == 404


def test_metrics_are_cached_until_the_log_grows(tmp_queue, monkeypatch):
    monkeypatch.setattr(tmp_queue, "allocate_gpus", lambda job_id, n: None)
    job = tmp_queue.submit_job(config={}, gpu_count=1)
    line = "Epoch 1/1 [x] 1% | step {}/9 | loss: 2.0 | avg: 2.0"
    tmp_queue._append_job_log(job.id, line.format(1))
    first = tmp_queue.get_job_metrics(job.id)
    assert tmp_queue.get_job_metrics(job.id) is first
    tmp_queue._append_job_log(job.id, line.format(2))
    assert len(tmp_queue.get_job_metrics(job.id)["points"]) == 2


def test_console_works_offline():
    """No page asset may come from another host: GPU servers are often offline."""
    for name in ("index.html", "styles.css"):
        text = (STATIC / name).read_text()
        assert "googleapis" not in text and "gstatic" not in text and "//cdn" not in text
    for url in __import__("re").findall(r"/static/([\w./-]+)", (STATIC / "index.html").read_text() + (STATIC / "styles.css").read_text()):
        assert (STATIC / url).exists(), url


# ---------------------------------------------------------------------------
# strategy: auto and the dataset check
# ---------------------------------------------------------------------------

import json

from tests.post_training_helpers import make_tokenizer, tiny_llama

FAKE_8xH100 = {"gpu_mem_gb": 80.0, "gpus_per_node": 8, "nodes": 1, "gpu_name": "NVIDIA H100 80GB HBM3", "detected": True}


@pytest.fixture
def eight_gpus(monkeypatch, tmp_queue):
    monkeypatch.setattr(ui_app, "_local_cluster", lambda: dict(FAKE_8xH100))
    monkeypatch.setattr(tmp_queue, "get_total_gpus", lambda: 8)
    monkeypatch.setattr(tmp_queue, "allocate_gpus", lambda *a, **k: [])      # never start a real process
    ui_app._DATA_CACHE.clear()
    return tmp_queue


def auto_cfg(**over):
    return {"model_name": "meta-llama/Llama-3.1-8B", "model_type": "llm", "strategy": "auto", "num_gpus": "auto",
            "dataset": {"name": "missing.jsonl", "subset": None, "split": "train"},
            "training": {"epochs": 1, "batch_size": 1, "max_length": 4096, "gradient_checkpointing": True},
            "peft": {"enabled": False}, "quantization": {"enabled": False}, "algorithm": {"type": "pretrain"}, **over}


def test_plan_of_an_auto_config_selects_the_recommendation(client):
    j = client.post("/api/plan", json={"config": auto_cfg(), "cluster": LOCAL_8xH100}).json()
    assert j["selected"]["auto"] and j["selected"]["label"] == j["recommended"]["label"] == "FSDP x8"
    assert j["advice"]["tech"] == "FSDP" and j["advice"]["others"][0]["workload"]["method"] == "lora"
    assert client.post("/api/config/validate", json={"config": auto_cfg()}).json()["valid"]


def test_unknown_dataset_size_gives_no_time_instead_of_a_guess(client):
    j = client.post("/api/plan", json={"config": auto_cfg(), "cluster": LOCAL_8xH100}).json()
    assert "total_seconds" not in j["recommended"]["perf"]
    j = client.post("/api/plan", json={"config": auto_cfg(dataset={"name": "x/y", "split": "train[:2000]"}),
                                       "cluster": LOCAL_8xH100}).json()
    assert j["recommended"]["perf"]["total_steps"] == 250
    j = client.post("/api/plan", json={"config": auto_cfg(), "cluster": LOCAL_8xH100, "num_samples": 800}).json()
    assert j["recommended"]["perf"]["total_steps"] == 100


def test_submitting_auto_queues_a_concrete_strategy(client, eight_gpus):
    r = client.post("/api/queue/submit", json={"config": auto_cfg(), "name": "a"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["strategy"] == "fsdp" and j["gpu_count"] == 8 and "split the model" in j["chosen"]
    job = eight_gpus.get_job(j["job_id"])
    assert job.config["strategy"] == "fsdp" and job.config["num_gpus"] == 8
    assert "Automatic strategy: fsdp on 8 GPUs" in "\n".join(eight_gpus.get_job_logs(j["job_id"]))


def test_submitting_auto_respects_the_gpu_limit(client, eight_gpus):
    j = client.post("/api/queue/submit", json={"config": auto_cfg(num_gpus=4)}).json()
    assert j["gpu_count"] == 4


def test_submitting_auto_explains_what_cannot_fit(client, eight_gpus):
    r = client.post("/api/queue/submit", json={"config": auto_cfg(model_name="meta-llama/Llama-3.1-70B")})
    assert r.status_code == 400 and "Does not fit on these 8 GPUs" in r.json()["detail"] and "LoRA" in r.json()["detail"]


def test_data_inspect_measures_and_feeds_the_plan(client, tmp_path):
    ui_app._DATA_CACHE.clear()
    tok = make_tokenizer()
    tiny_llama(len(tok)).save_pretrained(tmp_path / "m")
    tok.save_pretrained(tmp_path / "m")
    (tmp_path / "d.jsonl").write_text("".join(json.dumps({"prompt": "hello", "completion": "hello world"}) + "\n" for _ in range(12)))
    cfg = auto_cfg(model_name=str(tmp_path / "m"), algorithm={"type": "sft"},
                   dataset={"name": str(tmp_path / "d.jsonl"), "subset": None, "split": "train"})
    before = client.post("/api/plan", json={"config": cfg, "cluster": LOCAL_8xH100}).json()
    assert before["data"] is None
    j = client.post("/api/data/inspect", json={"config": cfg}).json()
    assert j["error"] == "" and j["data"]["rows"] == 12
    after = client.post("/api/plan", json={"config": cfg, "cluster": LOCAL_8xH100}).json()
    assert after["data"]["rows"] == 12 and "12 examples" in after["advice"]["data"]
    assert after["workload"]["seq_len"] == max(8, j["data"]["longest"]) < before["workload"]["seq_len"]


def test_data_inspect_reports_errors(client):
    ui_app._DATA_CACHE.clear()
    j = client.post("/api/data/inspect", json={"config": auto_cfg()}).json()
    assert j["data"] is None and "not found" in j["error"]
    assert not ui_app._DATA_CACHE                    # a failure may be temporary


def test_configs_without_num_gpus_still_mean_one_gpu(client):
    cfg = {k: v for k, v in auto_cfg(strategy="solo").items() if k != "num_gpus"}
    assert client.post("/api/config/render", json={"config": cfg}).json()["config"]["num_gpus"] == 1
    old = {"model": {"name": "gpt2"}, "distributed": {"strategy": "ddp"}}
    assert client.post("/api/config/render", json={"config": old}).json()["config"]["num_gpus"] == 1


def test_console_defaults_to_automatic():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "strategy: 'auto', num_gpus: 'auto'" in js and "/api/data/inspect" in js


# ---------------------------------------------------------------------------
# Planner page: the capacity map
# ---------------------------------------------------------------------------

def test_capacity_map_by_method(client):
    cfg = auto_cfg(model_name="meta-llama/Llama-3.1-8B")
    j = client.post("/api/plan/grid", json={"config": cfg, "cluster": {"gpu_mem_gb": 80, "gpu_name": "h100", "gpus_per_node": 8},
                                            "num_samples": 20000}).json()
    assert j["rows"] == [1, 2, 4, 8, 16, 32, 64] and [c["key"] for c in j["columns"]] == ["full", "lora", "qlora"]
    full = [row[0] for row in j["cells"]]
    assert not full[0]["fits"] and full[1]["strategy"] == "fsdp"       # 8B full fine-tune: FSDP from 2 GPUs
    assert j["cells"][0][1]["fits"] and j["cells"][0][1]["strategy"] == "solo"
    assert all(c["strategy"] != "fsdp" for row in j["cells"] for c in row[2:])   # QLoRA never on FSDP
    assert full[1]["advice"]["tech"] == "FSDP" and full[1]["total_seconds"] > full[3]["total_seconds"]


def test_capacity_map_by_sequence_length_reaches_3d(client):
    cfg = auto_cfg(model_name="meta-llama/Llama-3.1-8B", dataset={"name": "some/hub-dataset", "split": "train"})   # 5d reads no local files
    j = client.post("/api/plan/grid", json={"config": cfg, "axis": "seq",
                                            "cluster": {"gpu_mem_gb": 80, "gpu_name": "h100", "gpus_per_node": 8}}).json()
    assert [c["label"] for c in j["columns"]] == ["512", "2K", "8K", "32K", "128K"]
    eight = j["cells"][j["rows"].index(8)]
    assert eight[0]["strategy"] == "fsdp" and eight[3]["strategy"] == "5d" and eight[3]["parallelism"]["cp"] > 1
    assert "total_seconds" not in eight[0] or eight[0]["total_seconds"] is None        # no examples given: no time


def test_capacity_map_rows_follow_the_machine_size_and_skip_methods_a_model_lacks(client):
    cfg = auto_cfg(model_type="custom_transformer", model_name=None,
                   custom_transformer_args={"n_layers": 2, "dim": 64, "n_heads": 4, "vocab_size": 256, "max_seq_len": 64})
    j = client.post("/api/plan/grid", json={"config": cfg, "cluster": {"gpu_mem_gb": 24, "gpu_name": "4090", "gpus_per_node": 4}}).json()
    assert j["rows"] == [1, 2, 4, 8, 16, 32]
    assert j["columns"][1]["why"] and all(row[1] is None and row[2] is None for row in j["cells"])
    assert client.post("/api/plan/grid", json={"config": cfg, "cluster": {}, "axis": "nope"}).status_code == 400

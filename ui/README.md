# fsdp-mini-project UI

Small guide for running the web UI locally.

## 1) Install dependencies

From the fsdp-mini-project directory:

```bash
cd dist-train-project
source /dist-train-project/.venv/bin/activate
pip install fastapi uvicorn pydantic pyyaml
```

If you already have an environment set up, just activate it and install any missing packages.

## 2) Start the UI

Run the app from fsdp-mini-project:

```bash
cd ./dist-train-project
python -m ui.app
```

Or with `uvicorn`:

```bash
cd ./dist-train-project
uvicorn ui.app:app --reload --port 8000
```

For a Linux server or VM, bind the app to all interfaces:

```bash
cd /home/rachad_lakkis/projects/distributed-training/Local/omni_train/testing/fsdp-mini-project
uvicorn ui.app:app --host 0.0.0.0 --port 8000
```

## 3) Open it in the browser

Go to:

```text
http://localhost:8000
```

## Notes

- Templates are loaded from this folder's `config.yaml` (and optional `configs/*.yaml` if present).
- The UI serves static files from `ui/static/`.
- Training is launched from this folder using local `train.py` and `CONFIG_PATH`.

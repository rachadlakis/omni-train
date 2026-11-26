"""OMNI-Train: distributed fine-tuning with solo, DDP, FSDP2 and 5D parallelism.

Nothing is imported here: `omni_train.planner` stays torch-free, and each entry point
(train.py, ui/app.py, scripts/) imports only the modules it needs.
"""

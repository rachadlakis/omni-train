"""
Job Queue Manager for OMNI-Train

Manages training job queue, GPU allocation, and background job processing.
Uses SQLite for persistence and supports multiple concurrent users.
"""

import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

try:
    from .config_adapter import adapt_ui_config_to_mini
except ImportError:
    from ui.config_adapter import adapt_ui_config_to_mini


ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    config: dict
    gpu_count: int
    status: JobStatus = JobStatus.PENDING
    gpu_indices: List[int] = field(default_factory=list)
    priority: int = 0
    estimated_seconds: Optional[float] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None
    process_pid: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "config": self.config,
            "gpu_count": self.gpu_count,
            "status": self.status.value,
            "gpu_indices": self.gpu_indices,
            "priority": self.priority,
            "estimated_seconds": self.estimated_seconds,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "error_message": self.error_message,
            "process_pid": self.process_pid,
        }


class QueueManager:
    """Manages job queue, GPU allocation, and background processing."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else Path(__file__).parent / "queue.db"
        self._lock = threading.RLock()
        self._running_processes: Dict[str, subprocess.Popen] = {}
        self._job_logs: Dict[str, deque] = {}
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._on_job_status_change: Optional[Callable[[Job], None]] = None
        self._init_db()

    def _append_job_log(self, job_id: str, line: str) -> None:
        """Store a cleaned log line for a queued job."""
        cleaned = ANSI_ESCAPE_RE.sub("", line).rstrip()
        if cleaned and job_id in self._job_logs:
            self._job_logs[job_id].append(cleaned)

    def _init_db(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT NOT NULL,
                    gpu_count INTEGER NOT NULL,
                    gpu_indices TEXT,
                    config TEXT NOT NULL,
                    priority INTEGER DEFAULT 0,
                    estimated_seconds REAL,
                    started_at TIMESTAMP,
                    finished_at TIMESTAMP,
                    error_message TEXT,
                    process_pid INTEGER
                );

                CREATE TABLE IF NOT EXISTS gpu_allocations (
                    gpu_index INTEGER PRIMARY KEY,
                    job_id TEXT,
                    allocated_at TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES jobs(id)
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
                CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
                CREATE INDEX IF NOT EXISTS idx_gpu_allocations_job_id ON gpu_allocations(job_id);
            """)

            # Initialize GPU allocations if not present
            total_gpus = self._detect_gpu_count()
            for i in range(total_gpus):
                conn.execute(
                    "INSERT OR IGNORE INTO gpu_allocations (gpu_index, job_id) VALUES (?, NULL)",
                    (i,)
                )

    @contextmanager
    def _get_connection(self):
        """Thread-safe database connection context manager."""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _detect_gpu_count(self) -> int:
        """Detect number of available GPUs."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except ImportError:
            pass
        return 1  # Default to 1 if torch not available

    # =========================================================================
    # GPU Management
    # =========================================================================

    def get_total_gpus(self) -> int:
        """Return total number of GPUs in system."""
        with self._get_connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM gpu_allocations").fetchone()
            return result[0] if result else 0

    def get_available_gpus(self) -> List[int]:
        """Return list of available GPU indices."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT gpu_index FROM gpu_allocations WHERE job_id IS NULL ORDER BY gpu_index"
            ).fetchall()
            return [row["gpu_index"] for row in rows]

    def get_gpu_status(self) -> List[dict]:
        """Get status of all GPUs."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT ga.gpu_index, ga.job_id, ga.allocated_at
                FROM gpu_allocations ga
                ORDER BY ga.gpu_index
            """).fetchall()

            gpus = []
            for row in rows:
                gpu_info = {
                    "index": row["gpu_index"],
                    "status": "busy" if row["job_id"] else "free",
                    "job_id": row["job_id"],
                    "allocated_at": row["allocated_at"],
                }
                # Add GPU hardware info if available
                try:
                    import torch
                    if torch.cuda.is_available() and row["gpu_index"] < torch.cuda.device_count():
                        props = torch.cuda.get_device_properties(row["gpu_index"])
                        gpu_info["name"] = props.name
                        gpu_info["total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
                except Exception:
                    gpu_info["name"] = f"GPU {row['gpu_index']}"

                gpus.append(gpu_info)
            return gpus

    def allocate_gpus(self, job_id: str, count: int) -> Optional[List[int]]:
        """Allocate GPUs for a job. Returns indices or None if not enough available."""
        with self._lock:
            available = self.get_available_gpus()
            if len(available) < count:
                return None

            # Allocate lowest-indexed GPUs first
            allocated = available[:count]

            with self._get_connection() as conn:
                now = datetime.now().isoformat()
                for gpu_idx in allocated:
                    conn.execute(
                        "UPDATE gpu_allocations SET job_id = ?, allocated_at = ? WHERE gpu_index = ?",
                        (job_id, now, gpu_idx)
                    )

            return allocated

    def release_gpus(self, job_id: str) -> List[int]:
        """Release GPUs allocated to a job. Returns freed indices."""
        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT gpu_index FROM gpu_allocations WHERE job_id = ?",
                    (job_id,)
                ).fetchall()
                freed = [row["gpu_index"] for row in rows]

                conn.execute(
                    "UPDATE gpu_allocations SET job_id = NULL, allocated_at = NULL WHERE job_id = ?",
                    (job_id,)
                )

            return freed

    # =========================================================================
    # Job Management
    # =========================================================================

    def submit_job(
        self,
        config: dict,
        gpu_count: int,
        priority: int = 0,
        estimated_seconds: Optional[float] = None,
    ) -> Job:
        """Submit a new job. Starts immediately if GPUs available, else queues."""
        job_id = str(uuid.uuid4())
        now = datetime.now()

        job = Job(
            id=job_id,
            config=config,
            gpu_count=gpu_count,
            status=JobStatus.PENDING,
            priority=priority,
            estimated_seconds=estimated_seconds,
            created_at=now,
        )

        # Save to database
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO jobs
                   (id, status, gpu_count, config, priority, estimated_seconds, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    job.id,
                    job.status.value,
                    job.gpu_count,
                    json.dumps(job.config),
                    job.priority,
                    job.estimated_seconds,
                    now.isoformat(),
                )
            )

        # Initialize log buffer for this job
        self._job_logs[job_id] = deque(maxlen=5000)

        # Try to start immediately if GPUs available
        gpu_indices = self.allocate_gpus(job_id, gpu_count)
        if gpu_indices:
            job.gpu_indices = gpu_indices
            job.status = JobStatus.RUNNING
            self._start_job(job)

        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()

            if not row:
                return None

            return self._row_to_job(row)

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        """Convert database row to Job object."""
        gpu_indices = []
        if row["gpu_indices"]:
            gpu_indices = [int(x) for x in row["gpu_indices"].split(",")]

        return Job(
            id=row["id"],
            config=json.loads(row["config"]),
            gpu_count=row["gpu_count"],
            status=JobStatus(row["status"]),
            gpu_indices=gpu_indices,
            priority=row["priority"],
            estimated_seconds=row["estimated_seconds"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
            error_message=row["error_message"],
            process_pid=row["process_pid"],
        )

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Job]:
        """List jobs with optional status filter."""
        with self._get_connection() as conn:
            if status:
                rows = conn.execute(
                    """SELECT * FROM jobs WHERE status = ?
                       ORDER BY priority DESC, created_at ASC
                       LIMIT ? OFFSET ?""",
                    (status.value, limit, offset)
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM jobs
                       ORDER BY created_at DESC
                       LIMIT ? OFFSET ?""",
                    (limit, offset)
                ).fetchall()

            return [self._row_to_job(row) for row in rows]

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        with self._lock:
            job = self.get_job(job_id)
            if not job:
                return False

            if job.status == JobStatus.RUNNING:
                # Stop the running process
                proc = self._running_processes.get(job_id)
                if proc:
                    try:
                        if sys.platform == "win32":
                            proc.terminate()
                        else:
                            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    del self._running_processes[job_id]

                # Release GPUs
                self.release_gpus(job_id)

            elif job.status != JobStatus.PENDING:
                return False

            # Update job status
            self._update_job_status(job_id, JobStatus.CANCELLED)
            return True

    def delete_job(self, job_id: str) -> bool:
        """Delete a job from history (only completed/failed/cancelled)."""
        with self._lock:
            job = self.get_job(job_id)
            if not job:
                return False

            if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
                return False

            with self._get_connection() as conn:
                conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

            # Clean up logs
            if job_id in self._job_logs:
                del self._job_logs[job_id]

            return True

    def _update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: Optional[str] = None,
        gpu_indices: Optional[List[int]] = None,
        process_pid: Optional[int] = None,
    ):
        """Update job status in database."""
        now = datetime.now().isoformat()
        updates = ["status = ?", "updated_at = ?"]
        values = [status.value, now]

        if status == JobStatus.RUNNING:
            updates.append("started_at = ?")
            values.append(now)

        if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            updates.append("finished_at = ?")
            values.append(now)

        if error_message is not None:
            updates.append("error_message = ?")
            values.append(error_message)

        if gpu_indices is not None:
            updates.append("gpu_indices = ?")
            values.append(",".join(str(i) for i in gpu_indices))

        if process_pid is not None:
            updates.append("process_pid = ?")
            values.append(process_pid) # type: ignore

        values.append(job_id)

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?",
                values
            )

    # =========================================================================
    # Queue Operations
    # =========================================================================

    def get_queue_position(self, job_id: str) -> Optional[int]:
        """Get position in queue (1-indexed). None if not pending."""
        job = self.get_job(job_id)
        if not job or job.status != JobStatus.PENDING:
            return None

        pending_jobs = self.list_jobs(status=JobStatus.PENDING, limit=1000)
        for i, pj in enumerate(pending_jobs):
            if pj.id == job_id:
                return i + 1
        return None

    def get_queue_eta(self, job_id: str) -> Optional[float]:
        """Estimate wait time in seconds until job starts."""
        position = self.get_queue_position(job_id)
        if position is None:
            return None

        pending_jobs = self.list_jobs(status=JobStatus.PENDING, limit=position)
        running_jobs = self.list_jobs(status=JobStatus.RUNNING, limit=100)

        # Calculate time for running jobs to complete
        running_time_remaining = 0.0
        for rj in running_jobs:
            if rj.estimated_seconds and rj.started_at:
                elapsed = (datetime.now() - rj.started_at).total_seconds()
                remaining = max(0, rj.estimated_seconds - elapsed)
                running_time_remaining = max(running_time_remaining, remaining)

        # Calculate time for pending jobs ahead in queue
        pending_time = 0.0
        for pj in pending_jobs[: position - 1]:
            pending_time += pj.estimated_seconds or 0

        return running_time_remaining + pending_time

    def get_queue_status(self) -> dict:
        """Get overall queue status."""
        total_gpus = self.get_total_gpus()
        available_gpus = self.get_available_gpus()
        gpu_details = self.get_gpu_status()
        running_jobs = self.list_jobs(status=JobStatus.RUNNING, limit=100)
        pending_jobs = self.list_jobs(status=JobStatus.PENDING, limit=100)

        queue = []
        for i, job in enumerate(pending_jobs):
            eta = self.get_queue_eta(job.id)
            queue.append({
                "job_id": job.id,
                "position": i + 1,
                "gpu_count": job.gpu_count,
                "eta": eta,
                "estimated_duration": job.estimated_seconds,
            })

        return {
            "total_gpus": total_gpus,
            "available_gpus": len(available_gpus),
            "gpu_details": gpu_details,
            "running_jobs": len(running_jobs),
            "pending_jobs": len(pending_jobs),
            "queue": queue,
        }

    def get_job_logs(self, job_id: str) -> List[str]:
        """Get logs for a job."""
        if job_id in self._job_logs:
            return list(self._job_logs[job_id])
        return []

    # =========================================================================
    # Job Execution
    # =========================================================================

    def _start_job(self, job: Job):
        """Start a training subprocess for the job."""
        ui_dir = Path(__file__).parent
        project_root = ui_dir.parent
        mini_cfg = adapt_ui_config_to_mini(job.config, project_root)

        # Write config to temp file
        config_path = ui_dir / f"_job_{job.id}_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(mini_cfg, f, default_flow_style=False)

        # Build environment with GPU assignment
        env = os.environ.copy()

        # Set PYTHONPATH
        python_path = env.get("PYTHONPATH", "")
        project_root_str = str(project_root)
        if project_root_str not in python_path:
            env["PYTHONPATH"] = project_root_str + os.pathsep + python_path

        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env["CONFIG_PATH"] = str(config_path)

        # Set CUDA_VISIBLE_DEVICES to allocated GPUs
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in job.gpu_indices)

        # Build command
        gpu_count = len(job.gpu_indices)
        strategy = str(mini_cfg.get("strategy", "solo")).lower()
        if strategy in {"ddp", "fsdp"} and gpu_count > 1:
            # Multi-GPU: use torchrun
            cmd = [
                sys.executable, "-u", "-m", "torch.distributed.run",
                f"--nproc_per_node={gpu_count}",
                "--master_addr=localhost",
                "--master_port=29500",
                "train.py",
            ]
        else:
            # Single GPU
            cmd = [
                sys.executable, "-u", "train.py",
            ]

        self._append_job_log(job.id, f"🚀 Starting queued training job {job.id}")
        self._append_job_log(job.id, f"📂 Working directory: {project_root}")
        self._append_job_log(job.id, f"🖥️ CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
        self._append_job_log(job.id, f"📜 Command: {' '.join(cmd)}")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=project_root_str,
                env=env,
                start_new_session=sys.platform != "win32",
            )

            self._running_processes[job.id] = proc

            # Update job status
            self._update_job_status(
                job.id,
                JobStatus.RUNNING,
                gpu_indices=job.gpu_indices,
                process_pid=proc.pid,
            )

            # Start log reader thread
            self._start_log_reader(job.id, proc)

        except Exception as e:
            self.release_gpus(job.id)
            self._update_job_status(job.id, JobStatus.FAILED, error_message=str(e))

    def _start_log_reader(self, job_id: str, proc: subprocess.Popen):
        """Start a background thread to read process output."""
        def reader():
            pending = ""
            try:
                if proc.stdout:
                    while True:
                        chunk = proc.stdout.read(1)
                        if chunk == "":
                            break

                        pending += chunk.replace("\r", "\n")
                        while "\n" in pending:
                            line, pending = pending.split("\n", 1)
                            self._append_job_log(job_id, line)

                    if pending.strip():
                        self._append_job_log(job_id, pending)
            except Exception as exc:
                self._append_job_log(job_id, f"Log streaming error: {exc}")

            proc.wait()
            self._append_job_log(
                job_id,
                "✅ Training job completed successfully" if proc.returncode == 0 else f"❌ Process exited with code {proc.returncode}",
            )
            self._on_job_complete(job_id, proc.returncode == 0)

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()

    def _on_job_complete(self, job_id: str, success: bool, error: Optional[str] = None):
        """Handle job completion - release GPUs and update DB."""
        with self._lock:
            # Clean up process reference
            if job_id in self._running_processes:
                del self._running_processes[job_id]

            # Release GPUs
            self.release_gpus(job_id)

            # Update status
            if success:
                self._update_job_status(job_id, JobStatus.COMPLETED)
            else:
                job = self.get_job(job_id)
                if job and job.status == JobStatus.RUNNING:
                    self._update_job_status(
                        job_id,
                        JobStatus.FAILED,
                        error_message=error or "Process exited with non-zero code"
                    )

            # Clean up config file
            config_path = Path(__file__).parent / f"_job_{job_id}_config.yaml"
            if config_path.exists():
                try:
                    config_path.unlink()
                except Exception:
                    pass

    # =========================================================================
    # Background Worker
    # =========================================================================

    def start_worker(self):
        """Start the background worker thread."""
        if self._worker_thread and self._worker_thread.is_alive():
            return

        self._shutdown_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop_worker(self):
        """Stop the background worker thread gracefully."""
        self._shutdown_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5)

        # Cancel all running jobs
        for job_id in list(self._running_processes.keys()):
            self.cancel_job(job_id)

    def _worker_loop(self):
        """Main worker loop - process queue when GPUs available."""
        while not self._shutdown_event.is_set():
            try:
                self._process_next_job()
            except Exception as e:
                # Log error but continue
                print(f"Queue worker error: {e}")

            # Wait before checking again
            self._shutdown_event.wait(timeout=2.0)

    def _process_next_job(self) -> bool:
        """Try to start the next pending job if resources available."""
        with self._lock:
            available_gpus = self.get_available_gpus()
            if not available_gpus:
                return False

            pending_jobs = self.list_jobs(status=JobStatus.PENDING, limit=100)

            for job in pending_jobs:
                if job.gpu_count <= len(available_gpus):
                    # Allocate GPUs and start job
                    gpu_indices = self.allocate_gpus(job.id, job.gpu_count)
                    if gpu_indices:
                        job.gpu_indices = gpu_indices
                        job.status = JobStatus.RUNNING
                        self._start_job(job)
                        return True

            return False

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup_stale_jobs(self):
        """Clean up jobs that were running when the server crashed."""
        with self._lock:
            with self._get_connection() as conn:
                # Mark running jobs as failed (they were interrupted)
                conn.execute(
                    """UPDATE jobs SET status = ?, error_message = ?, finished_at = ?
                       WHERE status = ?""",
                    (
                        JobStatus.FAILED.value,
                        "Server was restarted while job was running",
                        datetime.now().isoformat(),
                        JobStatus.RUNNING.value,
                    )
                )

                # Release all GPU allocations
                conn.execute("UPDATE gpu_allocations SET job_id = NULL, allocated_at = NULL")

"""A small local job runner: background threads, retries for temporary failures, resumable after restart.

Steps are idempotent (chip measurements are cached by input hash), so re-running a job after a
crash or restart never repeats finished work.
"""

from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tinyml.pipeline import stages
from tinyml.store import Store
from tinyml.toolchains import TransientError

MAX_ATTEMPTS = 3
BACKOFF_S = (2, 10)


class JobRunner:
    def __init__(self, store: Store, workers: int = 1):
        self.store = store
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tinyml-job")
        self._lock = threading.Lock()

    def resume(self) -> int:
        """Re-queue jobs that were queued or running when the app last stopped."""
        rows = self.store.all("SELECT id FROM jobs WHERE status IN ('queued', 'running')")
        for row in rows:
            self.store.append_job_log(row["id"], "Resuming after restart")
            self.pool.submit(self._execute, row["id"])
        return len(rows)

    def submit(self, kind: str, project_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        job = self.store.create_job(kind, project_id, params or {})
        self.pool.submit(self._execute, job["id"])
        return job

    def _execute(self, job_id: str) -> None:
        job = self.store.job(job_id)
        if job is None:
            return
        kind, project_id, params = job["kind"], job["project_id"], job["params"] or {}

        def log(message: str) -> None:
            self.store.append_job_log(job_id, f"{time.strftime('%H:%M:%S')} {message}")
            self.store.update_job(job_id, step=message[:200])

        for attempt in range(job["attempts"] + 1, MAX_ATTEMPTS + 1):
            self.store.update_job(job_id, status="running", attempts=attempt, error=None)
            try:
                result = self._run(kind, project_id, params, log)
                self.store.update_job(job_id, status="done", progress=1.0, result=_jsonable(result), step="done")
                return
            except TransientError as exc:
                log(f"Temporary failure: {exc}")
                if attempt < MAX_ATTEMPTS:
                    time.sleep(BACKOFF_S[min(attempt - 1, len(BACKOFF_S) - 1)])
                    continue
                self.store.update_job(job_id, status="failed", error=str(exc))
                return
            except stages.StageError as exc:
                log(str(exc))
                self.store.update_job(job_id, status="failed", error=str(exc))
                return
            except Exception as exc:
                log("Unexpected error: " + "".join(traceback.format_exception_only(type(exc), exc)).strip())
                self.store.append_job_log(job_id, traceback.format_exc())
                self.store.update_job(job_id, status="failed", error=str(exc))
                return

    def _run(self, kind: str, project_id: str, params: dict[str, Any], log) -> dict[str, Any]:
        if kind == "train":
            return stages.train(self.store, project_id, log)
        if kind == "rank":
            return stages.rank_chips(self.store, project_id, params.get("chips"), log)
        if kind == "build":
            return stages.build_for_chip(self.store, project_id, params["chip_id"], log)
        raise ValueError(f"unknown job kind {kind}")


def _jsonable(value: Any) -> Any:
    import json

    return json.loads(json.dumps(value, default=str))

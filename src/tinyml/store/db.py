"""SQLite metadata store with a small repository API.

Only this module talks SQL, so moving to Postgres later means rewriting one file.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tinyml.config import workspace

SCHEMA = Path(__file__).with_name("schema.sql")
JSON_COLUMNS = {
    "spec", "files", "summary", "roles", "label_map", "issues", "stats", "metrics",
    "measurement", "reasons", "tradeoffs", "parity", "reproducibility", "options",
    "params", "result", "output",
}


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _encode(column: str, value: Any) -> Any:
    if column in JSON_COLUMNS and value is not None and not isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return int(value)
    return value


def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    out: dict[str, Any] = {}
    for key in row.keys():
        value = row[key]
        if key in JSON_COLUMNS and isinstance(value, str):
            value = json.loads(value)
        out[key] = value
    return out


class Store:
    def __init__(self, path: Path | None = None):
        self.path = path or workspace() / "tinyml.db"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA.read_text())

    def close(self) -> None:
        self._conn.close()

    # generic helpers -------------------------------------------------------------
    def insert(self, table: str, row: dict[str, Any]) -> None:
        cols = list(row)
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
        with self._lock:
            self._conn.execute(sql, [_encode(c, row[c]) for c in cols])

    def update(self, table: str, key: str, key_value: Any, changes: dict[str, Any]) -> None:
        if not changes:
            return
        assignments = ", ".join(f"{c} = ?" for c in changes)
        sql = f"UPDATE {table} SET {assignments} WHERE {key} = ?"
        with self._lock:
            self._conn.execute(sql, [_encode(c, v) for c, v in changes.items()] + [key_value])

    def one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self._lock:
            return _decode(self._conn.execute(sql, params).fetchone())

    def all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [_decode(r) for r in self._conn.execute(sql, params).fetchall()]  # type: ignore[misc]

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return cur.lastrowid or 0

    # projects -----------------------------------------------------------------
    def create_project(self, name: str, description: str, spec: dict[str, Any] | None) -> dict[str, Any]:
        row = {
            "id": new_id("prj"), "name": name, "description": description, "spec": spec,
            "status": "described", "created_at": now(), "updated_at": now(),
        }
        self.insert("projects", row)
        return self.project(row["id"])  # type: ignore[return-value]

    def project(self, project_id: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM projects WHERE id = ?", (project_id,))

    def projects(self) -> list[dict[str, Any]]:
        return self.all("SELECT * FROM projects ORDER BY created_at DESC")

    def update_project(self, project_id: str, **changes: Any) -> None:
        self.update("projects", "id", project_id, {**changes, "updated_at": now()})

    # datasets -----------------------------------------------------------------
    def save_dataset(self, row: dict[str, Any]) -> dict[str, Any]:
        row = {"id": new_id("ds"), "created_at": now(), **row}
        self.insert("datasets", row)
        return row

    def latest_dataset(self, project_id: str) -> dict[str, Any] | None:
        return self.one(
            "SELECT * FROM datasets WHERE project_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (project_id,),
        )

    # candidates ---------------------------------------------------------------
    def replace_candidates(self, project_id: str, rows: list[dict[str, Any]]) -> None:
        self.execute("DELETE FROM candidates WHERE project_id = ?", (project_id,))
        for row in rows:
            self.insert("candidates", {"id": new_id("cand"), "project_id": project_id, "created_at": now(), **row})

    def candidates(self, project_id: str) -> list[dict[str, Any]]:
        return self.all("SELECT * FROM candidates WHERE project_id = ? ORDER BY rung, rowid", (project_id,))

    def select_candidate(self, project_id: str, candidate_id: str) -> None:
        self.execute("UPDATE candidates SET selected = (id = ?) WHERE project_id = ?", (candidate_id, project_id))

    def selected_candidate(self, project_id: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM candidates WHERE project_id = ? AND selected = 1", (project_id,))

    # chip results -------------------------------------------------------------
    def replace_chip_results(self, project_id: str, rows: list[dict[str, Any]]) -> None:
        self.execute("DELETE FROM chip_results WHERE project_id = ?", (project_id,))
        for row in rows:
            self.insert("chip_results", {"id": new_id("chip"), "project_id": project_id, "created_at": now(), **row})

    def chip_results(self, project_id: str) -> list[dict[str, Any]]:
        return self.all(
            "SELECT * FROM chip_results WHERE project_id = ? ORDER BY rank IS NULL, rank, chip_id", (project_id,)
        )

    # builds -------------------------------------------------------------------
    def save_build(self, row: dict[str, Any]) -> dict[str, Any]:
        row = {"id": new_id("bld"), "created_at": now(), **row}
        self.insert("builds", row)
        return row

    def builds(self, project_id: str) -> list[dict[str, Any]]:
        return self.all("SELECT * FROM builds WHERE project_id = ? ORDER BY created_at DESC, rowid DESC", (project_id,))

    def build(self, build_id: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM builds WHERE id = ?", (build_id,))

    # decisions ----------------------------------------------------------------
    def log_decision(self, row: dict[str, Any]) -> int:
        row = {"created_at": now(), "updated_at": now(), **row}
        cols = list(row)
        sql = f"INSERT INTO decisions ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
        return self.execute(sql, tuple(_encode(c, row[c]) for c in cols))

    def decision(self, decision_id: int) -> dict[str, Any] | None:
        return self.one("SELECT * FROM decisions WHERE id = ?", (decision_id,))

    def decisions(self, project_id: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        if project_id:
            return self.all(
                "SELECT * FROM decisions WHERE project_id = ? ORDER BY id DESC LIMIT ?", (project_id, limit)
            )
        return self.all("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,))

    def finalize_decision(self, decision_id: int, choice: str, source: str) -> None:
        self.update("decisions", "id", decision_id, {"final_choice": choice, "final_source": source, "updated_at": now()})

    def mark_build_outcome(self, project_id: str, passed: bool) -> None:
        self.execute(
            "UPDATE decisions SET build_passed = ?, updated_at = ? WHERE project_id = ?",
            (int(passed), now(), project_id),
        )

    def corrected_examples(self, decision_type: str, limit: int = 50) -> list[dict[str, Any]]:
        return self.all(
            "SELECT * FROM decisions WHERE decision_type = ? AND final_source = 'user' "
            "AND final_choice IS NOT NULL ORDER BY build_passed DESC, id DESC LIMIT ?",
            (decision_type, limit),
        )

    # learned rules --------------------------------------------------------------
    def learned_rule(self, decision_type: str, key: str) -> dict[str, Any] | None:
        return self.one(
            "SELECT * FROM learned_rules WHERE decision_type = ? AND decision_key = ?", (decision_type, key)
        )

    def upsert_learned_rule(self, decision_type: str, key: str, choice: str, support: int) -> None:
        self.execute(
            "INSERT INTO learned_rules (decision_type, decision_key, choice, support, created_at) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(decision_type, decision_key) "
            "DO UPDATE SET choice = excluded.choice, support = excluded.support",
            (decision_type, key, choice, support, now()),
        )

    def learned_rules(self) -> list[dict[str, Any]]:
        return self.all("SELECT * FROM learned_rules ORDER BY decision_type, decision_key")

    # jobs ---------------------------------------------------------------------
    def create_job(self, kind: str, project_id: str | None, params: dict[str, Any]) -> dict[str, Any]:
        row = {
            "id": new_id("job"), "project_id": project_id, "kind": kind, "status": "queued",
            "params": params, "created_at": now(), "updated_at": now(),
        }
        self.insert("jobs", row)
        return self.job(row["id"])  # type: ignore[return-value]

    def job(self, job_id: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM jobs WHERE id = ?", (job_id,))

    def jobs(self, project_id: str) -> list[dict[str, Any]]:
        return self.all("SELECT * FROM jobs WHERE project_id = ? ORDER BY created_at DESC", (project_id,))

    def update_job(self, job_id: str, **changes: Any) -> None:
        self.update("jobs", "id", job_id, {**changes, "updated_at": now()})

    def append_job_log(self, job_id: str, line: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET log = log || ?, updated_at = ? WHERE id = ?", (line + "\n", now(), job_id)
            )

    # step cache -----------------------------------------------------------------
    def cached_step(self, cache_key: str) -> dict[str, Any] | None:
        row = self.one("SELECT * FROM step_cache WHERE cache_key = ?", (cache_key,))
        return row["output"] if row else None

    def cache_step(self, cache_key: str, step: str, output: dict[str, Any]) -> None:
        self.execute(
            "INSERT OR REPLACE INTO step_cache (cache_key, step, output, created_at) VALUES (?, ?, ?, ?)",
            (cache_key, step, json.dumps(output), now()),
        )


_default: Store | None = None
_default_lock = threading.Lock()


def get_store() -> Store:
    global _default
    with _default_lock:
        if _default is None or _default.path != workspace() / "tinyml.db":
            _default = Store()
        return _default

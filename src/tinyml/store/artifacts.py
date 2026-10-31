"""Per-project folders for raw uploads, models, builds and packages."""

from __future__ import annotations

from pathlib import Path

from tinyml.config import workspace


def project_dir(project_id: str, *parts: str) -> Path:
    path = workspace() / "projects" / project_id
    for part in parts:
        path = path / part
    path.mkdir(parents=True, exist_ok=True)
    return path

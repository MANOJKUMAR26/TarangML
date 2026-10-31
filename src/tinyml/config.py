"""Paths and settings. Everything lives on the local machine."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def repo_root() -> Path:
    override = os.environ.get("TINYML_ROOT")
    if override:
        return Path(override).resolve()
    return PACKAGE_DIR.parents[1]


def workspace() -> Path:
    path = Path(os.environ.get("TINYML_HOME", repo_root() / "workspace")).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def recipes_dir() -> Path:
    return repo_root() / "recipes"


def catalog_path() -> Path:
    return repo_root() / "catalog" / "chips.yaml"


def firmware_dir() -> Path:
    return repo_root() / "firmware"


def ui_dir() -> Path:
    return repo_root() / "ui"


def keys_dir() -> Path:
    path = Path(os.environ.get("TINYML_KEYS", Path.home() / ".tinyml" / "keys"))
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class AssistantSettings:
    """Who answers a judgement call that the rules are unsure about.

    provider: "off" (ask the user), "claude_code" (headless `claude -p`),
    or "claude_chat" (show a prompt to paste into Claude chat).
    """

    provider: str = "off"
    claude_command: str = "claude"
    claude_model: str | None = None
    timeout_s: int = 120
    cutoffs: dict[str, float] = field(
        default_factory=lambda: {
            "recipe": 0.6,
            "task": 0.6,
            "connectivity": 0.7,
            "power": 0.7,
            "priority": 0.6,
            "column_role": 0.75,
            "normal_label": 0.7,
            "issue_severity": 0.7,
            "starting_rung": 0.5,
        }
    )


@dataclass
class Settings:
    assistant: AssistantSettings = field(default_factory=AssistantSettings)
    seed: int = 42
    max_golden_windows: int = 32
    parallel_chip_builds: int = 2
    emulator_timeout_s: int = 300

    @staticmethod
    def path() -> Path:
        return workspace() / "settings.json"

    def save(self) -> None:
        self.path().write_text(json.dumps(asdict(self), indent=2))


@lru_cache(maxsize=1)
def _cached_settings(mtime: float) -> Settings:
    raw = json.loads(Settings.path().read_text())
    assistant = AssistantSettings(**{**asdict(AssistantSettings()), **raw.get("assistant", {})})
    rest = {k: v for k, v in raw.items() if k != "assistant" and k in Settings.__dataclass_fields__}
    return Settings(assistant=assistant, **rest)


def load_settings() -> Settings:
    path = Settings.path()
    if not path.exists():
        return Settings()
    return _cached_settings(path.stat().st_mtime)

from __future__ import annotations

from pathlib import Path

import pytest

from tinyml.datasets import synthetic


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    """Every test gets its own workspace, database and signing key."""
    monkeypatch.setenv("TINYML_HOME", str(tmp_path / "ws"))
    monkeypatch.setenv("TINYML_KEYS", str(tmp_path / "keys"))
    return tmp_path / "ws"


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("data")
    synthetic.pump(folder / "pump.csv", rate=2000, normal_recordings=3, normal_s=12, fault_recordings=2, fault_s=4)
    synthetic.driving(folder / "driving.csv", trips=4, trip_s=150)
    synthetic.activity(folder / "activity.csv", people=3, seconds_each=40)
    synthetic.pump(folder / "pump_weak_faults.csv", rate=2000, normal_recordings=4, normal_s=20, fault_recordings=4,
                   fault_s=6, fault_strength=0.1)
    return folder


@pytest.fixture
def store(workspace):
    from tinyml.store import Store

    workspace.mkdir(parents=True, exist_ok=True)
    return Store(workspace / "test.db")

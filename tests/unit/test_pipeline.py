"""Stage A end to end on the host: data -> ladder -> C code -> host parity."""

import json

import pytest

from tinyml.pipeline import stages
from tinyml.store import get_store, project_dir


@pytest.mark.parametrize("csv,description,expected_task", [
    ("pump.csv", "Alert me when my pump vibration looks abnormal. Battery, Wi-Fi.", "anomaly"),
    ("driving.csv", "Detect harsh braking, harsh acceleration and sharp turns in vans.", "classification"),
    ("activity.csv", "Tell whether a worker is resting, walking, running or on stairs.", "classification"),
])
def test_stage_a_produces_verified_c(data_dir, csv, description, expected_task):
    store = get_store()
    pid = stages.describe(store, description)["project"]["id"]
    ds = stages.ingest(store, pid, [data_dir / csv])
    assert not ds["blocked"], ds["issues"]
    result = stages.train(store, pid)
    assert result["host_parity"]["passed"]
    artifact = json.loads((project_dir(pid, "models") / "model.json").read_text())
    assert artifact["task"] == expected_task
    assert (project_dir(pid, "models", "c") / "tinyml_model.c").exists()
    assert result["chosen"]["passed"]


def test_ladder_climbs_when_rules_are_not_enough(data_dir, monkeypatch):
    from tinyml.ml import ladder

    original = ladder._candidate

    def rules_never_pass(model, inp, note=""):
        cand = original(model, inp, note)
        if cand.rung == 1:
            cand.passed = False
        return cand

    monkeypatch.setattr(ladder, "_candidate", rules_never_pass)
    store = get_store()
    pid = stages.describe(store, "Tell whether a worker is resting, walking, running or on stairs.")["project"]["id"]
    stages.ingest(store, pid, [data_dir / "activity.csv"])
    result = stages.train(store, pid)
    assert result["chosen"]["rung"] == 2 and result["chosen"]["algorithm"] == "random_forest"
    assert result["host_parity"]["passed"]


def test_blocked_data_stops_training(tmp_path):
    import numpy as np
    import pandas as pd

    path = tmp_path / "tiny.csv"
    pd.DataFrame({"time": np.arange(200) / 100, "acc": np.random.default_rng(0).normal(size=200)}).to_csv(path, index=False)
    store = get_store()
    pid = stages.describe(store, "pump vibration abnormal")["project"]["id"]
    ds = stages.ingest(store, pid, [path])
    assert ds["blocked"]
    with pytest.raises(stages.StageError):
        stages.train(store, pid)

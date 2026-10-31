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


def test_every_rung_is_tried_and_the_user_can_switch(data_dir):
    store = get_store()
    pid = stages.describe(store, "Alert me when my pump vibration looks abnormal. Battery, Wi-Fi.")["project"]["id"]
    stages.ingest(store, pid, [data_dir / "pump.csv"])
    result = stages.train(store, pid)
    assert result["chosen"]["rung"] == 1 and result["chosen"]["id"] == result["recommended"]
    rows = store.candidates(pid)
    assert {r["rung"] for r in rows} == {1, 2, 3}
    other = next(r for r in rows if r["rung"] == 3 and r["artifact_path"])
    switched = stages.select_model(store, pid, other["id"])
    assert switched["host_parity"]["passed"]
    assert store.selected_candidate(pid)["id"] == other["id"]
    artifact = json.loads((project_dir(pid, "models") / "model.json").read_text())
    assert artifact["algorithm"] == other["algorithm"]


def test_no_model_meets_the_target_lets_the_user_choose(data_dir):
    store = get_store()
    pid = stages.describe(store, "Alert me when my pump vibration looks abnormal. Battery, Wi-Fi.")["project"]["id"]
    stages.ingest(store, pid, [data_dir / "pump_weak_faults.csv"])
    result = stages.train(store, pid)
    assert result["chosen"] is None and result["data_request"]
    assert store.project(pid)["status"] == "needs_choice"
    assert not (project_dir(pid, "models") / "model.json").exists()
    with pytest.raises(stages.StageError):
        stages.rank_chips(store, pid)
    chosen = stages.select_model(store, pid, result["best"])
    assert not chosen["chosen"]["passed"] and chosen["host_parity"]["passed"]
    assert "below the target" in chosen["explanation"]
    assert stages._below_target(store, pid) == result["data_request"]


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

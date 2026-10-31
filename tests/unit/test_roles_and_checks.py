import numpy as np
import pandas as pd

from tinyml.data.ingest import SOURCE_COLUMN, summarize
from tinyml.data.prepare import prepare
from tinyml.data.roles import guess_normal_label, guess_role
from tinyml.recipes import get_recipe


def _frame(n=6000, rate=100.0, **extra):
    t = np.arange(n) / rate
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"time": t, "acc_x": rng.normal(0, 1, n), "acc_y": rng.normal(0, 1, n),
                       "label": "normal", **extra})
    df[SOURCE_COLUMN] = "file1"
    return df


def test_roles_are_guessed_from_names_and_values():
    df = _frame(session=["s1"] * 3000 + ["s2"] * 3000)
    summary = summarize(df, ["file1.csv"], "")
    roles = {c.name: guess_role(c) for c in summary.columns}
    assert roles["time"][0] == "time"
    assert roles["acc_x"][0] == "sensor_axis"
    assert roles["label"][0] == "label"
    assert roles["session"][0] == "recording_id"


def test_axis_named_y_is_not_a_label():
    df = _frame()
    df["ay"] = np.linspace(-1, 1, len(df))
    summary = summarize(df, ["f.csv"], "")
    role = {c.name: guess_role(c) for c in summary.columns}["ay"][0]
    assert role != "label"


def test_normal_label_hint():
    assert guess_normal_label(["fault", "healthy"], ["normal", "healthy"])[0] == "healthy"


def test_checks_fix_missing_values_and_flag_clipping():
    df = _frame()
    df.loc[10:20, "acc_x"] = np.nan
    df.loc[100:300, "acc_y"] = 4.0  # stuck at the sensor limit
    roles = {"time": "time", "acc_x": "sensor_axis", "acc_y": "sensor_axis", "label": "label"}
    data = prepare(df, roles, get_recipe("pump_health"), "anomaly", normal_label="normal")
    codes = {i.code: i.severity for i in data.issues}
    assert codes["missing_values"] == "fix"
    assert codes["clipped"] == "warn"
    assert not np.isnan(data.recordings[0].signals).any()


def test_gap_splits_segments_and_ms_time_is_converted():
    df = _frame()
    df["time"] = (df["time"] * 1000).astype(int)
    df = df.rename(columns={"time": "timestamp_ms"})
    df = df.drop(index=range(3000, 3500)).reset_index(drop=True)
    roles = {"timestamp_ms": "time", "acc_x": "sensor_axis", "acc_y": "sensor_axis", "label": "label"}
    data = prepare(df, roles, get_recipe("pump_health"), "anomaly", normal_label="normal")
    assert data.rate == 100.0
    assert len(data.recordings[0].segments) == 2
    assert any(i.code == "gaps" for i in data.issues)


def test_not_enough_data_blocks_with_a_number():
    df = _frame(n=600)
    roles = {"time": "time", "acc_x": "sensor_axis", "acc_y": "sensor_axis", "label": "label"}
    data = prepare(df, roles, get_recipe("pump_health"), "anomaly", normal_label="normal")
    issue = next(i for i in data.issues if i.code == "not_enough_normal")
    assert issue.severity == "block" and "more minutes" in issue.message


def test_rate_mismatch_is_resampled():
    a = _frame(n=4000, rate=100.0)
    b = _frame(n=2000, rate=50.0)
    b[SOURCE_COLUMN] = "file2"
    df = pd.concat([a, b], ignore_index=True)
    roles = {"time": "time", "acc_x": "sensor_axis", "acc_y": "sensor_axis", "label": "label"}
    data = prepare(df, roles, get_recipe("pump_health"), "anomaly", normal_label="normal")
    assert data.rate == 100.0
    assert {i.code for i in data.issues} >= {"rate_mismatch", "upsampled"}

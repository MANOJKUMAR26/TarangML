"""Decide what each CSV column is: time, sensor axis, label, recording id, or ignore."""

from __future__ import annotations

import re

from tinyml.contracts import ColumnSummary, CsvSummary
from tinyml.decide import Answer, Decider, Question

ROLES = ["time", "sensor_axis", "label", "recording_id", "ignore"]

TIME_NAMES = r"^(t|time|timestamp|ts|times?_?(s|sec|ms|us)?|seconds?|secs?|millis|ms|elapsed|datetime|date_?time)$"
LABEL_NAMES = r"(^|_)(label|labels|class|state|target|activity|event|condition|status|fault|category|y)(_|$)"
RECORDING_NAMES = r"(^|_)(session|recording|rec|file|run|trial|subject|experiment|id|take|capture|drive|trip)(_?id)?$"
AXIS_NAMES = r"(acc|gyro|gyr|vib|imu|sensor|signal|channel|(^|_)(ax|ay|az|gx|gy|gz|x|y|z|de|fe|ba|ch\d*)($|_))"


def guess_role(col: ColumnSummary) -> tuple[str, float]:
    name = col.name.strip().lower()
    if col.unique <= 1:
        if re.search(LABEL_NAMES, name):
            return "label", 0.8  # one label for all rows is normal for normal-only anomaly data
        return "ignore", 0.9
    if re.match(TIME_NAMES, name) or "timestamp" in name:
        return "time", 0.97 if col.always_increasing else 0.9
    many_numbers = col.dtype == "numeric" and col.unique > 50
    if re.search(LABEL_NAMES, name) and not many_numbers:
        return "label", 0.95 if col.unique <= 50 else 0.6
    if re.search(RECORDING_NAMES, name):
        return "recording_id", 0.9
    if col.dtype == "datetime":
        return "time", 0.85
    if col.dtype == "text":
        if col.unique <= 20:
            return "label", 0.65
        return "ignore", 0.6
    # numeric
    if col.always_increasing and col.step and col.step > 0:
        return "time", 0.6
    if col.unique <= 12 and col.count > 1000 and col.min is not None and float(col.min).is_integer():
        return "label", 0.5
    if re.search(AXIS_NAMES, name):
        return "sensor_axis", 0.92
    return "sensor_axis", 0.75


def assign_roles(summary: CsvSummary, decider: Decider) -> dict[str, Answer]:
    answers: dict[str, Answer] = {}
    for col in summary.columns:
        role, conf = guess_role(col)
        facts = col.model_dump(exclude={"count"})
        answers[col.name] = decider.decide(
            Question(
                type="column_role",
                key=col.name.strip().lower(),
                text=f"What is column '{col.name}'?",
                options=ROLES,
                summary=facts,
                use_case=summary.use_case,
            ),
            (role, conf),
        )
    return answers


def guess_normal_label(labels: list[str], hints: list[str]) -> tuple[str, float]:
    lowered = {lab: str(lab).strip().lower() for lab in labels}
    for hint in hints:
        for lab, low in lowered.items():
            if low == hint:
                return lab, 0.95
    for hint in hints:
        for lab, low in lowered.items():
            if hint in low:
                return lab, 0.8
    if "0" in lowered.values():
        return next(lab for lab, low in lowered.items() if low == "0"), 0.6
    return labels[0], 0.3

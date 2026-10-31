"""Save cleaned, steady-rate data as Parquet so later stages never re-read the raw CSV."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tinyml.contracts import DataIssue
from tinyml.data.prepare import PreparedData, Recording


def save_prepared(data: PreparedData, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for rec in data.recordings:
        frame = pd.DataFrame(rec.signals, columns=data.axes)
        frame.insert(0, "__rec", rec.id)
        frame["__label"] = rec.labels if rec.labels is not None else None
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["__rec", *data.axes, "__label"])
    table.to_parquet(folder / "prepared.parquet", index=False)
    meta = {
        "axes": data.axes,
        "rate": data.rate,
        "normal_label": data.normal_label,
        "segments": {rec.id: rec.segments for rec in data.recordings},
        "has_labels": {rec.id: rec.labels is not None for rec in data.recordings},
        "issues": [i.model_dump() for i in data.issues],
        "stats": data.stats,
    }
    (folder / "prepared.json").write_text(json.dumps(meta, indent=1))


def load_prepared(folder: Path) -> PreparedData:
    meta = json.loads((folder / "prepared.json").read_text())
    table = pd.read_parquet(folder / "prepared.parquet")
    axes = meta["axes"]
    recordings = []
    for rec_id, group in table.groupby("__rec", sort=False):
        labels = group["__label"].to_numpy().astype(str) if meta["has_labels"][rec_id] else None
        recordings.append(Recording(
            id=str(rec_id), rate=meta["rate"], signals=group[axes].to_numpy(dtype=np.float32),
            labels=labels, segments=[tuple(s) for s in meta["segments"][rec_id]],
        ))
    return PreparedData(axes=axes, rate=meta["rate"], recordings=recordings,
                        issues=[DataIssue(**i) for i in meta["issues"]], stats=meta["stats"],
                        normal_label=meta["normal_label"])

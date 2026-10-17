"""Step 3: checks, cleaning and resampling. Problems are explained in plain words.

Output is a set of recordings sampled at one steady rate, cut into gap-free segments,
plus a list of issues with severity fix (we fixed it), warn (you should know) or block (cannot continue).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from tinyml.contracts import DataIssue
from tinyml.data.ingest import SOURCE_COLUMN
from tinyml.decide import Decider, Question
from tinyml.recipes import Recipe

DEFAULT_SEVERITY = {
    "no_sensor_columns": "block",
    "no_time_or_rate": "block",
    "text_sensor_column": "warn",
    "time_in_ms": "fix",
    "time_unsorted": "fix",
    "duplicate_timestamps": "fix",
    "uneven_timing": "fix",
    "rate_mismatch": "fix",
    "upsampled": "warn",
    "gaps": "warn",
    "missing_values": "fix",
    "many_missing_values": "warn",
    "missing_labels": "warn",
    "clipped": "warn",
    "flat_axis": "warn",
    "single_class": "block",
    "not_enough_normal": "block",
    "not_enough_class": "block",
    "no_fault_data": "warn",
    "single_recording": "warn",
    "no_label_for_classification": "block",
}


@dataclass
class Recording:
    id: str
    rate: float
    signals: np.ndarray  # float32, shape (n, axes), steady rate
    labels: np.ndarray | None  # str per sample
    segments: list[tuple[int, int]]  # [start, end) without gaps


@dataclass
class PreparedData:
    axes: list[str]
    rate: float
    recordings: list[Recording]
    issues: list[DataIssue]
    stats: dict[str, Any] = field(default_factory=dict)
    normal_label: str | None = None

    @property
    def blocked(self) -> bool:
        return any(i.severity == "block" for i in self.issues)


class _Issues:
    def __init__(self, decider: Decider | None):
        self.decider = decider
        self.items: list[DataIssue] = []

    def add(self, code: str, message: str, columns: list[str] | None = None, **detail: Any) -> None:
        default = DEFAULT_SEVERITY.get(code, "warn")
        severity = default
        if self.decider is not None:
            answer = self.decider.decide(
                Question("issue_severity", code, f"How serious is this data problem? {message}",
                         ["fix", "warn", "block"], {"code": code, **detail}),
                (default, 0.9),
            )
            severity = answer.choice
        self.items.append(DataIssue(code=code, severity=severity, message=message,  # type: ignore[arg-type]
                                    columns=columns or [], detail=detail))


def _to_seconds(series: pd.Series, name: str, issues: _Issues) -> np.ndarray:
    if not pd.api.types.is_numeric_dtype(series):
        parsed = pd.to_datetime(series.astype(str), errors="coerce", format="mixed")
        return parsed.astype("int64").to_numpy() / 1e9
    values = series.to_numpy(dtype=np.float64)
    low = name.lower()
    if "ms" in low or "milli" in low:
        return values / 1e3
    if low.endswith("us") or "micro" in low:
        return values / 1e6
    if low.endswith("ns") or "nano" in low:
        return values / 1e9
    finite = values[np.isfinite(values)]
    peak = np.nanmax(np.abs(finite)) if len(finite) else 0.0
    if peak > 1e17:
        return values / 1e9
    if peak > 1e14:
        return values / 1e6
    if peak > 1e11:
        return values / 1e3
    diffs = np.diff(finite)
    diffs = diffs[diffs > 0]
    if len(diffs) and np.median(diffs) >= 1.0 and 1000.0 / np.median(diffs) >= 5:
        issues.add("time_in_ms", f"Column '{name}' looks like milliseconds; converted to seconds.", [name])
        return values / 1e3
    return values


def _segments(t: np.ndarray, median_dt: float) -> list[tuple[int, int]]:
    breaks = np.where(np.diff(t) > 2.5 * median_dt)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks + 1, [len(t)]])
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True) if e - s >= 2]


def _segment_rate(t: np.ndarray, median_dt: float) -> float:
    """Rate from the longest gap-free stretch, snapped to a whole number when it is that close.

    (n - 1) / span is far less sensitive to rounded timestamps than 1 / median(dt).
    """
    segs = _segments(t, median_dt)
    s, e = max(segs, key=lambda se: se[1] - se[0]) if segs else (0, len(t))
    span = t[e - 1] - t[s]
    rate = (e - s - 1) / span if span > 0 else 1.0 / median_dt
    nearest = round(rate)
    if nearest > 0 and abs(rate - nearest) <= 5e-4 * rate:
        return float(nearest)
    return float(rate)


def _resample_segment(t: np.ndarray, x: np.ndarray, src_rate: float, target: float) -> np.ndarray:
    n_uniform = int(np.floor((t[-1] - t[0]) * src_rate)) + 1
    grid = t[0] + np.arange(n_uniform) / src_rate
    uniform = np.column_stack([np.interp(grid, t, x[:, a]) for a in range(x.shape[1])])
    if abs(src_rate - target) / target <= 0.01:
        return uniform.astype(np.float32)
    ratio = Fraction(target / src_rate).limit_denominator(512)
    out = resample_poly(uniform, ratio.numerator, ratio.denominator, axis=0)
    return out.astype(np.float32)


def prepare(
    df: pd.DataFrame,
    roles: dict[str, str],
    recipe: Recipe,
    task: str,
    decider: Decider | None = None,
    normal_label: str | None = None,
    sample_rate_hz: float | None = None,
) -> PreparedData:
    issues = _Issues(decider)
    axes = [c for c, r in roles.items() if r == "sensor_axis"]
    text_axes = [c for c in axes if not pd.api.types.is_numeric_dtype(df[c])]
    for col in text_axes:
        issues.add("text_sensor_column", f"Column '{col}' holds text, so it cannot be a sensor reading. Ignored.", [col])
    axes = [c for c in axes if c not in text_axes]
    if not axes:
        issues.add("no_sensor_columns", "No sensor columns were found. Mark at least one column as a sensor axis.")
        return PreparedData(axes=[], rate=0.0, recordings=[], issues=issues.items)

    time_cols = [c for c, r in roles.items() if r == "time"]
    label_cols = [c for c, r in roles.items() if r == "label"]
    rec_cols = [c for c, r in roles.items() if r == "recording_id"]
    time_col = time_cols[0] if time_cols else None
    label_col = label_cols[0] if label_cols else None

    if time_col is None and not sample_rate_hz:
        issues.add("no_time_or_rate", "There is no time column. Tell us how many readings per second the sensor takes.")
        return PreparedData(axes=axes, rate=0.0, recordings=[], issues=issues.items)
    if task == "classification" and label_col is None:
        issues.add("no_label_for_classification",
                   "To tell events apart we need a column that names the event for each row (for example 'label').")
        return PreparedData(axes=axes, rate=0.0, recordings=[], issues=issues.items)

    work = pd.DataFrame({a: pd.to_numeric(df[a], errors="coerce") for a in axes})
    work["__t"] = _to_seconds(df[time_col], time_col, issues) if time_col else np.nan
    multi_file = df[SOURCE_COLUMN].nunique() > 1
    if rec_cols:
        rec = df[rec_cols[0]].astype(str)
        work["__rec"] = (df[SOURCE_COLUMN].astype(str) + "/" + rec) if multi_file else rec
    else:
        work["__rec"] = df[SOURCE_COLUMN].astype(str)
    if label_col:
        work["__label"] = df[label_col].astype(str).where(df[label_col].notna(), None)
        missing_labels = int(work["__label"].isna().sum())
        if missing_labels:
            issues.add("missing_labels", f"{missing_labels} rows have no label and were left out.", [label_col],
                       rows=missing_labels)
            work = work[work["__label"].notna()]

    for axis in axes:
        frac = float(work[axis].isna().mean())
        if frac > 0.05:
            issues.add("many_missing_values", f"{frac:.0%} of '{axis}' readings are empty. Check the sensor wiring "
                       "or logger; gaps were filled by joining the neighbouring readings.", [axis], fraction=frac)
        elif frac > 0:
            issues.add("missing_values", f"{frac:.2%} of '{axis}' readings were empty and were filled in.",
                       [axis], fraction=frac)

    for axis in axes:
        col = work[axis].dropna().to_numpy()
        if len(col) < 10:
            continue
        lo, hi = col.min(), col.max()
        if hi - lo == 0:
            issues.add("flat_axis", f"'{axis}' never changes, so it carries no information. Ignored.", [axis])
            continue
        at_edges = np.mean(col == hi) + np.mean(col == lo)
        if at_edges > 0.002 and (np.sum(col == hi) > 10 or np.sum(col == lo) > 10):
            issues.add("clipped", f"'{axis}' is stuck at its limit ({lo:g} / {hi:g}) in {at_edges:.1%} of readings. "
                       "The sensor range is probably too small; set a wider range (for example ±16 g).",
                       [axis], fraction=float(at_edges))
    flat = {i.columns[0] for i in issues.items if i.code == "flat_axis"}
    axes = [a for a in axes if a not in flat]

    recordings_raw = []
    unsorted = duplicates = 0
    for rec_id, group in work.groupby("__rec", sort=False):
        if time_col:
            if not group["__t"].is_monotonic_increasing:
                unsorted += 1
                group = group.sort_values("__t", kind="stable")
            dup = group["__t"].duplicated()
            if dup.any():
                duplicates += int(dup.sum())
                group = group[~dup]
        if len(group) < 4:
            continue
        group = group.copy()
        group[axes] = group[axes].interpolate(limit_direction="both")
        recordings_raw.append((str(rec_id), group))
    if unsorted:
        issues.add("time_unsorted", f"Time went backwards in {unsorted} recording(s); rows were put in time order.")
    if duplicates:
        issues.add("duplicate_timestamps", f"{duplicates} rows repeated an earlier timestamp and were dropped.")

    rates: dict[str, float] = {}
    jittery = []
    for rec_id, group in recordings_raw:
        if time_col:
            dt = np.diff(group["__t"].to_numpy())
            median_dt = float(np.median(dt)) if len(dt) else 0.0
            if median_dt <= 0:
                continue
            rates[rec_id] = _segment_rate(group["__t"].to_numpy(), median_dt)
            spread = (np.percentile(dt, 95) - np.percentile(dt, 5)) / median_dt
            if spread > 0.1:
                jittery.append(rec_id)
        else:
            rates[rec_id] = float(sample_rate_hz)  # type: ignore[arg-type]
    if not rates:
        issues.add("no_time_or_rate", "Could not work out the sampling rate from the time column.")
        return PreparedData(axes=axes, rate=0.0, recordings=[], issues=issues.items)

    weights: dict[float, int] = {}
    for rec_id, group in recordings_raw:
        if rec_id in rates:
            key = round(rates[rec_id], 1)
            weights[key] = weights.get(key, 0) + len(group)
    target = max(weights.items(), key=lambda kv: kv[1])[0]
    if jittery:
        issues.add("uneven_timing", f"Readings arrive at uneven intervals in {len(jittery)} recording(s); "
                   f"they were resampled to a steady {target:g} per second.", recordings=jittery[:10])
    mismatched = [r for r, v in rates.items() if abs(v - target) / target > 0.01]
    if mismatched:
        issues.add("rate_mismatch", f"{len(mismatched)} recording(s) use a different sampling rate; all were converted "
                   f"to {target:g} per second.", recordings=mismatched[:10])
    slower = [r for r in mismatched if rates[r] < target]
    if slower:
        issues.add("upsampled", f"{len(slower)} recording(s) were sampled slower than the rest. Fast details are "
                   "missing in them; record at the same rate if you can.", recordings=slower[:10])

    recordings: list[Recording] = []
    gap_count, longest_gap = 0, 0.0
    for rec_id, group in recordings_raw:
        if rec_id not in rates:
            continue
        src_rate = rates[rec_id]
        values = group[axes].to_numpy(dtype=np.float64)
        labels_raw = group["__label"].to_numpy() if label_col else None
        if time_col:
            t = group["__t"].to_numpy()
            segs = _segments(t, 1.0 / src_rate)
            if len(segs) > 1:
                gap_count += len(segs) - 1
                gaps = [t[segs[i + 1][0]] - t[segs[i][1] - 1] for i in range(len(segs) - 1)]
                longest_gap = max(longest_gap, max(gaps))
        else:
            t = np.arange(len(group)) / src_rate
            segs = [(0, len(group))]
        out_parts, label_parts, out_segs, cursor = [], [], [], 0
        for s, e in segs:
            seg = _resample_segment(t[s:e], values[s:e], src_rate, target)
            if len(seg) < 2:
                continue
            out_parts.append(seg)
            if labels_raw is not None:
                grid = t[s] + np.arange(len(seg)) / target
                idx = np.clip(np.searchsorted(t[s:e], grid), 0, e - s - 1)
                label_parts.append(labels_raw[s:e][idx])
            out_segs.append((cursor, cursor + len(seg)))
            cursor += len(seg)
        if not out_parts:
            continue
        recordings.append(Recording(
            id=rec_id, rate=target, signals=np.vstack(out_parts),
            labels=np.concatenate(label_parts).astype(str) if label_parts else None, segments=out_segs,
        ))
    if gap_count:
        issues.add("gaps", f"Found {gap_count} gap(s) in the recordings (longest {longest_gap:.2f} s). "
                   "Windows never cross a gap.", gaps=gap_count, longest_s=round(longest_gap, 3))

    stats = _data_stats(recordings, target, axes)
    prepared = PreparedData(axes=axes, rate=target, recordings=recordings, issues=issues.items, stats=stats,
                            normal_label=normal_label)
    _check_amount(prepared, recipe, task, normal_label, issues)
    prepared.issues = issues.items
    return prepared


def _data_stats(recordings: list[Recording], rate: float, axes: list[str]) -> dict[str, Any]:
    per_class: dict[str, float] = {}
    rec_per_class: dict[str, set[str]] = {}
    total = 0.0
    for rec in recordings:
        total += len(rec.signals) / rate
        if rec.labels is None:
            continue
        values, counts = np.unique(rec.labels, return_counts=True)
        for value, count in zip(values, counts, strict=True):
            per_class[str(value)] = per_class.get(str(value), 0.0) + count / rate
            rec_per_class.setdefault(str(value), set()).add(rec.id)
    return {
        "sample_rate_hz": rate,
        "axes": axes,
        "recordings": len(recordings),
        "duration_s": round(float(total), 2),
        "class_seconds": {k: round(float(v), 2) for k, v in sorted(per_class.items())},
        "class_recordings": {k: len(v) for k, v in sorted(rec_per_class.items())},
    }


def _check_amount(data: PreparedData, recipe: Recipe, task: str, normal_label: str | None, issues: _Issues) -> None:
    class_seconds: dict[str, float] = data.stats["class_seconds"]
    class_recs: dict[str, int] = data.stats["class_recordings"]
    window_s = min(recipe.window_ms / 1000.0, recipe.max_window_samples / max(data.rate, 1e-9))
    hop_s = window_s * recipe.hop_ratio
    if task == "anomaly":
        normal_s = class_seconds.get(normal_label, 0.0) if normal_label else data.stats["duration_s"]
        need = recipe.min_minutes["normal"] * 60
        if normal_s < need:
            issues.add("not_enough_normal", f"You have {normal_s / 60:.1f} minutes of normal data; this needs about "
                       f"{need / 60:.1f}. Record {max(need - normal_s, 0) / 60:.1f} more minutes of the machine running "
                       "normally.", have_s=round(normal_s, 1), need_s=need)
        others = [c for c in class_seconds if c != normal_label] if normal_label else []
        if not others:
            issues.add("no_fault_data", "There are no recordings of faults, so we can measure false alarms but not "
                       "how many real faults get caught. Add a few fault recordings if you have any.")
        if normal_label and class_recs.get(normal_label, 0) < 2:
            issues.add("single_recording", "All normal data comes from one recording. We will test on a held-back "
                       "part of it, so results may look better than in the field.")
        return
    if len(class_seconds) < 2:
        issues.add("single_class", "Only one kind of event is in the data. Add recordings of the other events "
                   "you want to tell apart.")
        return
    short = []
    for label, seconds in class_seconds.items():
        windows = int(max(0.0, seconds - window_s) / hop_s) + 1 if seconds >= window_s else 0
        need_s = max(recipe.min_minutes["per_class"] * 60, recipe.min_windows_per_class * hop_s + window_s)
        if seconds < need_s:
            short.append(f"'{label}' has {seconds:.0f} s (about {windows} examples), needs about {need_s:.0f} s")
    if short:
        issues.add("not_enough_class", "Some events have too little data: " + "; ".join(short) + ".",
                   classes=short)
    lonely = [c for c, n in class_recs.items() if n < 2]
    if lonely:
        issues.add("single_recording", f"{', '.join(lonely)} come from a single recording each. We will test on a "
                   "held-back part, so results may look better than in the field.", classes=lonely)

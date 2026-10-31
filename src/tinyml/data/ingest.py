"""Read uploaded CSVs and build the per-column summary (the only thing the assistant ever sees)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from tinyml.contracts import ColumnSummary, CsvSummary

SOURCE_COLUMN = "__source_file__"


def file_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        digest.update(path.name.encode())
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def read_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path, engine="pyarrow")
        except Exception:  # pyarrow is strict about odd files; fall back to the C parser
            frame = pd.read_csv(path, sep=None, engine="python")
        frame.columns = [str(c).strip() for c in frame.columns]
        frame[SOURCE_COLUMN] = path.stem
        frames.append(frame)
    if not frames:
        raise ValueError("No CSV files given")
    columns = list(dict.fromkeys(c for f in frames for c in f.columns))
    return pd.concat([f.reindex(columns=columns) for f in frames], ignore_index=True)


def _is_datetime_text(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(20)
    if sample.empty or sample.str.fullmatch(r"-?\d+(\.\d+)?").all():
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.9


def _within_group_diffs(values: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
    if len(values) < 2:
        return np.array([])
    diffs = np.diff(values)
    if groups is not None:
        diffs = diffs[groups[1:] == groups[:-1]]
    return diffs


def summarize_column(name: str, series: pd.Series, groups: np.ndarray | None = None) -> ColumnSummary:
    """Summarise one column. `groups` marks source files so a clock that restarts per file still counts as increasing."""
    missing = int(series.isna().sum())
    present = series.dropna()
    sample = [str(v) for v in present.head(3).tolist()]
    unique = int(present.nunique(dropna=True)) if len(present) < 2_000_000 else int(present.head(200_000).nunique())
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        values = present.to_numpy(dtype=np.float64)
        present_groups = groups[present.index.to_numpy()] if groups is not None else None
        diffs = _within_group_diffs(values, present_groups)
        increasing = bool(len(diffs) > 0 and np.all(diffs > 0))
        step = float(np.median(diffs)) if len(diffs) else None
        return ColumnSummary(
            name=name, dtype="numeric", count=len(series), missing=missing,
            min=float(values.min()) if len(values) else None,
            max=float(values.max()) if len(values) else None,
            mean=float(values.mean()) if len(values) else None,
            std=float(values.std()) if len(values) else None,
            unique=unique, always_increasing=increasing, step=step, sample=sample,
        )
    if _is_datetime_text(series):
        parsed = pd.to_datetime(present.astype(str), errors="coerce", format="mixed")
        parsed = parsed.dropna()
        seconds = parsed.astype("int64").to_numpy() / 1e9
        present_groups = groups[parsed.index.to_numpy()] if groups is not None else None
        diffs = _within_group_diffs(seconds, present_groups)
        return ColumnSummary(
            name=name, dtype="datetime", count=len(series), missing=missing,
            min=float(seconds.min()) if len(seconds) else None,
            max=float(seconds.max()) if len(seconds) else None,
            unique=unique, always_increasing=bool(len(diffs) and np.all(diffs > 0)),
            step=float(np.median(diffs)) if len(diffs) else None, sample=sample,
        )
    return ColumnSummary(
        name=name, dtype="text", count=len(series), missing=missing, unique=unique,
        always_increasing=False, sample=sample,
    )


def summarize(df: pd.DataFrame, files: list[str], use_case: str) -> CsvSummary:
    groups = df[SOURCE_COLUMN].to_numpy() if SOURCE_COLUMN in df else None
    columns = [summarize_column(c, df[c], groups) for c in df.columns if c != SOURCE_COLUMN]
    return CsvSummary(use_case=use_case, files=files, rows=len(df), columns=columns)

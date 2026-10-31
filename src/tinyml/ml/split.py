"""Train/test split by recording session.

Neighbouring windows from one recording are near-copies. Splitting windows at random puts
near-copies on both sides and gives fake-good accuracy, so whole recordings go to one side.
When a class has only one recording, the end of each recording is held back instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tinyml.data.windows import WindowSet

TEST_SHARE = 0.25


@dataclass
class Split:
    train: np.ndarray  # bool mask over windows
    test: np.ndarray
    method: str


def _hold_back_end(ws: WindowSet, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each (recording, label) run, the last quarter goes to test; windows overlapping the cut are dropped."""
    train = np.zeros(len(ws), dtype=bool)
    test = np.zeros(len(ws), dtype=bool)
    keys = np.char.add(np.char.add(ws.recordings.astype(str), "\x00"), ws.labels.astype(str))
    for key in np.unique(keys[rows]):
        idx = np.where(rows & (keys == key))[0]
        idx = idx[np.argsort(ws.starts[idx])]
        if len(idx) < 4:
            train[idx] = True
            continue
        cut = int(np.ceil(len(idx) * (1 - TEST_SHARE)))
        cut_start = ws.starts[idx[cut]]
        test[idx[cut:]] = True
        safe = ws.starts[idx[:cut]] + ws.window_samples <= cut_start
        train[idx[:cut][safe]] = True
    return train, test


def _pick_test_recordings(recs: list[str], rng: np.random.Generator) -> set[str]:
    order = list(recs)
    rng.shuffle(order)
    count = max(1, int(round(len(order) * TEST_SHARE)))
    return set(order[:count])


def split_windows(ws: WindowSet, task: str, normal_label: str | None, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    if task == "anomaly":
        normal = ws.labels == normal_label if normal_label is not None else np.ones(len(ws), dtype=bool)
        normal_recs = sorted(set(ws.recordings[normal]))
        if len(normal_recs) >= 2:
            test_recs = _pick_test_recordings(normal_recs, rng)
            in_test = np.isin(ws.recordings, list(test_recs))
            return Split(normal & ~in_test, (normal & in_test) | ~normal, "whole recordings held back for testing")
        train, test = _hold_back_end(ws, normal)
        return Split(train, test | ~normal, "the end of each normal recording held back for testing")

    classes = sorted(set(ws.labels))
    recs_by_class = {c: sorted(set(ws.recordings[ws.labels == c])) for c in classes}
    if all(len(r) >= 2 for r in recs_by_class.values()):
        for _ in range(50):
            test_recs: set[str] = set()
            for c in classes:
                if not test_recs & set(recs_by_class[c]):
                    test_recs |= _pick_test_recordings(recs_by_class[c], rng)
            ok = all(set(recs_by_class[c]) - test_recs for c in classes)
            if ok:
                in_test = np.isin(ws.recordings, list(test_recs))
                return Split(~in_test, in_test, "whole recordings held back for testing")
    train, test = _hold_back_end(ws, np.ones(len(ws), dtype=bool))
    return Split(train, test, "the end of each recording held back for testing")

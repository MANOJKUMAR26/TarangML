"""Cut the steady-rate stream into fixed windows. The model judges one window at a time."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tinyml.data.prepare import PreparedData
from tinyml.recipes import Recipe

MIN_LABEL_SHARE = 0.8  # a window gets a label only if most of it agrees


@dataclass
class WindowSet:
    X: np.ndarray  # (windows, samples, axes) float32
    labels: np.ndarray  # str
    recordings: np.ndarray  # str
    starts: np.ndarray  # first sample of each window inside its recording
    window_samples: int
    hop: int
    rate: float
    axes: list[str]

    def __len__(self) -> int:
        return len(self.X)

    def subset(self, mask: np.ndarray) -> WindowSet:
        return WindowSet(self.X[mask], self.labels[mask], self.recordings[mask], self.starts[mask],
                         self.window_samples, self.hop, self.rate, self.axes)


def window_plan(recipe: Recipe, rate: float) -> tuple[int, int]:
    samples = int(round(rate * recipe.window_ms / 1000.0))
    samples = max(16, min(samples, recipe.max_window_samples))
    hop = max(1, int(round(samples * recipe.hop_ratio)))
    return samples, hop


def make_windows(data: PreparedData, recipe: Recipe, default_label: str = "normal") -> WindowSet:
    n, hop = window_plan(recipe, data.rate)
    xs, labels, recs, starts = [], [], [], []
    for rec in data.recordings:
        for seg_start, seg_end in rec.segments:
            for start in range(seg_start, seg_end - n + 1, hop):
                if rec.labels is not None:
                    values, counts = np.unique(rec.labels[start:start + n], return_counts=True)
                    top = int(np.argmax(counts))
                    if counts[top] / n < MIN_LABEL_SHARE:
                        continue
                    label = str(values[top])
                else:
                    label = default_label
                xs.append(rec.signals[start:start + n])
                labels.append(label)
                recs.append(rec.id)
                starts.append(start)
    if not xs:
        empty = np.zeros((0, n, len(data.axes)), dtype=np.float32)
        return WindowSet(empty, np.array([], dtype=str), np.array([], dtype=str), np.array([], dtype=int),
                         n, hop, data.rate, data.axes)
    return WindowSet(
        X=np.stack(xs).astype(np.float32),
        labels=np.array(labels, dtype=str),
        recordings=np.array(recs, dtype=str),
        starts=np.array(starts, dtype=int),
        window_samples=n,
        hop=hop,
        rate=data.rate,
        axes=list(data.axes),
    )

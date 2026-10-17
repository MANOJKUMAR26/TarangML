"""Deterministic, physically plausible demo datasets.

They let you try the whole flow offline and keep the test suite fast. For real
pump data use the CWRU converter in tinyml.datasets.cwru.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

G = 1.0  # readings are in g


def _raised_cosine(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / max(n - 1, 1))


def pump(path: Path, seed: int = 7, rate: int = 4000, normal_recordings: int = 4, normal_s: float = 30.0,
         fault_recordings: int = 3, fault_s: float = 10.0) -> Path:
    """Two-axis vibration of a pump. Faults add bearing impacts that ring a structural resonance."""
    rng = np.random.default_rng(seed)
    frames = []

    def recording(rec_id: str, seconds: float, fault: bool) -> pd.DataFrame:
        n = int(seconds * rate)
        t = np.arange(n) / rate
        shaft = rng.uniform(28.0, 30.0)
        load = rng.uniform(0.8, 1.2)
        base = np.zeros((n, 2))
        for h, amp in ((1, 0.08), (2, 0.03), (3, 0.015)):
            phase = rng.uniform(0, 2 * np.pi, 2)
            base[:, 0] += load * amp * np.sin(2 * np.pi * h * shaft * t + phase[0])
            base[:, 1] += load * amp * 0.7 * np.sin(2 * np.pi * h * shaft * t + phase[1])
        base += rng.normal(0, 0.02, size=(n, 2))
        if fault:
            bpfo = 3.58 * shaft
            ring = np.exp(-np.arange(int(0.004 * rate)) / (0.0008 * rate))
            ring = ring * np.sin(2 * np.pi * 1200.0 * np.arange(len(ring)) / rate)
            impacts = np.zeros(n)
            for k in np.arange(0, seconds, 1.0 / bpfo):
                idx = int(k * rate + rng.normal(0, 0.0004 * rate))
                if 0 <= idx < n:
                    impacts[idx] = rng.uniform(0.6, 1.0)
            ringing = np.convolve(impacts, ring, mode="same") * load * 0.9
            base[:, 0] += ringing
            base[:, 1] += 0.6 * ringing
        return pd.DataFrame({"time": np.round(t, 6), "accel_x": np.round(base[:, 0], 5),
                             "accel_y": np.round(base[:, 1], 5), "recording": rec_id,
                             "label": "fault" if fault else "normal"})

    for i in range(normal_recordings):
        frames.append(recording(f"normal_{i}", normal_s, False))
    for i in range(fault_recordings):
        frames.append(recording(f"fault_{i}", fault_s, True))
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


def driving(path: Path, seed: int = 11, rate: int = 50, trips: int = 8, trip_s: float = 240.0) -> Path:
    """Three-axis accelerometer in a car: x forward, y left, z up. Events are labelled per sample."""
    rng = np.random.default_rng(seed)
    frames = []
    events = ("hard_brake", "hard_accel", "sharp_turn")
    for trip in range(trips):
        n = int(trip_s * rate)
        t = np.arange(n) / rate
        ax = 0.06 * np.sin(2 * np.pi * rng.uniform(0.02, 0.05) * t) + rng.normal(0, 0.03, n)
        ay = 0.04 * np.sin(2 * np.pi * rng.uniform(0.02, 0.05) * t + 1.0) + rng.normal(0, 0.03, n)
        az = G + rng.normal(0, 0.04, n) + 0.03 * np.sin(2 * np.pi * rng.uniform(8, 14) * t)
        labels = np.array(["normal"] * n, dtype=object)
        cursor = int(rng.uniform(5, 10) * rate)
        while cursor < n - 6 * rate:
            kind = events[rng.integers(0, 3)]
            length = int(rng.uniform(2.6, 4.2) * rate)
            shape = _raised_cosine(length)
            sl = slice(cursor, cursor + length)
            if kind == "hard_brake":
                ax[sl] -= rng.uniform(0.35, 0.6) * shape
                az[sl] += 0.05 * shape
            elif kind == "hard_accel":
                ax[sl] += rng.uniform(0.25, 0.45) * shape
            else:
                ay[sl] += rng.choice([-1, 1]) * rng.uniform(0.3, 0.55) * shape
                ax[sl] -= 0.05 * shape
            core = slice(cursor + length // 8, cursor + length - length // 8)
            labels[core] = kind
            cursor += length + int(rng.uniform(6, 14) * rate)
        frames.append(pd.DataFrame({"timestamp_ms": np.round(t * 1000).astype(int), "ax": np.round(ax, 4),
                                    "ay": np.round(ay, 4), "az": np.round(az, 4), "trip": f"trip_{trip}",
                                    "event": labels}))
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


def activity(path: Path, seed: int = 5, rate: int = 50, people: int = 6, seconds_each: float = 60.0) -> Path:
    """Wrist accelerometer for resting, walking, running and climbing stairs."""
    rng = np.random.default_rng(seed)
    frames = []
    for person in range(people):
        tilt = rng.normal(0, 0.15, 3)
        gravity = np.array([0.1, 0.2, 0.97]) + tilt
        gravity = gravity / np.linalg.norm(gravity)
        parts, labels = [], []
        for activity_name in rng.permutation(["resting", "walking", "running", "stairs"]):
            n = int(seconds_each * rate)
            t = np.arange(n) / rate
            sig = np.tile(gravity, (n, 1)) + rng.normal(0, 0.015, (n, 3))
            if activity_name != "resting":
                freq, amp = {"walking": (rng.uniform(1.6, 2.0), rng.uniform(0.25, 0.4)),
                             "running": (rng.uniform(2.5, 3.0), rng.uniform(0.8, 1.2)),
                             "stairs": (rng.uniform(1.2, 1.5), rng.uniform(0.35, 0.5))}[activity_name]
                step = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(4 * np.pi * freq * t + 0.4)
                sway = np.sin(np.pi * freq * t + rng.uniform(0, np.pi))
                sig[:, 2] += amp * step
                sig[:, 0] += 0.4 * amp * sway
                sig[:, 1] += 0.25 * amp * np.sin(2 * np.pi * freq * t + 1.3)
                if activity_name == "stairs":
                    sig[:, 2] += 0.12 * np.sign(np.sin(2 * np.pi * freq * t)) * amp
                if activity_name == "running":
                    sig += rng.normal(0, 0.08, (n, 3))
            parts.append(sig)
            labels += [activity_name] * n
        data = np.vstack(parts)
        t = np.arange(len(data)) / rate
        frames.append(pd.DataFrame({"t": np.round(t, 4), "acc_x": np.round(data[:, 0], 4),
                                    "acc_y": np.round(data[:, 1], 4), "acc_z": np.round(data[:, 2], 4),
                                    "session": f"person_{person}", "activity": labels}))
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


GENERATORS = {"pump": pump, "driving": driving, "activity": activity}

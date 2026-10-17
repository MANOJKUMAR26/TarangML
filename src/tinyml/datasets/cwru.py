"""Case Western Reserve University bearing data (12 kHz drive-end) to a pump-health CSV.

The CWRU files are MATLAB .mat files. This converts a few normal and fault recordings
into one CSV with columns: time, accel_de, accel_fe, recording, label.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

MIRROR = "https://github.com/XiongMeijing/CWRU-1.git"
RATE = 12000
NORMAL = ["Normal_0", "Normal_1", "Normal_2", "Normal_3"]
FAULTS = ["IR007_0", "B007_0", "OR007@6_0", "IR021_1", "B021_2", "OR021@6_3"]


def fetch(dest: Path) -> Path:
    """Sparse-clone the public GitHub mirror (only the folders we need)."""
    if (dest / "Data" / "Normal").exists():
        return dest
    if shutil.which("git") is None:
        raise RuntimeError("git is required to download the CWRU mirror")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", MIRROR, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", "Data/Normal", "Data/12k_DE"], check=True)
    return dest


def _find(source: Path, name: str) -> Path:
    matches = list(source.rglob(f"{name}.mat"))
    if not matches:
        raise FileNotFoundError(f"{name}.mat not found under {source}")
    return matches[0]


def _signals(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = loadmat(path)
    de = next(v for k, v in data.items() if k.endswith("_DE_time")).ravel()
    fe_keys = [k for k in data if k.endswith("_FE_time")]
    fe = data[fe_keys[0]].ravel() if fe_keys else np.zeros_like(de)
    n = min(len(de), len(fe))
    return de[:n], fe[:n]


def convert(source: Path, out: Path, normal_s: float = 10.0, fault_s: float = 4.0) -> Path:
    frames = []
    for name, seconds, label in [(n, normal_s, "normal") for n in NORMAL] + [(f, fault_s, "fault") for f in FAULTS]:
        de, fe = _signals(_find(source, name))
        n = min(len(de), int(seconds * RATE))
        t = np.arange(n) / RATE
        frames.append(pd.DataFrame({
            "time": np.round(t, 7),
            "accel_de": np.round(de[:n], 5),
            "accel_fe": np.round(fe[:n], 5),
            "recording": name,
            "label": label,
        }))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)
    return out

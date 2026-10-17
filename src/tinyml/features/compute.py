"""Python reference implementation of docs/contracts/features.md (float64 maths on float32 inputs)."""

from __future__ import annotations

import numpy as np

from tinyml.features.spec import NUM_BANDS, TINY, FeatureDef, fft_size

DOMFREQ_TIE = 1e-3  # if the two strongest bins are this close, the strongest frequency is unstable


def channel_data(X: np.ndarray, channel_index: int) -> np.ndarray:
    """X is (windows, samples, axes) float32. Returns (windows, samples) float64."""
    if channel_index >= 0:
        return X[:, :, channel_index].astype(np.float64)
    squares = np.sum(X.astype(np.float64) ** 2, axis=2)
    return np.sqrt(squares)


def _time_features(x: np.ndarray) -> dict[str, np.ndarray]:
    n = x.shape[1]
    mean = x.mean(axis=1)
    d = x - mean[:, None]
    m2 = np.mean(d * d, axis=1)
    m4 = np.mean(d ** 4, axis=1)
    rms = np.sqrt(np.mean(x * x, axis=1))
    peak = np.max(np.abs(x), axis=1)
    p2p = x.max(axis=1) - x.min(axis=1)
    crest = np.where(rms > TINY, peak / np.where(rms > TINY, rms, 1.0), 0.0)
    flat = m2 <= TINY * np.maximum(peak * peak, TINY)
    kurt = np.where(flat, 0.0, m4 / np.where(flat, 1.0, m2 * m2))
    negative = d < 0
    zcr = np.sum(negative[:, 1:] != negative[:, :-1], axis=1) / max(n - 1, 1)
    return {"mean": mean, "std": np.sqrt(m2), "rms": rms, "peak": peak, "p2p": p2p,
            "crest": crest, "kurt": kurt, "zcr": zcr}


def _spectral_features(x: np.ndarray, rate: float) -> tuple[dict[str, np.ndarray], np.ndarray]:
    n = x.shape[1]
    n_fft = fft_size(n)
    half = n_fft // 2
    d = x - x.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(d, n=n_fft, axis=1)
    power = (spectrum.real ** 2 + spectrum.imag ** 2)[:, 1:half + 1]  # bins 1..N/2
    total = power.sum(axis=1)
    safe_total = np.where(total > 0, total, 1.0)
    bin_hz = rate / n_fft
    top = np.argmax(power, axis=1)
    domfreq = (top + 1) * bin_hz
    ordered = np.sort(power, axis=1)
    top1, top2 = ordered[:, -1], ordered[:, -2]
    unstable = (top1 - top2) <= DOMFREQ_TIE * np.maximum(top1, TINY)
    k = np.arange(1, half + 1, dtype=np.float64)
    centroid = np.where(total > 0, (power @ k) / safe_total * bin_hz, 0.0)
    out = {"domfreq": domfreq, "centroid": centroid}
    width = half // NUM_BANDS
    for b in range(NUM_BANDS):
        band = power[:, b * width:(b + 1) * width].sum(axis=1)
        out[f"band{b}"] = np.where(total > 0, band / safe_total, 0.0)
    return out, unstable


def compute_features(
    X: np.ndarray, defs: list[FeatureDef], rate: float, return_unstable: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return (windows, features) float64, and optionally a mask of values that are numerically unstable."""
    out = np.zeros((X.shape[0], len(defs)), dtype=np.float64)
    unstable = np.zeros_like(out, dtype=bool)
    by_channel: dict[int, list[tuple[int, FeatureDef]]] = {}
    for j, f in enumerate(defs):
        by_channel.setdefault(f.channel_index, []).append((j, f))
    for channel_index, items in by_channel.items():
        x = channel_data(X, channel_index)
        values = _time_features(x)
        spectral_unstable = None
        if any(f.spectral for _, f in items):
            spectral, spectral_unstable = _spectral_features(x, rate)
            values.update(spectral)
        for j, f in items:
            out[:, j] = values[f.kind]
            if f.kind == "domfreq" and spectral_unstable is not None:
                unstable[:, j] = spectral_unstable
    if return_unstable:
        return out, unstable
    return out

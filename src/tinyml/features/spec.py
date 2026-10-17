"""The feature spec (docs/contracts/features.md) in code form.

Python (compute.py) and C (codegen templates) are both written from this list.
"""

from __future__ import annotations

from dataclasses import dataclass

TIME_KINDS = ["mean", "std", "rms", "peak", "p2p", "crest", "kurt", "zcr"]
SPECTRAL_KINDS = ["domfreq", "centroid", "band0", "band1", "band2", "band3"]
AMPLITUDE_KINDS = {"mean", "std", "rms", "peak", "p2p"}
NUM_BANDS = 4
TINY = 1e-12


@dataclass(frozen=True)
class FeatureDef:
    name: str
    channel: str  # axis name, or "mag" for the vector magnitude of all axes
    channel_index: int  # index into the window's axes; -1 for "mag"
    kind: str

    @property
    def spectral(self) -> bool:
        return self.kind in SPECTRAL_KINDS

    @property
    def cost_rank(self) -> int:
        """Rough compute cost, used to prefer cheap features when two rules tie."""
        if self.spectral:
            return 3
        if self.kind in ("kurt", "std", "crest", "zcr"):
            return 2
        return 1


def fft_size(window_samples: int) -> int:
    size = 16
    while size < window_samples:
        size *= 2
    return size


def feature_space(axes: list[str], include_magnitude: bool, families: list[str]) -> list[FeatureDef]:
    channels = [(axis, i) for i, axis in enumerate(axes)]
    if include_magnitude and len(axes) >= 2:
        channels.append(("mag", -1))
    kinds = list(TIME_KINDS) + (list(SPECTRAL_KINDS) if "spectral" in families else [])
    return [FeatureDef(f"{kind}_{channel}", channel, index, kind) for channel, index in channels for kind in kinds]


def describe(feature: FeatureDef) -> str:
    """Plain-English name used in explanations."""
    channel = "overall motion" if feature.channel == "mag" else f"'{feature.channel}'"
    words = {
        "mean": "average level", "std": "spread", "rms": "vibration strength (RMS)", "peak": "peak level",
        "p2p": "peak-to-peak swing", "crest": "spikiness (crest factor)", "kurt": "sharp impacts (kurtosis)",
        "zcr": "how often it crosses its average", "domfreq": "strongest frequency", "centroid": "typical frequency",
        "band0": "energy in the lowest frequency band", "band1": "energy in the low-middle band",
        "band2": "energy in the high-middle band", "band3": "energy in the highest band",
    }
    return f"{words[feature.kind]} of {channel}"


def tolerance(feature: FeatureDef, window_samples: int, rate: float, channel_scale: float) -> dict[str, float]:
    """How far the C value may differ from Python's float64 value for this feature."""
    n_fft = fft_size(window_samples)
    if feature.kind in AMPLITUDE_KINDS:
        return {"rtol": 1e-3, "atol": 1e-4 * max(channel_scale, 1e-9)}
    if feature.kind == "zcr":
        return {"rtol": 0.0, "atol": 2.5 / max(window_samples - 1, 1)}
    if feature.kind == "domfreq":
        return {"rtol": 0.0, "atol": rate / n_fft * 1.01}
    if feature.kind == "centroid":
        return {"rtol": 1e-3, "atol": rate / n_fft * 0.01}
    if feature.kind.startswith("band"):
        return {"rtol": 1e-3, "atol": 1e-4}
    if feature.kind == "kurt":
        return {"rtol": 2e-3, "atol": 1e-4}
    return {"rtol": 1e-3, "atol": 1e-5}


FLOAT32_EPS = 5.96e-8


def decision_tolerance(feature: FeatureDef, window_samples: int, parity: dict[str, float]) -> dict[str, float]:
    """Realistic float32-vs-float64 difference, used to flag golden windows whose answer could flip.

    Rounding error of n sequential float32 additions grows roughly with sqrt(n); higher powers and
    FFT stages make it larger. This is far tighter than the parity tolerance, which is a hard limit.
    """
    import math

    growth = math.sqrt(window_samples) * FLOAT32_EPS
    factor = {"kurt": 64.0, "crest": 16.0}.get(feature.kind, 8.0)
    if feature.spectral:
        factor *= math.log2(fft_size(window_samples))
    return {"rtol": factor * growth, "atol": parity["atol"] * 0.01}

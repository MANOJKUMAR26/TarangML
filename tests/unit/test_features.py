import numpy as np

from tinyml.features.compute import compute_features
from tinyml.features.spec import feature_space, fft_size


def _window(signal: np.ndarray) -> np.ndarray:
    return signal.astype(np.float32)[None, :, None]


def test_sine_features():
    rate, n = 1000.0, 1000
    t = np.arange(n) / rate
    x = 2.0 * np.sin(2 * np.pi * 50 * t)
    defs = feature_space(["a"], include_magnitude=False, families=["time", "spectral"])
    f = dict(zip([d.kind for d in defs], compute_features(_window(x), defs, rate)[0], strict=True))
    assert abs(f["rms"] - 2.0 / np.sqrt(2)) < 1e-4
    assert abs(f["mean"]) < 1e-6
    assert abs(f["peak"] - 2.0) < 1e-3
    assert abs(f["crest"] - np.sqrt(2)) < 1e-3
    assert abs(f["kurt"] - 1.5) < 1e-3  # a sine wave has kurtosis 1.5
    assert abs(f["domfreq"] - 50.0) <= rate / fft_size(n)
    assert abs(f["zcr"] - 100 / (n - 1)) < 2 / (n - 1)
    assert abs(sum(f[f"band{b}"] for b in range(4)) - 1.0) < 1e-9


def test_constant_signal_is_safe():
    defs = feature_space(["a"], include_magnitude=False, families=["time", "spectral"])
    f = compute_features(_window(np.full(64, 3.0)), defs, 100.0)[0]
    assert np.all(np.isfinite(f))
    by_kind = dict(zip([d.kind for d in defs], f, strict=True))
    assert by_kind["kurt"] == 0.0 and by_kind["std"] == 0.0 and by_kind["centroid"] == 0.0


def test_magnitude_channel():
    defs = feature_space(["x", "y"], include_magnitude=True, families=["time"])
    X = np.zeros((1, 10, 2), dtype=np.float32)
    X[0, :, 0] = 3.0
    X[0, :, 1] = 4.0
    f = dict(zip([d.name for d in defs], compute_features(X, defs, 10.0)[0], strict=True))
    assert abs(f["mean_mag"] - 5.0) < 1e-9


def test_domfreq_near_tie_is_flagged():
    n, rate = 64, 64.0
    t = np.arange(n) / rate
    x = np.sin(2 * np.pi * 8 * t) + np.sin(2 * np.pi * 16 * t)
    defs = [d for d in feature_space(["a"], False, ["time", "spectral"]) if d.kind == "domfreq"]
    _, unstable = compute_features(_window(x), defs, rate, return_unstable=True)
    assert unstable[0, 0]

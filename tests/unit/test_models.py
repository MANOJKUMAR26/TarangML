import numpy as np
import pytest

from tinyml.ml.models import (
    QuantizedMLP,
    ThresholdRule,
    average_path_length,
    f32_floor,
    quantize_multiplier,
    requantize,
)


def test_f32_floor_preserves_comparisons():
    rng = np.random.default_rng(0)
    for t in rng.normal(0, 10, 200):
        thr = f32_floor(float(t))
        x = rng.normal(t, 1e-5, 500).astype(np.float32)
        assert np.array_equal(x <= t, x <= np.float32(thr))


def test_quantize_multiplier_and_requantize():
    for real in (0.0003, 0.07, 0.5, 0.999):
        m0, shift = quantize_multiplier(real)
        assert (1 << 30) <= m0 < (1 << 31)
        acc = np.arange(-5000, 5000, 37)
        got = requantize(acc, m0, shift)
        assert np.max(np.abs(got - np.floor(acc * real + 0.5))) <= 1


def test_average_path_length_matches_definition():
    assert average_path_length(1) == 0.0
    assert average_path_length(2) == 1.0
    assert average_path_length(256) == pytest.approx(10.2448, abs=1e-3)


def test_threshold_rule_and_borderline():
    rule = ThresholdRule([0], [ThresholdRule.make_condition(0, "gt", mean=1.0, std=0.1, k=3.0)])
    F = np.array([[1.0], [1.5], [1.3]])
    pred = rule.predict(F)
    assert pred.labels.tolist() == [0, 1, 0] or pred.labels.tolist() == [0, 1, 1]
    assert pred.labels[1] == 1


def test_quantized_inputs_round_half_away_from_zero():
    q = {"in_mean": [0.0], "in_mul": [1.0], "w1": [[1]], "b1": [0], "m1": 1 << 30, "shift1": 30,
         "w2": [[1]], "b2": [0], "score_scale": 1.0}
    model = QuantizedMLP("classifier", ["a", "b"], [0], q)
    q_in, _ = model.quantize_inputs(np.array([[0.5], [-0.5], [1.49], [200.0], [-200.0]]), None)
    assert q_in.ravel().tolist() == [1, -1, 1, 127, -127]

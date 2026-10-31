"""Deployable models and their Python reference predictions.

Each model's `predict` mirrors exactly what the generated C code does: the same
float32 thresholds, the same comparison operators, the same integer maths for
8-bit networks. That is what makes the golden-vector parity test meaningful.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

EULER_GAMMA = 0.5772156649015329


@dataclass
class Prediction:
    labels: np.ndarray  # int
    scores: np.ndarray  # float64
    borderline: np.ndarray  # bool: a legitimate float difference could flip this answer


@dataclass
class DecisionTolerance:
    """How much a C feature value may plausibly differ from Python's, per selected feature."""

    rtol: np.ndarray
    atol: np.ndarray
    unstable: np.ndarray | None = None  # (windows, features) bool for discrete features

    def near(self, values: np.ndarray, thresholds: np.ndarray | float, j: int | np.ndarray) -> np.ndarray:
        return np.abs(values - thresholds) <= self.atol[j] + self.rtol[j] * np.abs(thresholds)

    def unstable_any(self, rows: int, cols: list[int]) -> np.ndarray:
        if self.unstable is None or not cols:
            return np.zeros(rows, dtype=bool)
        return self.unstable[:, cols].any(axis=1)


def f32_floor(value: float) -> float:
    """Largest float32 <= value, so float32 inputs compare exactly as they would against the float64 value."""
    v = np.float32(value)
    if float(v) > value:
        v = np.nextafter(v, np.float32(-np.inf), dtype=np.float32)
    return float(v)


def f32(value: float) -> float:
    return float(np.float32(value))


class Model:
    algorithm = "base"
    rung = 0
    quantization = "float32"

    def __init__(self, classes: list[str], feature_idx: list[int]):
        self.classes = classes
        self.feature_idx = feature_idx

    def predict(self, F: np.ndarray, tol: DecisionTolerance | None = None) -> Prediction:
        raise NotImplementedError

    def params(self) -> dict[str, Any]:
        raise NotImplementedError

    def size_bytes(self) -> int:
        return len(json.dumps(self.params()))

    def explain(self, features: list[Any], passed: bool = True) -> str:
        return ""


# ---------------------------------------------------------------------------- rung 1: rules
class ThresholdRule(Model):
    """Anomaly when any condition fires. Each condition: feature above (or below) mean + k * std."""

    algorithm = "threshold_rule"
    rung = 1

    def __init__(self, feature_idx: list[int], conditions: list[dict[str, Any]]):
        super().__init__(["normal", "anomaly"], feature_idx)
        self.conditions = conditions  # feature is the position inside feature_idx

    @staticmethod
    def make_condition(position: int, op: str, mean: float, std: float, k: float) -> dict[str, Any]:
        raw = mean + k * std if op == "gt" else mean - k * std
        return {"feature": position, "op": op, "threshold": f32_floor(raw) if op == "gt" else -f32_floor(-raw),
                "mean": f32(mean), "std": f32(std), "k": f32(k)}

    def predict(self, F: np.ndarray, tol: DecisionTolerance | None = None) -> Prediction:
        n = F.shape[0]
        fired = np.zeros(n, dtype=bool)
        score = np.full(n, -np.inf)
        border = np.zeros(n, dtype=bool)
        for c in self.conditions:
            x = F[:, c["feature"]]
            if c["op"] == "gt":
                fired |= x > c["threshold"]
                z = (x - c["mean"]) / c["std"]
            else:
                fired |= x < c["threshold"]
                z = (c["mean"] - x) / c["std"]
            score = np.maximum(score, z)
            if tol is not None:
                border |= tol.near(x, c["threshold"], c["feature"])
        if tol is not None:
            border |= tol.unstable_any(n, [c["feature"] for c in self.conditions])
        return Prediction(fired.astype(int), score, border)

    def params(self) -> dict[str, Any]:
        return {"conditions": self.conditions, "combine": "any"}

    def size_bytes(self) -> int:
        return 24 * len(self.conditions)

    def explain(self, features: list[Any], passed: bool = True) -> str:
        from tinyml.features.spec import describe

        parts = []
        for c in self.conditions:
            fdef = features[c["feature"]]
            word = "above" if c["op"] == "gt" else "below"
            ratio = c["threshold"] / c["mean"] if c["mean"] not in (0.0,) else None
            times = f" ({ratio:.1f}x normal)" if ratio and ratio > 0 and c["op"] == "gt" else ""
            parts.append(f"{describe(fdef)} goes {word} {c['threshold']:.4g}{times}")
        lead = "Rules were enough" if passed else "Rules, below the target"
        return f"{lead}: an alert fires when " + " or ".join(parts) + "."


# ---------------------------------------------------------------------------- trees (rungs 1 and 2)
class TreeEnsemble(Model):
    """One flat node table for decision trees, random forests, XGBoost and isolation forests.

    Internal node: feature >= 0, go left when x <= threshold ("le", scikit-learn) or x < threshold ("lt", XGBoost).
    Leaf: feature == -1 and right holds the row in the leaf table.
    """

    def __init__(
        self,
        algorithm: str,
        rung: int,
        classes: list[str],
        feature_idx: list[int],
        op: str,
        roots: list[int],
        feature: list[int],
        threshold: list[float],
        left: list[int],
        right: list[int],
        leaves: np.ndarray,
        aggregate: str,
        base: list[float] | None = None,
        extra: dict[str, Any] | None = None,
    ):
        super().__init__(classes, feature_idx)
        self.algorithm = algorithm
        self.rung = rung
        self.op = op
        self.roots = roots
        self.feature = np.array(feature, dtype=np.int32)
        self.threshold = np.array(threshold, dtype=np.float64)
        self.left = np.array(left, dtype=np.int32)
        self.right = np.array(right, dtype=np.int32)
        self.leaves = np.asarray(leaves, dtype=np.float32)
        self.aggregate = aggregate  # proba | margin | depth
        self.base = base or [0.0] * self.leaves.shape[1]
        self.extra = extra or {}

    def _walk(self, F: np.ndarray, tol: DecisionTolerance | None) -> tuple[np.ndarray, np.ndarray]:
        n = F.shape[0]
        out = np.zeros((n, self.leaves.shape[1]), dtype=np.float64)
        border = np.zeros(n, dtype=bool)
        used: set[int] = set()
        for root in self.roots:
            node = np.full(n, root, dtype=np.int64)
            active = self.feature[node] >= 0
            while active.any():
                idx = np.where(active)[0]
                cur = node[idx]
                feats = self.feature[cur]
                used.update(int(f) for f in np.unique(feats))
                x = F[idx, feats]
                thr = self.threshold[cur]
                go_left = x <= thr if self.op == "le" else x < thr
                if tol is not None:
                    border[idx] |= tol.near(x, thr, feats)
                node[idx] = np.where(go_left, self.left[cur], self.right[cur])
                active = self.feature[node] >= 0
            out += self.leaves[self.right[node]].astype(np.float64)
        if tol is not None:
            border |= tol.unstable_any(n, sorted(used))
        return out, border

    def predict(self, F: np.ndarray, tol: DecisionTolerance | None = None) -> Prediction:
        sums, border = self._walk(F, tol)
        n_trees = len(self.roots)
        if self.aggregate == "depth":
            avg = sums[:, 0] / n_trees
            labels = (avg < self.extra["depth_threshold"]).astype(int)
            scores = 1.0 - avg / self.extra["c_norm"]
            border |= np.abs(avg - self.extra["depth_threshold"]) <= 1e-5 * max(1.0, abs(self.extra["depth_threshold"]))
            return Prediction(labels, scores, border)
        if self.aggregate == "margin":
            sums = sums + np.asarray(self.base)[None, :]
        if sums.shape[1] == 1:
            labels = (sums[:, 0] > 0).astype(int)
            scores = sums[:, 0]
            border |= np.abs(sums[:, 0]) <= 1e-5 * (1.0 + np.abs(sums[:, 0]))
            return Prediction(labels, scores, border)
        order = np.sort(sums, axis=1)
        labels = np.argmax(sums, axis=1)
        top, second = order[:, -1], order[:, -2]
        border |= (top - second) <= 1e-5 * (1.0 + np.abs(top))
        scores = top / n_trees if self.aggregate == "proba" else top - second
        return Prediction(labels, scores, border)

    def params(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "aggregate": self.aggregate,
            "roots": self.roots,
            "feature": self.feature.tolist(),
            "threshold": [float(t) for t in self.threshold],
            "left": self.left.tolist(),
            "right": self.right.tolist(),
            "leaves": self.leaves.tolist(),
            "base": [float(b) for b in self.base],
            **self.extra,
        }

    @property
    def n_nodes(self) -> int:
        return len(self.feature)

    def size_bytes(self) -> int:
        index_bytes = 2 if self.n_nodes < 32768 else 4
        return self.n_nodes * (2 + 4 + 2 * index_bytes) + self.leaves.size * 4 + len(self.roots) * index_bytes

    def explain(self, features: list[Any], passed: bool = True) -> str:
        from tinyml.features.spec import describe

        if self.algorithm == "decision_tree":
            return ("Rules were enough: " if passed else "Rules, below the target: ") + self._rules_text(features)
        used = sorted({int(f) for f in self.feature if f >= 0})
        top = ", ".join(describe(features[i]) for i in used[:4])
        label = {
            "random_forest": f"A random forest of {len(self.roots)} small trees",
            "gradient_boosting": f"A gradient-boosted model of {len(self.roots)} small trees",
            "isolation_forest": f"An isolation forest of {len(self.roots)} trees that learned normal behaviour",
        }.get(self.algorithm, self.algorithm)
        verdict = "It reached the target." if passed else "It is below the target."
        return f"{label}, looking mostly at {top}. {verdict}"

    def _rules_text(self, features: list[Any]) -> str:
        from tinyml.features.spec import describe

        def leaf_class(node: int) -> int:
            return int(np.argmax(self.leaves[self.right[node]]))

        def pure(node: int) -> int | None:
            """The single class every leaf under this node predicts, or None."""
            if self.feature[node] < 0:
                return leaf_class(node)
            a, b = pure(int(self.left[node])), pure(int(self.right[node]))
            return a if a is not None and a == b else None

        lines: list[str] = []

        def visit(node: int, conds: list[str]) -> None:
            cls = pure(node)
            if cls is not None:
                lines.append(("if " + " and ".join(conds) if conds else "always") + f" then '{self.classes[cls]}'")
                return
            name = describe(features[self.feature[node]])
            thr = self.threshold[node]
            visit(int(self.left[node]), conds + [f"{name} <= {thr:.4g}"])
            visit(int(self.right[node]), conds + [f"{name} > {thr:.4g}"])

        visit(self.roots[0], [])
        return "; ".join(lines) + "."


def sklearn_tree_tables(
    trees: list[Any], feature_maps: list[list[int]] | None, value_fn
) -> tuple[list[int], list[int], list[float], list[int], list[int], list[list[float]]]:
    """Flatten scikit-learn tree_ objects into one table. value_fn(tree, node, depth) gives a leaf row."""
    roots, feature, threshold, left, right, leaves = [], [], [], [], [], []
    for t_i, est in enumerate(trees):
        tree = est.tree_
        offset = len(feature)
        roots.append(offset)
        depth = np.zeros(tree.node_count, dtype=int)
        for node in range(tree.node_count):
            for child in (tree.children_left[node], tree.children_right[node]):
                if child >= 0:
                    depth[child] = depth[node] + 1
        fmap = feature_maps[t_i] if feature_maps is not None else None
        for node in range(tree.node_count):
            if tree.children_left[node] < 0:
                feature.append(-1)
                threshold.append(0.0)
                left.append(-1)
                right.append(len(leaves))
                leaves.append(value_fn(tree, node, int(depth[node])))
            else:
                f = int(tree.feature[node])
                feature.append(int(fmap[f]) if fmap is not None else f)
                threshold.append(f32_floor(float(tree.threshold[node])))
                left.append(offset + int(tree.children_left[node]))
                right.append(offset + int(tree.children_right[node]))
    return roots, feature, threshold, left, right, leaves


def average_path_length(n: float) -> float:
    if n <= 1:
        return 0.0
    if n <= 2:
        return 1.0
    return 2.0 * (math.log(n - 1.0) + EULER_GAMMA) - 2.0 * (n - 1.0) / n


# ---------------------------------------------------------------------------- rung 3: 8-bit networks
def quantize_multiplier(real: float) -> tuple[int, int]:
    """real ~= m0 * 2**-shift with m0 in [2**30, 2**31)."""
    if real <= 0:
        raise ValueError("multiplier must be positive")
    shift = 30 - math.floor(math.log2(real))
    m0 = int(round(real * (1 << shift)))
    if m0 >= 1 << 31:
        m0 //= 2
        shift -= 1
    return m0, shift


def requantize(acc: np.ndarray, m0: int, shift: int) -> np.ndarray:
    acc64 = acc.astype(np.int64)
    return (acc64 * m0 + (1 << (shift - 1))) >> shift


class QuantizedMLP(Model):
    """One hidden ReLU layer, int8 weights, int32 accumulators, fixed-point requantization.

    kind="classifier": outputs are class logits (argmax, or > 0 for a single output).
    kind="autoencoder": outputs reconstruct the input; anomaly when the squared error exceeds a threshold.
    """

    rung = 3
    quantization = "int8"

    def __init__(self, kind: str, classes: list[str], feature_idx: list[int], q: dict[str, Any]):
        super().__init__(classes, feature_idx)
        self.kind = kind
        self.algorithm = "mlp_int8" if kind == "classifier" else "autoencoder_int8"
        self.q = q

    def quantize_inputs(self, F: np.ndarray, tol: DecisionTolerance | None) -> tuple[np.ndarray, np.ndarray]:
        mean = np.asarray(self.q["in_mean"], dtype=np.float64)
        mul = np.asarray(self.q["in_mul"], dtype=np.float64)
        v = (F - mean[None, :]) * mul[None, :]
        rounded = np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5))
        q_in = np.clip(rounded, -127, 127).astype(np.int64)
        border = np.zeros(F.shape[0], dtype=bool)
        if tol is not None:
            frac = np.abs(v - np.trunc(v))
            slack = (tol.atol[None, :] + tol.rtol[None, :] * np.abs(F)) * np.abs(mul)[None, :] + 1e-6
            inside = np.abs(v) < 127.5
            border = (inside & (np.abs(frac - 0.5) <= slack)).any(axis=1)
            border |= tol.unstable_any(F.shape[0], list(range(F.shape[1])))
        return q_in, border

    def forward(self, q_in: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w1 = np.asarray(self.q["w1"], dtype=np.int64)
        b1 = np.asarray(self.q["b1"], dtype=np.int64)
        acc1 = q_in @ w1.T + b1[None, :]
        acc1 = np.maximum(acc1, 0)
        q_h = np.clip(requantize(acc1, self.q["m1"], self.q["shift1"]), 0, 127)
        w2 = np.asarray(self.q["w2"], dtype=np.int64)
        b2 = np.asarray(self.q["b2"], dtype=np.int64)
        out = q_h @ w2.T + b2[None, :]
        return q_h, out

    def predict(self, F: np.ndarray, tol: DecisionTolerance | None = None) -> Prediction:
        q_in, border = self.quantize_inputs(F, tol)
        _, out = self.forward(q_in)
        if self.kind == "autoencoder":
            recon = np.clip(requantize(out, self.q["m2"], self.q["shift2"]), -32768, 32767)
            err = np.sum((recon - q_in) ** 2, axis=1)
            labels = (err > self.q["err_threshold"]).astype(int)
            return Prediction(labels, err.astype(np.float64), border)
        if out.shape[1] == 1:
            labels = (out[:, 0] > 0).astype(int)
            return Prediction(labels, out[:, 0] * self.q["score_scale"], border)
        order = np.sort(out, axis=1)
        labels = np.argmax(out, axis=1)
        return Prediction(labels, (order[:, -1] - order[:, -2]) * self.q["score_scale"], border)

    def params(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.q}

    def size_bytes(self) -> int:
        w1 = np.asarray(self.q["w1"])
        w2 = np.asarray(self.q["w2"])
        return int(w1.size + w2.size + 4 * (w1.shape[0] + w2.shape[0]) + 8 * w1.shape[1] + 32)

    def explain(self, features: list[Any], passed: bool = True) -> str:
        hidden = len(self.q["b1"])
        verdict = "" if passed else " It is below the target."
        if self.kind == "autoencoder":
            return (f"A tiny 8-bit network ({hidden} hidden units) learned what normal looks like and raises an "
                    f"alert when a window does not fit that pattern.{verdict}")
        if passed:
            return f"A tiny 8-bit neural network with {hidden} hidden units reached the target."
        return f"A tiny 8-bit neural network with {hidden} hidden units.{verdict}"

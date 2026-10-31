"""Step 4: try every rung of the algorithm ladder and recommend the simplest option that meets the target.

Rung 1: rules (thresholds for anomalies, a depth-2/3 decision tree for events).
Rung 2: classic ML (isolation forest; random forest, then XGBoost), smallest size first.
Rung 3: a tiny 8-bit neural network (autoencoder for anomalies, MLP for events).
Candidates over the memory budget are dropped before they are scored. Every rung is tried so the
user can compare them and pick another one than the recommendation.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tinyml.features.spec import FeatureDef, describe
from tinyml.ml.models import (
    DecisionTolerance,
    Model,
    QuantizedMLP,
    ThresholdRule,
    TreeEnsemble,
    average_path_length,
    f32,
    quantize_multiplier,
    requantize,
    sklearn_tree_tables,
)
from tinyml.ml.split import Split
from tinyml.recipes import Recipe

RUNG_NAMES = {1: "rules", 2: "classic ML", 3: "tiny neural network"}
MAX_SELECTED_FEATURES = 12


@dataclass
class Candidate:
    rung: int
    algorithm: str
    model: Model | None
    metrics: dict[str, Any]
    passed: bool
    note: str
    size_bytes: int

    def summary(self) -> dict[str, Any]:
        return {"rung": self.rung, "algorithm": self.algorithm, "metrics": self.metrics, "passed": self.passed,
                "note": self.note, "size_bytes": self.size_bytes}


@dataclass
class LadderResult:
    candidates: list[Candidate]
    chosen: Candidate | None  # the recommendation: the simplest candidate that meets the target
    classes: list[str]
    data_request: str | None = None
    explanation: str = ""
    best: Candidate | None = None  # the strongest candidate, whether or not it meets the target
    extras: dict[str, Any] = field(default_factory=dict)


class LadderInputs:
    def __init__(self, F: np.ndarray, defs: list[FeatureDef], labels: np.ndarray, split: Split, task: str,
                 normal_label: str | None, recipe: Recipe, seed: int, tol_rtol: np.ndarray, tol_atol: np.ndarray,
                 unstable: np.ndarray):
        self.F, self.defs, self.labels, self.split = F, defs, labels, split
        self.task, self.normal_label, self.recipe, self.seed = task, normal_label, recipe, seed
        self.tol_rtol, self.tol_atol, self.unstable = tol_rtol, tol_atol, unstable
        if task == "anomaly":
            self.classes = ["normal", "anomaly"]
            is_normal = labels == normal_label if normal_label is not None else np.ones(len(labels), dtype=bool)
            self.y = (~is_normal).astype(int)
        else:
            self.classes = sorted(set(labels.tolist()))
            index = {c: i for i, c in enumerate(self.classes)}
            self.y = np.array([index[c] for c in labels], dtype=int)
        self.budget = recipe.model_budget_kb * 1024
        self.target = recipe.target(task)

    def tol(self, idx: list[int], rows: np.ndarray | None = None) -> DecisionTolerance:
        unstable = self.unstable[:, idx] if rows is None else self.unstable[rows][:, idx]
        return DecisionTolerance(self.tol_rtol[idx], self.tol_atol[idx], unstable)


# ---------------------------------------------------------------------------- scoring
def evaluate(model: Model, inp: LadderInputs) -> dict[str, Any]:
    test = inp.split.test
    pred = model.predict(inp.F[test][:, model.feature_idx]).labels
    y = inp.y[test]
    if inp.task == "anomaly":
        normal = y == 0
        far = float(np.mean(pred[normal] == 1)) if normal.any() else None
        detection = float(np.mean(pred[~normal] == 1)) if (~normal).any() else None
        return {"false_alarm_rate": far, "detection_rate": detection, "test_normal_windows": int(normal.sum()),
                "test_fault_windows": int((~normal).sum())}
    k = len(inp.classes)
    confusion = np.zeros((k, k), dtype=int)
    for t, p in zip(y, pred, strict=True):
        confusion[t, p] += 1
    recall = {inp.classes[i]: float(confusion[i, i] / confusion[i].sum()) if confusion[i].sum() else None
              for i in range(k)}
    present = [v for v in recall.values() if v is not None]
    return {"accuracy": float(np.mean(pred == y)) if len(y) else None,
            "balanced_accuracy": float(np.mean(present)) if present else None, "recall": recall,
            "confusion": confusion.tolist(), "test_windows": int(len(y))}


def meets_target(metrics: dict[str, Any], inp: LadderInputs) -> bool:
    if inp.task == "anomaly":
        far = metrics["false_alarm_rate"]
        det = metrics["detection_rate"]
        if far is None or far > inp.target["false_alarm_rate"]:
            return False
        return det is None or det >= inp.target["detection_rate"]
    acc, balanced = metrics["accuracy"], metrics["balanced_accuracy"]
    return acc is not None and balanced is not None and min(acc, balanced) >= inp.target["accuracy"]


def _candidate(model: Model, inp: LadderInputs, note: str = "") -> Candidate:
    size = model.size_bytes()
    if size > inp.budget:
        return Candidate(model.rung, model.algorithm, None, {}, False,
                         f"dropped: needs {size / 1024:.1f} KB, budget is {inp.budget / 1024:.0f} KB", size)
    metrics = evaluate(model, inp)
    return Candidate(model.rung, model.algorithm, model, metrics, meets_target(metrics, inp), note, size)


# ---------------------------------------------------------------------------- rung 1
def _rule_conditions(inp: LadderInputs) -> list[Candidate]:
    train = inp.split.train & (inp.y == 0)
    F_train = inp.F[train]
    far_target = inp.target["false_alarm_rate"]
    singles: list[tuple[int, str, float, float, float]] = []
    for j in range(F_train.shape[1]):
        x = F_train[:, j]
        mu, sigma = float(x.mean()), float(x.std())
        if not np.isfinite(sigma) or sigma <= 1e-12 * max(1.0, abs(mu)):
            continue
        for op in ("gt", "lt"):
            z = (x - mu) / sigma if op == "gt" else (mu - x) / sigma
            k = max(3.0, float(np.quantile(z, 1 - far_target / 2)) + 0.5)
            singles.append((j, op, mu, sigma, k))

    has_faults = bool((inp.y[inp.split.test] == 1).any())
    candidates: list[Candidate] = []
    scored: list[tuple[float, Candidate, tuple]] = []
    for j, op, mu, sigma, k in singles:
        model = ThresholdRule([j], [ThresholdRule.make_condition(0, op, mu, sigma, k)])
        cand = _candidate(model, inp)
        det = cand.metrics.get("detection_rate") or 0.0
        far = cand.metrics.get("false_alarm_rate") or 0.0
        scored.append((det - 2 * far - 0.01 * inp.defs[j].cost_rank, cand, (j, op, mu, sigma, k)))
    if not scored:
        return []
    scored.sort(key=lambda item: item[0], reverse=True)

    if not has_faults:
        preferred = [s for s in scored if inp.defs[s[2][0]].kind in ("rms", "kurt") and s[2][1] == "gt"
                     and inp.defs[s[2][0]].channel_index in (0, -1)]
        picks = preferred[:2] if len(preferred) >= 2 else scored[:2]
        model = _combine([p[2] for p in picks])
        cand = _candidate(model, inp, "no fault data: watches vibration strength and sharp impacts")
        return [cand]

    passing = [s for s in scored if s[1].passed]
    if passing:
        candidates.append(passing[0][1])
        return candidates
    candidates.append(scored[0][1])
    top = [s for s in scored[:12] if (s[1].metrics.get("false_alarm_rate") or 0) <= far_target]
    best_pair: Candidate | None = None
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            if top[a][2][0] == top[b][2][0]:
                continue
            cand = _candidate(_combine([top[a][2], top[b][2]]), inp, "two rules combined")
            if cand.passed and (best_pair is None or
                                (cand.metrics["detection_rate"] or 0) > (best_pair.metrics["detection_rate"] or 0)):
                best_pair = cand
    if best_pair is not None:
        candidates.append(best_pair)
    return candidates


def _combine(specs: list[tuple[int, str, float, float, float]]) -> ThresholdRule:
    idx = []
    conditions = []
    for j, op, mu, sigma, k in specs:
        if j not in idx:
            idx.append(j)
        conditions.append(ThresholdRule.make_condition(idx.index(j), op, mu, sigma, k))
    return ThresholdRule(idx, conditions)


def _decision_tree(inp: LadderInputs, depth: int) -> TreeEnsemble:
    from sklearn.tree import DecisionTreeClassifier

    train = inp.split.train
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=max(3, int(0.01 * train.sum())),
                                 random_state=inp.seed)
    clf.fit(inp.F[train].astype(np.float32), inp.y[train])
    model = _sk_classifier_tables("decision_tree", 1, [clf], clf.classes_, list(range(inp.F.shape[1])), inp)
    return _compact(model)


# ---------------------------------------------------------------------------- rung 2
def _select_features(inp: LadderInputs) -> list[int]:
    train = inp.split.train
    F = inp.F[train]
    std = F.std(axis=0)
    usable = [j for j in range(F.shape[1]) if np.isfinite(std[j]) and std[j] > 1e-12]
    if inp.task == "classification":
        from sklearn.feature_selection import f_classif

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores, _ = f_classif(F[:, usable], inp.y[train])
        scores = np.nan_to_num(scores, nan=0.0)
        order = [usable[i] for i in np.argsort(-scores)]
    else:
        order = sorted(usable, key=lambda j: (inp.defs[j].cost_rank, j))
    chosen: list[int] = []
    for j in order:
        if len(chosen) >= MAX_SELECTED_FEATURES:
            break
        if any(abs(np.corrcoef(F[:, j], F[:, c])[0, 1]) > 0.97 for c in chosen):
            continue
        chosen.append(j)
    return sorted(chosen)


def _sk_classifier_tables(algorithm: str, rung: int, estimators: list[Any], classes: np.ndarray, feature_idx: list[int],
                          inp: LadderInputs) -> TreeEnsemble:
    k = len(inp.classes)

    def leaf(tree, node, depth):
        row = np.zeros(k, dtype=np.float64)
        values = tree.value[node][0]
        total = values.sum()
        for ci, cls in enumerate(classes):
            row[int(cls)] = values[ci] / total if total else 0.0
        return row.tolist()

    roots, feature, threshold, left, right, leaves = sklearn_tree_tables(estimators, None, leaf)
    return TreeEnsemble(algorithm, rung, inp.classes, feature_idx, "le", roots, feature, threshold, left, right,
                        np.array(leaves), "proba")


def _compact(model: TreeEnsemble) -> TreeEnsemble:
    used = sorted({int(f) for f in model.feature if f >= 0})
    remap = {old: new for new, old in enumerate(used)}
    model.feature = np.array([remap[int(f)] if f >= 0 else -1 for f in model.feature], dtype=np.int32)
    model.feature_idx = [model.feature_idx[u] for u in used]
    return model


def _random_forest(inp: LadderInputs, idx: list[int], n: int, depth: int) -> TreeEnsemble:
    from sklearn.ensemble import RandomForestClassifier

    train = inp.split.train
    clf = RandomForestClassifier(n_estimators=n, max_depth=depth, min_samples_leaf=2, random_state=inp.seed, n_jobs=1)
    clf.fit(inp.F[train][:, idx].astype(np.float32), inp.y[train])
    return _compact(_sk_classifier_tables("random_forest", 2, clf.estimators_, clf.classes_, idx, inp))


def _xgboost(inp: LadderInputs, idx: list[int], n: int, depth: int) -> TreeEnsemble:
    import xgboost as xgb

    train = inp.split.train
    X = inp.F[train][:, idx].astype(np.float32)
    y = inp.y[train]
    k = len(inp.classes)
    params: dict[str, Any] = {"max_depth": depth, "eta": 0.3, "seed": inp.seed, "nthread": 1, "tree_method": "exact"}
    if k == 2:
        params["objective"] = "binary:logistic"
    else:
        params.update(objective="multi:softprob", num_class=k)
    booster = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=n)
    raw = json.loads(booster.save_raw("json"))
    model_json = raw["learner"]["gradient_booster"]["model"]
    out_dim = 1 if k == 2 else k
    roots, feature, threshold, left, right, leaves = [], [], [], [], [], []
    for t_i, tree in enumerate(model_json["trees"]):
        cls = int(model_json["tree_info"][t_i]) if out_dim > 1 else 0
        offset = len(feature)
        roots.append(offset)
        lc, rc = tree["left_children"], tree["right_children"]
        for node in range(len(lc)):
            if lc[node] == -1:
                row = [0.0] * out_dim
                row[cls] = float(np.float32(tree["split_conditions"][node]))
                feature.append(-1)
                threshold.append(0.0)
                left.append(-1)
                right.append(len(leaves))
                leaves.append(row)
            else:
                feature.append(int(tree["split_indices"][node]))
                threshold.append(float(np.float32(tree["split_conditions"][node])))
                left.append(offset + lc[node])
                right.append(offset + rc[node])
    model = TreeEnsemble("gradient_boosting", 2, inp.classes, idx, "lt", roots, feature, threshold, left, right,
                         np.array(leaves), "margin", base=[0.0] * out_dim)
    margins = booster.predict(xgb.DMatrix(X), output_margin=True).reshape(len(X), -1)
    sums, _ = model._walk(X.astype(np.float64), None)
    base = np.median(margins - sums, axis=0)
    if np.max(np.abs(margins - sums - base[None, :])) > 1e-3:
        raise RuntimeError("Could not reproduce XGBoost margins from its tree dump")
    model.base = [f32(b) for b in base]
    return _compact(model)


def _isolation_forest(inp: LadderInputs, idx: list[int], n: int) -> TreeEnsemble:
    from sklearn.ensemble import IsolationForest

    train = inp.split.train & (inp.y == 0)
    X = inp.F[train][:, idx].astype(np.float32)
    forest = IsolationForest(n_estimators=n, max_samples=min(256, len(X)), random_state=inp.seed)
    forest.fit(X)

    def leaf(tree, node, depth):
        return [depth + average_path_length(float(tree.n_node_samples[node]))]

    maps = [list(map(int, f)) for f in forest.estimators_features_]
    roots, feature, threshold, left, right, leaves = sklearn_tree_tables(forest.estimators_, maps, leaf)
    c_norm = average_path_length(float(forest.max_samples_))
    model = TreeEnsemble("isolation_forest", 2, inp.classes, idx, "le", roots, feature, threshold, left, right,
                         np.array(leaves), "depth", extra={"depth_threshold": 0.0, "c_norm": f32(c_norm)})
    sums, _ = model._walk(X.astype(np.float64), None)
    avg = sums[:, 0] / n
    model.extra["depth_threshold"] = f32(float(np.quantile(avg, inp.target["false_alarm_rate"] / 2)))
    model.extra["train_mean_depth"] = f32(float(avg.mean()))
    return _compact(model)


# ---------------------------------------------------------------------------- rung 3
def _standardize(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = F.mean(axis=0)
    scale = F.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    return mean, scale


def _quantize(coefs: list[np.ndarray], intercepts: list[np.ndarray], mean: np.ndarray, scale: np.ndarray,
              Z_train: np.ndarray) -> dict[str, Any]:
    s_in = max(float(np.percentile(np.abs(Z_train), 99.9)) / 127.0, 1e-6)
    in_mean = [f32(m) for m in mean]
    in_mul = [f32(1.0 / (s * s_in)) for s in scale]
    w1 = coefs[0].T
    s_w1 = max(float(np.abs(w1).max()) / 127.0, 1e-12)
    w1q = np.clip(np.round(w1 / s_w1), -127, 127).astype(int)
    b1q = np.round(intercepts[0] / (s_in * s_w1)).astype(np.int64)
    q = {"in_mean": in_mean, "in_mul": in_mul, "w1": w1q.tolist(), "b1": b1q.tolist()}
    probe = QuantizedMLP("classifier", [], [], {**q, "m1": 1 << 30, "shift1": 30, "w2": [[0]], "b2": [0]})
    F_train = Z_train * scale + mean
    q_in, _ = probe.quantize_inputs(F_train, None)
    acc1 = np.maximum(q_in @ w1q.T.astype(np.int64) + b1q[None, :], 0)
    real = acc1 * (s_in * s_w1)
    s_h = max(float(np.percentile(real, 99.9)) / 127.0, 1e-9)
    m1, shift1 = quantize_multiplier(s_in * s_w1 / s_h)
    w2 = coefs[1].T
    s_w2 = max(float(np.abs(w2).max()) / 127.0, 1e-12)
    w2q = np.clip(np.round(w2 / s_w2), -127, 127).astype(int)
    b2q = np.round(intercepts[1] / (s_h * s_w2)).astype(np.int64)
    q.update(m1=m1, shift1=shift1, w2=w2q.tolist(), b2=b2q.tolist(), score_scale=f32(s_h * s_w2),
             s_in=s_in, s_h=s_h, s_w2=s_w2)
    return q


def _mlp(inp: LadderInputs, idx: list[int], hidden: int) -> QuantizedMLP:
    from sklearn.neural_network import MLPClassifier

    train = inp.split.train
    F = inp.F[train][:, idx]
    mean, scale = _standardize(F)
    Z = (F - mean) / scale
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), activation="relu", alpha=1e-3, max_iter=1500,
                        random_state=inp.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(Z, inp.y[train])
    q = _quantize(clf.coefs_, clf.intercepts_, mean, scale, Z)
    if len(clf.classes_) != len(inp.classes) and clf.coefs_[1].shape[1] != 1:
        raise RuntimeError("training data is missing a class")
    return QuantizedMLP("classifier", inp.classes, idx, q)


def _autoencoder(inp: LadderInputs, idx: list[int], hidden: int) -> QuantizedMLP:
    from sklearn.neural_network import MLPRegressor

    train = inp.split.train & (inp.y == 0)
    F = inp.F[train][:, idx]
    mean, scale = _standardize(F)
    Z = (F - mean) / scale
    reg = MLPRegressor(hidden_layer_sizes=(hidden,), activation="relu", alpha=1e-3, max_iter=2000,
                       random_state=inp.seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.fit(Z, Z)
    q = _quantize(reg.coefs_, reg.intercepts_, mean, scale, Z)
    m2, shift2 = quantize_multiplier(q["s_h"] * q["s_w2"] / q["s_in"])
    q.update(m2=m2, shift2=shift2, err_threshold=0)
    model = QuantizedMLP("autoencoder", inp.classes, idx, q)
    q_in, _ = model.quantize_inputs(F, None)
    _, out = model.forward(q_in)
    recon = np.clip(requantize(out, m2, shift2), -32768, 32767)
    err = np.sum((recon - q_in) ** 2, axis=1)
    q["err_threshold"] = int(np.ceil(np.quantile(err, 1 - inp.target["false_alarm_rate"] / 2)))
    q["train_mean_err"] = f32(float(err.mean()))
    return model


# ---------------------------------------------------------------------------- the ladder
def run_ladder(inp: LadderInputs, log=lambda msg: None) -> LadderResult:
    candidates: list[Candidate] = []

    log("Rung 1: trying plain rules")
    if inp.task == "anomaly":
        candidates.extend(_rule_conditions(inp))
    else:
        candidates.extend(_candidate(_decision_tree(inp, d), inp, f"decision rules, depth {d}") for d in (2, 3))

    log("Rung 2: trying classic machine learning")
    idx = _select_features(inp)
    if inp.task == "anomaly":
        for n in (25, 50, 100):
            candidates.append(_candidate(_isolation_forest(inp, idx, n), inp, f"{n} trees"))
    else:
        for n, depth in ((10, 4), (20, 6), (40, 8)):
            candidates.append(_candidate(_random_forest(inp, idx, n, depth), inp, f"{n} trees, depth {depth}"))
        for n, depth in ((20, 3), (50, 4)):
            try:
                cand = _candidate(_xgboost(inp, idx, n, depth), inp, f"{n} rounds, depth {depth}")
            except Exception as exc:  # keep climbing if XGBoost is unavailable or misbehaves
                cand = Candidate(2, "gradient_boosting", None, {}, False, f"skipped: {exc}", 0)
            candidates.append(cand)

    log("Rung 3: trying a tiny 8-bit neural network")
    builder = _autoencoder if inp.task == "anomaly" else _mlp
    for hidden in (8, 16, 32):
        try:
            cand = _candidate(builder(inp, idx, hidden), inp, f"{hidden} hidden units, 8-bit")
        except Exception as exc:
            cand = Candidate(3, "neural_network", None, {}, False, f"skipped: {exc}", 0)
        candidates.append(cand)

    best = max((c for c in candidates if c.model is not None), key=lambda c: strength(c.metrics, inp.task),
               default=None)
    chosen = next((c for c in candidates if c.passed), None)
    if chosen is not None:
        log(f"Recommended: {chosen.algorithm.replace('_', ' ')} (rung {chosen.rung}), the simplest that meets "
            "the target")
        return LadderResult(candidates, chosen, inp.classes, explanation=explain(chosen, inp), best=best)
    return LadderResult(candidates, None, inp.classes, data_request=_data_request(candidates, inp), best=best)


def strength(metrics: dict[str, Any], task: str) -> float:
    """One number to compare candidates when none meets the target; false alarms count double."""
    if not metrics:
        return float("-inf")
    if task == "anomaly":
        return (metrics.get("detection_rate") or 0.0) - 2 * (metrics.get("false_alarm_rate") or 0.0)
    return metrics.get("balanced_accuracy") or 0.0


def explain(cand: Candidate, inp: LadderInputs) -> str:
    model = cand.model
    assert model is not None
    return model.explain([inp.defs[i] for i in model.feature_idx], cand.passed)


def _data_request(candidates: list[Candidate], inp: LadderInputs) -> str:
    scored = [c for c in candidates if c.metrics]
    if inp.task == "anomaly":
        best = max(scored, key=lambda c: strength(c.metrics, inp.task), default=None)
        if best is None:
            return "No model could be trained. Upload more normal recordings."
        det = best.metrics.get("detection_rate")
        far = best.metrics.get("false_alarm_rate")
        det_text = f"caught {det:.0%} of faults" if det is not None else "could not be checked on faults"
        return (f"Even the best model {det_text} with {far:.0%} false alarms (targets: catch "
                f"{inp.target['detection_rate']:.0%}, at most {inp.target['false_alarm_rate']:.0%} false alarms). "
                "Record more normal operation across the speeds and loads the machine really sees, and check that "
                "fault recordings come from the same sensor position.")
    best = max(scored, key=lambda c: c.metrics.get("balanced_accuracy") or 0, default=None)
    if best is None:
        return "No model could be trained. Upload more labelled recordings."
    recall = {k: v for k, v in best.metrics["recall"].items() if v is not None}
    weakest = sorted(recall.items(), key=lambda kv: kv[1])[:2]
    weak_text = ", ".join(f"'{k}' ({v:.0%} recognised)" for k, v in weakest)
    return (f"The best model recognised {best.metrics['balanced_accuracy']:.0%} of events on average; the target is "
            f"{inp.target['accuracy']:.0%}. The weakest events are {weak_text}. Record more examples of them, "
            "ideally from more sessions or people.")


def describe_features(defs: list[FeatureDef]) -> list[str]:
    return [describe(d) for d in defs]

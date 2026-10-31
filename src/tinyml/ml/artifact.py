"""Model artifact (docs/contracts/model-artifact.md): everything codegen needs, nothing else."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tinyml import __version__
from tinyml.features.spec import FeatureDef, fft_size
from tinyml.ml.models import Model

SCHEMA = "tinyml.model/1"


def build_artifact(
    model: Model,
    defs: list[FeatureDef],
    task: str,
    window_samples: int,
    hop: int,
    rate: float,
    axes: list[str],
    recipe_id: str,
    recipe_version: int,
    seed: int,
    dataset_sha256: str,
    explanation: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    used = [defs[i] for i in model.feature_idx]
    spectral = any(f.spectral for f in used)
    return {
        "schema": SCHEMA,
        "tinyml_version": __version__,
        "task": task,
        "algorithm": model.algorithm,
        "rung": model.rung,
        "quantization": model.quantization,
        "classes": model.classes,
        "window": {
            "samples": window_samples,
            "hop": hop,
            "axes": axes,
            "sample_rate_hz": rate,
            "fft_size": fft_size(window_samples) if spectral else None,
        },
        "features": [{"name": f.name, "channel": f.channel, "channel_index": f.channel_index, "kind": f.kind}
                     for f in used],
        "params": model.params(),
        "explanation": explanation,
        "metrics": metrics,
        "recipe": {"id": recipe_id, "version": recipe_version},
        "seed": seed,
        "dataset_sha256": dataset_sha256,
    }


def save(artifact: dict[str, Any], path: Path) -> Path:
    path.write_text(json.dumps(artifact, indent=1))
    return path


def load(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text())
    if artifact.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} artifact")
    return artifact


def model_from_artifact(artifact: dict[str, Any]) -> Model:
    """Rebuild the Python reference model from an artifact (feature positions are 0..n-1)."""
    import numpy as np

    from tinyml.ml.models import QuantizedMLP, ThresholdRule, TreeEnsemble

    alg = artifact["algorithm"]
    p = artifact["params"]
    idx = list(range(len(artifact["features"])))
    classes = artifact["classes"]
    if alg == "threshold_rule":
        return ThresholdRule(idx, p["conditions"])
    if alg in ("mlp_int8", "autoencoder_int8"):
        q = {k: v for k, v in p.items() if k != "kind"}
        return QuantizedMLP(p["kind"], classes, idx, q)
    extra_keys = ("depth_threshold", "c_norm", "train_mean_depth")
    extra = {k: p[k] for k in extra_keys if k in p}
    return TreeEnsemble(alg, artifact["rung"], classes, idx, p["op"], p["roots"], p["feature"], p["threshold"],
                        p["left"], p["right"], np.asarray(p["leaves"]), p["aggregate"], base=p.get("base"),
                        extra=extra)

# Model artifact (`model.json`)

The only input codegen reads. Produced by Stage A (`src/tinyml/ml/artifact.py`), schema `tinyml.model/1`.

```json
{
  "schema": "tinyml.model/1",
  "task": "anomaly | classification",
  "algorithm": "threshold_rule | decision_tree | random_forest | gradient_boosting | isolation_forest | mlp_int8 | autoencoder_int8",
  "rung": 1,
  "quantization": "float32 | int8",
  "classes": ["normal", "anomaly"],
  "window": {"samples": 2048, "hop": 1024, "axes": ["accel_de", "accel_fe"], "sample_rate_hz": 12000, "fft_size": null},
  "features": [{"name": "rms_accel_de", "channel": "accel_de", "channel_index": 0, "kind": "rms"}],
  "params": {},
  "explanation": "Rules were enough: …",
  "metrics": {},
  "recipe": {"id": "pump_health", "version": 1},
  "seed": 42,
  "dataset_sha256": "…"
}
```

`features` is the model's input order. `channel_index` is −1 for `mag`. Anomaly models always use the classes `["normal", "anomaly"]`, and label 1 means anomaly.

## `params` by algorithm

All thresholds are float32 values. A tree threshold `t` from scikit-learn is stored as the largest float32 ≤ t, so float32 inputs compare exactly as they did in training.

**threshold_rule.** `conditions`: a list of `{feature, op: "gt" | "lt", threshold, mean, std, k}`; `combine` is always `"any"`. The label is 1 if any condition fires. The score is the largest `(x − mean) / std` (or `(mean − x) / std` for `lt`). Calibration recomputes `mean` and `std` on site and sets `threshold = mean ± k·std`.

**Tree models** (decision_tree, random_forest, gradient_boosting, isolation_forest). One flat node table: `feature[]`, `threshold[]`, `left[]`, `right[]`, `roots[]`. A node with `feature = −1` is a leaf, and `right` then holds its row in `leaves` (a list of rows of length `out_dim`). `op` is `"le"` (go left when x ≤ threshold, scikit-learn) or `"lt"` (x < threshold, XGBoost). `aggregate`:

- `proba`: sum leaf rows over trees, label = first index of the maximum, score = max / number of trees.
- `margin`: sum leaf rows, add `base`; with one output, label = sum > 0; otherwise argmax, score = top − second.
- `depth` (isolation forest): each leaf row holds `depth + c(n_leaf)`. Average over trees; label 1 when the average < `depth_threshold`. Score = 1 − average / `c_norm`. Calibration scales `depth_threshold` by the site's mean depth over `train_mean_depth`.

**mlp_int8 / autoencoder_int8.** `in_mean`, `in_mul` (float32 per input); `q = round_half_away((x − in_mean) · in_mul)` clamped to ±127. `w1` (hidden × inputs, int8), `b1` (int32), ReLU, then `q_h = clamp((acc·m1 + 2^(shift1−1)) >> shift1, 0, 127)`. `w2` (outputs × hidden, int8), `b2` (int32). The classifier takes the argmax (or > 0 for a single output); score = logit gap × `score_scale`. The autoencoder requantizes outputs with `m2`/`shift2` (clamped to int16), sums squared differences from `q`, and flags an anomaly when the sum > `err_threshold`. Calibration scales `err_threshold` by the site's mean error over `train_mean_err`.

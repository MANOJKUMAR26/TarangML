# Golden vectors (`golden.json`)

A set of test windows plus Python's expected answers. The parity test runs them through the compiled firmware (host, then emulator) and compares. Schema `tinyml.golden/1`, model `GoldenSet` in `src/tinyml/contracts.py`.

```json
{
  "schema": "tinyml.golden/1",
  "window_samples": 2048,
  "axes": ["accel_de", "accel_fe"],
  "feature_names": ["rms_accel_de"],
  "feature_tolerances": [{"rtol": 0.001, "atol": 0.00012}],
  "score_rtol": 0.002,
  "score_exact": false,
  "vectors": [
    {"id": 0, "recording": "Normal_3", "true_label": "normal", "input_b64": "…",
     "expected_label": 0, "expected_score": -0.41, "expected_features": [0.0712],
     "margin": 0.0, "borderline": false}
  ]
}
```

`input_b64` is the window as little-endian float32, `samples × axes`, interleaved by axis. In C the same values appear as exact hex float literals in `golden_vectors.h`.

## Selection

Up to 32 windows (fewer when a window is large; the whole set stays under 512 KB) from the test split only, covering every combination of true label and predicted label.

## Pass rule

A build passes when, for every vector:

1. the firmware printed a result for it and printed `TINYML DONE`;
2. the label equals `expected_label`, unless the vector is `borderline`;
3. every feature is within its tolerance from `features.md`;
4. the score is within `score_rtol · (1 + |expected|)` (skipped for borderline vectors).

A vector is **borderline** when a feature the model compares sits within the decision tolerance of a threshold, a discrete feature (`domfreq`, `zcr`) is numerically unstable for that window, a near-tie decides an argmax, or an 8-bit input lies on a rounding boundary. Borderline mismatches are reported but do not fail the build; everything else does, and a failed build is never packaged.

## Harness output format

The harness prints only hex bits, so no `printf` is needed and the comparison is exact:

```
TINYML BEGIN
T <counter ticks for one prediction>
C <instructions> <ticks>          (only where a calibration loop is needed, e.g. ESP32 in QEMU)
W <id> <label> <score-bits> <feature-bits> …
TINYML DONE
```

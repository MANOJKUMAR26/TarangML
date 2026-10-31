# Feature spec

Python (`src/tinyml/features/compute.py`) and the generated C (`src/tinyml/codegen/emit.py`) are both written from this file. If they disagree, this file wins and the other one is the bug. Change it only through a pull request the lead approves, together with both implementations and a passing parity run.

## Window

A window is `n` consecutive samples of every axis at one steady rate, stored interleaved by axis: `window[i * axes + a]`. The recipe sets the duration (`window_ms`, capped at `max_window_samples`); the hop between windows is `hop_ratio * n`.

## Channels

Every axis is a channel. When the recipe sets `include_magnitude` and there are at least two axes, there is one more channel, `mag`, the vector length `sqrt(x² + y² + z² …)` of each sample. C computes it once per window into a buffer.

## Time features (per channel, `d_i = x_i - mean`)

| name | formula |
|---|---|
| `mean` | Σx / n |
| `std` | sqrt(Σd² / n) (population, not sample) |
| `rms` | sqrt(Σx² / n) |
| `peak` | max \|x\| |
| `p2p` | max x − min x |
| `crest` | peak / rms, or 0 when rms ≤ 1e-12 |
| `kurt` | (Σd⁴ / n) / (Σd² / n)², or 0 when Σd²/n ≤ 1e-12 · max(peak², 1e-12) |
| `zcr` | number of i in 1..n−1 where (d_{i−1} < 0) ≠ (d_i < 0), divided by n − 1 |

## Spectral features (per channel)

`N` is the smallest power of two ≥ max(n, 16). Take the FFT of `d` zero-padded to `N`; `P_k = re_k² + im_k²` for bins `k = 1 … N/2` (bin 0 is skipped). `bin_hz = rate / N`. `total = Σ P_k`.

| name | formula |
|---|---|
| `domfreq` | k* · bin_hz, where k* is the first k with the largest P_k |
| `centroid` | (Σ k·P_k) / total · bin_hz, or 0 when total = 0 |
| `band0..band3` | Σ P_k over bins `1 + b·N/8 … (b+1)·N/8`, divided by total (0 when total = 0) |

## Numeric rules for C

All C arithmetic is float32 with sequential accumulation in sample order. Compile with `-ffp-contract=off` so multiply-adds are not fused. Square roots must be correctly rounded: hardware VSQRT on Arm cores with an FPU, otherwise the integer routine in the template.

## Tolerances (C versus Python float64)

The parity test compares every feature with `|c − py| ≤ atol + rtol · |py|`.

| kind | rtol | atol |
|---|---|---|
| mean, std, rms, peak, p2p | 1e-3 | 1e-4 × the largest absolute input value of that channel in the data |
| crest, others | 1e-3 | 1e-5 |
| kurt | 2e-3 | 1e-4 |
| zcr | 0 | 2.5 / (n − 1) |
| domfreq | 0 | 1.01 × bin_hz |
| centroid | 1e-3 | 0.01 × bin_hz |
| band0..band3 | 1e-3 | 1e-4 |

Separately, a tighter "decision tolerance" (about 8·√n·ε32, larger for kurtosis and spectral features) flags golden windows whose answer could legitimately flip; see `golden-vectors.md`.

# Package

What the user downloads: `tinyml-<project>-<chip>-v<version>.zip`, everything under `tinyml/`.

| path | content |
|---|---|
| `include/tinyml_model.h` | the API: `model_predict(window) -> tinyml_result_t`, `model_features`, constants, and calibration for anomaly models |
| `src/tinyml_model.c` | the whole library in one portable C99 file, no heap, no libc maths |
| `cpp/TinyMLModel.hpp` | header-only C++ wrapper with a streaming `push()` helper |
| `example/main.c` | sensor loop with calibration on first install |
| `test/` | parity harness, host port, golden vectors (`.h` and `.json`), `check_parity.py`, README |
| `model.json` | the artifact the code was generated from |
| `report.md` | plain-words report: request, data checks, choice, test results, tagged numbers, parity, reproducibility |
| `manifest.json` | SHA-256 of every file, versions, seed, dataset hash, toolchain |
| `manifest.sig` | Ed25519 signature of `manifest.json` |
| `signing_key.pub.pem` | the public key; `tinyml verify <zip>` checks the signature and every hash |

The private signing key lives in `~/.tinyml/keys/` on the machine that built the package and is never committed.

Every number in `report.md` carries a tag: **measured** (linker output of the chip's own compiler), **emulated** (emulator instruction count plus a speed factor measured on a real board) or **estimated** (datasheet values, or a speed factor not yet calibrated). Power is always estimated.

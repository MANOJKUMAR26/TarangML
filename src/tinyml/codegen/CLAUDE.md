Owner: Dev B. Turns `model.json` into C. Reads only the artifact, never training data.

- Feature code must follow `docs/contracts/features.md` exactly: float32, sequential sums, same tie-breaking and bin ranges as `src/tinyml/features/compute.py`.
- Constants are written with `cf()` so they are exact float32 literals.
- Emit only what the model uses: unused helpers or variables become warnings in users' firmware.
- After any change: `uv run pytest tests/unit -q` (host parity for every recipe) and `make parity`.

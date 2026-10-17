# Project rules for people and coding agents

tinyml turns a plain-English goal and a sensor CSV into a C library that is proven to give the same answers on a chip as in Python, plus a ranking of chips. Read `docs/architecture.md` once before your first change.

## Non-negotiables

1. Every number shown to a user comes from code: a compile, a linker output, an emulator run or a catalog value. Never from an AI. Every number carries a `measured`, `emulated` or `estimated` tag.
2. No package ships unless the golden-vector parity test passes in the emulator. Do not weaken tolerances to make a test pass; find the bug.
3. The contracts in `docs/contracts/` change only through a pull request the lead approves. Update the contract, `src/tinyml/contracts.py`, and both the Python and C sides in the same PR.
4. Python features and C features are written from `docs/contracts/features.md`. If you touch one side, run `make parity`.
5. Raw sensor data never goes to an assistant and never into the decision log. Only column summaries.
6. Everything runs locally. No new network calls in the engine. Secrets and signing keys never enter the repo.
7. The product must keep working with the assistant set to `off`.

## Who owns what

| path | owner |
|---|---|
| `docs/contracts/`, `recipes/`, `catalog/`, `src/tinyml/decide/`, `src/tinyml/intake/` | lead |
| `src/tinyml/data/`, `src/tinyml/features/`, `src/tinyml/ml/` | Dev A (data and ML) |
| `firmware/`, `src/tinyml/codegen/`, `src/tinyml/toolchains/` | Dev B (embedded) |
| `src/tinyml/cli/`, `api/`, `pipeline/`, `packaging/`, `store/`, `ui/`, `ci/`, `.github/`, `tests/parity/` | Dev C (platform) |

Stay inside your folders. If you need a change elsewhere, open an issue for its owner or ask in the PR.

## Commands

```sh
uv sync                     # install
uv run tinyml doctor        # which chips this machine can build and emulate
uv run tinyml serve         # web app on http://127.0.0.1:8765
make test                   # lint + unit tests (fast, host only)
make parity                 # build and emulate every chip whose toolchain is installed
```

## Code style

Python 3.11+, typed, formatted to `ruff` defaults with a 110-column limit. Write code a reviewer can follow without the PR description: clear names, no commented-out code, no comments that restate the code. Generated C must compile cleanly with `-Wall -Wextra -Wpedantic` on GCC and Clang. Messages shown to users are plain English with a number and a next step ("You have 0.8 minutes of normal data; this needs about 2").

## Pull requests

One issue per PR, with the issue's "Works when" check in the description. CI runs lint, unit tests and the emulator parity tests; all must be green to merge.

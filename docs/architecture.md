# How tinyml works

tinyml runs entirely on your machine. One Python package holds the engine; a local web app and a command line sit on top of the same functions. The metadata lives in one SQLite file (`workspace/tinyml.db`), and the files for each project live under `workspace/projects/<id>/`. Nothing calls a server while you work, except Claude when you explicitly turn it on as the assistant.

## The flow

The user goes through the six steps of the PRD. Steps 1 and 2 turn a paragraph into a `ProblemSpec` (`intake/parse.py`). Step 3 reads the CSV, summarises every column, decides what each column is, runs the data checks, resamples to one steady rate and saves the result as Parquet (`data/`). Steps 4 to 6 are the three stages of the technical pipeline in `pipeline/stages.py`.

Stage A (`train`) cuts windows, computes features, splits train and test by recording, climbs the algorithm ladder (`ml/ladder.py`), writes `model.json`, picks golden windows, generates C (`codegen/`) and immediately checks that C against Python on the host compiler. If the C does not match on the host, nothing else runs.

Stage B (`rank_chips`) drops chips that fail the hard filters, then for every remaining chip builds a size probe with the chip's compiler (flash and RAM from the linker output, stack from GCC's call-graph data), runs the model once in the emulator to count instructions, and runs four golden windows to catch broken builds early. The ranking (`ranking/score.py`) applies the post-build filters and a weighted score for the user's priority, and writes reasons in plain words.

Stage C (`build_for_chip`) repeats the build with all golden windows, re-checks the memory budget, and only if everything passes writes the signed package (`packaging/`).

## Folders and owners

| path | what | owner |
|---|---|---|
| `docs/contracts/`, `recipes/`, `catalog/` | the contracts, recipe defaults, chip catalog | lead |
| `src/tinyml/data/`, `features/`, `ml/` | ingest, checks, windows, Python features, ladder | Dev A |
| `firmware/`, `src/tinyml/codegen/`, `src/tinyml/toolchains/` | C generation, harness, ports, builds, emulators | Dev B |
| `src/tinyml/cli/`, `api/`, `pipeline/`, `packaging/`, `store/`, `ui/`, `ci/`, `tests/parity/` | CLI, REST API, jobs, packaging, web app, CI | Dev C |
| `src/tinyml/decide/`, `intake/` | judgement calls and the assistant layer | lead (Jev in Phase 5) |

## Emulators and what each number means

Espressif chips build real ESP-IDF firmware and run it in Espressif's QEMU fork (chip-level). Under `-icount` the CPU cycle counter follows virtual time rather than instructions, so the harness also times a loop of exactly 40,031 instructions and converts.

Arm chips run on upstream QEMU's MPS2 boards that use the same CPU core (Cortex-M4F for STM32F4, STM32L4 and nRF52840; Cortex-M33 for nRF54L15 and RP2350). The compiled instructions are identical to what the chip runs, so parity and instruction counts carry over; flash wait states and bus timing do not, which is what the per-core speed factor (`cpi` in the catalog) covers. Instructions are counted exactly from QEMU's execution trace between two marker functions. The RP2040 runs Cortex-M0+ code on the Cortex-M3 board and stays "estimated", as the PRD requires for community-level emulation. Renode profiles for chip-level STM32 and nRF emulation are the planned next step.

When ESP-IDF is missing, the ESP32-C3 falls back to a bare-metal RV32IMC build on QEMU's `virt` board.

Flash and RAM are **measured**: they come from linking a probe that calls `model_predict` and one that does not, with the chip's own compiler and flags, and subtracting. RAM includes the window buffer and the worst-case stack. Instruction counts are **emulated** on chip- and core-level emulators. Time per prediction is instructions × `cpi` / clock, so it stays **estimated** until `cpi.source` in the catalog is set to `measured` after a real-board calibration. Energy and battery life are always **estimated**.

## Reliability

Jobs run in background threads (`pipeline/jobs.py`). Temporary failures (timeouts) are retried with back-off; a build error on one chip never stops the others. Each chip measurement is cached by a hash of the model, the golden set, the firmware sources and the build code, so re-running a job after a crash or restart never repeats finished work, and queued jobs resume when the app starts again.

## Moving to Postgres later

Only `store/db.py` speaks SQL, and `store/schema.sql` is plain SQL. Swapping SQLite for Postgres means reimplementing that one class; nothing else changes.

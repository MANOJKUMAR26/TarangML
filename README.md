# tinyml

Describe what a sensor should detect, upload a CSV, and get back a C library that has been **proven** to give the same answers on the chip as in Python, plus a ranking of microcontrollers it runs on. Everything runs on your own machine.

```
"Alert me when my pump vibration looks abnormal. Battery, Wi-Fi."  +  pump.csv
        │
        ▼
 checks the data ─► rules → classic ML → tiny 8-bit network (stops at the simplest that works)
        │
        ▼
 C99 library ─► compiled with each chip's own compiler ─► run in an emulator ─► parity with Python
        │
        ▼
 top 3 chips with reasons  +  signed .zip (library, C++ wrapper, example, calibration, parity test, report)
```

## Quick start (macOS)

```sh
scripts/setup_macos.sh --espidf     # uv, QEMU, Arm GCC, RISC-V GCC, ESP-IDF v5.5.6 + Espressif QEMU
uv run tinyml doctor                # shows which chips are ready
uv run tinyml serve                 # opens http://127.0.0.1:8765
```

Leave out `--espidf` for a quicker first install (about 2 GB less). Arm chips then work right away, the ESP32-C3 falls back to a bare-metal RISC-V build, and the ESP32-S3 stays unavailable until you run `scripts/setup_espidf.sh`. On Linux use `scripts/setup_linux.sh`. For a fully pinned environment, use `ci/docker/Dockerfile`.

## Try it

In the web app, click **New project**, pick one of the example sentences, then on the Data step press **pump**, **driving** or **activity** to load demo data. Then train, rank and build.

The same flow from the terminal:

```sh
uv run tinyml demo-data cwru --out workspace/demo      # real CWRU bearing data (downloads from GitHub)
uv run tinyml build workspace/demo/pump_cwru.csv --chip esp32s3
uv run tinyml rank workspace/demo/pump_cwru.csv --any-vendor --priority battery
uv run tinyml verify workspace/projects/<id>/packages/<package>.zip
```

`build` is the Phase 1 "done when" check from the PRD: it produces a zip whose emulator output matches Python. `rank` is the Phase 3 check: it prints the top three chips with reasons.

## What you get

The six screens follow the PRD. **Describe**: a sentence, turned into a problem spec, with one follow-up question if something important is missing. **Vendor**: a vendor or "any", plus connectivity, power, budget and priority, pre-filled from the sentence. **Data**: column roles you can correct, data checks in plain words (missing values, gaps, clipping, uneven or mixed sampling rates, not enough data), and a look at the signal. **Algorithm**: the ladder with every candidate tried, the chosen one explained in one sentence, and how it separates the classes. **Rank chips**: a real build and emulator run per chip, the top three with reasons and trade-offs, and why the others were left out. **Build**: the final build, parity per golden window, the report, the code, a download, and "ask about this package".

Every number is tagged **measured** (linker output from the chip's compiler), **emulated** (emulator instruction count) or **estimated** (datasheet values or an uncalibrated speed factor). See `docs/architecture.md` for exactly where each one comes from.

## Chips

| chip | how it is built and run | numbers |
|---|---|---|
| ESP32-S3, ESP32-C3 | ESP-IDF v5.5.6, Espressif QEMU (chip-level) | measured flash and RAM, emulated instructions |
| STM32F411, STM32L432, nRF52840 | Arm GCC, QEMU MPS2 Cortex-M4F (core-level) | measured flash and RAM, emulated instructions |
| nRF54L15 | Arm GCC, QEMU MPS2 Cortex-M33 (core-level) | measured flash and RAM, emulated instructions |
| RP2040, RP2350 | Arm GCC, community-level emulation | estimated until onboarded |

Time per prediction stays estimated until each core's speed factor is measured on a real board (`cpi` in `catalog/chips.yaml`). The catalog is hand-edited for V1.

## The assistant

Rules make every judgement call first. When they are unsure, tinyml asks the assistant you chose: **Ask me** (no AI), **Claude Code** (runs `claude -p` on this machine with your login) or **Claude chat** (gives you a prompt to paste). Only column summaries are ever sent, never sensor data. Every decision is logged, and corrections you repeat three times become rules. Jev plugs into the same slot later. See `docs/decisions.md`.

## For the team

Read `AGENTS.md` (rules and folder ownership) and `docs/contracts/` (the five formats everyone builds against). `make test` runs lint and unit tests in seconds, and `make parity` builds and emulates every installed chip. CI runs both on every pull request.

```
docs/contracts/   problem spec, CSV summary, features, model artifact, golden vectors, package
recipes/          per-use-case defaults (pump health, driving events, human activity)
catalog/          chip catalog
src/tinyml/       engine: data, features, ml, codegen, toolchains, ranking, packaging, decide, pipeline, api, cli
firmware/         parity harness, size probe, ports for Arm, RISC-V, host and ESP-IDF
ui/               the web app (plain HTML, CSS and JS, works offline)
tests/            unit tests and per-chip emulator parity tests
```

Metadata lives in SQLite (`workspace/tinyml.db`); only `src/tinyml/store/db.py` speaks SQL, so moving to Postgres later is one file. Pick a license before publishing the repository; Apache-2.0 fits an open-core plan.

"""Stage B steps 2-3 for one chip: build with the chip's toolchain, measure, and check parity early."""

from __future__ import annotations

from pathlib import Path

from tinyml.catalog import Chip
from tinyml.codegen import golden as G
from tinyml.contracts import ChipMeasurement, GoldenSet, TaggedNumber
from tinyml.toolchains import backend_for

BATTERY_MAH = 2000.0
VOLTAGE = 3.3


def instruction_tag(chip: Chip) -> str:
    return "estimated" if chip.emulator.fidelity == "community" else "emulated"


def latency_tag(chip: Chip) -> str:
    if chip.emulator.fidelity == "community":
        return "estimated"
    return "emulated" if chip.cpi.get("source") == "measured" else "estimated"


def derive(chip: Chip, flash: int, ram: int, stack: int, instructions: int | None, hop_s: float,
           toolchain: str, emulator: str, sensor_ma: float = 0.0) -> dict:
    out = {
        "flash_bytes": TaggedNumber(value=flash, unit="bytes", tag="measured", source="linker output of a size probe"),
        "ram_bytes": TaggedNumber(value=ram, unit="bytes", tag="measured",
                                  source="static data + window buffer from the linker, plus worst-case stack"),
        "stack_bytes": TaggedNumber(value=stack, unit="bytes", tag="measured", source="GCC call-graph stack usage"),
        "price_usd": TaggedNumber(value=chip.price_usd, unit="USD", tag="estimated", source="catalog"),
    }
    if instructions is not None:
        latency_ms = instructions * float(chip.cpi["value"]) / (chip.clock_mhz * 1e3)
        energy_uj = VOLTAGE * chip.active_ma * latency_ms
        duty = min(1.0, latency_ms / 1000.0 / max(hop_s, 1e-9))
        avg_ma = chip.sleep_ma + (chip.active_ma - chip.sleep_ma) * duty + sensor_ma
        out.update(
            instructions=TaggedNumber(value=instructions, unit="instructions", tag=instruction_tag(chip),
                                      source=emulator),
            latency_ms=TaggedNumber(value=round(latency_ms, 4), unit="ms", tag=latency_tag(chip),
                                    source=f"instructions x {chip.cpi['value']} cycles/instruction "
                                           f"({chip.cpi.get('source', 'estimated')}) at {chip.clock_mhz:g} MHz"),
            energy_uj=TaggedNumber(value=round(energy_uj, 3), unit="uJ", tag="estimated",
                                   source=f"{chip.active_ma:g} mA active at {VOLTAGE} V"),
            battery_days=TaggedNumber(value=round(BATTERY_MAH / avg_ma / 24.0, 1), unit="days", tag="estimated",
                                      source=f"MCU plus a {sensor_ma:g} mA sensor on {BATTERY_MAH:g} mAh, sleeping at "
                                             f"{chip.sleep_ma:g} mA between predictions; radio not included"),
        )
    return out


def subset(golden: GoldenSet, count: int) -> GoldenSet:
    copy = golden.model_copy(deep=True)
    copy.vectors = copy.vectors[:count]
    return copy


def measure_chip(chip: Chip, gen_dir: Path, golden: GoldenSet, work: Path, window_bytes: int, hop_s: float,
                 parity_windows: int | None = 4, timeout: int = 300, sensor_ma: float = 0.0) -> ChipMeasurement:
    backend, why = backend_for(chip)
    if backend is None:
        return ChipMeasurement(chip_id=chip.id, status="skipped", backend=chip.backend,
                               error=f"toolchain not installed ({why}); run `tinyml doctor`")
    work.mkdir(parents=True, exist_ok=True)
    size = backend.measure(chip, gen_dir, work, window_bytes)
    check = subset(golden, parity_windows) if parity_windows else golden
    instructions = None
    if backend.name == "espidf":
        result = backend.run(chip, gen_dir, G.header(check), work, timing=True, parity=True, timeout=timeout)
        instructions = result.instructions
        rows, _, finished = G.parse_output(result.output)
        parity = G.compare(check, rows, finished)
    else:
        timing = backend.run(chip, gen_dir, G.header(subset(golden, 1)), work, timing=True, parity=False,
                             timeout=timeout)
        instructions = timing.instructions
        result = backend.run(chip, gen_dir, G.header(check), work, timing=False, parity=True, timeout=timeout)
        rows, _, finished = G.parse_output(result.output)
        parity = G.compare(check, rows, finished)
    numbers = derive(chip, size.flash_bytes, size.ram_bytes, size.stack_bytes, instructions, hop_s,
                     size.toolchain, backend.emulator_name(chip), sensor_ma)
    (work / "emulator_output.txt").write_text(result.output)
    return ChipMeasurement(
        chip_id=chip.id, status="ok", backend=backend.name, parity_ok=parity["passed"], parity_detail=parity,
        toolchain=size.toolchain, emulator=backend.emulator_name(chip), **numbers,
    )

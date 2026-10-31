"""Bare-metal Arm Cortex-M and RISC-V builds, run on upstream QEMU.

Arm chips run on QEMU's MPS2 boards that use the same CPU core (core-level emulation):
the compiled instructions are identical to the chip's, so parity and instruction counts
carry over. Flash and RAM come from the linker output of a size probe, measured with the
chip's compiler flags.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

from tinyml.catalog import Chip, Core, get_catalog
from tinyml.toolchains.common import (
    COMMON_CFLAGS,
    HOST_CFLAGS,
    RunResult,
    SizeReport,
    TransientError,
    berkeley_size,
    firmware,
    run,
    symbol_address,
    tool_version,
    worst_stack,
)

TRACE_LINE = re.compile(rb"Trace \d+: 0x[0-9a-f]+ \[[0-9a-f]+/([0-9a-f]+)/")
MB = 1024 * 1024


def _which(*names: str) -> str | None:
    for name in names:
        if shutil.which(name):
            return name
    return None


class BareMetalBackend:
    name = "baremetal"
    gcc_prefixes: tuple[str, ...] = ()
    qemu_binary = ""

    def __init__(self) -> None:
        self.prefix = next((p for p in self.gcc_prefixes if shutil.which(p + "gcc")), self.gcc_prefixes[0])

    # --- description ---------------------------------------------------------------------
    def core(self, chip: Chip) -> Core:
        return get_catalog().cores[chip.core]

    def machine(self, chip: Chip) -> str:
        machine = self.core(chip).qemu_machine
        assert machine is not None
        return machine

    def available(self, chip: Chip) -> tuple[bool, str]:
        if not shutil.which(self.prefix + "gcc"):
            return False, f"{self.prefix}gcc not found"
        if not shutil.which(self.qemu_binary):
            return False, f"{self.qemu_binary} not found"
        return True, ""

    def toolchain_version(self, chip: Chip) -> str:
        return tool_version(self.prefix + "gcc")

    def emulator_name(self, chip: Chip) -> str:
        return f"{tool_version(self.qemu_binary)} -M {self.machine(chip)}"

    # --- building ------------------------------------------------------------------------
    def cflags(self, chip: Chip) -> list[str]:
        return list(self.core(chip).gcc_flags) + COMMON_CFLAGS

    def startup_sources(self) -> list[Path]:
        raise NotImplementedError

    def linker_script(self, flash_len: int, ram_len: int, chip: Chip) -> str:
        raise NotImplementedError

    def compile(self, chip: Chip, src: Path, obj: Path, includes: list[Path], defines: list[str] | None = None,
                extra: list[str] | None = None) -> Path:
        obj.parent.mkdir(parents=True, exist_ok=True)
        cmd = [self.prefix + "gcc", *self.cflags(chip), *(extra or []), *[f"-I{p}" for p in includes],
               *[f"-D{d}" for d in defines or []], "-c", str(src), "-o", str(obj)]
        run(cmd, cwd=obj.parent)
        return obj

    def link(self, chip: Chip, objs: list[Path], elf: Path, flash_len: int, ram_len: int) -> Path:
        script = elf.with_suffix(".ld")
        script.write_text(self.linker_script(flash_len, ram_len, chip))
        cmd = [self.prefix + "gcc", *self.cflags(chip), "-nostdlib", "-nostartfiles", "-Wl,--gc-sections",
               f"-Wl,-Map={elf.with_suffix('.map')}", "-T", str(script), *map(str, objs), "-lgcc", "-o", str(elf)]
        run(cmd, cwd=elf.parent)
        return elf

    def runtime_objects(self, chip: Chip, work: Path, includes: list[Path]) -> list[Path]:
        sources = self.startup_sources() + [firmware("common", "libc_min.c")]
        return [self.compile(chip, s, work / "rt" / (s.stem + ".o"), includes) for s in sources]

    def model_object(self, chip: Chip, gen_dir: Path, work: Path) -> Path:
        return self.compile(chip, gen_dir / "tinyml_model.c", work / "model" / "tinyml_model.o", [gen_dir],
                            extra=["-fstack-usage", "-fcallgraph-info=su"])

    # --- measuring -----------------------------------------------------------------------
    def measure(self, chip: Chip, gen_dir: Path, work: Path, window_bytes: int) -> SizeReport:
        includes = [gen_dir, firmware("common")]
        model = self.model_object(chip, gen_dir, work)
        runtime = self.runtime_objects(chip, work, includes)
        probe = self.compile(chip, firmware("common", "probe.c"), work / "probe" / "probe.o", includes)
        empty = self.compile(chip, firmware("common", "probe.c"), work / "probe" / "empty.o", includes,
                             ["TINYML_PROBE_EMPTY"])
        big = 64 * MB
        full_elf = self.link(chip, [model, probe, *runtime], work / "probe" / "probe.elf", big, big)
        base_elf = self.link(chip, [empty, *runtime], work / "probe" / "empty.elf", big, big)
        t1, d1, b1 = berkeley_size(self.prefix + "size", full_elf)
        t0, d0, b0 = berkeley_size(self.prefix + "size", base_elf)
        stack, info = worst_stack(sorted((work / "model").glob("*.ci")))
        return SizeReport(
            flash_bytes=(t1 + d1) - (t0 + d0),
            ram_static_bytes=(d1 + b1) - (d0 + b0),
            window_bytes=window_bytes,
            stack_bytes=stack,
            toolchain=self.toolchain_version(chip),
            detail={"probe_map": str(full_elf.with_suffix(".map")), **info},
        )

    # --- emulating -----------------------------------------------------------------------
    def qemu_args(self, chip: Chip, elf: Path) -> list[str]:
        raise NotImplementedError

    def emulator_layout(self, chip: Chip) -> tuple[int, int]:
        return 4 * MB, 4 * MB

    def build_harness(self, chip: Chip, gen_dir: Path, golden_header: str, work: Path, timing: bool,
                      parity: bool) -> Path:
        tag = ("t" if timing else "") + ("p" if parity else "")
        hdir = work / f"harness_{tag}"
        hdir.mkdir(parents=True, exist_ok=True)
        (hdir / "golden_vectors.h").write_text(golden_header)
        includes = [hdir, gen_dir, firmware("common")]
        model = self.model_object(chip, gen_dir, work)
        runtime = self.runtime_objects(chip, work, includes)
        harness = self.compile(chip, firmware("common", "harness.c"), hdir / "harness.o", includes,
                               [f"TINYML_TIMING={int(timing)}", f"TINYML_PARITY={int(parity)}"])
        flash_len, ram_len = self.emulator_layout(chip)
        return self.link(chip, [model, harness, *runtime], hdir / "harness.elf", flash_len, ram_len)

    def run(self, chip: Chip, gen_dir: Path, golden_header: str, work: Path, timing: bool, parity: bool,
            timeout: int = 300) -> RunResult:
        elf = self.build_harness(chip, gen_dir, golden_header, work, timing, parity)
        cmd = self.qemu_args(chip, elf)
        trace = elf.with_suffix(".trace")
        if timing:
            cmd += ["-accel", "tcg,one-insn-per-tb=on", "-d", "exec,nochain", "-D", str(trace)]
        start = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise TransientError(f"QEMU timed out after {timeout}s") from exc
        output = proc.stdout.decode(errors="replace") + proc.stderr.decode(errors="replace")
        instructions = None
        if timing and trace.exists():
            instructions = self._count(elf, trace)
            trace.unlink()
        return RunResult(output, instructions, time.time() - start)

    def _count(self, elf: Path, trace: Path) -> int | None:
        begin = symbol_address(self.prefix + "nm", elf, "tinyml_mark_begin")
        end = symbol_address(self.prefix + "nm", elf, "tinyml_mark_end")
        if begin is None or end is None:
            return None
        count, counting = 0, False
        with trace.open("rb") as fh:
            for line in fh:
                match = TRACE_LINE.search(line)
                if not match:
                    continue
                pc = int(match.group(1), 16)
                if not counting:
                    if pc == begin:
                        counting = True
                    continue
                if pc == end:
                    return count
                count += 1
        return None


class ArmBackend(BareMetalBackend):
    name = "arm"
    gcc_prefixes = ("arm-none-eabi-",)
    qemu_binary = "qemu-system-arm"
    LAYOUT = {
        "mps2-an385": (0x00000000, 0x20000000),
        "mps2-an386": (0x00000000, 0x20000000),
        "mps2-an500": (0x00000000, 0x20000000),
        "mps2-an505": (0x10000000, 0x38000000),
    }

    def startup_sources(self) -> list[Path]:
        return [firmware("arm", "startup_cortexm.c"), firmware("arm", "port_semihost.c")]

    def linker_script(self, flash_len: int, ram_len: int, chip: Chip) -> str:
        flash_origin, ram_origin = self.LAYOUT[self.machine(chip)]
        return f"""ENTRY(Reset_Handler)
MEMORY
{{
    FLASH (rx) : ORIGIN = {flash_origin:#x}, LENGTH = {flash_len:#x}
    RAM (rwx) : ORIGIN = {ram_origin:#x}, LENGTH = {ram_len:#x}
}}
_estack = ORIGIN(RAM) + LENGTH(RAM);
SECTIONS
{{
    .isr_vector : {{ KEEP(*(.isr_vector)) }} > FLASH
    .text : {{ *(.text*) *(.rodata*) . = ALIGN(4); }} > FLASH
    .ARM.exidx : {{ *(.ARM.exidx*) }} > FLASH
    _sidata = LOADADDR(.data);
    .data : {{ _sdata = .; *(.data*) . = ALIGN(4); _edata = .; }} > RAM AT> FLASH
    .bss (NOLOAD) : {{ _sbss = .; *(.bss*) *(COMMON) . = ALIGN(4); _ebss = .; }} > RAM
    /DISCARD/ : {{ *(.comment) *(.ARM.attributes) }}
}}
"""

    def qemu_args(self, chip: Chip, elf: Path) -> list[str]:
        return [self.qemu_binary, "-M", self.machine(chip), "-display", "none", "-monitor", "none",
                "-serial", "none", "-semihosting-config", "enable=on,target=native", "-kernel", str(elf)]


class RiscvBackend(BareMetalBackend):
    name = "riscv"
    gcc_prefixes = ("riscv64-unknown-elf-", "riscv64-elf-", "riscv-none-elf-", "riscv32-unknown-elf-")
    qemu_binary = "qemu-system-riscv32"

    def core(self, chip: Chip) -> Core:
        return get_catalog().cores[chip.fallback_core or chip.core]

    def available(self, chip: Chip) -> tuple[bool, str]:
        ok, why = super().available(chip)
        if not ok:
            return ok, why
        probe = subprocess.run([self.prefix + "gcc", *self.core(chip).gcc_flags, "-print-libgcc-file-name"],
                               capture_output=True, text=True)
        if "rv32" not in probe.stdout:
            return False, f"{self.prefix}gcc has no 32-bit RISC-V libraries"
        return True, ""

    def startup_sources(self) -> list[Path]:
        return [firmware("riscv", "start.S"), firmware("riscv", "port_semihost.c")]

    def linker_script(self, flash_len: int, ram_len: int, chip: Chip) -> str:
        return f"""ENTRY(_start)
MEMORY
{{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = {flash_len + ram_len:#x}
}}
SECTIONS
{{
    .text : {{ KEEP(*(.text.start)) *(.text*) }} > RAM
    .rodata : {{ *(.rodata*) *(.srodata*) . = ALIGN(4); }} > RAM
    .data : {{ *(.data*) . = ALIGN(8); __global_pointer$ = . + 0x800; *(.sdata*) . = ALIGN(4); }} > RAM
    .bss (NOLOAD) : {{ _sbss = .; *(.sbss*) *(.bss*) *(COMMON) . = ALIGN(4); _ebss = .; }} > RAM
    . = ALIGN(16);
    _estack = ORIGIN(RAM) + LENGTH(RAM);
    /DISCARD/ : {{ *(.comment) *(.riscv.attributes) }}
}}
"""

    def qemu_args(self, chip: Chip, elf: Path) -> list[str]:
        return [self.qemu_binary, "-M", "virt", "-bios", "none", "-display", "none", "-monitor", "none",
                "-serial", "none", "-semihosting-config", "enable=on,target=native", "-kernel", str(elf)]

    def machine(self, chip: Chip) -> str:
        return "virt"


class HostBackend(BareMetalBackend):
    """Native build of the same C: a fast first check before any cross build."""

    name = "host"
    gcc_prefixes = ("",)

    def available(self, chip: Chip | None = None) -> tuple[bool, str]:
        cc = _which("cc", "gcc", "clang")
        return (cc is not None, "" if cc else "no C compiler found")

    def toolchain_version(self, chip: Chip | None = None) -> str:
        return tool_version(_which("cc", "gcc", "clang") or "cc")

    def run_host(self, gen_dir: Path, golden_header: str, work: Path, timeout: int = 120) -> RunResult:
        cc = _which("cc", "gcc", "clang") or "cc"
        work.mkdir(parents=True, exist_ok=True)
        (work / "golden_vectors.h").write_text(golden_header)
        exe = work / "harness_host"
        run([cc, *HOST_CFLAGS, "-DTINYML_TIMING=0", "-DTINYML_PARITY=1", f"-I{work}", f"-I{gen_dir}",
             f"-I{firmware('common')}", str(gen_dir / "tinyml_model.c"), str(firmware("common", "harness.c")),
             str(firmware("host", "port_host.c")), "-o", str(exe)])
        start = time.time()
        try:
            proc = subprocess.run([str(exe)], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise TransientError("host harness timed out") from exc
        return RunResult(proc.stdout + proc.stderr, None, time.time() - start)

"""Espressif chips: ESP-IDF builds run on Espressif's QEMU fork (chip-level emulation).

Parity and timing run the real ESP-IDF firmware in QEMU with -icount, where the CPU cycle
counter advances once per instruction. Flash and RAM come from a size probe linked with the
same vendor compiler that ESP-IDF uses.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import time
from functools import lru_cache
from pathlib import Path

from tinyml.catalog import Chip
from tinyml.toolchains.common import (
    RunResult,
    SizeReport,
    ToolchainError,
    TransientError,
    firmware,
    run,
    wait_for_text,
    worst_stack,
)

TARGETS = {
    "esp32s3": {
        "gcc": "xtensa-esp32s3-elf-",
        "flags": ["-mlongcalls"],
        "qemu": "qemu-system-xtensa",
        "machine": ["-M", "esp32s3", "-m", "32M"],
    },
    "esp32c3": {
        "gcc": "riscv32-esp-elf-",
        "flags": ["-march=rv32imc_zicsr_zifencei", "-mabi=ilp32"],
        "qemu": "qemu-system-riscv32",
        "machine": ["-M", "esp32c3"],
    },
}
CFLAGS = ["-O2", "-std=gnu99", "-ffunction-sections", "-fdata-sections", "-ffp-contract=off", "-fno-math-errno",
          "-Wall", "-Wno-unused-function"]


def idf_paths() -> tuple[Path | None, Path | None]:
    candidates = [os.environ.get("TINYML_IDF_PATH"), os.environ.get("IDF_PATH"),
                  str(Path.home() / "esp" / "esp-idf"), "/opt/esp/esp-idf"]
    idf = next((Path(c) for c in candidates if c and (Path(c) / "export.sh").exists()), None)
    tool_candidates = [os.environ.get("TINYML_IDF_TOOLS_PATH"), os.environ.get("IDF_TOOLS_PATH"),
                       str(Path.home() / ".espressif"), "/opt/esp/tools"]
    tools = next((Path(c) for c in tool_candidates if c and Path(c).is_dir()), None)
    return idf, tools


def _prefix() -> str:
    idf, tools = idf_paths()
    parts = []
    if tools:
        parts.append(f"export IDF_TOOLS_PATH={shlex.quote(str(tools))}")
    parts.append(f"export IDF_PATH={shlex.quote(str(idf))}")
    parts.append(f"source {shlex.quote(str(idf / 'export.sh'))} >/dev/null 2>&1")
    return " && ".join(parts) + " && "


@lru_cache(maxsize=4)
def _probe_env(target: str) -> tuple[bool, str, str]:
    idf, _ = idf_paths()
    if idf is None:
        return False, "ESP-IDF not found (set IDF_PATH or install it to ~/esp/esp-idf)", ""
    info = TARGETS[target]
    cmd = _prefix() + f"which idf.py {info['qemu']} {info['gcc']}gcc && {info['gcc']}gcc --version | head -1"
    proc = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        return False, f"ESP-IDF at {idf} is missing idf.py, {info['qemu']} or {info['gcc']}gcc", ""
    lines = proc.stdout.strip().splitlines()
    idf_version = _idf_version(idf)
    return True, "", f"ESP-IDF {idf_version}; {lines[-1] if lines else ''}"


def _idf_version(idf: Path) -> str:
    version_file = idf / "version.txt"
    if version_file.exists():
        return version_file.read_text().strip()
    cmake = idf / "tools" / "cmake" / "version.cmake"
    if cmake.exists():
        import re

        parts = dict(re.findall(r"set\(IDF_VERSION_(MAJOR|MINOR|PATCH) (\d+)\)", cmake.read_text()))
        if len(parts) == 3:
            return f"v{parts['MAJOR']}.{parts['MINOR']}.{parts['PATCH']}"
    return idf.name


class EspIdfBackend:
    name = "espidf"

    def target(self, chip: Chip) -> str:
        assert chip.idf_target in TARGETS, f"unsupported ESP-IDF target {chip.idf_target}"
        return chip.idf_target

    def available(self, chip: Chip) -> tuple[bool, str]:
        ok, why, _ = _probe_env(self.target(chip))
        return ok, why

    def toolchain_version(self, chip: Chip) -> str:
        return _probe_env(self.target(chip))[2]

    def emulator_name(self, chip: Chip) -> str:
        return f"Espressif QEMU -M {self.target(chip)} (-icount)"

    def shell(self, command: str, cwd: Path, timeout: int = 900) -> subprocess.CompletedProcess:
        return run(_prefix() + command, cwd=cwd, timeout=timeout, shell=True)

    # --- measuring -----------------------------------------------------------------------
    def measure(self, chip: Chip, gen_dir: Path, work: Path, window_bytes: int) -> SizeReport:
        info = TARGETS[self.target(chip)]
        gcc, size = info["gcc"] + "gcc", info["gcc"] + "size"
        out = work / "probe"
        out.mkdir(parents=True, exist_ok=True)
        flags = " ".join(CFLAGS + info["flags"])
        inc = f"-I{shlex.quote(str(gen_dir))} -I{shlex.quote(str(firmware('common')))}"
        null_port = firmware("common", "port_null.c")
        libc_min = firmware("common", "libc_min.c")
        model_c = gen_dir / "tinyml_model.c"
        probe_c = firmware("common", "probe.c")
        cmds = [
            f"{gcc} {flags} {inc} -fstack-usage -fcallgraph-info=su -c {model_c} -o model.o",
            f"{gcc} {flags} {inc} -c {probe_c} -o probe.o",
            f"{gcc} {flags} {inc} -DTINYML_PROBE_EMPTY -c {probe_c} -o empty.o",
            f"{gcc} {flags} {inc} -c {null_port} -o port.o",
            f"{gcc} {flags} -fno-builtin -fno-tree-loop-distribute-patterns -c {libc_min} -o libc.o",
            f"{gcc} {flags} -nostdlib -Wl,--gc-sections -Wl,-e,harness_main -Wl,-Map=probe.map model.o probe.o port.o libc.o -lgcc -o probe.elf",
            f"{gcc} {flags} -nostdlib -Wl,--gc-sections -Wl,-e,harness_main empty.o port.o libc.o -lgcc -o empty.elf",
            f"{size} -B probe.elf empty.elf",
        ]
        proc = self.shell(" && ".join(cmds), cwd=out, timeout=300)
        rows = [line.split() for line in proc.stdout.strip().splitlines()[-2:]]
        (t1, d1, b1), (t0, d0, b0) = ((int(r[0]), int(r[1]), int(r[2])) for r in rows)
        stack, detail = worst_stack(sorted(out.glob("*.ci")))
        return SizeReport(
            flash_bytes=(t1 + d1) - (t0 + d0),
            ram_static_bytes=(d1 + b1) - (d0 + b0),
            window_bytes=window_bytes,
            stack_bytes=stack,
            toolchain=self.toolchain_version(chip),
            detail={"probe_map": str(out / "probe.map"), **detail},
        )

    # --- emulating -----------------------------------------------------------------------
    def build(self, chip: Chip, gen_dir: Path, golden_header: str, work: Path) -> Path:
        target = self.target(chip)
        project = work / "espidf"
        shutil.copytree(firmware("esp-idf"), project, dirs_exist_ok=True, copy_function=shutil.copy)
        main = project / "main"
        for name in ("tinyml_model.c", "tinyml_model.h"):
            shutil.copy(gen_dir / name, main / name)
        shutil.copy(firmware("common", "harness.c"), main / "harness.c")
        shutil.copy(firmware("common", "tinyml_port.h"), main / "tinyml_port.h")
        (main / "golden_vectors.h").write_text(golden_header)
        marker = project / "build" / ".tinyml_target"
        if not marker.exists() or marker.read_text() != target:
            if (project / "build").exists():
                shutil.rmtree(project / "build")
            (project / "sdkconfig").unlink(missing_ok=True)
            self.shell(f"idf.py -B build set-target {target}", cwd=project)
            marker.write_text(target)
        try:
            self.shell("idf.py -B build build", cwd=project)
        except ToolchainError as exc:
            logs = sorted((project / "build" / "log").glob("idf_py_stdout_output_*"), key=lambda p: p.stat().st_mtime)
            detail = logs[-1].read_text(errors="replace")[-3000:] if logs else ""
            raise ToolchainError(f"ESP-IDF build failed for {target}:\n{detail}") from exc
        build = project / "build"
        flash = build / "qemu_flash.bin"
        # ESP-IDF v5.5 ships esptool 4.x (merge_bin, --fill-flash-size); esptool 5 renamed both.
        self.shell(f"python -m esptool --chip={target} merge_bin --output={flash} --fill-flash-size=4MB @flash_args"
                   f" || python -m esptool --chip={target} merge-bin --output={flash} --pad-to-size=4MB @flash_args",
                   cwd=build, timeout=300)
        efuse = build / "qemu_efuse.bin"
        if not efuse.exists():
            script = (
                "import os,sys; sys.path.insert(0, os.path.join(os.environ['IDF_PATH'], 'tools')); "
                "from idf_py_actions.qemu_ext import QEMU_TARGETS; "
                f"open('qemu_efuse.bin','wb').write(QEMU_TARGETS['{target}'].default_efuse)"
            )
            self.shell(f"python -c {shlex.quote(script)}", cwd=build, timeout=120)
        return build

    def run(self, chip: Chip, gen_dir: Path, golden_header: str, work: Path, timing: bool, parity: bool,
            timeout: int = 300) -> RunResult:
        target = self.target(chip)
        info = TARGETS[target]
        build = self.build(chip, gen_dir, golden_header, work)
        uart = build / "uart.log"
        uart.unlink(missing_ok=True)
        qemu = (
            f"exec {info['qemu']} {' '.join(info['machine'])} -icount shift=0 "
            f"-drive file=qemu_flash.bin,if=mtd,format=raw "
            f"-drive file=qemu_efuse.bin,if=none,format=raw,id=efuse "
            f"-global driver=nvram.{target}.efuse,property=drive,value=efuse "
            f"-global driver=timer.{target}.timg,property=wdt_disable,value=true "
            f"-display none -monitor none -serial file:uart.log"
        )
        start = time.time()
        proc = subprocess.Popen(_prefix() + qemu, cwd=build, shell=True, executable="/bin/bash",
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            text = wait_for_text(uart, "TINYML DONE", proc, timeout)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
        if "TINYML BEGIN" not in text:
            if time.time() - start >= timeout:
                raise TransientError("ESP32 firmware produced no output before the timeout")
            raise ToolchainError("ESP32 firmware did not start in QEMU:\n" + text[-2000:])
        return RunResult(text, ticks_to_instructions(text) if timing else None, time.time() - start)


def ticks_to_instructions(text: str) -> int | None:
    """Convert the harness's counter ticks into instructions using its calibration loop."""
    ticks = cal_instr = cal_ticks = None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "T" and parts[1].isdigit():
            ticks = int(parts[1])
        elif len(parts) >= 3 and parts[0] == "C" and parts[1].isdigit() and parts[2].isdigit():
            cal_instr, cal_ticks = int(parts[1]), int(parts[2])
    if ticks is None:
        return None
    if cal_instr and cal_ticks:
        return int(round(ticks * cal_instr / cal_ticks))
    return ticks

"""Build backends. Each chip names a primary backend and optionally a fallback."""

from __future__ import annotations

import glob
import os
from pathlib import Path

from tinyml.catalog import Chip
from tinyml.toolchains.baremetal import ArmBackend, HostBackend, RiscvBackend
from tinyml.toolchains.common import RunResult, SizeReport, ToolchainError, TransientError
from tinyml.toolchains.espidf import EspIdfBackend


def _extend_path() -> None:
    """Find toolchains installed by scripts/setup_*.sh even if the shell PATH was not updated."""
    patterns = [
        str(Path.home() / ".tinyml" / "toolchains" / "*" / "bin"),
        "/Applications/ArmGNUToolchain/*/arm-none-eabi/bin",
        "/opt/homebrew/bin",
    ]
    current = os.environ.get("PATH", "").split(os.pathsep)
    extra = [d for pattern in patterns for d in sorted(glob.glob(pattern)) if d not in current]
    if extra:
        os.environ["PATH"] = os.pathsep.join(current + extra)


_extend_path()

BACKENDS = {"arm": ArmBackend, "riscv": RiscvBackend, "espidf": EspIdfBackend}


def backend_for(chip: Chip):
    """The first available backend for this chip, or (None, reasons)."""
    reasons = []
    for name in [chip.backend, chip.fallback_backend]:
        if not name:
            continue
        backend = BACKENDS[name]()
        ok, why = backend.available(chip)
        if ok:
            return backend, ""
        reasons.append(f"{name}: {why}")
    return None, "; ".join(reasons)


__all__ = [
    "ArmBackend", "EspIdfBackend", "HostBackend", "RiscvBackend", "RunResult", "SizeReport", "ToolchainError",
    "TransientError", "backend_for",
]

"""What can this machine build and emulate right now?"""

from __future__ import annotations

import shutil
from typing import Any

from tinyml.catalog import get_catalog
from tinyml.config import load_settings
from tinyml.toolchains import BACKENDS, HostBackend
from tinyml.toolchains.common import tool_version

INSTALL_HINTS = {
    "arm": "macOS: brew install --cask gcc-arm-embedded && brew install qemu | Ubuntu: apt install gcc-arm-none-eabi qemu-system-arm",
    "riscv": "macOS: scripts/setup_macos.sh installs the xPack RISC-V GCC | Ubuntu: apt install gcc-riscv64-unknown-elf qemu-system-misc",
    "espidf": "Run scripts/setup_espidf.sh (installs ESP-IDF v5.5.6 and Espressif QEMU into ~/esp)",
}


def report() -> dict[str, Any]:
    host = HostBackend()
    host_ok, host_why = host.available()
    chips = []
    for chip in get_catalog().chips:
        options = []
        for name in [chip.backend, chip.fallback_backend]:
            if not name:
                continue
            backend = BACKENDS[name]()
            ok, why = backend.available(chip)
            options.append({"backend": name, "ok": ok, "why": why,
                            "toolchain": backend.toolchain_version(chip) if ok else None,
                            "hint": None if ok else INSTALL_HINTS[name]})
        usable = next((o for o in options if o["ok"]), None)
        chips.append({"chip": chip.id, "name": chip.name, "ready": usable is not None,
                      "backend": usable["backend"] if usable else None,
                      "fidelity": chip.emulator.fidelity if usable and usable["backend"] == chip.backend else "core",
                      "options": options})
    settings = load_settings()
    claude = shutil.which(settings.assistant.claude_command)
    return {
        "host_compiler": {"ok": host_ok, "why": host_why, "version": host.toolchain_version() if host_ok else None},
        "chips": chips,
        "assistant": {"provider": settings.assistant.provider, "claude_code_found": bool(claude),
                      "claude_version": tool_version(settings.assistant.claude_command) if claude else None},
    }

"""Shared helpers for compiling, measuring and emulating."""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from tinyml.config import firmware_dir

HOST_CFLAGS = ["-O2", "-std=c99", "-ffp-contract=off", "-fno-math-errno", "-Wall", "-Wno-unused-function"]
COMMON_CFLAGS = [
    "-O2", "-std=c99", "-ffunction-sections", "-fdata-sections", "-ffp-contract=off", "-fno-math-errno",
    "-fno-tree-loop-distribute-patterns", "-Wall", "-Wno-unused-function",
]


class ToolchainError(RuntimeError):
    """A build or emulator step failed in a way a retry will not fix."""


class TransientError(RuntimeError):
    """A step failed for a temporary reason (timeout, resource) and may be retried."""


@dataclass
class SizeReport:
    flash_bytes: int
    ram_static_bytes: int
    window_bytes: int
    stack_bytes: int
    toolchain: str
    detail: dict = field(default_factory=dict)

    @property
    def ram_bytes(self) -> int:
        return self.ram_static_bytes + self.stack_bytes


@dataclass
class RunResult:
    output: str
    instructions: int | None
    seconds: float


def run(cmd: list[str] | str, cwd: Path | None = None, timeout: int = 600, shell: bool = False,
        env: dict | None = None) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, shell=shell,
                              env=env, executable="/bin/bash" if shell else None)
    except subprocess.TimeoutExpired as exc:
        raise TransientError(f"timed out after {timeout}s: {cmd if isinstance(cmd, str) else ' '.join(cmd)}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-2500:]
        shown = cmd if isinstance(cmd, str) else " ".join(cmd)
        raise ToolchainError(f"command failed ({proc.returncode}): {shown[:300]}\n{tail}")
    return proc


@lru_cache(maxsize=32)
def tool_version(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        return "missing"
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=20)
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def firmware(*parts: str) -> Path:
    return firmware_dir().joinpath(*parts)


def berkeley_size(size_tool: str, elf: Path) -> tuple[int, int, int]:
    """(text, data, bss) from `size -B`. Flash = text + data, RAM = data + bss."""
    out = run([size_tool, "-B", str(elf)]).stdout.strip().splitlines()
    text, data, bss = (int(v) for v in out[-1].split()[:3])
    return text, data, bss


def symbol_address(nm_tool: str, elf: Path, symbol: str) -> int | None:
    for line in run([nm_tool, str(elf)]).stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == symbol:
            return int(parts[0], 16) & ~1
    return None


_NODE = re.compile(r'node:\s*\{\s*title:\s*"([^"]+)"\s*label:\s*"([^"]*)"')
_EDGE = re.compile(r'edge:\s*\{\s*sourcename:\s*"([^"]+)"\s*targetname:\s*"([^"]+)"')
_BYTES = re.compile(r"(\d+) bytes \((static|dynamic|bounded)")


def worst_stack(ci_files: list[Path], root: str = "model_predict") -> tuple[int, dict]:
    """Worst-case stack depth from GCC -fcallgraph-info=su files, starting at `root`.

    Functions outside the library (compiler helpers) are counted as 64 bytes each.
    """
    usage: dict[str, int] = {}
    edges: dict[str, set[str]] = {}
    dynamic = False
    for path in ci_files:
        text = path.read_text(errors="replace")
        for title, label in _NODE.findall(text):
            name = title.split(":")[-1]
            match = _BYTES.search(label.replace("\\n", "\n"))
            if match:
                usage[name] = int(match.group(1))
                dynamic |= match.group(2) == "dynamic"
            else:
                usage.setdefault(name, -1)
        for src, dst in _EDGE.findall(text):
            edges.setdefault(src.split(":")[-1], set()).add(dst.split(":")[-1])

    memo: dict[str, int] = {}

    def depth(fn: str, seen: frozenset) -> int:
        if fn in memo:
            return memo[fn]
        own = usage.get(fn, -1)
        own = 64 if own < 0 else own
        children = [depth(c, seen | {fn}) for c in edges.get(fn, ()) if c not in seen]
        memo[fn] = own + (max(children) if children else 0)
        return memo[fn]

    if root not in usage:
        return 0, {"note": "call graph not found"}
    return depth(root, frozenset()), {"dynamic_frames": dynamic}


def wait_for_text(path: Path, needle: str, proc: subprocess.Popen, timeout: float) -> str:
    deadline = time.time() + timeout
    text = ""
    while time.time() < deadline:
        if path.exists():
            text = path.read_text(errors="replace")
            if needle in text:
                return text
        if proc.poll() is not None:
            return path.read_text(errors="replace") if path.exists() else text
        time.sleep(0.2)
    return text

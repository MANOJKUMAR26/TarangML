#!/usr/bin/env bash
# One-time setup on macOS (Apple silicon or Intel). Everything installs locally; nothing runs on a server.
#   scripts/setup_macos.sh            # Python app + Arm and RISC-V toolchains + QEMU
#   scripts/setup_macos.sh --espidf   # also ESP-IDF v5.5.6 and Espressif QEMU (about 2 GB)
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v brew >/dev/null; then
  echo "Homebrew is required: https://brew.sh" >&2
  exit 1
fi

echo "==> Python tooling and QEMU"
brew install uv qemu cmake ninja

echo "==> Arm GCC (bare-metal, includes newlib)"
brew list --cask gcc-arm-embedded >/dev/null 2>&1 || brew install --cask gcc-arm-embedded

echo "==> RISC-V GCC (xPack build with 32-bit libraries)"
XPACK_VERSION=14.2.0-3
ARCH=$(uname -m); [ "$ARCH" = "arm64" ] && XARCH=darwin-arm64 || XARCH=darwin-x64
DEST="$HOME/.tinyml/toolchains"
mkdir -p "$DEST"
if [ ! -x "$DEST/xpack-riscv-none-elf-gcc-$XPACK_VERSION/bin/riscv-none-elf-gcc" ]; then
  URL="https://github.com/xpack-dev-tools/riscv-none-elf-gcc-xpack/releases/download/v$XPACK_VERSION/xpack-riscv-none-elf-gcc-$XPACK_VERSION-$XARCH.tar.gz"
  curl -fL "$URL" | tar -xz -C "$DEST"
  xattr -dr com.apple.quarantine "$DEST/xpack-riscv-none-elf-gcc-$XPACK_VERSION" 2>/dev/null || true
fi

echo "==> Python environment"
uv sync

if [ "${1:-}" = "--espidf" ]; then
  scripts/setup_espidf.sh
fi

echo
echo "Done. Check what is ready with:  uv run tinyml doctor"
echo "Start the app with:              uv run tinyml serve"

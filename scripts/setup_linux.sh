#!/usr/bin/env bash
# One-time setup on Ubuntu/Debian.  Add --espidf for ESP-IDF and Espressif QEMU.
set -euo pipefail
cd "$(dirname "$0")/.."
sudo apt-get update
sudo apt-get install -y gcc-arm-none-eabi gcc-riscv64-unknown-elf qemu-system-arm qemu-system-misc \
  cmake ninja-build git curl python3 python3-venv libslirp0 libsdl2-2.0-0 libgcrypt20 libglib2.0-0 libpixman-1-0
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
if [ "${1:-}" = "--espidf" ]; then
  scripts/setup_espidf.sh
fi
echo "Done. Run: uv run tinyml doctor"

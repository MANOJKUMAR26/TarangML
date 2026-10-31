#!/usr/bin/env bash
# ESP-IDF (pinned) plus Espressif's QEMU for ESP32-S3 and ESP32-C3, installed under ~/esp.
set -euo pipefail
IDF_VERSION=v5.5.6
IDF_DIR="${IDF_PATH:-$HOME/esp/esp-idf}"
export IDF_TOOLS_PATH="${IDF_TOOLS_PATH:-$HOME/.espressif}"

if [ "$(uname)" = "Darwin" ]; then
  brew install libgcrypt glib pixman sdl2 libslirp dfu-util
fi
if [ ! -d "$IDF_DIR" ]; then
  mkdir -p "$(dirname "$IDF_DIR")"
  git clone --depth 1 --branch "$IDF_VERSION" --recurse-submodules --shallow-submodules \
    https://github.com/espressif/esp-idf.git "$IDF_DIR"
fi
cd "$IDF_DIR"
./install.sh esp32s3,esp32c3
python3 tools/idf_tools.py install qemu-xtensa qemu-riscv32
echo "ESP-IDF $IDF_VERSION ready in $IDF_DIR (tinyml finds it there automatically)."

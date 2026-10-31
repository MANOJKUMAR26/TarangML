# Pinned toolchain image: every chip's compiler and emulator in one place, so any build can be
# reproduced months later. Works on x86_64 and arm64 hosts (including Docker on Apple silicon).
#   docker build -t tinyml/toolchains:2026.10 -f ci/docker/Dockerfile .
#   docker run --rm -p 8765:8765 -v "$PWD/workspace:/app/workspace" tinyml/toolchains:2026.10
FROM ubuntu:24.04

ARG IDF_VERSION=v5.5.6
ENV DEBIAN_FRONTEND=noninteractive \
    IDF_PATH=/opt/esp/esp-idf \
    IDF_TOOLS_PATH=/opt/esp/tools \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git python3 python3-venv python3-pip cmake ninja-build build-essential \
        gcc-arm-none-eabi gcc-riscv64-unknown-elf qemu-system-arm qemu-system-misc \
        libslirp0 libsdl2-2.0-0 libgcrypt20 libglib2.0-0 libpixman-1-0 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${IDF_VERSION} --recurse-submodules --shallow-submodules \
        https://github.com/espressif/esp-idf.git ${IDF_PATH} \
    && ${IDF_PATH}/install.sh esp32s3,esp32c3 \
    && python3 ${IDF_PATH}/tools/idf_tools.py install qemu-xtensa qemu-riscv32 \
    && rm -rf ${IDF_TOOLS_PATH}/dist

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
WORKDIR /app
COPY . /app
RUN uv sync --frozen || uv sync

ENV TINYML_TOOLCHAIN_IMAGE=tinyml/toolchains:2026.10
EXPOSE 8765
ENTRYPOINT ["tinyml"]
CMD ["serve", "--host", "0.0.0.0", "--no-open"]

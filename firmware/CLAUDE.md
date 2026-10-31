Owner: Dev B. C sources shared by every chip build.

- `common/harness.c` prints parity and timing lines in the exact format of `docs/contracts/golden-vectors.md`. Changing that format means changing `src/tinyml/codegen/golden.py` and `src/tinyml/packaging/check_parity.py` in the same PR.
- `common/probe.c` must stay the smallest program that keeps the model alive; flash and RAM numbers are the difference between the two probe builds.
- One port per platform (`arm/`, `riscv/`, `host/`, `esp-idf/main/port_esp.c`) implements `common/tinyml_port.h`. No printf in bare-metal ports.
- Keep everything C99 and warning-free with `-Wall -Wextra`.

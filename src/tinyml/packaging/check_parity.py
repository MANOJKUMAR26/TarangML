#!/usr/bin/env python3
"""Check harness output against golden.json. Standard library only, so it runs anywhere.

    cc -O2 -ffp-contract=off -I../include -I. ../src/tinyml_model.c parity_main.c port_host.c -o parity
    ./parity | python3 check_parity.py golden.json

On a device, capture the harness output from the serial port and pipe it in the same way.
"""

import json
import struct
import sys


def bits_to_float(text):
    return struct.unpack("<f", struct.pack("<I", int(text, 16)))[0]


def main():
    golden = json.load(open(sys.argv[1]))
    rows = {}
    finished = False
    for line in sys.stdin:
        parts = line.split()
        if parts[:1] == ["W"]:
            rows[int(parts[1])] = (int(parts[2]), bits_to_float(parts[3]), [bits_to_float(p) for p in parts[4:]])
        finished |= "TINYML DONE" in line
    failures = []
    for vec in golden["vectors"]:
        got = rows.get(vec["id"])
        if got is None:
            failures.append(f"window {vec['id']}: no output")
            continue
        label, score, features = got
        if label != vec["expected_label"] and not vec["borderline"]:
            failures.append(f"window {vec['id']}: label {label} != {vec['expected_label']}")
        for j, (c_val, py_val) in enumerate(zip(features, vec["expected_features"])):
            tol = golden["feature_tolerances"][j]
            if abs(c_val - py_val) > tol["atol"] + tol["rtol"] * abs(py_val):
                failures.append(f"window {vec['id']}: {golden['feature_names'][j]} {c_val:.7g} vs {py_val:.7g}")
    if not finished:
        failures.append("harness did not print TINYML DONE")
    if failures:
        print("PARITY FAILED")
        print("\n".join(failures[:20]))
        sys.exit(1)
    print(f"PARITY OK: {len(golden['vectors'])} windows match Python")


if __name__ == "__main__":
    main()

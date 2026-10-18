"""Stage C step 3: bundle the library, wrapper, example, calibration, parity test and report; sign it."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tinyml import __version__
from tinyml.catalog import Chip
from tinyml.codegen import golden as G
from tinyml.config import firmware_dir
from tinyml.contracts import ChipMeasurement, GoldenSet, ProblemSpec, TaggedNumber
from tinyml.packaging import signing

PACKAGE_ROOT = "tinyml"


def _tag(n: TaggedNumber | None, fmt: str = "{:,.0f}") -> str:
    if n is None or n.value is None:
        return "not measured"
    return f"{fmt.format(n.value)} {n.unit} [{n.tag}]"


def report_md(spec: ProblemSpec, artifact: dict[str, Any], chip: Chip, m: ChipMeasurement, parity: dict[str, Any],
              dataset: dict[str, Any], reproducibility: dict[str, Any], version: str,
              below_target: str | None = None) -> str:
    metrics = artifact.get("metrics", {})
    lines = [
        f"# TinyML package: {spec.recipe.replace('_', ' ')} on {chip.name}",
        "",
        f"Version {version}, built {reproducibility['built_at']} by tinyml {__version__}.",
        "",
        "## What you asked for",
        "",
        f"> {spec.description or '(no description given)'}",
        "",
        f"Task: **{spec.task}**. Connectivity: {spec.connectivity}. Power: {spec.power}. Priority: {spec.priority}.",
        "",
        "## Your data",
        "",
        f"- {dataset['stats'].get('recordings', '?')} recordings, {dataset['stats'].get('duration_s', 0) / 60:.1f} minutes, "
        f"{dataset['stats'].get('sample_rate_hz', 0):g} readings per second, axes: {', '.join(dataset['stats'].get('axes', []))}",
        "- Time per kind: " + ", ".join(f"{k} {v / 60:.1f} min" for k, v in dataset["stats"].get("class_seconds", {}).items()),
    ]
    issues = dataset.get("issues", [])
    if issues:
        lines.append("- Checks:")
        lines += [f"  - [{i['severity']}] {i['message']}" for i in issues]
    lines += [
        "",
        "## What the platform chose",
        "",
    ]
    if below_target:
        lines += [f"> **Below the target.** {below_target} You chose to use this model anyway.", ""]
    lines += [
        artifact.get("explanation", ""),
        "",
        f"Algorithm: `{artifact['algorithm']}` (rung {artifact['rung']}, {artifact['quantization']}). "
        f"Window: {artifact['window']['samples']} samples ({artifact['window']['samples'] / artifact['window']['sample_rate_hz'] * 1000:.0f} ms), "
        f"a new prediction every {artifact['window']['hop']} samples.",
        "",
        "Test results on recordings the model never saw:",
        "",
    ]
    if spec.task == "anomaly":
        det = metrics.get("detection_rate")
        far = metrics.get("false_alarm_rate")
        lines.append(f"- Faults caught: {det:.1%}" if det is not None else "- Faults caught: not measured (no fault recordings)")
        lines.append(f"- False alarms: {far:.1%}" if far is not None else "- False alarms: not measured")
    else:
        lines.append(f"- Accuracy: {metrics.get('accuracy', 0):.1%}, average per event: {metrics.get('balanced_accuracy', 0):.1%}")
        for cls, rec in (metrics.get("recall") or {}).items():
            lines.append(f"  - {cls}: {rec:.1%} recognised" if rec is not None else f"  - {cls}: not in test data")
    lines += [
        "",
        f"## On {chip.name}",
        "",
        "Each number says where it came from: **measured** (linker output), **emulated** (emulator plus a "
        "per-core speed factor measured on a real board) or **estimated** (datasheet values or an uncalibrated factor).",
        "",
        f"- Flash used by the model: {_tag(m.flash_bytes)}",
        f"- RAM used by the model (incl. window buffer and stack): {_tag(m.ram_bytes)}",
        f"- Instructions per prediction: {_tag(m.instructions)}",
        f"- Time per prediction: {_tag(m.latency_ms, '{:.3f}')}",
        f"- Energy per prediction: {_tag(m.energy_uj, '{:.2f}')}",
        f"- Battery life, MCU and sensor (radio not included): {_tag(m.battery_days, '{:.0f}')}",
        f"- Chip price: {_tag(m.price_usd, '{:.2f}')}",
        "",
        "## Proof: parity test",
        "",
        f"{parity['windows']} golden windows ran through the compiled firmware in {m.emulator}. "
        f"{parity['label_matches']} of {parity['windows']} answers match Python exactly"
        + (f"; {parity['borderline']} were flagged borderline (a legitimate float difference could flip them)"
           if parity["borderline"] else "")
        + f". Result: **{'PASSED' if parity['passed'] else 'FAILED'}**.",
        "",
        "Re-run it yourself: see `test/README.md`.",
        "",
        "## Reproduce this build",
        "",
    ]
    lines += [f"- {k}: `{v}`" for k, v in reproducibility.items()]
    lines += [
        "",
        "## How to use",
        "",
        "1. Copy `include/` and `src/` into your firmware and compile `tinyml_model.c` with `-O2 -ffp-contract=off`.",
        f"2. Sample at {artifact['window']['sample_rate_hz']:g} readings per second and fill a window of "
        f"{artifact['window']['samples']} samples (axes interleaved: {', '.join(artifact['window']['axes'])}).",
        "3. Call `model_predict(window)`; see `example/main.c`, or `cpp/TinyMLModel.hpp` for C++ with a streaming helper.",
    ]
    if spec.task == "anomaly":
        lines.append("4. On first install, run the calibration routine while the machine runs normally "
                     "(`model_calibrate_begin/add/end`), and store the result with `model_calibration_get`.")
    return "\n".join(lines) + "\n"


def test_readme() -> str:
    return (
        "# Parity test\n\n"
        "Proves this library gives the same answers as the Python model it came from.\n\n"
        "```sh\n"
        "cc -O2 -ffp-contract=off -I../include -I. ../src/tinyml_model.c parity_main.c port_host.c -o parity\n"
        "./parity | python3 check_parity.py golden.json\n"
        "```\n\n"
        "On a device: build `parity_main.c` with your own `tinyml_port.h` implementation (print over UART), "
        "capture the output and pipe it into `check_parity.py`.\n"
    )


def build_package(out_dir: Path, version: str, project_name: str, chip: Chip, gen_dir: Path,
                  artifact: dict[str, Any], golden: GoldenSet, report: str, reproducibility: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in project_name.lower())[:40] or "model"
    zip_path = out_dir / f"tinyml-{safe}-{chip.id}-v{version}.zip"
    fw = firmware_dir()
    files: dict[str, bytes] = {
        "include/tinyml_model.h": (gen_dir / "tinyml_model.h").read_bytes(),
        "src/tinyml_model.c": (gen_dir / "tinyml_model.c").read_bytes(),
        "cpp/TinyMLModel.hpp": (gen_dir / "TinyMLModel.hpp").read_bytes(),
        "example/main.c": (gen_dir / "example_main.c").read_bytes(),
        "test/parity_main.c": (fw / "common" / "harness.c").read_bytes(),
        "test/tinyml_port.h": (fw / "common" / "tinyml_port.h").read_bytes(),
        "test/port_host.c": (fw / "host" / "port_host.c").read_bytes(),
        "test/golden_vectors.h": G.header(golden).encode(),
        "test/golden.json": golden.model_dump_json(by_alias=True, indent=1).encode(),
        "test/check_parity.py": (Path(__file__).with_name("check_parity.py")).read_bytes(),
        "test/README.md": test_readme().encode(),
        "model.json": json.dumps(artifact, indent=1).encode(),
        "report.md": report.encode(),
        "signing_key.pub.pem": signing.public_pem(),
    }
    manifest = {
        "package": f"tinyml-{safe}-{chip.id}",
        "version": version,
        "chip": chip.id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "key_id": signing.key_id(),
        "reproducibility": reproducibility,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())},
    }
    manifest_bytes = json.dumps(manifest, indent=1, sort_keys=True).encode()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(f"{PACKAGE_ROOT}/{name}", data)
        zf.writestr(f"{PACKAGE_ROOT}/manifest.json", manifest_bytes)
        zf.writestr(f"{PACKAGE_ROOT}/manifest.sig", signing.sign(manifest_bytes))
    return zip_path


def verify_package(zip_path: Path) -> tuple[bool, str]:
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        need = {f"{PACKAGE_ROOT}/manifest.json", f"{PACKAGE_ROOT}/manifest.sig", f"{PACKAGE_ROOT}/signing_key.pub.pem"}
        if not need <= names:
            return False, "manifest, signature or public key missing"
        manifest_bytes = zf.read(f"{PACKAGE_ROOT}/manifest.json")
        signature = zf.read(f"{PACKAGE_ROOT}/manifest.sig").decode()
        public = zf.read(f"{PACKAGE_ROOT}/signing_key.pub.pem")
        if not signing.verify(manifest_bytes, signature, public):
            return False, "signature does not match the manifest"
        manifest = json.loads(manifest_bytes)
        for name, digest in manifest["files"].items():
            data = zf.read(f"{PACKAGE_ROOT}/{name}")
            if hashlib.sha256(data).hexdigest() != digest:
                return False, f"{name} was changed after signing"
    trusted = public.strip() == signing.public_pem().strip()
    note = "signed by this machine's key" if trusted else "signed by a different key (check key_id with the sender)"
    return True, f"valid signature, all {len(manifest['files'])} files intact; {note}"

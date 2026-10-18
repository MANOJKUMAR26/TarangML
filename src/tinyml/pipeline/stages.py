"""The three stages of PRD section 7, plus intake and CSV ingest.

Every function is plain, deterministic code that reads and writes the project folder and
the store. The CLI calls them directly; the web app runs them as background jobs.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tinyml import __version__
from tinyml.catalog import Chip, get_catalog
from tinyml.codegen import golden as G
from tinyml.codegen.emit import generate, write
from tinyml.config import load_settings
from tinyml.contracts import ChipMeasurement, GoldenSet, ProblemSpec
from tinyml.data.ingest import file_sha256, read_csvs, summarize
from tinyml.data.persist import load_prepared, save_prepared
from tinyml.data.prepare import prepare
from tinyml.data.roles import ROLES, assign_roles, guess_normal_label
from tinyml.data.windows import make_windows
from tinyml.decide import Decider, Question
from tinyml.features.compute import compute_features
from tinyml.features.spec import decision_tolerance, feature_space, tolerance
from tinyml.intake.parse import parse_use_case
from tinyml.ml.artifact import build_artifact, load, model_from_artifact, save
from tinyml.ml.ladder import RUNG_NAMES, LadderInputs, explain, run_ladder
from tinyml.ml.split import split_windows
from tinyml.packaging.package import build_package, report_md
from tinyml.pipeline.measure import measure_chip
from tinyml.ranking.score import pre_filter, rank
from tinyml.recipes import get_recipe
from tinyml.store import Store, new_id, project_dir
from tinyml.toolchains import HostBackend, TransientError

Log = Callable[[str], None]


class StageError(RuntimeError):
    """A stage cannot continue; the message is written for the user."""


def _noop(msg: str) -> None:
    pass


def _spec(store: Store, project_id: str) -> ProblemSpec:
    project = store.project(project_id)
    if project is None:
        raise StageError(f"Project {project_id} not found")
    if not project["spec"]:
        raise StageError("Describe the use case first")
    return ProblemSpec(**project["spec"])


# ------------------------------------------------------------------------------------- step 1-2
def describe(store: Store, description: str, name: str | None = None, vendor: str | None = None,
             ask_user=None) -> dict[str, Any]:
    project = store.create_project(name or description.strip()[:60] or "Untitled", description, None)
    decider = Decider(store, project["id"], ask_user=ask_user)
    result = parse_use_case(description, decider, vendor)
    store.update_project(project["id"], spec=result.spec.model_dump())
    return {"project": store.project(project["id"]), "questions": result.questions}


def update_spec(store: Store, project_id: str, changes: dict[str, Any]) -> ProblemSpec:
    spec = _spec(store, project_id)
    data = spec.model_dump()
    allowed = {"recipe", "task", "vendor", "connectivity", "power", "budget_usd_per_unit", "priority",
               "max_latency_ms", "description"}
    decider = Decider(store, project_id)
    for decision in store.decisions(project_id):
        field = decision["decision_type"]
        if field in changes and decision["final_source"] != "user" and changes[field] is not None:
            decider.record_user_choice(decision["id"], str(changes[field]))
    data.update({k: v for k, v in changes.items() if k in allowed})
    new = ProblemSpec(**data)
    get_recipe(new.recipe)
    store.update_project(project_id, spec=new.model_dump())
    return new


# ------------------------------------------------------------------------------------- step 3
def ingest(store: Store, project_id: str, files: list[Path], sample_rate_hz: float | None = None,
           ask_user=None, roles_override: dict[str, str] | None = None, normal_label: str | None = None,
           log: Log = _noop) -> dict[str, Any]:
    spec = _spec(store, project_id)
    recipe = get_recipe(spec.recipe)
    raw_dir = project_dir(project_id, "data", "raw")
    stored: list[Path] = []
    for src in files:
        dest = raw_dir / Path(src).name
        if Path(src).resolve() != dest.resolve():
            shutil.copy2(src, dest)
        stored.append(dest)
    return _prepare_dataset(store, project_id, spec, recipe, stored, sample_rate_hz, ask_user, roles_override,
                            normal_label, log)


def _prepare_dataset(store, project_id, spec, recipe, stored, sample_rate_hz, ask_user, roles_override,
                     normal_label, log) -> dict[str, Any]:
    decider = Decider(store, project_id, ask_user=ask_user)
    log(f"Reading {len(stored)} file(s)")
    df = read_csvs(stored)
    summary = summarize(df, [p.name for p in stored], spec.description)
    role_answers = assign_roles(summary, decider)
    roles = {name: ans.choice for name, ans in role_answers.items()}
    questions = [
        {"field": f"role:{name}", "decision_id": ans.decision_id, "question": ans.follow_up or f"What is column '{name}'?",
         "guess": ans.choice, "confidence": round(ans.confidence, 2), "options": ROLES, "prompt": ans.prompt}
        for name, ans in role_answers.items() if ans.needs_user
    ]
    if roles_override:
        for name, role in roles_override.items():
            if name in roles and roles[name] != role:
                did = role_answers[name].decision_id
                if did:
                    decider.record_user_choice(did, role)
            roles[name] = role
        questions = [q for q in questions if q["field"].split(":", 1)[1] not in roles_override]

    label_cols = [c for c, r in roles.items() if r == "label"]
    labels = sorted(df[label_cols[0]].dropna().astype(str).unique().tolist()) if label_cols else []
    if spec.task == "anomaly" and labels:
        if normal_label is None:
            guess = guess_normal_label(labels, recipe.normal_label_hints)
            answer = decider.decide(Question("normal_label", label_cols[0].lower(),
                                             "Which label means the machine is working normally?", labels,
                                             {"labels": labels[:20]}, spec.description), guess)
            normal_label = answer.choice
            if answer.needs_user:
                questions.append({"field": "normal_label", "decision_id": answer.decision_id,
                                  "question": "Which label means normal operation?", "guess": answer.choice,
                                  "confidence": round(answer.confidence, 2), "options": labels, "prompt": answer.prompt})
    elif spec.task == "anomaly":
        normal_label = None

    log("Checking the data")
    data = prepare(df, roles, recipe, spec.task, decider, normal_label, sample_rate_hz)
    save_prepared(data, project_dir(project_id, "data"))
    row = store.save_dataset({
        "project_id": project_id,
        "files": [p.name for p in stored],
        "sha256": file_sha256(stored),
        "summary": summary.model_dump(),
        "roles": roles,
        "label_map": {"normal_label": normal_label, "labels": labels, "sample_rate_hz": sample_rate_hz},
        "issues": [i.model_dump() for i in data.issues],
        "stats": data.stats,
    })
    store.update_project(project_id, status="failed" if data.blocked else "data_ready")
    preview = _preview(data)
    (project_dir(project_id, "data") / "preview.json").write_text(json.dumps(preview))
    return {**row, "questions": questions, "blocked": data.blocked, "preview": preview}


def update_roles(store: Store, project_id: str, roles: dict[str, str], normal_label: str | None = None,
                 sample_rate_hz: float | None = None, log: Log = _noop) -> dict[str, Any]:
    spec = _spec(store, project_id)
    dataset = store.latest_dataset(project_id)
    if dataset is None:
        raise StageError("Upload a CSV first")
    stored = [project_dir(project_id, "data", "raw") / name for name in dataset["files"]]
    return _prepare_dataset(store, project_id, spec, get_recipe(spec.recipe), stored, sample_rate_hz, None, roles,
                            normal_label, log)


def _preview(data, points: int = 400) -> dict[str, Any]:
    """Two seconds of each kind of data (first occurrence of each label), downsampled for the signal chart."""
    out = []
    seen: set[str] = set()
    span = int(data.rate * 2)
    for rec in data.recordings:
        labels = rec.labels if rec.labels is not None else np.array(["data"] * len(rec.signals))
        for label in dict.fromkeys(labels.tolist()):
            if label in seen or len(out) >= 4:
                continue
            start = int(np.argmax(labels == label))
            end = min(len(rec.signals), start + span)
            if end - start < 8:
                continue
            seen.add(label)
            step = max(1, (end - start) // points)
            out.append({"recording": rec.id, "label": str(label), "rate": data.rate / step,
                        "series": {a: rec.signals[start:end:step, i].round(5).tolist() for i, a in enumerate(data.axes)}})
    return {"axes": data.axes, "samples": out}


# ------------------------------------------------------------------------------------- stage A
@dataclass
class _Training:
    """Everything a model needs to be scored, turned into C and checked; rebuilt deterministically from the data."""

    spec: ProblemSpec
    recipe: Any
    dataset: dict[str, Any]
    ws: Any
    defs: list[Any]
    F: np.ndarray
    tols: list[dict[str, float]]
    split: Any
    inp: LadderInputs


def _training_inputs(store: Store, project_id: str, log: Log) -> _Training:
    spec = _spec(store, project_id)
    recipe = get_recipe(spec.recipe)
    dataset = store.latest_dataset(project_id)
    if dataset is None:
        raise StageError("Upload a CSV first")
    data = load_prepared(project_dir(project_id, "data"))
    if data.blocked:
        blockers = "; ".join(i.message for i in data.issues if i.severity == "block")
        raise StageError(f"The data has problems that stop training: {blockers}")
    normal = data.normal_label
    log("Cutting windows")
    ws = make_windows(data, recipe, default_label=normal or "normal")
    if len(ws) < 10:
        raise StageError(f"Only {len(ws)} windows could be cut from the data; upload more.")
    defs = feature_space(ws.axes, recipe.include_magnitude, recipe.feature_families)
    log(f"Computing {len(defs)} features on {len(ws)} windows")
    F, unstable = compute_features(ws.X, defs, ws.rate, return_unstable=True)
    scale = {i: float(np.abs(ws.X[:, :, i]).max()) for i in range(len(ws.axes))}
    scale[-1] = float(np.sqrt(np.sum(ws.X.astype(np.float64) ** 2, axis=2)).max()) if len(ws.axes) > 1 else scale[0]
    tols = [tolerance(d, ws.window_samples, ws.rate, scale[d.channel_index]) for d in defs]
    decision_tols = [decision_tolerance(d, ws.window_samples, t) for d, t in zip(defs, tols, strict=True)]
    split = split_windows(ws, spec.task, normal, spec.seed)
    log(f"Split: {split.method} ({int(split.train.sum())} train, {int(split.test.sum())} test windows)")
    inp = LadderInputs(F, defs, ws.labels, split, spec.task, normal, recipe, spec.seed,
                       np.array([t["rtol"] for t in decision_tols]), np.array([t["atol"] for t in decision_tols]),
                       unstable)
    return _Training(spec, recipe, dataset, ws, defs, F, tols, split, inp)


def train(store: Store, project_id: str, log: Log = _noop) -> dict[str, Any]:
    """Try every rung, keep every model, and put the recommended one to use.

    When no candidate meets the target nothing is put to use; the user picks one with `select_model`
    (knowing it is below the target) or improves the data.
    """
    t = _training_inputs(store, project_id, log)
    result = run_ladder(t.inp, log)

    candidates_dir = project_dir(project_id, "models", "candidates")
    for old in candidates_dir.glob("*.json"):
        old.unlink()
    rows = []
    for cand in result.candidates:
        cand_id = new_id("cand")
        path = None
        if cand.model is not None:
            artifact = _artifact(t, cand.model, explain(cand, t.inp), cand.metrics)
            path = str(save(artifact, candidates_dir / f"{cand_id}.json"))
        rows.append({"id": cand_id, "rung": cand.rung, "algorithm": cand.algorithm,
                     "metrics": {**cand.metrics, "size_bytes": cand.size_bytes}, "passed": cand.passed,
                     "selected": False, "note": cand.note, "artifact_path": path})
    store.replace_candidates(project_id, rows)
    by_cand = {id(c): row["id"] for c, row in zip(result.candidates, rows, strict=True)}
    recommended = by_cand[id(result.chosen)] if result.chosen is not None else None
    best = by_cand[id(result.best)] if result.best is not None else None

    models_dir = project_dir(project_id, "models")
    for stale in ("model.json", "golden.json", "host_parity.json"):
        (models_dir / stale).unlink(missing_ok=True)
    shutil.rmtree(models_dir / "c", ignore_errors=True)
    (models_dir / "training.json").write_text(json.dumps({
        "split": t.split.method, "windows": len(t.ws), "window_samples": t.ws.window_samples, "hop": t.ws.hop,
        "classes": result.classes, "data_request": result.data_request, "explanation": "",
        "recommended": recommended, "best": best, "viz": {}, "rung_names": RUNG_NAMES,
    }, indent=1, default=float))
    summaries = [{"id": row["id"], **c.summary()} for c, row in zip(result.candidates, rows, strict=True)]
    store.replace_chip_results(project_id, [])
    if recommended is None:
        store.update_project(project_id, status="needs_choice")
        log(result.data_request or "No model met the target.")
        return {"chosen": None, "recommended": None, "best": best, "data_request": result.data_request,
                "candidates": summaries}
    return {**_activate(store, project_id, t, recommended, log), "recommended": recommended, "best": best,
            "candidates": summaries}


def select_model(store: Store, project_id: str, candidate_id: str, log: Log = _noop) -> dict[str, Any]:
    """Put a candidate the user picked to use: C code, golden windows and host parity, as for the recommendation."""
    if not any(c["id"] == candidate_id for c in store.candidates(project_id)):
        raise StageError("That model is not one of this project's candidates; train again.")
    t = _training_inputs(store, project_id, log)
    store.replace_chip_results(project_id, [])
    return _activate(store, project_id, t, candidate_id, log)


def _artifact(t: _Training, model, explanation: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return build_artifact(model, t.defs, t.spec.task, t.ws.window_samples, t.ws.hop, t.ws.rate, t.ws.axes,
                          t.recipe.id, t.recipe.version, t.spec.seed, t.dataset["sha256"], explanation, metrics)


def _activate(store: Store, project_id: str, t: _Training, candidate_id: str, log: Log) -> dict[str, Any]:
    rows = store.candidates(project_id)
    row = next(r for r in rows if r["id"] == candidate_id)
    if not row["artifact_path"]:
        raise StageError(f"{row['algorithm'].replace('_', ' ')} was not trained ({row['note']}); pick another model.")
    artifact = load(Path(row["artifact_path"]))
    model = model_from_artifact(artifact)
    model.feature_idx = [next(i for i, d in enumerate(t.defs) if d.name == f["name"]) for f in artifact["features"]]
    label = row["algorithm"].replace("_", " ")
    log(f"Using {label} (rung {row['rung']})" + ("" if row["passed"] else ", below the target"))

    models_dir = project_dir(project_id, "models")
    save(artifact, models_dir / "model.json")
    log("Choosing golden test windows")
    settings = load_settings()
    test_idx = np.where(t.split.test)[0]
    F_model = t.F[:, model.feature_idx]
    pred_test = model.predict(F_model[test_idx], t.inp.tol(model.feature_idx, test_idx))
    window_bytes = t.ws.window_samples * len(t.ws.axes) * 4
    pick = test_idx[G.select_windows(t.ws.labels[test_idx], pred_test, settings.max_golden_windows, window_bytes,
                                     t.spec.seed)]
    pred = model.predict(F_model[pick], t.inp.tol(model.feature_idx, pick))
    golden = G.build_golden(t.ws.X[pick], t.ws.recordings[pick], t.ws.labels[pick], F_model[pick], pred, model,
                            [f["name"] for f in artifact["features"]],
                            [t.tols[i] for i in model.feature_idx], t.ws.axes)
    (models_dir / "golden.json").write_text(golden.model_dump_json(by_alias=True, indent=1))

    log("Generating C code")
    gen_dir = models_dir / "c"
    shutil.rmtree(gen_dir, ignore_errors=True)
    write(generate(artifact, _title(t.spec)), gen_dir)
    host = HostBackend()
    ok, why = host.available()
    host_parity: dict[str, Any] = {"skipped": why}
    if ok:
        log("Checking the C code against Python on this computer")
        out = host.run_host(gen_dir, G.header(golden), project_dir(project_id, "work", "host"))
        rows_out, _, finished = G.parse_output(out.output)
        host_parity = G.compare(golden, rows_out, finished)
        if not host_parity["passed"]:
            raise StageError("The generated C code does not match Python on this computer: "
                             + "; ".join(f["reason"] for f in host_parity["failures"][:3]))
    (models_dir / "host_parity.json").write_text(json.dumps(host_parity, indent=1))

    store.select_candidate(project_id, candidate_id)
    training_path = models_dir / "training.json"
    training = json.loads(training_path.read_text())
    training.update(explanation=artifact["explanation"], selected=candidate_id,
                    viz=_visual_data(t.inp, model, t.defs))
    training_path.write_text(json.dumps(training, indent=1, default=float))
    store.update_project(project_id, status="trained")
    chosen = {"id": candidate_id, "rung": row["rung"], "algorithm": row["algorithm"], "metrics": row["metrics"],
              "passed": bool(row["passed"]), "note": row["note"]}
    return {"chosen": chosen, "explanation": artifact["explanation"], "host_parity": host_parity,
            "golden_windows": len(golden.vectors)}


def _title(spec: ProblemSpec) -> str:
    return f"{spec.recipe.replace('_', ' ').title()} ({spec.task})"


def _visual_data(inp: LadderInputs, model, defs) -> dict[str, Any]:
    """Small data for the UI: distribution of the model's first feature for each class, plus thresholds."""
    j = model.feature_idx[0]
    values = inp.F[:, j]
    lo, hi = float(np.percentile(values, 0.5)), float(np.percentile(values, 99.5))
    if hi <= lo:
        hi = lo + 1e-9
    bins = np.linspace(lo, hi, 31)
    hist = {}
    for c, name in enumerate(inp.classes):
        mask = inp.y == c
        if mask.any():
            counts, _ = np.histogram(np.clip(values[mask], lo, hi), bins=bins)
            hist[name] = counts.tolist()
    thresholds = []
    if model.algorithm == "threshold_rule":
        thresholds = [c["threshold"] for c in model.conditions if c["feature"] == 0]
    from tinyml.features.spec import describe

    return {"feature": defs[j].name, "feature_label": describe(defs[j]), "bins": bins.tolist(),
            "hist": hist, "thresholds": thresholds}


# ------------------------------------------------------------------------------------- stage B
def _load_model_files(project_id: str) -> tuple[dict[str, Any], GoldenSet, Path]:
    models_dir = project_dir(project_id, "models")
    if not (models_dir / "model.json").exists():
        raise StageError("Train a model first")
    artifact = load(models_dir / "model.json")
    golden = GoldenSet.model_validate_json((models_dir / "golden.json").read_text())
    return artifact, golden, models_dir / "c"


def _artifact_hash(project_id: str) -> str:
    """Changes whenever the model, the golden set, the firmware sources or tinyml itself change."""
    models_dir = project_dir(project_id, "models")
    digest = hashlib.sha256((models_dir / "model.json").read_bytes())
    digest.update((models_dir / "golden.json").read_bytes())
    digest.update(_firmware_hash().encode())
    digest.update(__version__.encode())
    return digest.hexdigest()[:16]


def _firmware_hash() -> str:
    """Hash of everything that affects a chip measurement: firmware sources and the build/measure code."""
    from tinyml.config import PACKAGE_DIR, firmware_dir

    roots = [firmware_dir(), PACKAGE_DIR / "toolchains", PACKAGE_DIR / "codegen"]
    digest = hashlib.sha256((PACKAGE_DIR / "pipeline" / "measure.py").read_bytes())
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def _measure_cached(store: Store, project_id: str, chip: Chip, gen_dir: Path, golden: GoldenSet, window_bytes: int,
                    hop_s: float, parity_windows: int | None, log: Log) -> ChipMeasurement:
    sensor_ma = get_recipe(_spec(store, project_id).recipe).sensor_ma
    key = f"measure:{project_id}:{chip.id}:{_artifact_hash(project_id)}:{parity_windows}"
    cached = store.cached_step(key)
    if cached:
        log(f"{chip.id}: reusing earlier measurement")
        return ChipMeasurement(**cached)
    work = project_dir(project_id, "work", chip.id)
    settings = load_settings()
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            log(f"{chip.id}: building and running in the emulator" + (f" (attempt {attempt})" if attempt > 1 else ""))
            m = measure_chip(chip, gen_dir, golden, work, window_bytes, hop_s, parity_windows,
                             settings.emulator_timeout_s, sensor_ma)
            break
        except TransientError as exc:
            if attempt == attempts:
                m = ChipMeasurement(chip_id=chip.id, status="failed", backend=chip.backend, error=str(exc))
        except Exception as exc:  # a build error on one chip must not stop the others
            m = ChipMeasurement(chip_id=chip.id, status="failed", backend=chip.backend, error=str(exc)[:1500])
            break
    if m.status == "ok":
        store.cache_step(key, "measure", m.model_dump())
    return m


def rank_chips(store: Store, project_id: str, chip_ids: list[str] | None = None, log: Log = _noop) -> dict[str, Any]:
    spec = _spec(store, project_id)
    artifact, golden, gen_dir = _load_model_files(project_id)
    catalog = get_catalog()
    chips = [catalog.chip(c) for c in chip_ids] if chip_ids else list(catalog.chips)
    window = artifact["window"]
    window_bytes = window["samples"] * len(window["axes"]) * 4
    hop_s = window["hop"] / window["sample_rate_hz"]
    measurements: dict[str, ChipMeasurement] = {}
    to_build = []
    for chip in chips:
        reason = pre_filter(chip, spec)
        if reason:
            measurements[chip.id] = ChipMeasurement(chip_id=chip.id, status="filtered", backend=chip.backend,
                                                    filtered_reason=reason)
            log(f"{chip.id}: filtered out ({reason})")
        else:
            to_build.append(chip)
    settings = load_settings()
    with ThreadPoolExecutor(max_workers=max(1, settings.parallel_chip_builds)) as pool:
        futures = {chip.id: pool.submit(_measure_cached, store, project_id, chip, gen_dir, golden, window_bytes,
                                        hop_s, 4, log) for chip in to_build}
        for chip_id, fut in futures.items():
            measurements[chip_id] = fut.result()
    ranked, filtered = rank(chips, measurements, spec, hop_s * 1000.0)
    rows = []
    for r in ranked:
        rows.append({"candidate_id": artifact["algorithm"], "chip_id": r.chip_id, "measurement": r.measurement.model_dump(),
                     "score": r.score, "rank": r.rank, "reasons": r.reasons, "tradeoffs": r.tradeoffs})
    for chip in chips:
        if chip.id in filtered:
            rows.append({"candidate_id": artifact["algorithm"], "chip_id": chip.id,
                         "measurement": measurements[chip.id].model_dump(), "score": None, "rank": None,
                         "reasons": [], "tradeoffs": []})
    store.replace_chip_results(project_id, rows)
    store.update_project(project_id, status="ranked")
    return {"ranked": [r.model_dump() for r in ranked], "filtered": filtered}


# ------------------------------------------------------------------------------------- stage C
def build_for_chip(store: Store, project_id: str, chip_id: str, log: Log = _noop) -> dict[str, Any]:
    spec = _spec(store, project_id)
    project = store.project(project_id)
    artifact, golden, gen_dir = _load_model_files(project_id)
    chip = get_catalog().chip(chip_id)
    window = artifact["window"]
    window_bytes = window["samples"] * len(window["axes"]) * 4
    hop_s = window["hop"] / window["sample_rate_hz"]

    log(f"Final build for {chip.name} with the pinned toolchain and seed {spec.seed}")
    m = _measure_cached(store, project_id, chip, gen_dir, golden, window_bytes, hop_s, None, log)
    if m.status != "ok":
        raise StageError(f"Could not build for {chip.name}: {m.error}")
    parity = m.parity_detail
    over = _over_budget(chip, m)
    passed = bool(m.parity_ok) and over is None
    previous = [b for b in store.builds(project_id) if b["chip_id"] == chip_id]
    version = f"1.0.{len(previous)}"
    dataset = store.latest_dataset(project_id) or {}
    reproducibility = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tinyml_version": __version__,
        "dataset_sha256": dataset.get("sha256"),
        "recipe": f"{artifact['recipe']['id']} v{artifact['recipe']['version']}",
        "seed": artifact["seed"],
        "toolchain": m.toolchain,
        "toolchain_image": os.environ.get("TINYML_TOOLCHAIN_IMAGE") or f"local install (pinned image: {chip.toolchain_image})",
        "emulator": m.emulator,
        "model_sha256": hashlib.sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest(),
    }
    package_path = None
    if passed:
        log("Parity passed; packaging and signing")
        report = report_md(spec, artifact, chip, m, parity, dataset, reproducibility, version,
                           _below_target(store, project_id))
        package_path = build_package(project_dir(project_id, "packages"), version, project["name"], chip, gen_dir,
                                     artifact, golden, report, reproducibility)
        (project_dir(project_id, "packages") / f"report-{chip.id}-v{version}.md").write_text(report)
    else:
        log("Blocked: " + (over or "C output did not match Python in the emulator"))
    row = store.save_build({
        "project_id": project_id, "chip_id": chip_id, "version": version,
        "package_path": str(package_path) if package_path else None,
        "parity": {**parity, "measurement": m.model_dump(), "blocked_reason": over}, "passed": passed,
        "reproducibility": reproducibility,
    })
    store.mark_build_outcome(project_id, passed)
    store.update_project(project_id, status="built" if passed else "failed")
    return {**row, "passed": passed}


def _below_target(store: Store, project_id: str) -> str | None:
    selected = store.selected_candidate(project_id)
    if selected is None or selected["passed"]:
        return None
    training = json.loads((project_dir(project_id, "models") / "training.json").read_text())
    return training.get("data_request") or "The model did not meet the recipe's target on the test recordings."


def _over_budget(chip: Chip, m: ChipMeasurement) -> str | None:
    from tinyml.ranking.score import MAX_MEMORY_SHARE

    if m.flash_bytes and m.flash_bytes.value > MAX_MEMORY_SHARE * chip.flash_kb * 1024:
        return "the model uses more than half of the chip's flash"
    if m.ram_bytes and m.ram_bytes.value > MAX_MEMORY_SHARE * chip.ram_kb * 1024:
        return "the model uses more than half of the chip's RAM"
    return None


def model_for_project(project_id: str):
    artifact, _, _ = _load_model_files(project_id)
    return model_from_artifact(artifact)

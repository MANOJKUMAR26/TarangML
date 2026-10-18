"""Local REST API and the web UI. Binds to 127.0.0.1 by default; nothing leaves the machine."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tinyml import __version__
from tinyml.catalog import get_catalog
from tinyml.config import AssistantSettings, Settings, load_settings, ui_dir
from tinyml.datasets import synthetic
from tinyml.decide import Decider
from tinyml.decide.providers import ClaudeCodeProvider, ProviderUnavailable
from tinyml.pipeline import stages
from tinyml.pipeline.jobs import JobRunner
from tinyml.recipes import load_recipes
from tinyml.store import get_store, project_dir
from tinyml.toolchains import doctor

runner: JobRunner | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runner
    runner = JobRunner(get_store())
    runner.resume()
    yield


app = FastAPI(title="tinyml", version=__version__, lifespan=lifespan)


def _store():
    return get_store()


def _project(project_id: str) -> dict[str, Any]:
    project = _store().project(project_id)
    if project is None:
        raise HTTPException(404, "project not found")
    return project


def _stage(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except stages.StageError as exc:
        raise HTTPException(400, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc


# ---------------------------------------------------------------------------------------- system
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "version": __version__}


@app.get("/api/doctor")
def get_doctor() -> dict[str, Any]:
    return doctor.report()


class SettingsBody(BaseModel):
    provider: str | None = None
    claude_command: str | None = None
    claude_model: str | None = None


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    return asdict(load_settings())


@app.put("/api/settings")
def put_settings(body: SettingsBody) -> dict[str, Any]:
    settings = load_settings()
    assistant = asdict(settings.assistant)
    if body.provider is not None:
        if body.provider not in ("off", "claude_code", "claude_chat"):
            raise HTTPException(400, "provider must be off, claude_code or claude_chat")
        assistant["provider"] = body.provider
    if body.claude_command:
        assistant["claude_command"] = body.claude_command
    if body.claude_model is not None:
        assistant["claude_model"] = body.claude_model or None
    new = Settings(**{**asdict(settings), "assistant": AssistantSettings(**assistant)})
    new.save()
    return asdict(new)


# ---------------------------------------------------------------------------------------- catalog API
@app.get("/api/vendors")
def vendors() -> list[dict[str, Any]]:
    return [v.model_dump() for v in get_catalog().vendors]


@app.get("/api/vendors/{vendor_id}/chips")
def vendor_chips(vendor_id: str) -> list[dict[str, Any]]:
    return [c.model_dump() for c in get_catalog().query(vendor=vendor_id)]


@app.get("/api/chips")
def chips(radio: str | None = None, max_price: float | None = None, vendor: str | None = None,
          fpu: bool | None = None) -> list[dict[str, Any]]:
    return [c.model_dump() for c in get_catalog().query(vendor=vendor, radio=radio, max_price=max_price, fpu=fpu)]


@app.get("/api/chips/{chip_id}")
def chip(chip_id: str) -> dict[str, Any]:
    try:
        return get_catalog().chip(chip_id).model_dump()
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/recipes")
def recipes() -> list[dict[str, Any]]:
    return [r.model_dump() for r in load_recipes().values()]


# ---------------------------------------------------------------------------------------- projects
class DescribeBody(BaseModel):
    description: str
    name: str | None = None
    vendor: str | None = None


@app.get("/api/projects")
def list_projects() -> list[dict[str, Any]]:
    return _store().projects()


@app.post("/api/projects")
def create_project(body: DescribeBody) -> dict[str, Any]:
    if not body.description.strip():
        raise HTTPException(400, "Describe what the device should detect")
    return stages.describe(_store(), body.description, body.name, body.vendor)


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    store = _store()
    project = _project(project_id)
    dataset = store.latest_dataset(project_id)
    training_path = project_dir(project_id, "models") / "training.json"
    preview_path = project_dir(project_id, "data") / "preview.json"
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for d in store.decisions(project_id):  # newest first
        latest.setdefault((d["decision_type"], d["decision_key"]), d)
    pending = [d for d in latest.values() if d["final_choice"] is None]
    return {
        "preview": json.loads(preview_path.read_text()) if preview_path.exists() else None,
        "pending_decisions": pending,
        "project": project,
        "dataset": dataset,
        "candidates": store.candidates(project_id),
        "training": json.loads(training_path.read_text()) if training_path.exists() else None,
        "chip_results": store.chip_results(project_id),
        "builds": store.builds(project_id),
        "jobs": store.jobs(project_id)[:10],
    }


@app.patch("/api/projects/{project_id}/spec")
def patch_spec(project_id: str, changes: dict[str, Any]) -> dict[str, Any]:
    _project(project_id)
    return _stage(stages.update_spec, _store(), project_id, changes).model_dump()


@app.post("/api/projects/{project_id}/datasets")
async def upload(project_id: str, files: list[UploadFile] = File(...),
                 sample_rate_hz: float | None = Form(None)) -> dict[str, Any]:
    _project(project_id)
    tmp = Path(tempfile.mkdtemp(prefix="tinyml-upload-"))
    paths = []
    try:
        for upload_file in files:
            name = Path(upload_file.filename or "data.csv").name
            if not name.lower().endswith(".csv"):
                raise HTTPException(400, f"{name} is not a CSV file")
            dest = tmp / name
            with dest.open("wb") as fh:
                shutil.copyfileobj(upload_file.file, fh)
            paths.append(dest)
        return _stage(stages.ingest, _store(), project_id, paths, sample_rate_hz)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class DemoBody(BaseModel):
    dataset: str


@app.post("/api/projects/{project_id}/demo-data")
def demo_data(project_id: str, body: DemoBody) -> dict[str, Any]:
    _project(project_id)
    if body.dataset not in synthetic.GENERATORS:
        raise HTTPException(400, f"unknown demo dataset; choose from {list(synthetic.GENERATORS)}")
    folder = Path(tempfile.mkdtemp(prefix="tinyml-demo-"))
    path = synthetic.GENERATORS[body.dataset](folder / f"{body.dataset}_demo.csv")
    try:
        return _stage(stages.ingest, _store(), project_id, [path])
    finally:
        shutil.rmtree(folder, ignore_errors=True)


class RolesBody(BaseModel):
    roles: dict[str, str]
    normal_label: str | None = None
    sample_rate_hz: float | None = None


@app.put("/api/projects/{project_id}/datasets/roles")
def put_roles(project_id: str, body: RolesBody) -> dict[str, Any]:
    _project(project_id)
    return _stage(stages.update_roles, _store(), project_id, body.roles, body.normal_label, body.sample_rate_hz)


# ---------------------------------------------------------------------------------------- jobs
class RankBody(BaseModel):
    chips: list[str] | None = None


class BuildBody(BaseModel):
    chip_id: str


def _runner() -> JobRunner:
    assert runner is not None
    return runner


@app.post("/api/projects/{project_id}/train")
def train(project_id: str) -> dict[str, Any]:
    _project(project_id)
    return _runner().submit("train", project_id)


class SelectBody(BaseModel):
    candidate_id: str


@app.post("/api/projects/{project_id}/select")
def select(project_id: str, body: SelectBody) -> dict[str, Any]:
    _project(project_id)
    if not any(c["id"] == body.candidate_id for c in _store().candidates(project_id)):
        raise HTTPException(404, "candidate not found")
    return _runner().submit("select", project_id, {"candidate_id": body.candidate_id})


@app.post("/api/projects/{project_id}/rank")
def rank(project_id: str, body: RankBody) -> dict[str, Any]:
    _project(project_id)
    return _runner().submit("rank", project_id, {"chips": body.chips})


@app.post("/api/projects/{project_id}/build")
def build(project_id: str, body: BuildBody) -> dict[str, Any]:
    _project(project_id)
    get_catalog().chip(body.chip_id)
    return _runner().submit("build", project_id, {"chip_id": body.chip_id})


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict[str, Any]:
    row = _store().job(job_id)
    if row is None:
        raise HTTPException(404, "job not found")
    return row


# ---------------------------------------------------------------------------------------- builds
def _build(build_id: str) -> dict[str, Any]:
    row = _store().build(build_id)
    if row is None:
        raise HTTPException(404, "build not found")
    return row


@app.get("/api/builds/{build_id}/download")
def download(build_id: str) -> FileResponse:
    row = _build(build_id)
    if not row["package_path"] or not Path(row["package_path"]).exists():
        raise HTTPException(404, "this build has no package (parity failed or file removed)")
    return FileResponse(row["package_path"], filename=Path(row["package_path"]).name, media_type="application/zip")


@app.get("/api/builds/{build_id}/report", response_class=PlainTextResponse)
def report(build_id: str) -> str:
    row = _build(build_id)
    if not row["package_path"]:
        raise HTTPException(404, "no report: this build was blocked")
    with zipfile.ZipFile(row["package_path"]) as zf:
        return zf.read("tinyml/report.md").decode()


@app.get("/api/builds/{build_id}/files/{name}", response_class=PlainTextResponse)
def build_file(build_id: str, name: str) -> str:
    row = _build(build_id)
    allowed = {"tinyml_model.h": "include/tinyml_model.h", "tinyml_model.c": "src/tinyml_model.c",
               "TinyMLModel.hpp": "cpp/TinyMLModel.hpp", "main.c": "example/main.c"}
    if name not in allowed or not row["package_path"]:
        raise HTTPException(404, "file not available")
    with zipfile.ZipFile(row["package_path"]) as zf:
        return zf.read(f"tinyml/{allowed[name]}").decode()


class AskBody(BaseModel):
    question: str
    build_id: str | None = None


@app.post("/api/projects/{project_id}/ask")
def ask(project_id: str, body: AskBody) -> dict[str, Any]:
    """Questions about a package go to Claude Code (reading the unpacked package) or become a prompt for Claude chat."""
    store = _store()
    _project(project_id)
    builds = [b for b in store.builds(project_id) if b["package_path"]]
    if body.build_id:
        builds = [b for b in builds if b["id"] == body.build_id]
    if not builds:
        raise HTTPException(400, "Build a package first")
    pkg = Path(builds[0]["package_path"])
    with zipfile.ZipFile(pkg) as zf:
        report_text = zf.read("tinyml/report.md").decode()
        header = zf.read("tinyml/include/tinyml_model.h").decode()
    settings = load_settings()
    prompt = (
        "You are helping a firmware engineer use a generated TinyML C library. Answer briefly and concretely.\n\n"
        f"Question: {body.question}\n\n--- report.md ---\n{report_text}\n\n--- tinyml_model.h ---\n{header}\n"
    )
    if settings.assistant.provider == "claude_code":
        unpack = project_dir(project_id, "work", "ask", pkg.stem)
        with zipfile.ZipFile(pkg) as zf:
            zf.extractall(unpack)
        try:
            answer = ClaudeCodeProvider(settings.assistant).run(
                f"{body.question}\n\nThe package is in ./tinyml (start with report.md). Answer briefly.",
                cwd=str(unpack), tools="Read,Grep,Glob")
            return {"answer": answer, "source": "claude_code"}
        except ProviderUnavailable as exc:
            return {"answer": None, "source": "unavailable", "error": str(exc), "prompt": prompt}
    return {"answer": None, "source": "claude_chat", "prompt": prompt}


# ---------------------------------------------------------------------------------------- decisions
@app.get("/api/decisions")
def decisions(project_id: str | None = None) -> list[dict[str, Any]]:
    return _store().decisions(project_id)


class DecisionBody(BaseModel):
    choice: str


@app.patch("/api/decisions/{decision_id}")
def decide_override(decision_id: int, body: DecisionBody) -> dict[str, Any]:
    store = _store()
    row = store.decision(decision_id)
    if row is None:
        raise HTTPException(404, "decision not found")
    if body.choice not in row["options"]:
        raise HTTPException(400, f"choice must be one of {row['options']}")
    Decider(store, row["project_id"]).record_user_choice(decision_id, body.choice)
    return store.decision(decision_id)  # type: ignore[return-value]


@app.get("/api/learned-rules")
def learned_rules() -> list[dict[str, Any]]:
    return _store().learned_rules()


app.mount("/", StaticFiles(directory=str(ui_dir()), html=True), name="ui")

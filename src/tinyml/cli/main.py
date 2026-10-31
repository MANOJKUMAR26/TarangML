"""Command line: the same engine as the web app, one command per job."""

from __future__ import annotations

import json
import sys
import webbrowser
from dataclasses import asdict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from tinyml import __version__
from tinyml.config import AssistantSettings, Settings, load_settings, workspace

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Goal-first TinyML: sentence + CSV -> verified C library.")
console = Console()


def _store():
    from tinyml.store import get_store

    return get_store()


def _ask_user_factory(interactive: bool):
    if not interactive or not sys.stdin.isatty():
        return None

    def ask(question, best, chat_prompt):
        console.print(f"\n[bold]{question.text}[/bold]  (best guess: {best.choice}, confidence {best.confidence:.0%})")
        if chat_prompt:
            console.print("[dim]Paste this into Claude chat, then paste its JSON reply (or just type an option):[/dim]")
            console.print(chat_prompt)
        return typer.prompt(f"Choose one of {', '.join(question.options)}", default=best.choice)

    return ask


def _describe(description: str | None, recipe: str | None, vendor: str | None, interactive: bool):
    from tinyml.pipeline import stages
    from tinyml.recipes import get_recipe

    text = description or (get_recipe(recipe).summary if recipe else get_recipe("pump_health").summary)
    result = stages.describe(_store(), text, vendor=vendor, ask_user=_ask_user_factory(interactive))
    project = result["project"]
    spec = project["spec"]
    console.print(f"Project [bold]{project['id']}[/bold]: {spec['recipe']} / {spec['task']}, "
                  f"connectivity {spec['connectivity']}, power {spec['power']}, priority {spec['priority']}")
    for q in result["questions"]:
        console.print(f"[yellow]Unsure:[/yellow] {q['question']} Using '{q['guess']}' (change with --connectivity etc.)")
    return project


def _apply_overrides(project_id: str, **overrides) -> None:
    from tinyml.pipeline import stages

    changes = {k: v for k, v in overrides.items() if v is not None}
    if changes:
        stages.update_spec(_store(), project_id, changes)


def _ingest(project_id: str, data: list[Path], sample_rate: float | None, interactive: bool) -> dict:
    from tinyml.pipeline import stages

    result = stages.ingest(_store(), project_id, data, sample_rate, ask_user=_ask_user_factory(interactive),
                           log=lambda m: console.print(f"[dim]{m}[/dim]"))
    roles = ", ".join(f"{c}={r}" for c, r in result["roles"].items())
    console.print(f"Columns: {roles}")
    for issue in result["issues"]:
        color = {"fix": "green", "warn": "yellow", "block": "red"}[issue["severity"]]
        console.print(f"[{color}]{issue['severity']:>5}[/{color}] {issue['message']}")
    if result["blocked"]:
        console.print("[red]Cannot continue until the blocking problems are fixed.[/red]")
        raise typer.Exit(2)
    return result


MODEL_HELP = ("Which trained model to use: 'recommended' (the simplest that meets the target), 'best' (the "
              "strongest, even below the target), an algorithm such as random_forest, or a candidate id")


def _train(project_id: str, model: str = "recommended") -> dict:
    from tinyml.pipeline import stages

    log = lambda m: console.print(f"[dim]{m}[/dim]")  # noqa: E731
    try:
        result = stages.train(_store(), project_id, log=log)
        _print_candidates(project_id)
        if model != "recommended" or result["chosen"] is None:
            if model == "recommended":
                console.print("[red]No model met the target.[/red] To continue anyway, pick one with --model best, "
                              "--model <algorithm> or --model <id>.")
                raise typer.Exit(3)
            result = stages.select_model(_store(), project_id, _resolve_model(project_id, model), log=log)
    except stages.StageError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(3) from exc
    if not result["chosen"]["passed"]:
        console.print("[yellow]This model is below the target; the report will say so.[/yellow]")
    console.print(f"[bold green]{result['explanation']}[/bold green]")
    return result


def _resolve_model(project_id: str, choice: str) -> str:
    from tinyml.ml.ladder import strength

    task = _store().project(project_id)["spec"]["task"]
    trained = [r for r in _store().candidates(project_id) if r["artifact_path"]]
    if choice == "recommended":
        matching = [r for r in trained if r["passed"]][:1]
    elif choice == "best":
        matching = trained
    else:
        matching = [r for r in trained if choice in (r["id"], r["algorithm"])]
    if not matching:
        names = ", ".join(sorted({r["algorithm"] for r in trained}))
        hint = "No model meets the target" if choice == "recommended" else f"No trained model called '{choice}'"
        console.print(f"[red]{hint}. Choose best or one of: {names}.[/red]")
        raise typer.Exit(3)
    return max(matching, key=lambda r: strength(r["metrics"], task))["id"]


def _print_candidates(project_id: str) -> None:
    from rich.table import Table

    table = Table(title="Models tried")
    for col in ("id", "rung", "algorithm", "settings", "test result", "size", "target"):
        table.add_column(col)
    for r in _store().candidates(project_id):
        m = r["metrics"] or {}
        if "false_alarm_rate" in m:
            caught = "n/a" if m["detection_rate"] is None else f"{m['detection_rate']:.0%}"
            score = f"caught {caught}, false alarms {m['false_alarm_rate']:.0%}"
        elif "accuracy" in m:
            score = f"accuracy {m['accuracy']:.0%}, per event {m['balanced_accuracy']:.0%}"
        else:
            score = "-"
        size_bytes = m.get("size_bytes") or 0
        size = "-" if not size_bytes else f"{size_bytes} B" if size_bytes < 1024 else f"{size_bytes / 1024:.1f} KB"
        table.add_row(r["id"], str(r["rung"]), r["algorithm"], r["note"] or "", score, size,
                      "meets" if r["passed"] else "below")
    console.print(table)


def _num(n, fmt="{:,.0f}"):
    if not n or n.get("value") is None:
        return "-"
    return f"{fmt.format(n['value'])} ({n['tag']})"


@app.command()
def build(
    data: list[Path] = typer.Argument(..., exists=True, help="One or more CSV files"),
    chip: str = typer.Option(..., "--chip", help="Chip id from `tinyml catalog`, e.g. esp32s3"),
    describe: str | None = typer.Option(None, "--describe", "-d", help="What the device should detect, in plain English"),
    recipe: str | None = typer.Option(None, help="Recipe to assume when --describe is not given"),
    sample_rate: float | None = typer.Option(None, help="Readings per second, if the CSV has no time column"),
    out: Path | None = typer.Option(None, help="Copy the package to this folder"),
    interactive: bool = typer.Option(True, help="Ask when unsure (needs a terminal)"),
    model: str = typer.Option("recommended", help=MODEL_HELP),
):
    """Sentence + CSV -> verified C library for one chip (zip)."""
    from tinyml.pipeline import stages

    project = _describe(describe, recipe, None, interactive)
    _ingest(project["id"], data, sample_rate, interactive)
    _train(project["id"], model)
    try:
        result = stages.build_for_chip(_store(), project["id"], chip, log=lambda m: console.print(f"[dim]{m}[/dim]"))
    except stages.StageError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(4) from exc
    parity = result["parity"]
    m = parity["measurement"]
    console.print(f"Flash {_num(m['flash_bytes'])} B, RAM {_num(m['ram_bytes'])} B, "
                  f"{_num(m['instructions'])} instructions, {_num(m['latency_ms'], '{:.3f}')} ms per prediction")
    console.print(f"Parity: {parity['label_matches']}/{parity['windows']} answers match Python "
                  f"({parity['borderline']} borderline) in {m['emulator']}")
    if not result["passed"]:
        console.print(f"[red]Blocked: {parity.get('blocked_reason') or 'parity failed'}[/red]")
        for failure in parity.get("failures", [])[:5]:
            console.print(f"  {failure['reason']}")
        raise typer.Exit(5)
    path = Path(result["package_path"])
    if out:
        out.mkdir(parents=True, exist_ok=True)
        path = Path(__import__("shutil").copy2(path, out / path.name))
    console.print(f"[bold green]Package:[/bold green] {path}")


@app.command()
def rank(
    data: list[Path] = typer.Argument(..., exists=True),
    any_vendor: bool = typer.Option(False, "--any-vendor", help="Compare chips from every vendor"),
    vendor: str | None = typer.Option(None, help="espressif, st, nordic or raspberrypi"),
    priority: str | None = typer.Option(None, help="battery, cost or speed"),
    connectivity: str | None = typer.Option(None, help="wifi, ble, cellular, none or any"),
    budget: float | None = typer.Option(None, help="Max chip price in USD"),
    describe: str | None = typer.Option(None, "--describe", "-d"),
    recipe: str | None = typer.Option(None),
    chips: str | None = typer.Option(None, help="Comma-separated chip ids to consider (default: all)"),
    sample_rate: float | None = typer.Option(None),
    interactive: bool = typer.Option(True),
    model: str = typer.Option("recommended", help=MODEL_HELP),
):
    """Rank microcontrollers for this data and goal; prints the top 3 with reasons."""
    from tinyml.pipeline import stages

    project = _describe(describe, recipe, None if any_vendor else vendor, interactive)
    _apply_overrides(project["id"], priority=priority, connectivity=connectivity, budget_usd_per_unit=budget,
                     vendor="any" if any_vendor else vendor)
    _ingest(project["id"], data, sample_rate, interactive)
    _train(project["id"], model)
    result = stages.rank_chips(_store(), project["id"], chips.split(",") if chips else None,
                               log=lambda m: console.print(f"[dim]{m}[/dim]"))
    table = Table(title="Chips that passed your filters")
    for col in ("#", "chip", "score", "flash B", "RAM B", "ms/prediction", "battery days"):
        table.add_column(col)
    for r in result["ranked"]:
        m = r["measurement"]
        table.add_row(str(r["rank"]), r["name"], f"{r['score']:.0f}", _num(m["flash_bytes"]), _num(m["ram_bytes"]),
                      _num(m["latency_ms"], "{:.3f}"), _num(m["battery_days"]))
    console.print(table)
    for r in result["ranked"][:3]:
        console.print(f"[bold]{r['rank']}. {r['name']}[/bold]: " + "; ".join(r["reasons"]))
        if r["tradeoffs"]:
            console.print("   [dim]trade-offs: " + "; ".join(r["tradeoffs"]) + "[/dim]")
    for chip_id, reason in result["filtered"].items():
        console.print(f"[dim]{chip_id}: {reason}[/dim]")
    console.print(f"Project {project['id']}: build with `tinyml package {project['id']} --chip <id>`")


@app.command()
def package(project_id: str, chip: str = typer.Option(..., "--chip")):
    """Final build, parity test and signed package for a project that was already ranked or trained."""
    from tinyml.pipeline import stages

    result = stages.build_for_chip(_store(), project_id, chip, log=lambda m: console.print(f"[dim]{m}[/dim]"))
    console.print(f"Passed: {result['passed']}  Package: {result['package_path']}")


@app.command("use-model")
def use_model(project_id: str, choice: str = typer.Argument(..., help=MODEL_HELP)):
    """Switch an already trained project to another of its models (see the table printed by training)."""
    from tinyml.pipeline import stages

    if _store().project(project_id) is None:
        console.print(f"[red]Project {project_id} not found[/red]")
        raise typer.Exit(1)
    chosen = _resolve_model(project_id, choice)
    try:
        result = stages.select_model(_store(), project_id, chosen, log=lambda m: console.print(f"[dim]{m}[/dim]"))
    except stages.StageError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(3) from exc
    console.print(f"[bold green]{result['explanation']}[/bold green]")


@app.command()
def check(data: list[Path] = typer.Argument(..., exists=True), describe: str | None = typer.Option(None, "-d"),
          recipe: str | None = typer.Option(None), sample_rate: float | None = typer.Option(None)):
    """Only run the CSV checks."""
    project = _describe(describe, recipe, None, False)
    _ingest(project["id"], data, sample_rate, False)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = typer.Option(True, "--open/--no-open")):
    """Start the local web app."""
    import uvicorn

    url = f"http://{host}:{port}"
    console.print(f"tinyml {__version__} on [bold]{url}[/bold]  (workspace: {workspace()})")
    if open_browser:
        webbrowser.open(url)
    uvicorn.run("tinyml.api.app:app", host=host, port=port, log_level="warning")


@app.command("doctor")
def doctor_cmd():
    """Which chips this machine can build and emulate."""
    from tinyml.toolchains import doctor

    rep = doctor.report()
    hc = rep["host_compiler"]
    console.print(f"Host C compiler: {'ok ' + str(hc['version']) if hc['ok'] else 'MISSING (' + hc['why'] + ')'}")
    table = Table(title="Chips")
    for col in ("chip", "ready", "backend", "toolchain / what to install"):
        table.add_column(col)
    for c in rep["chips"]:
        usable = next((o for o in c["options"] if o["ok"]), None)
        detail = usable["toolchain"] if usable else "; ".join(f"{o['why']} -> {o['hint']}" for o in c["options"])
        table.add_row(c["name"], "yes" if c["ready"] else "no", c["backend"] or "-", detail or "")
    console.print(table)
    a = rep["assistant"]
    console.print(f"Assistant: {a['provider']}  (Claude Code {'found: ' + str(a['claude_version']) if a['claude_code_found'] else 'not found'})")


@app.command()
def catalog(radio: str | None = None, max_price: float | None = None, vendor: str | None = None):
    """List chips in the catalog."""
    from tinyml.catalog import get_catalog

    table = Table()
    for col in ("id", "chip", "core", "MHz", "RAM KB", "flash KB", "radios", "price", "status"):
        table.add_column(col)
    for c in get_catalog().query(vendor=vendor, radio=radio, max_price=max_price):
        table.add_row(c.id, c.name, c.core, f"{c.clock_mhz:g}", f"{c.ram_kb:g}", f"{c.flash_kb:g}",
                      ",".join(c.radios) or "-", f"${c.price_usd:.2f}", c.onboarding)
    console.print(table)


@app.command("demo-data")
def demo_data(kind: str = typer.Argument(..., help="pump, driving, activity or cwru"),
              out: Path = typer.Option(Path("."), help="Output folder"),
              source: Path | None = typer.Option(None, help="CWRU only: folder with the .mat files")):
    """Write a demo CSV (synthetic, or real CWRU bearing data)."""
    from tinyml.datasets import cwru, synthetic

    out.mkdir(parents=True, exist_ok=True)
    if kind == "cwru":
        src = source or cwru.fetch(workspace() / "datasets" / "cwru")
        path = cwru.convert(src, out / "pump_cwru.csv")
    elif kind in synthetic.GENERATORS:
        path = synthetic.GENERATORS[kind](out / f"{kind}.csv")
    else:
        raise typer.BadParameter("choose pump, driving, activity or cwru")
    console.print(f"Wrote {path}")


@app.command()
def verify(package_zip: Path = typer.Argument(..., exists=True)):
    """Check a package's signature and file hashes."""
    from tinyml.packaging.package import verify_package

    ok, message = verify_package(package_zip)
    console.print(("[green]OK[/green] " if ok else "[red]INVALID[/red] ") + message)
    raise typer.Exit(0 if ok else 1)


@app.command()
def projects():
    """List projects in the workspace."""
    for p in _store().projects():
        console.print(f"{p['id']}  {p['status']:<10} {p['name']}")


@app.command()
def decisions(project: str | None = None, limit: int = 30):
    """Show the decision log (what was decided, by whom, and what the user changed)."""
    table = Table()
    for col in ("id", "type", "key", "answer", "conf", "source", "final", "build ok"):
        table.add_column(col)
    for d in _store().decisions(project, limit):
        table.add_row(str(d["id"]), d["decision_type"], str(d["decision_key"])[:24], str(d["answer"]),
                      f"{d['confidence'] or 0:.2f}", d["source"], str(d["final_choice"] or "-"),
                      {None: "-", 1: "yes", 0: "no"}[d["build_passed"]])
    console.print(table)


@app.command()
def settings(assistant: str | None = typer.Option(None, help="off, claude_code or claude_chat"),
             claude_model: str | None = typer.Option(None)):
    """Show or change settings (for example which assistant handles unsure decisions)."""
    current = load_settings()
    if assistant or claude_model:
        a = asdict(current.assistant)
        if assistant:
            if assistant not in ("off", "claude_code", "claude_chat"):
                raise typer.BadParameter("off, claude_code or claude_chat")
            a["provider"] = assistant
        if claude_model:
            a["claude_model"] = claude_model
        current = Settings(**{**asdict(current), "assistant": AssistantSettings(**a)})
        current.save()
    console.print_json(json.dumps(asdict(current)))


if __name__ == "__main__":
    app()

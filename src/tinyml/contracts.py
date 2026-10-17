"""Typed versions of the data contracts in docs/contracts/.

Each model here mirrors one contract file. Change the contract document first,
then this file, in the same pull request.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Task = Literal["anomaly", "classification"]
Connectivity = Literal["wifi", "ble", "cellular", "none", "any"]
Power = Literal["battery", "plugged"]
Priority = Literal["battery", "cost", "speed"]
Severity = Literal["fix", "warn", "block"]
Tag = Literal["measured", "emulated", "estimated"]
ColumnRole = Literal["time", "sensor_axis", "label", "recording_id", "ignore"]


class ProblemSpec(BaseModel):
    """docs/contracts/problem-spec.md"""

    recipe: str
    task: Task
    vendor: str = "any"
    connectivity: Connectivity = "any"
    power: Power = "battery"
    budget_usd_per_unit: float | None = None
    priority: Priority = "battery"
    max_latency_ms: float = 1000.0
    description: str = ""
    seed: int = 42


class ColumnSummary(BaseModel):
    """docs/contracts/csv-summary.md: one column, never raw rows."""

    name: str
    dtype: Literal["numeric", "text", "datetime"]
    count: int
    missing: int
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    unique: int
    always_increasing: bool
    step: float | None = None
    sample: list[str]


class CsvSummary(BaseModel):
    use_case: str
    files: list[str]
    rows: int
    columns: list[ColumnSummary]


class DataIssue(BaseModel):
    code: str
    severity: Severity
    message: str
    columns: list[str] = Field(default_factory=list)
    detail: dict[str, Any] = Field(default_factory=dict)


class TaggedNumber(BaseModel):
    value: float | None
    unit: str
    tag: Tag
    source: str = ""


class GoldenVector(BaseModel):
    """docs/contracts/golden-vectors.md: one test window and Python's answer."""

    id: int
    recording: str
    true_label: str | None
    input_b64: str
    expected_label: int
    expected_score: float
    expected_features: list[float]
    margin: float
    borderline: bool


class GoldenSet(BaseModel):
    schema_: str = Field("tinyml.golden/1", alias="schema")
    window_samples: int
    axes: list[str]
    feature_names: list[str]
    feature_tolerances: list[dict[str, float]]
    score_rtol: float
    score_exact: bool
    vectors: list[GoldenVector]

    model_config = {"populate_by_name": True}


class ChipMeasurement(BaseModel):
    chip_id: str
    status: Literal["ok", "failed", "skipped", "filtered"]
    backend: str
    flash_bytes: TaggedNumber | None = None
    ram_bytes: TaggedNumber | None = None
    stack_bytes: TaggedNumber | None = None
    instructions: TaggedNumber | None = None
    latency_ms: TaggedNumber | None = None
    energy_uj: TaggedNumber | None = None
    battery_days: TaggedNumber | None = None
    price_usd: TaggedNumber | None = None
    parity_ok: bool | None = None
    parity_detail: dict[str, Any] = Field(default_factory=dict)
    toolchain: str = ""
    emulator: str = ""
    error: str | None = None
    filtered_reason: str | None = None


class RankedChip(BaseModel):
    chip_id: str
    name: str
    score: float
    rank: int
    reasons: list[str]
    tradeoffs: list[str]
    measurement: ChipMeasurement

"""The hand-edited chip catalog (PRD section 8)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml
from pydantic import BaseModel

from tinyml.config import catalog_path


class Core(BaseModel):
    id: str
    arch: str
    fpu: str
    description: str
    gcc_flags: list[str] = []
    qemu_machine: str | None = None


class Emulator(BaseModel):
    name: str
    machine: str
    fidelity: str  # chip | core | community


class Chip(BaseModel):
    id: str
    vendor: str
    family: str
    part_number: str
    core: str
    clock_mhz: float
    ram_kb: float
    flash_kb: float
    radios: list[str]
    active_ma: float
    sleep_ma: float
    price_usd: float
    lifecycle: str
    backend: str
    idf_target: str | None = None
    fallback_backend: str | None = None
    fallback_core: str | None = None
    emulator: Emulator
    cpi: dict[str, Any]
    toolchain_image: str
    onboarding: str
    notes: str = ""

    @property
    def name(self) -> str:
        return f"{self.family} ({self.part_number})"

    @property
    def has_fpu(self) -> bool:
        return get_catalog().cores[self.core].fpu != "none"


class Vendor(BaseModel):
    id: str
    name: str


class Catalog(BaseModel):
    vendors: list[Vendor]
    cores: dict[str, Core]
    chips: list[Chip]

    def chip(self, chip_id: str) -> Chip:
        for chip in self.chips:
            if chip.id == chip_id:
                return chip
        raise KeyError(f"Unknown chip '{chip_id}'. Known: {', '.join(c.id for c in self.chips)}")

    def vendor(self, vendor_id: str) -> Vendor:
        for vendor in self.vendors:
            if vendor.id == vendor_id:
                return vendor
        raise KeyError(f"Unknown vendor '{vendor_id}'")

    def query(
        self,
        vendor: str | None = None,
        radio: str | None = None,
        max_price: float | None = None,
        fpu: bool | None = None,
    ) -> list[Chip]:
        chips = self.chips
        if vendor and vendor != "any":
            chips = [c for c in chips if c.vendor == vendor]
        if radio:
            chips = [c for c in chips if radio in c.radios]
        if max_price is not None:
            chips = [c for c in chips if c.price_usd <= max_price]
        if fpu is not None:
            chips = [c for c in chips if c.has_fpu == fpu]
        return chips


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    raw: dict[str, Any] = yaml.safe_load(catalog_path().read_text())
    cores = {cid: Core(id=cid, **body) for cid, body in raw["cores"].items()}
    return Catalog(
        vendors=[Vendor(**v) for v in raw["vendors"]],
        cores=cores,
        chips=[Chip(**c) for c in raw["chips"]],
    )

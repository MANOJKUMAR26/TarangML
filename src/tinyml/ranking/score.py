"""Stage B, step 4: hard filters, a weighted score from the user's priority, and reasons in plain words."""

from __future__ import annotations

from tinyml.catalog import Chip
from tinyml.contracts import ChipMeasurement, ProblemSpec, RankedChip

RADIO_NAMES = {"wifi": "Wi-Fi", "ble": "Bluetooth LE", "thread": "Thread", "zigbee": "Zigbee"}
MAX_MEMORY_SHARE = 0.5  # the model may use at most half of the chip's RAM and flash; the rest is for your firmware
WEIGHTS = {
    "battery": {"energy": 0.55, "price": 0.15, "latency": 0.10, "memory": 0.20},
    "cost": {"energy": 0.15, "price": 0.55, "latency": 0.10, "memory": 0.20},
    "speed": {"energy": 0.15, "price": 0.15, "latency": 0.50, "memory": 0.20},
}


def pre_filter(chip: Chip, spec: ProblemSpec) -> str | None:
    """Filters that need no build. Returns the reason a chip is out, or None."""
    if spec.vendor not in ("any", "", None) and chip.vendor != spec.vendor:
        return "different vendor"
    if spec.connectivity in ("wifi", "ble") and spec.connectivity not in chip.radios:
        return f"no {RADIO_NAMES[spec.connectivity]}"
    if spec.budget_usd_per_unit is not None and chip.price_usd > spec.budget_usd_per_unit:
        return f"costs about ${chip.price_usd:.2f}, over your ${spec.budget_usd_per_unit:g} budget"
    if chip.lifecycle != "active":
        return f"lifecycle: {chip.lifecycle}"
    return None


def post_filter(chip: Chip, m: ChipMeasurement, spec: ProblemSpec, hop_ms: float) -> str | None:
    """Filters that need the measured numbers."""
    if m.status != "ok":
        return m.error or m.filtered_reason or "could not be built"
    if m.parity_ok is False:
        return "C output did not match Python in the emulator"
    flash = m.flash_bytes.value if m.flash_bytes else None
    ram = m.ram_bytes.value if m.ram_bytes else None
    if flash is not None and flash > MAX_MEMORY_SHARE * chip.flash_kb * 1024:
        return f"model needs {flash / 1024:.0f} KB of flash, too much of the chip's {chip.flash_kb:g} KB"
    if ram is not None and ram > MAX_MEMORY_SHARE * chip.ram_kb * 1024:
        return f"model needs {ram / 1024:.0f} KB of RAM, too much of the chip's {chip.ram_kb:g} KB"
    latency = m.latency_ms.value if m.latency_ms else None
    if latency is not None:
        limit = min(spec.max_latency_ms, hop_ms)
        if latency > limit:
            return f"too slow: {latency:.1f} ms per prediction, needs under {limit:.0f} ms"
    return None


def _lower_is_better(values: list[float]) -> list[float]:
    """Best value scores 1; a value twice as bad scores 0.5. Ratios, so one outlier cannot flatten the rest."""
    best = min(values)
    return [1.0 if v <= 0 else (best / v if best > 0 else 0.0) for v in values]


def _higher_is_better(values: list[float]) -> list[float]:
    best = max(values)
    return [v / best if best > 0 else 1.0 for v in values]


def memory_share(chip: Chip, m: ChipMeasurement) -> float:
    flash = (m.flash_bytes.value or 0) / (chip.flash_kb * 1024)
    ram = (m.ram_bytes.value or 0) / (chip.ram_kb * 1024)
    return max(flash, ram)


def rank(chips: list[Chip], measurements: dict[str, ChipMeasurement], spec: ProblemSpec, hop_ms: float
         ) -> tuple[list[RankedChip], dict[str, str]]:
    filtered: dict[str, str] = {}
    passing: list[tuple[Chip, ChipMeasurement]] = []
    for chip in chips:
        m = measurements[chip.id]
        reason = m.filtered_reason or post_filter(chip, m, spec, hop_ms)
        if reason:
            filtered[chip.id] = reason
            m.filtered_reason = reason
        else:
            passing.append((chip, m))
    if not passing:
        return [], filtered

    energy = [m.energy_uj.value or 0.0 for _, m in passing]
    price = [c.price_usd for c, _ in passing]
    latency = [m.latency_ms.value or 0.0 for _, m in passing]
    memory = [memory_share(c, m) for c, m in passing]
    days = [m.battery_days.value if m.battery_days and m.battery_days.value else 0.0 for _, m in passing]
    norms = {
        "energy": _higher_is_better(days) if any(days) else _lower_is_better(energy),
        "price": _lower_is_better(price),
        "latency": _lower_is_better(latency),
        "memory": [1.0 - min(share, 1.0) for share in memory],
    }
    weights = WEIGHTS[spec.priority]
    scored = []
    for i, (chip, m) in enumerate(passing):
        score = sum(weights[k] * norms[k][i] for k in weights) * 100
        scored.append((score, chip, m, i))
    # Chips whose numbers are still estimates are listed after every chip with proven numbers.
    scored.sort(key=lambda s: (s[1].onboarding == "estimated", -s[0], s[1].price_usd))

    best_i = scored[0][3]
    ranked: list[RankedChip] = []
    for position, (score, chip, m, i) in enumerate(scored, start=1):
        reasons, tradeoffs = _reasons(chip, m, spec, i, best_i, energy, price, latency, memory, passing)
        ranked.append(RankedChip(chip_id=chip.id, name=chip.name, score=round(score, 1), rank=position,
                                 reasons=reasons, tradeoffs=tradeoffs, measurement=m))
    return ranked, filtered


def _reasons(chip: Chip, m: ChipMeasurement, spec: ProblemSpec, i: int, best_i: int, energy: list[float],
             price: list[float], latency: list[float], memory: list[float], passing: list) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    if spec.connectivity in ("wifi", "ble"):
        reasons.append(f"{RADIO_NAMES[spec.connectivity]} built in")
    elif chip.radios:
        reasons.append(" and ".join(RADIO_NAMES.get(r, r) for r in chip.radios[:2]) + " built in")
    days_all = [x.battery_days.value if x.battery_days else 0 for _, x in passing]
    if (m.battery_days and m.battery_days.value == max(days_all)) or (not any(days_all) and energy[i] == min(energy)):
        days = m.battery_days.value if m.battery_days else None
        extra = f" (about {days:.0f} days on 2000 mAh with the sensor, estimated)" if days else ""
        reasons.append(f"lowest battery use of the shortlist{extra}")
    if price[i] == min(price):
        reasons.append(f"cheapest (about ${chip.price_usd:.2f})")
    if latency[i] == min(latency):
        reasons.append(f"fastest ({latency[i]:.2f} ms per prediction)")
    reasons.append(f"model uses {memory[i] * 100:.1f}% of memory")
    if not chip.has_fpu:
        reasons.append("no FPU, so float maths runs in software")
    if chip.onboarding == "estimated":
        reasons.append("numbers are estimates until this chip family is onboarded")

    tradeoffs: list[str] = []
    if i != best_i:
        best_chip = passing[best_i][0]
        if energy[best_i] > 0 and energy[i] > 1.2 * energy[best_i]:
            tradeoffs.append(f"uses {energy[i] / energy[best_i]:.1f}x the energy per prediction of {best_chip.family}")
        if price[i] > price[best_i] + 0.05:
            tradeoffs.append(f"costs about ${price[i] - price[best_i]:.2f} more than {best_chip.family}")
        if latency[best_i] > 0 and latency[i] > 1.5 * latency[best_i]:
            tradeoffs.append(f"{latency[i] / latency[best_i]:.1f}x slower than {best_chip.family}")
    return reasons, tradeoffs

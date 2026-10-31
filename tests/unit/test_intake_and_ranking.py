from tinyml.catalog import get_catalog
from tinyml.config import Settings
from tinyml.contracts import ChipMeasurement, ProblemSpec, TaggedNumber
from tinyml.decide import Decider
from tinyml.intake.parse import parse_use_case
from tinyml.ranking.score import pre_filter, rank


def test_prd_example_sentence(store):
    text = "Alert me when my pump vibration looks abnormal. It runs on battery and reports over Wi-Fi."
    res = parse_use_case(text, Decider(store, None, Settings()))
    s = res.spec
    assert (s.recipe, s.task, s.connectivity, s.power, s.priority) == ("pump_health", "anomaly", "wifi", "battery", "battery")
    assert res.questions == []


def test_budget_and_classification(store):
    text = "Tell apart harsh braking and sharp turns in vans, Bluetooth to the phone, budget $4"
    s = parse_use_case(text, Decider(store, None, Settings())).spec
    assert s.recipe == "driving_events" and s.task == "classification"
    assert s.connectivity == "ble" and s.budget_usd_per_unit == 4.0


def test_missing_connectivity_becomes_one_question(store):
    res = parse_use_case("watch my motor bearings", Decider(store, None, Settings()))
    assert res.spec.connectivity == "any"
    assert len(res.questions) == 1 and res.questions[0]["field"] == "connectivity"


def _m(chip, latency, energy, flash=2000, ram=4000, days=100):
    t = lambda v, u, tag="measured": TaggedNumber(value=v, unit=u, tag=tag)  # noqa: E731
    return ChipMeasurement(chip_id=chip, status="ok", backend="arm", parity_ok=True, flash_bytes=t(flash, "B"),
                           ram_bytes=t(ram, "B"), latency_ms=t(latency, "ms", "estimated"),
                           energy_uj=t(energy, "uJ", "estimated"), battery_days=t(days, "d", "estimated"))


def test_filters_and_priority():
    cat = get_catalog()
    spec = ProblemSpec(recipe="pump_health", task="anomaly", connectivity="ble", priority="battery")
    assert pre_filter(cat.chip("stm32f411"), spec) == "no Bluetooth LE"
    assert pre_filter(cat.chip("nrf52840"), spec) is None
    chips = [cat.chip("nrf52840"), cat.chip("nrf54l15"), cat.chip("esp32s3")]
    ms = {"nrf52840": _m("nrf52840", 0.3, 3.0, days=250), "nrf54l15": _m("nrf54l15", 0.2, 2.0, days=300),
          "esp32s3": _m("esp32s3", 0.05, 6.0, days=120)}
    ranked, filtered = rank(chips, ms, spec, hop_ms=100)
    assert ranked[0].chip_id == "nrf54l15" and not filtered
    assert any("lowest battery" in r for r in ranked[0].reasons)


def test_too_slow_and_too_big_are_filtered():
    cat = get_catalog()
    spec = ProblemSpec(recipe="pump_health", task="anomaly", max_latency_ms=1.0)
    chips = [cat.chip("stm32l432"), cat.chip("nrf52840")]
    ms = {"stm32l432": _m("stm32l432", 0.1, 1.0, ram=60 * 1024), "nrf52840": _m("nrf52840", 5.0, 1.0)}
    ranked, filtered = rank(chips, ms, spec, hop_ms=100)
    assert not ranked
    assert "RAM" in filtered["stm32l432"] and "too slow" in filtered["nrf52840"]


def test_estimated_chips_come_after_proven_ones():
    cat = get_catalog()
    spec = ProblemSpec(recipe="pump_health", task="anomaly", priority="speed")
    chips = [cat.chip("rp2350"), cat.chip("stm32f411")]
    ms = {"rp2350": _m("rp2350", 0.01, 0.5, days=500), "stm32f411": _m("stm32f411", 0.2, 3.0, days=100)}
    ranked, _ = rank(chips, ms, spec, hop_ms=100)
    assert [r.chip_id for r in ranked] == ["stm32f411", "rp2350"]

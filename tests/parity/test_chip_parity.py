"""Build, emulate and package for every chip whose toolchain is installed (CI runs this in pinned images)."""

import zipfile

import pytest

from tinyml.catalog import get_catalog
from tinyml.pipeline import stages
from tinyml.store import get_store
from tinyml.toolchains import backend_for

CHIPS = [c.id for c in get_catalog().chips]


@pytest.fixture(scope="module")
def trained(data_dir, tmp_path_factory):
    import os

    home = tmp_path_factory.mktemp("parity_ws")
    os.environ["TINYML_HOME"] = str(home)
    store = get_store()
    pid = stages.describe(store, "Detect harsh braking, harsh acceleration and sharp turns in vans.")["project"]["id"]
    stages.ingest(store, pid, [data_dir / "driving.csv"])
    stages.train(store, pid)
    return home, pid


@pytest.mark.emulator
@pytest.mark.parametrize("chip_id", CHIPS)
def test_chip_build_matches_python(trained, chip_id, monkeypatch):
    home, pid = trained
    monkeypatch.setenv("TINYML_HOME", str(home))
    chip = get_catalog().chip(chip_id)
    backend, why = backend_for(chip)
    if backend is None:
        pytest.skip(why)
    result = stages.build_for_chip(get_store(), pid, chip_id)
    assert result["passed"], result["parity"]
    m = result["parity"]["measurement"]
    assert m["flash_bytes"]["value"] > 0 and m["instructions"]["value"] > 0
    with zipfile.ZipFile(result["package_path"]) as zf:
        assert "tinyml/test/golden.json" in zf.namelist()

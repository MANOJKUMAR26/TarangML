from fastapi.testclient import TestClient


def test_catalog_and_project_flow():
    from tinyml.api.app import app

    with TestClient(app) as client:
        assert client.get("/api/health").json()["ok"]
        ble = client.get("/api/chips", params={"radio": "ble", "max_price": 5}).json()
        assert {c["id"] for c in ble} >= {"nrf52840", "esp32s3"}
        assert all("ble" in c["radios"] for c in ble)
        assert client.get("/api/vendors/nordic/chips").status_code == 200
        assert client.get("/api/chips/esp32s3").json()["family"] == "ESP32-S3"
        res = client.post("/api/projects", json={"description": "Detect harsh braking and sharp turns in vans"})
        pid = res.json()["project"]["id"]
        ds = client.post(f"/api/projects/{pid}/demo-data", json={"dataset": "driving"}).json()
        assert ds["roles"]["event"] == "label"
        roles = dict(ds["roles"])
        roles["az"] = "ignore"
        ds2 = client.put(f"/api/projects/{pid}/datasets/roles", json={"roles": roles}).json()
        assert ds2["roles"]["az"] == "ignore"
        project = client.get(f"/api/projects/{pid}").json()
        assert project["preview"]["samples"]
        assert client.get("/").status_code == 200


def test_select_rejects_unknown_candidate():
    from tinyml.api.app import app

    with TestClient(app) as client:
        pid = client.post("/api/projects", json={"description": "pump vibration abnormal"}).json()["project"]["id"]
        res = client.post(f"/api/projects/{pid}/select", json={"candidate_id": "cand_missing"})
        assert res.status_code == 404

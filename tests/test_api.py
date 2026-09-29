"""End-to-end API tests (FastAPI TestClient + SQLite)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tests.conftest import (
    BASE_TIME,
    make_waveform_b64,
    station_ring,
    synthetic_event_payload,
)

BASE_DT = datetime(2026, 9, 22, tzinfo=timezone.utc)


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #
def test_process_end_to_end(client):
    body, truth = synthetic_event_payload(prefix="E2E")
    resp = client.post("/v1/process", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["created"] is True
    assert data["model"] == {"name": "default", "version": "1.0"}
    assert len(data["events"]) >= 1

    best = max(data["events"], key=lambda e: e["station_count"])
    assert best["station_count"] >= 8
    assert best["latitude"] == pytest.approx(truth["lat"], abs=0.02)
    assert best["longitude"] == pytest.approx(truth["lon"], abs=0.02)
    assert best["depth_km"] == pytest.approx(truth["depth_km"], abs=3.0)
    expected_origin = BASE_DT + timedelta(seconds=truth["origin_offset_s"])
    assert abs((_parse(best["origin_time"]) - expected_origin).total_seconds()) < 1.5
    assert best["quality"] in ("A", "B", "C")
    assert best["rms_residual_s"] <= 1.0

    phases = {p["phase"] for p in best["picks"]}
    assert phases == {"P", "S"}
    for pick in best["picks"]:
        assert abs(pick["residual_s"]) <= 1.0
        assert pick["source"] == "stalta"
    assert isinstance(data["unassociated_picks"], list)


def test_idempotent_same_window(client):
    body, _ = synthetic_event_payload(prefix="IDEM", origin_offset_s=35.0)
    first = client.post("/v1/process", json=body)
    second = client.post("/v1/process", json=body)
    assert first.status_code == 200 and second.status_code == 200
    d1, d2 = first.json(), second.json()
    assert d1["created"] is True
    assert d2["created"] is False
    assert d1["run_id"] == d2["run_id"]
    assert d1["ingestion_id"] == d2["ingestion_id"]
    assert [e["event_id"] for e in d1["events"]] == [e["event_id"] for e in d2["events"]]


def test_manual_picks_associate(client):
    from app.config import DEFAULT_MODEL_LAYERS
    from app.services.association import distance_km
    from app.services.traveltime import LayeredModel

    model = LayeredModel.from_dicts(DEFAULT_MODEL_LAYERS)
    ev_lat, ev_lon, ev_depth, t0_off = 35.02, -118.03, 10.0, 40.0
    waveforms = []
    manual = []
    for code, lat, lon, elev in station_ring("MAN", n=6):
        b64 = make_waveform_b64(code, 100.0, BASE_TIME, [], duration=120.0, noise=0.01)
        waveforms.append(
            {
                "station": code,
                "latitude": lat,
                "longitude": lon,
                "elevation_m": elev,
                "sampling_rate": 100.0,
                "start_time": BASE_TIME,
                "data_b64": b64,
            }
        )
        dist = distance_km(ev_lat, ev_lon, lat, lon)
        tp = model.travel_time(ev_depth + elev / 1000.0, dist, "P")[0]
        manual.append(
            {
                "station": code,
                "phase": "P",
                "time": (BASE_DT + timedelta(seconds=t0_off + tp)).strftime(
                    "%Y-%m-%dT%H:%M:%S.%f"
                )[:-3]
                + "Z"
            }
        )
    body = {
        "waveforms": waveforms,
        "manual_picks": manual,
        "model": {"name": "default", "version": "1.0"},
        "params": {"max_residual_s": 1.0, "min_stations": 4},
    }
    resp = client.post("/v1/process", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    manual_events = [
        e
        for e in data["events"]
        if sum(1 for p in e["picks"] if p["source"] == "manual") >= 6
    ]
    assert manual_events, "expected an event built from the manual picks"
    event = manual_events[0]
    assert event["latitude"] == pytest.approx(ev_lat, abs=0.02)
    assert event["depth_km"] == pytest.approx(ev_depth, abs=3.0)


# --------------------------------------------------------------------------- #
# model versioning, replay, comparison
# --------------------------------------------------------------------------- #
def test_model_version_immutable(client):
    layers = [{"thickness_km": None, "vp_kms": 6.0, "vs_kms": 3.5}]
    r1 = client.post("/v1/models", json={"name": "regional", "version": "1.0", "layers": layers})
    assert r1.status_code == 201
    r2 = client.post("/v1/models", json={"name": "regional", "version": "1.0", "layers": layers})
    assert r2.status_code == 200 and r2.json()["created"] is False
    r3 = client.post(
        "/v1/models",
        json={
            "name": "regional",
            "version": "1.0",
            "layers": [{"thickness_km": None, "vp_kms": 6.6, "vs_kms": 3.8}],
        },
    )
    assert r3.status_code == 409
    assert r3.json()["detail"]["code"] == "model_version_conflict"


def test_replay_and_compare(client):
    body, truth = synthetic_event_payload(prefix="RPL", origin_offset_s=32.0)
    first = client.post("/v1/process", json=body)
    assert first.status_code == 200, first.text
    run_a = first.json()["run_id"]
    ingestion_id = first.json()["ingestion_id"]

    slower = [
        {"thickness_km": 15.0, "vp_kms": 5.2, "vs_kms": 3.0},
        {"thickness_km": 20.0, "vp_kms": 5.9, "vs_kms": 3.4},
        {"thickness_km": None, "vp_kms": 7.2, "vs_kms": 4.2},
    ]
    r = client.post("/v1/models", json={"name": "default", "version": "2.0", "layers": slower})
    assert r.status_code == 201

    replay = client.post(
        "/v1/runs",
        json={
            "ingestion_id": ingestion_id,
            "model": {"name": "default", "version": "2.0"},
            "params": {"max_residual_s": 1.0, "min_stations": 4},
        },
    )
    assert replay.status_code == 201, replay.text
    run_b = replay.json()["run_id"]
    assert run_b != run_a
    assert replay.json()["events"], "replay should still find the event"

    # replaying again is idempotent
    again = client.post(
        "/v1/runs",
        json={
            "ingestion_id": ingestion_id,
            "model": {"name": "default", "version": "2.0"},
            "params": {"max_residual_s": 1.0, "min_stations": 4},
        },
    )
    assert again.status_code == 200
    assert again.json()["run_id"] == run_b

    cmp_resp = client.get("/v1/runs/compare", params={"run_a": run_a, "run_b": run_b})
    assert cmp_resp.status_code == 200
    cmp_data = cmp_resp.json()
    assert cmp_data["model_a"] == "1.0" and cmp_data["model_b"] == "2.0"
    assert cmp_data["matches"], "expected matched events across model versions"
    delta = cmp_data["matches"][0]["delta"]
    assert abs(delta["origin_time_s"]) > 0.05 or abs(delta["depth_km"]) > 0.5

    run_view = client.get(f"/v1/runs/{run_a}")
    assert run_view.status_code == 200
    assert run_view.json()["run_id"] == run_a

    event_id = first.json()["events"][0]["event_id"]
    event_view = client.get(f"/v1/events/{event_id}")
    assert event_view.status_code == 200
    assert event_view.json()["event_id"] == event_id


# --------------------------------------------------------------------------- #
# explicit rejections
# --------------------------------------------------------------------------- #
def _single_waveform_payload(**overrides):
    wf = {
        "station": "BAD1",
        "latitude": 35.0,
        "longitude": -118.0,
        "elevation_m": 0.0,
        "sampling_rate": 100.0,
        "start_time": BASE_TIME,
        "data_b64": make_waveform_b64("BAD1", 100.0, BASE_TIME, [], duration=30.0),
    }
    wf.update(overrides)
    return {
        "waveforms": [wf],
        "manual_picks": [],
        "model": {"name": "default", "version": "1.0"},
        "params": {"max_residual_s": 1.0, "min_stations": 2},
    }


def test_missing_sampling_rate_rejected(client):
    body = _single_waveform_payload()
    del body["waveforms"][0]["sampling_rate"]
    resp = client.post("/v1/process", json=body)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "missing_sampling_rate"


def test_coordinate_out_of_bounds_rejected(client):
    resp = client.post("/v1/process", json=_single_waveform_payload(latitude=95.0))
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "coordinate_out_of_bounds"
    resp = client.post("/v1/process", json=_single_waveform_payload(longitude=-181.0))
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "coordinate_out_of_bounds"


def test_insufficient_data_rejected(client):
    short = make_waveform_b64("BAD1", 100.0, BASE_TIME, [], duration=5.0)
    resp = client.post("/v1/process", json=_single_waveform_payload(data_b64=short))
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "insufficient_data"


def test_invalid_base64_rejected(client):
    resp = client.post("/v1/process", json=_single_waveform_payload(data_b64="!!!not-base64!!!"))
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "invalid_base64"


def test_sampling_rate_mismatch_rejected(client):
    resp = client.post("/v1/process", json=_single_waveform_payload(sampling_rate=50.0))
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "sampling_rate_mismatch"


def test_sampling_rate_too_low_rejected(client):
    resp = client.post("/v1/process", json=_single_waveform_payload(sampling_rate=20.0))
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "sampling_rate_too_low"


def test_unknown_model_rejected(client):
    body = _single_waveform_payload()
    body["model"] = {"name": "default", "version": "9.9"}
    resp = client.post("/v1/process", json=body)
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "model_not_found"


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

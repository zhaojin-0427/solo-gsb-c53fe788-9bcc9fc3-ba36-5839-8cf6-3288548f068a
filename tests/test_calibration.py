"""End-to-end tests for station arrival-time calibrations."""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest

from tests.conftest import make_waveform_b64, station_ring

from app.config import DEFAULT_MODEL_LAYERS
from app.services.association import distance_km
from app.services.traveltime import LayeredModel

SR = 50.0
DURATION = 60.0
ORIGIN_OFFSET = 15.0
EV = {"lat": 35.05, "lon": -117.95, "depth_km": 12.0}
BASE_DT = datetime(2026, 9, 1, tzinfo=timezone.utc)

# Zero-sum per-station P/S clock biases the estimator must recover.
P_BIAS = {"%02d" % i: v for i, v in enumerate(
    [0.35, -0.25, 0.20, -0.30, 0.15, -0.10, 0.05, -0.10]
)}
S_BIAS = {"%02d" % i: v for i, v in enumerate(
    [-0.20, 0.30, -0.10, 0.25, -0.15, 0.05, -0.05, -0.10]
)}

MODEL = LayeredModel.from_dicts(DEFAULT_MODEL_LAYERS)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _event_window(day: int, prefix: str = "CA", p_bias=None, s_bias=None,
                  extra_station: bool = False, scramble: bool = False):
    """One /v1/process body with a synthetic event on day ``day``."""
    tails = list(P_BIAS.keys())
    if scramble:
        # Rotate biases across stations for holdout events: the trained terms
        # then mis-correct those arrivals.
        shift = (day % 7) + 1
        p_bias = {tails[i]: P_BIAS[tails[(i + shift) % len(tails)]] for i in range(8)}
        s_bias = {tails[i]: S_BIAS[tails[(i + shift) % len(tails)]] for i in range(8)}
    waveforms = []
    for code, lat, lon, elev in station_ring(prefix, n=8):
        tail = code[-2:]
        dist = distance_km(EV["lat"], EV["lon"], lat, lon)
        z = EV["depth_km"] + elev / 1000.0
        tp = MODEL.travel_time(z, dist, "P")[0] + (p_bias or {}).get(tail, 0.0)
        ts = MODEL.travel_time(z, dist, "S")[0] + (s_bias or {}).get(tail, 0.0)
        start = BASE_DT + timedelta(days=day)
        b64 = make_waveform_b64(
            code, SR, _iso(start),
            [(ORIGIN_OFFSET + tp, 4.0), (ORIGIN_OFFSET + ts, 6.0)],
            duration=DURATION,
        )
        waveforms.append(
            {
                "station": code, "latitude": lat, "longitude": lon,
                "elevation_m": elev, "sampling_rate": SR,
                "start_time": _iso(start), "data_b64": b64,
            }
        )
    if extra_station:
        # ninth station only present in these windows (holdout)
        code = f"{prefix}08"
        lat, lon, elev = 35.45, -117.55, 0.0
        dist = distance_km(EV["lat"], EV["lon"], lat, lon)
        z = EV["depth_km"] + elev / 1000.0
        tp = MODEL.travel_time(z, dist, "P")[0]
        ts = MODEL.travel_time(z, dist, "S")[0]
        start = BASE_DT + timedelta(days=day)
        b64 = make_waveform_b64(
            code, SR, _iso(start),
            [(ORIGIN_OFFSET + tp, 4.0), (ORIGIN_OFFSET + ts, 6.0)],
            duration=DURATION,
        )
        waveforms.append(
            {
                "station": code, "latitude": lat, "longitude": lon,
                "elevation_m": elev, "sampling_rate": SR,
                "start_time": _iso(start), "data_b64": b64,
            }
        )
    return {
        "waveforms": waveforms,
        "manual_picks": [],
        "model": {"name": "default", "version": "1.0"},
        "params": {"max_residual_s": 1.0, "min_stations": 4},
    }


def _post_events(client, days, prefix, p_bias=None, s_bias=None, extra=False,
                 scramble=False):
    for d in days:
        resp = client.post(
            "/v1/process",
            json=_event_window(d, prefix, p_bias, s_bias, extra_station=extra,
                               scramble=scramble),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["events"], f"no event located on day {d}"


def _cal_request(**overrides):
    body = {
        "version": "cal-1",
        "train_start": _iso(BASE_DT + timedelta(days=0)),
        "train_end": _iso(BASE_DT + timedelta(days=6)),
        "holdout_start": _iso(BASE_DT + timedelta(days=7)),
        "holdout_end": _iso(BASE_DT + timedelta(days=10)),
        "min_quality": "B",
        "min_events": 5,
        "min_span_days": 1.0,
        "min_station_coverage": 0.6,
        "huber_delta_s": 0.5,
    }
    body.update(overrides)
    return body


# --------------------------------------------------------------------------- #
# happy path: create candidate -> validate metrics -> activate -> use
# --------------------------------------------------------------------------- #
def test_calibration_create_activate_and_apply(client):
    _post_events(client, range(0, 7), "CA", P_BIAS, S_BIAS)
    _post_events(client, [8, 9], "CA", P_BIAS, S_BIAS)

    resp = client.post(
        "/v1/models/default/1.0/calibrations",
        json=_cal_request(
            train_end=_iso(BASE_DT + timedelta(days=7)),
            holdout_start=_iso(BASE_DT + timedelta(days=8)),
            holdout_end=_iso(BASE_DT + timedelta(days=10)),
        ),
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["created"] is True
    assert data["status"] == "candidate"
    assert data["version"] == "cal-1"

    metrics = data["metrics"]
    assert metrics["train_event_count"] == 7
    assert metrics["holdout_event_count"] == 2
    assert len(metrics["station_universe"]) == 8

    for phase in ("P", "S"):
        m = metrics["phases"][phase]
        assert m["sample_count"] >= 7 * 6
        assert m["station_count"] == 8
        assert m["holdout_sample_count"] >= 2 * 6
        assert m["holdout_rms_calibrated_s"] < m["holdout_rms_baseline_s"]
        assert m["train_rms_calibrated_s"] < m["train_rms_baseline_s"]

    # per-station, per-phase terms: counts / offsets / uncertainty.
    # P picks are sharp; S picks carry more AIC jitter, so S tolerances are
    # looser.  The zero-sum constraint and RMS improvement are exact/strict.
    by_key = {(s["station"], s["phase"]): s for s in data["stations"]}
    assert len(by_key) == 16
    offsets = {"P": {}, "S": {}}
    for (station, phase), row in by_key.items():
        tail = station[-2:]
        truth = P_BIAS[tail] if phase == "P" else S_BIAS[tail]
        assert row["sample_count"] >= 7
        assert row["uncertainty_s"] > 0
        assert abs(row["offset_s"] - truth) < (0.15 if phase == "P" else 0.25)
        offsets[phase][station] = row["offset_s"]
    assert sum(offsets["P"].values()) == pytest.approx(0.0, abs=1e-6)
    assert sum(offsets["S"].values()) == pytest.approx(0.0, abs=1e-6)

    # activation
    act = client.post(
        "/v1/models/default/1.0/calibrations/cal-1/activate"
    )
    assert act.status_code == 200, act.text
    assert act.json()["status"] == "active"

    # a calibrated run must beat the uncalibrated baseline on a fresh window
    fresh = _event_window(12, "CA", P_BIAS, S_BIAS)
    base = client.post("/v1/process", json=fresh).json()
    cal = client.post(
        "/v1/process", json={**fresh, "calibration": {"version": "cal-1"}}
    ).json()
    assert cal["calibration"] == {"version": "cal-1", "status": "active"}
    assert base["calibration"] is None
    be = max(base["events"], key=lambda e: e["station_count"])
    ce = max(cal["events"], key=lambda e: e["station_count"])
    assert ce["rms_residual_s"] < be["rms_residual_s"]
    # station count must not degrade when corrections are applied
    assert ce["station_count"] >= be["station_count"] - 1


def test_calibrated_replay_echoes_version(client):
    _post_events(client, range(0, 5), "RP", P_BIAS, S_BIAS)
    _post_events(client, [7, 8], "RP", P_BIAS, S_BIAS)
    r = client.post(
        "/v1/models/default/1.0/calibrations", json=_cal_request(version="cal-r")
    )
    assert r.status_code == 201
    client.post("/v1/models/default/1.0/calibrations/cal-r/activate")

    first = client.post("/v1/process", json=_event_window(11, "RP", P_BIAS, S_BIAS)).json()
    replay = client.post(
        "/v1/runs",
        json={
            "ingestion_id": first["ingestion_id"],
            "model": {"name": "default", "version": "1.0"},
            "params": {"max_residual_s": 1.0, "min_stations": 4},
            "calibration": {"version": "cal-r"},
        },
    )
    assert replay.status_code == 201, replay.text
    body = replay.json()
    assert body["calibration"]["version"] == "cal-r"
    # run id differs from the uncalibrated baseline run
    assert body["run_id"] != first["run_id"]
    ev = client.get(f"/v1/events/{body['events'][0]['event_id']}")
    assert ev.json()["calibration"]["version"] == "cal-r"


# --------------------------------------------------------------------------- #
# idempotency / immutability
# --------------------------------------------------------------------------- #
def test_calibration_creation_is_idempotent(client):
    _post_events(client, range(0, 5), "ID", P_BIAS, S_BIAS)
    _post_events(client, [7], "ID", P_BIAS, S_BIAS)
    body = _cal_request(version="dup")
    r1 = client.post("/v1/models/default/1.0/calibrations", json=body)
    r2 = client.post("/v1/models/default/1.0/calibrations", json=body)
    assert r1.status_code == 201 and r2.status_code == 200
    assert r1.json()["calibration_id"] == r2.json()["calibration_id"]
    assert r2.json()["created"] is False

    # same version label, different parameters -> conflict
    conflict = client.post(
        "/v1/models/default/1.0/calibrations",
        json=_cal_request(version="dup", min_events=6),
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "calibration_version_conflict"


def test_rejected_creation_leaves_no_state(client):
    _post_events(client, [0, 1], "RJ", P_BIAS, S_BIAS)
    _post_events(client, [7], "RJ", P_BIAS, S_BIAS)
    resp = client.post(
        "/v1/models/default/1.0/calibrations", json=_cal_request(version="ghost")
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "insufficient_events"
    listing = client.get("/v1/models/default/1.0/calibrations").json()
    assert listing["calibrations"] == []


# --------------------------------------------------------------------------- #
# explicit rejections
# --------------------------------------------------------------------------- #
def test_gate_time_span_too_short(client):
    _post_events(client, range(0, 5), "SP", P_BIAS, S_BIAS)
    _post_events(client, [7], "SP", P_BIAS, S_BIAS)
    # all five training events fall on the same five consecutive days;
    # demand an impossible span
    resp = client.post(
        "/v1/models/default/1.0/calibrations",
        json=_cal_request(
            version="sp", train_end=_iso(BASE_DT + timedelta(days=5)),
            min_span_days=100.0,
        ),
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "time_span_too_short"


def test_gate_station_coverage_too_low(client):
    _post_events(client, range(0, 5), "CV", P_BIAS, S_BIAS, extra=False)
    _post_events(client, [7], "CV", P_BIAS, S_BIAS, extra=True)
    resp = client.post(
        "/v1/models/default/1.0/calibrations",
        json=_cal_request(
            version="cv",
            train_end=_iso(BASE_DT + timedelta(days=5)),
            holdout_start=_iso(BASE_DT + timedelta(days=7)),
            holdout_end=_iso(BASE_DT + timedelta(days=8)),
            min_station_coverage=0.95,
        ),
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "station_coverage_too_low"


def test_invalid_windows_rejected(client):
    resp = client.post(
        "/v1/models/default/1.0/calibrations",
        json=_cal_request(version="bad", train_end=_iso(BASE_DT - timedelta(days=1))),
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "invalid_time_window"
    resp = client.post(
        "/v1/models/default/1.0/calibrations",
        json=_cal_request(
            version="bad2",
            holdout_start=_iso(BASE_DT + timedelta(days=3)),
        ),
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "invalid_time_window"


def test_holdout_gate_blocks_activation(client):
    # train on stable station biases, hold out events whose biases are rotated
    # across stations: the learned terms mis-correct the holdout arrivals.
    _post_events(client, range(0, 5), "HG", P_BIAS, S_BIAS)
    _post_events(client, [7], "HG", scramble=True)
    r = client.post(
        "/v1/models/default/1.0/calibrations",
        json=_cal_request(
            version="hg",
            train_end=_iso(BASE_DT + timedelta(days=5)),
            holdout_end=_iso(BASE_DT + timedelta(days=8)),
        ),
    )
    assert r.status_code == 201, r.text
    worse = [
        ph for ph, m in r.json()["metrics"]["phases"].items()
        if not (m["holdout_rms_calibrated_s"] < m["holdout_rms_baseline_s"])
    ]
    assert worse, "expected at least one phase to fail the holdout gate"
    act = client.post("/v1/models/default/1.0/calibrations/hg/activate")
    assert act.status_code == 409
    assert act.json()["detail"]["code"] == "holdout_gate_failed"
    # candidates cannot be used for processing
    use = client.post(
        "/v1/process",
        json={**_event_window(11, "HG", P_BIAS, S_BIAS),
              "calibration": {"version": "hg"}},
    )
    assert use.status_code == 409
    assert use.json()["detail"]["code"] == "calibration_not_active"


# --------------------------------------------------------------------------- #
# lifecycle: active/published, not found
# --------------------------------------------------------------------------- #
def test_activation_lifecycle_and_published_immutability(client):
    _post_events(client, range(0, 5), "LC", P_BIAS, S_BIAS)
    _post_events(client, [7], "LC", P_BIAS, S_BIAS)
    for ver in ("v-a", "v-b"):
        r = client.post(
            "/v1/models/default/1.0/calibrations",
            json=_cal_request(version=ver, huber_delta_s=0.4 if ver == "v-b" else 0.5),
        )
        assert r.status_code == 201, r.text

    a1 = client.post("/v1/models/default/1.0/calibrations/v-a/activate")
    assert a1.status_code == 200 and a1.json()["status"] == "active"
    # repeat activation is idempotent
    a1b = client.post("/v1/models/default/1.0/calibrations/v-a/activate")
    assert a1b.status_code == 200 and a1b.json()["status"] == "active"

    a2 = client.post("/v1/models/default/1.0/calibrations/v-b/activate")
    assert a2.status_code == 200 and a2.json()["status"] == "active"

    listing = {
        c["version"]: c["status"]
        for c in client.get("/v1/models/default/1.0/calibrations").json()["calibrations"]
    }
    assert listing["v-a"] == "published"
    assert listing["v-b"] == "active"

    # published versions cannot be re-activated
    again = client.post("/v1/models/default/1.0/calibrations/v-a/activate")
    assert again.status_code == 409
    assert again.json()["detail"]["code"] == "calibration_published_immutable"
    # ... but remain usable for processing/replay
    use = client.post(
        "/v1/process",
        json={**_event_window(11, "LC", P_BIAS, S_BIAS),
              "calibration": {"version": "v-a"}},
    )
    assert use.status_code == 200
    assert use.json()["calibration"]["status"] == "published"


def test_calibrated_process_is_idempotent(client):
    _post_events(client, range(0, 5), "IP", P_BIAS, S_BIAS)
    _post_events(client, [7], "IP", P_BIAS, S_BIAS)
    r = client.post(
        "/v1/models/default/1.0/calibrations", json=_cal_request(version="ip")
    )
    assert r.status_code == 201
    client.post("/v1/models/default/1.0/calibrations/ip/activate")

    fresh = _event_window(14, "IP", P_BIAS, S_BIAS)
    body = {**fresh, "calibration": {"version": "ip"}}
    r1 = client.post("/v1/process", json=body)
    r2 = client.post("/v1/process", json=body)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["created"] is True
    assert r2.json()["created"] is False
    assert r1.json()["run_id"] == r2.json()["run_id"]


def test_calibration_not_found(client):
    r = client.get("/v1/models/default/1.0/calibrations/nope")
    assert r.status_code == 404 and r.json()["detail"]["code"] == "calibration_not_found"
    act = client.post("/v1/models/default/1.0/calibrations/nope/activate")
    assert act.status_code == 404
    use = client.post(
        "/v1/process",
        json={**_event_window(11, "ZZ"), "calibration": {"version": "nope"}},
    )
    assert use.status_code == 404 and use.json()["detail"]["code"] == "calibration_not_found"


# --------------------------------------------------------------------------- #
# concurrency: simultaneous activation of two candidates
# --------------------------------------------------------------------------- #
def test_concurrent_activation_is_atomic(client):
    _post_events(client, range(0, 5), "CC", P_BIAS, S_BIAS)
    _post_events(client, [7], "CC", P_BIAS, S_BIAS)
    for ver, delta in (("c-1", 0.5), ("c-2", 0.4)):
        r = client.post(
            "/v1/models/default/1.0/calibrations",
            json=_cal_request(
                version=ver,
                huber_delta_s=delta,
                train_end=_iso(BASE_DT + timedelta(days=5)),
                holdout_end=_iso(BASE_DT + timedelta(days=8)),
            ),
        )
        assert r.status_code == 201, r.text

    barrier = threading.Barrier(2)
    results = {}

    def worker(ver, key):
        barrier.wait()
        results[key] = client.post(
            f"/v1/models/default/1.0/calibrations/{ver}/activate"
        ).status_code

    t1 = threading.Thread(target=worker, args=("c-1", "a"))
    t2 = threading.Thread(target=worker, args=("c-2", "b"))
    t1.start(); t2.start(); t1.join(); t2.join()

    assert sorted(results.values()) == [200, 409], results
    listing = client.get("/v1/models/default/1.0/calibrations").json()["calibrations"]
    active = [c for c in listing if c["status"] == "active"]
    assert len(active) == 1

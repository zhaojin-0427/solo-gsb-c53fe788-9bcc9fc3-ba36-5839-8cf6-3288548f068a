"""End-to-end tests for the station impact audit (leave-one-station-out) module."""
from __future__ import annotations

import threading

from tests.conftest import synthetic_event_payload

from tests.test_calibration import (  # noqa: E402
    P_BIAS,
    S_BIAS,
    _cal_request,
    _event_window as _cal_event_window,
    _post_events,
)


def _process(client, prefix, **kwargs):
    resp = client.post(
        "/v1/process", json=synthetic_event_payload(prefix=prefix, **kwargs)[0]
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _best(data):
    return max(data["events"], key=lambda e: e["station_count"])


def _audit(client, event_id, min_keep_stations=4):
    return client.post(
        "/v1/station-audits",
        json={"event_id": event_id, "min_keep_stations": min_keep_stations},
    )


# --------------------------------------------------------------------------- #
# freezing + report binding
# --------------------------------------------------------------------------- #
def test_audit_freezes_baseline_picks_model_and_params(client):
    data = _process(client, "AF", n_stations=8)
    event = _best(data)
    resp = _audit(client, event["event_id"])
    assert resp.status_code == 201, resp.text
    rep = resp.json()

    assert rep["created"] is True
    assert rep["event_id"] == event["event_id"]
    assert rep["run_id"] == data["run_id"]
    assert rep["ingestion_id"] == data["ingestion_id"]
    assert rep["station_count"] == 8
    assert rep["result_count"] == 8

    # frozen context: original velocity model, (uncalibrated) calibration
    # coordinate and processing parameters
    frozen = rep["frozen"]
    assert frozen["model"]["name"] == "default"
    assert frozen["model"]["version"] == "1.0"
    assert isinstance(frozen["model"]["layers"], list) and frozen["model"]["layers"]
    assert frozen["calibration"] is None
    assert frozen["params"] == {"max_residual_s": 1.0, "min_stations": 4}
    assert frozen["min_keep_stations"] == 4
    assert len(frozen["stations"]) == 8

    # baseline snapshot equals the stored original event solution and carries
    # the original associated pick subset (P + S of every participating station)
    baseline = rep["baseline"]
    assert baseline["event_id"] == event["event_id"]
    assert baseline["quality"] == event["quality"]
    assert baseline["station_count"] == event["station_count"]
    assert baseline["rms_residual_s"] == event["rms_residual_s"]
    assert baseline["origin_time"] == event["origin_time"]
    baseline_pick_ids = {p["pick_id"] for p in baseline["picks"]}
    event_pick_ids = {p["pick_id"] for p in event["picks"]}
    assert baseline_pick_ids == event_pick_ids
    stations_of_picks = {p["station"] for p in baseline["picks"]}
    assert stations_of_picks == set(baseline["stations"])

    # report is bound to event/run and carries a stable params hash
    assert len(rep["params_hash"]) == 64
    assert rep["request"]["event_id"] == event["event_id"]
    assert rep["request"]["run_id"] == data["run_id"]


def test_audit_freeze_includes_calibration_version(client):
    """A calibrated run's audit freezes and re-applies the calibration terms."""
    _post_events(client, range(0, 5), "CA", P_BIAS, S_BIAS)
    _post_events(client, [7], "CA", P_BIAS, S_BIAS)
    created = client.post(
        "/v1/models/default/1.0/calibrations",
        json=_cal_request(version="cal-audit"),
    )
    assert created.status_code == 201, created.text
    activated = client.post(
        "/v1/models/default/1.0/calibrations/cal-audit/activate"
    )
    assert activated.status_code == 200

    window = _cal_event_window(11, "CA", P_BIAS, S_BIAS)
    data = client.post(
        "/v1/process", json={**window, "calibration": {"version": "cal-audit"}}
    ).json()
    event = _best(data)
    assert data["calibration"] == {"version": "cal-audit", "status": "active"}

    resp = _audit(client, event["event_id"])
    assert resp.status_code == 201, resp.text
    cal = resp.json()["frozen"]["calibration"]
    assert cal["version"] == "cal-audit"
    assert cal["status"] in ("active", "published")
    assert cal["terms"], "frozen calibration carries its per-station P/S terms"
    assert {(t["station"], t["phase"]) for t in cal["terms"]}
    assert resp.json()["baseline"]["quality"] == event["quality"]


# --------------------------------------------------------------------------- #
# outcomes: event loss, quality downgrade, stable; key stations
# --------------------------------------------------------------------------- #
def test_eight_station_event_every_removal_downgrades_quality_and_is_key(client):
    event = _best(_process(client, "AK", n_stations=8))
    # grade A requires >= 8 stations, so removing any one station drops to B
    assert event["quality"] == "A"
    rep = _audit(client, event["event_id"]).json()
    assert rep["key_station_count"] == 8
    assert len(rep["key_stations"]) == 8
    for r in rep["results"]:
        assert r["outcome"] == "survived"
        assert r["is_key_station"] is True
        assert r["quality_downgraded"] is True
        assert r["solution"]["quality"] == "B"
        assert r["solution"]["station_count"] == 7
        assert r["delta"]["station_count"] == -1
        assert set(r["delta"]) == {
            "origin_time_s", "latitude", "longitude", "depth_km",
            "rms_s", "station_count",
        }


def test_four_station_event_every_removal_loses_event(client):
    event = _best(_process(client, "AL", n_stations=4, min_stations=4))
    rep = _audit(client, event["event_id"], min_keep_stations=2).json()
    assert rep["key_station_count"] == 4
    for r in rep["results"]:
        assert r["outcome"] == "event_lost"
        assert r["is_key_station"] is True
        assert r["quality_downgraded"] is True
        assert r["solution"] is None
        assert r["delta"] is None
        assert r["rms_increase_s"] is None
        # three stations remain -> the 4-unknown location is under-determined
        assert r["unresolvable_reason"] == "location_failed"
        assert r["reason_counts"]
        assert r["associated_pick_ids"] == []
        # every remaining pick is reported as dropped from the lost event
        assert len(r["dropped_pick_ids"]) == 6
        assert len(r["removed_pick_ids"]) == 2  # the removed station's P + S
        assert r["stations"] == []


def test_stable_quality_stations_are_not_key(client):
    event = _best(_process(client, "AS", n_stations=10, min_stations=4))
    assert event["quality"] == "A"
    rep = _audit(client, event["event_id"]).json()
    assert rep["key_station_count"] == 0
    for r in rep["results"]:
        assert r["outcome"] == "survived"
        assert r["is_key_station"] is False
        assert r["quality_downgraded"] is False
        assert r["solution"]["quality"] == "A"


def test_each_rerun_removes_only_that_station_ps_picks(client):
    event = _best(_process(client, "AP", n_stations=8))
    rep = _audit(client, event["event_id"]).json()
    base_ids = {p["pick_id"]: p for p in rep["baseline"]["picks"]}
    for r in rep["results"]:
        removed = set(r["removed_pick_ids"])
        retained = set(r["retained_pick_ids"])
        dropped = set(r["dropped_pick_ids"])
        # removed/retained/dropped partition the original pick set exactly
        assert removed | retained | dropped == set(base_ids)
        assert not (removed & retained) and not (retained & dropped)
        # only the audited station's P and S picks are removed
        assert len(removed) == 2
        assert {base_ids[i]["station"] for i in removed} == {r["station"]}
        assert {base_ids[i]["phase"] for i in removed} == {"P", "S"}
        # retained picks all belong to the relocated event; stations excludes
        # the removed station
        assert r["station"] not in r["stations"]
        assert len(r["stations"]) == r["solution"]["station_count"]


# --------------------------------------------------------------------------- #
# stable ordering: event loss, downgrade depth, RMS increase, station code
# --------------------------------------------------------------------------- #
def test_results_are_stably_sorted(client):
    # 8 stations: every removal downgrades A -> B, so results sit in one
    # severity group ordered by RMS increase (desc), then station code.
    event = _best(_process(client, "AO", n_stations=8))
    rep = _audit(client, event["event_id"]).json()
    rows = rep["results"]
    # Monotone non-increasing RMS increase except where 4-decimal display
    # rounding merges two full-precision keys into the same bucket; those
    # buckets must still follow ascending station codes (final tie-break).
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        assert a["rms_increase_s"] >= b["rms_increase_s"] or (
            abs(a["rms_increase_s"] - b["rms_increase_s"]) < 1e-4
            and a["station"] < b["station"]
        )

    # sort_index is gapless and follows the presentation order
    assert [r["sort_index"] for r in rows] == list(range(8))

    # all-lost report: equal severity/RMS, so station code is the total order
    lost_event = _best(_process(client, "AX", n_stations=4, min_stations=4))
    lost_rep = _audit(client, lost_event["event_id"], min_keep_stations=2).json()
    lost_stations = [r["station"] for r in lost_rep["results"]]
    assert lost_stations == sorted(lost_stations)


def test_event_lost_sorts_before_survived(client):
    """Outcome severity dominates the order key (lost first)."""
    from app.services import station_audit as svc

    lost = {"station": "Z9", "_sort": (svc._SEVERITY_LOST, 0, 0.0)}
    downgraded = {"station": "A1", "_sort": (svc._SEVERITY_DOWNGRADED, 0, 0.01)}
    stable = {"station": "A2", "_sort": (svc._SEVERITY_STABLE, 0, 5.0)}
    ordered = sorted([stable, downgraded, lost], key=lambda r: (*r["_sort"], r["station"]))
    assert [r["station"] for r in ordered] == ["Z9", "A1", "A2"]


# --------------------------------------------------------------------------- #
# idempotency / concurrency
# --------------------------------------------------------------------------- #
def test_audit_is_idempotent(client):
    event = _best(_process(client, "AI", n_stations=8))
    first = _audit(client, event["event_id"])
    assert first.status_code == 201
    second = _audit(client, event["event_id"])
    assert second.status_code == 200
    assert second.json()["created"] is False
    assert second.json()["audit_id"] == first.json()["audit_id"]
    assert second.json()["params_hash"] == first.json()["params_hash"]
    assert second.json()["results"] == first.json()["results"]


def test_different_min_keep_is_a_distinct_audit(client):
    event = _best(_process(client, "AM", n_stations=6, min_stations=2))
    a = _audit(client, event["event_id"], min_keep_stations=2)
    b = _audit(client, event["event_id"], min_keep_stations=3)
    assert a.status_code == 201 and b.status_code == 201
    assert a.json()["audit_id"] != b.json()["audit_id"]
    assert a.json()["params_hash"] != b.json()["params_hash"]


def test_concurrent_identical_audits_collapse(client):
    event = _best(_process(client, "AC", n_stations=8))
    results = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        r = _audit(client, event["event_id"])
        results.append((r.status_code, r.json()["audit_id"]))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start(); t2.start(); t1.join(); t2.join()

    statuses = sorted(s for s, _ in results)
    assert statuses == [200, 201], results
    assert results[0][1] == results[1][1]


# --------------------------------------------------------------------------- #
# pagination
# --------------------------------------------------------------------------- #
def test_results_pagination_stable_cursor(client):
    event = _best(_process(client, "APG", n_stations=8))
    rep = _audit(client, event["event_id"]).json()
    audit_id = rep["audit_id"]
    seen, cursor = [], None
    for _ in range(10):
        params = {"limit": 3}
        if cursor:
            params["cursor"] = cursor
        page = client.get(f"/v1/station-audits/{audit_id}/results", params=params)
        assert page.status_code == 200, page.text
        data = page.json()
        seen.extend(r["station"] for r in data["results"])
        cursor = data["next_cursor"]
        if cursor is None:
            break
    assert seen == [r["station"] for r in rep["results"]]
    assert len(seen) == len(set(seen)) == 8


def test_result_cursor_bound_to_audit(client):
    e1 = _best(_process(client, "AB1", n_stations=8))
    e2 = _best(_process(client, "AB2", n_stations=8))
    a1 = _audit(client, e1["event_id"]).json()
    a2 = _audit(client, e2["event_id"]).json()
    page = client.get(
        f"/v1/station-audits/{a1['audit_id']}/results", params={"limit": 2}
    ).json()
    bad = client.get(
        f"/v1/station-audits/{a2['audit_id']}/results",
        params={"limit": 2, "cursor": page["next_cursor"]},
    )
    assert bad.status_code == 422
    assert bad.json()["detail"]["code"] == "invalid_cursor"


def test_audit_list_pagination_and_filters(client):
    e1 = _best(_process(client, "AL1", n_stations=8))
    e2 = _best(_process(client, "AL2", n_stations=8))
    a1 = _audit(client, e1["event_id"]).json()
    a2 = _audit(client, e2["event_id"]).json()

    by_event = client.get(
        "/v1/station-audits", params={"event_id": e1["event_id"]}
    ).json()
    assert [a["audit_id"] for a in by_event["audits"]] == [a1["audit_id"]]

    by_run = client.get(
        "/v1/station-audits", params={"run_id": e2 and _run_id(client, e2)}
    ).json()
    assert {a["audit_id"] for a in by_run["audits"]} == {a2["audit_id"]}

    # keyset pagination walks every audit exactly once
    seen, cursor = set(), None
    for _ in range(10):
        params = {"limit": 1}
        if cursor:
            params["cursor"] = cursor
        data = client.get("/v1/station-audits", params=params).json()
        seen.update(a["audit_id"] for a in data["audits"])
        cursor = data["next_cursor"]
        if cursor is None:
            break
    assert seen == {a1["audit_id"], a2["audit_id"]}

    # list cursor is bound to the event/run filter fingerprint
    page = client.get(
        "/v1/station-audits", params={"limit": 1, "event_id": e1["event_id"]}
    ).json()
    mismatched = client.get(
        "/v1/station-audits",
        params={"limit": 1, "event_id": e2["event_id"], "cursor": page["next_cursor"]},
    )
    assert mismatched.status_code == 422
    assert mismatched.json()["detail"]["code"] == "invalid_cursor"


def _run_id(client, event):
    return client.get(f"/v1/events/{event['event_id']}").json()["run_id"]


# --------------------------------------------------------------------------- #
# rejections
# --------------------------------------------------------------------------- #
def test_unknown_event_returns_404(client):
    resp = _audit(client, "0" * 32)
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "event_not_found"


def test_min_keep_stations_too_high_is_rejected(client):
    event = _best(_process(client, "AH", n_stations=4, min_stations=4))
    # removing one station leaves 3 < 4, so the audit would be trivially lost
    resp = _audit(client, event["event_id"], min_keep_stations=4)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "min_keep_stations_too_high"


def test_audit_not_found(client):
    resp = client.get("/v1/station-audits/deadbeef")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "station_audit_not_found"
    results = client.get("/v1/station-audits/deadbeef/results")
    assert results.status_code == 404
    assert results.json()["detail"]["code"] == "station_audit_not_found"


def test_list_unknown_event_filter_404(client):
    resp = client.get("/v1/station-audits", params={"event_id": "f" * 32})
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "event_not_found"


# --------------------------------------------------------------------------- #
# read-only: original event, run and catalog content never change
# --------------------------------------------------------------------------- #
def test_audit_does_not_mutate_event_run_or_catalog(client):
    data = _process(client, "AR", n_stations=8)
    event = _best(data)
    run_before = client.get(f"/v1/runs/{data['run_id']}").json()
    event_before = client.get(f"/v1/events/{event['event_id']}").json()

    _audit(client, event["event_id"])
    # repeat as well
    _audit(client, event["event_id"])

    run_after = client.get(f"/v1/runs/{data['run_id']}").json()
    event_after = client.get(f"/v1/events/{event['event_id']}").json()
    assert run_after == run_before
    assert event_after == event_before

    # the audit created no runs/events/picks of its own
    assert client.get(f"/v1/runs/{data['run_id']}").json()["events"] == run_before["events"]

    # process/replay/compare semantics are untouched
    again = client.post("/v1/process", json=synthetic_event_payload(prefix="AR")[0])
    assert again.json()["run_id"] == data["run_id"]
    assert again.json()["created"] is False

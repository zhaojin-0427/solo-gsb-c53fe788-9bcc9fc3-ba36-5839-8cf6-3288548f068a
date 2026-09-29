"""End-to-end tests for the event location uncertainty assessment module."""
from __future__ import annotations

import math
import threading
from datetime import datetime

import numpy as np
import pytest

from app.config import DEFAULT_MODEL_LAYERS
from app.services import location_uncertainty as lu
from app.services.location import Observation, locate_event
from app.services.traveltime import LayeredModel

from tests.conftest import synthetic_event_payload
from tests.test_calibration import (  # noqa: E402
    P_BIAS,
    S_BIAS,
    _cal_request,
    _event_window as _cal_event_window,
    _post_events,
)

EPOCH = datetime(1970, 1, 1)


def _process(client, prefix, **kwargs):
    resp = client.post(
        "/v1/process", json=synthetic_event_payload(prefix=prefix, **kwargs)[0]
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _best(data):
    return max(data["events"], key=lambda e: e["station_count"])


def _assess(client, event_id, n_resamples=200, alpha=0.9, seed=42):
    return client.post(
        "/v1/location-uncertainties",
        json={
            "event_id": event_id,
            "n_resamples": n_resamples,
            "alpha": alpha,
            "seed": seed,
        },
    )


def _all_samples(client, assessment_id, status="all", page=50):
    """Walk every page and return the full ordered sample list."""
    out, cursor = [], None
    while True:
        url = f"/v1/location-uncertainties/{assessment_id}/samples?limit={page}&status={status}"
        if cursor:
            url += f"&cursor={cursor}"
        resp = client.get(url)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        out.extend(body["samples"])
        cursor = body["next_cursor"]
        if not cursor:
            return out


def _epoch(iso: str) -> float:
    return (datetime.strptime(iso, "%Y-%m-%dT%H:%M:%S.%fZ") - EPOCH).total_seconds()


# --------------------------------------------------------------------------- #
# unit: nearest-rank quantiles and multiplicity weighting
# --------------------------------------------------------------------------- #
def test_nearest_rank_quantile_unit():
    xs = list(range(1, 11))  # 1..10
    assert lu._nearest_rank(xs, 0.1) == 1    # ceil(1.0)  -> rank 1
    assert lu._nearest_rank(xs, 0.25) == 3   # ceil(2.5)  -> rank 3
    assert lu._nearest_rank(xs, 0.5) == 5    # ceil(5.0)  -> rank 5
    assert lu._nearest_rank(xs, 0.95) == 10  # ceil(9.5)  -> rank 10
    assert lu._nearest_rank(xs, 0.01) == 1   # clamped to rank 1
    assert lu._nearest_rank(xs, 1.0) == 10
    lo, hi = lu._interval(list(range(1, 21)), 0.8)  # tail = 0.1
    assert lo == 2   # ceil(0.1 * 20)
    assert hi == 18  # ceil(0.9 * 20)


def test_weight_multiplication_matches_observation_duplication():
    """A station weighted by its draw multiplicity localizes like the same
    station's observations duplicated that many times (and unlike the
    unweighted set, given noisy arrivals)."""
    model = LayeredModel.from_dicts(DEFAULT_MODEL_LAYERS)
    stations = {
        "EX00": (35.25, -118.0, 0.0),
        "EX01": (35.0, -117.75, 50.0),
        "EX02": (34.75, -118.0, 100.0),
        "EX03": (35.0, -118.25, 150.0),
    }
    from app.services.association import distance_km

    obs = []
    t0 = 1_000_000.0
    for code, (lat, lon, elev) in stations.items():
        dist = distance_km(35.05, -117.95, lat, lon)
        z = 12.0 + elev / 1000.0
        # perturb EX00's arrivals so the weighting actually moves the solution
        jitter = 0.4 if code == "EX00" else 0.0
        obs.append(Observation(code, "P", t0 + model.travel_time(z, dist, "P")[0] + jitter, 1.0))
        obs.append(Observation(code, "S", t0 + model.travel_time(z, dist, "S")[0] + jitter, 0.5))

    weighted = [
        Observation(o.station, o.phase, o.t_obs, o.weight * (3 if o.station == "EX00" else 1))
        for o in obs
    ]
    duplicated = []
    for o in obs:
        duplicated.extend([o] * (3 if o.station == "EX00" else 1))

    lw = locate_event(model, weighted, stations)
    ld = locate_event(model, duplicated, stations)
    l_plain = locate_event(model, obs, stations)
    assert lw is not None and ld is not None and l_plain is not None

    # Weighting and duplication solve the same normal equations; the iterative
    # locator settles within its own convergence precision (~1e-2 s, ~100 m).
    assert lw.t0 == pytest.approx(ld.t0, abs=1e-2)
    assert lw.lat == pytest.approx(ld.lat, abs=1e-3)
    assert lw.lon == pytest.approx(ld.lon, abs=1e-3)
    assert lw.depth_km == pytest.approx(ld.depth_km, abs=1e-1)

    # ...while dropping the multiplicity weight moves the solution clearly
    # further than the weighted/duplicated agreement.
    assert abs(lw.t0 - l_plain.t0) > 10.0 * abs(lw.t0 - ld.t0) + 1e-3
    assert abs(lw.depth_km - l_plain.depth_km) > 10.0 * abs(lw.depth_km - ld.depth_km) + 1e-3


# --------------------------------------------------------------------------- #
# freezing + report binding
# --------------------------------------------------------------------------- #
def test_assessment_freezes_picks_weights_model_and_params(client):
    data = _process(client, "UF", n_stations=8)
    event = _best(data)
    resp = _assess(client, event["event_id"])
    assert resp.status_code == 201, resp.text
    rep = resp.json()

    assert rep["created"] is True
    assert rep["event_id"] == event["event_id"]
    assert rep["run_id"] == data["run_id"]
    assert rep["ingestion_id"] == data["ingestion_id"]
    assert rep["n_resamples"] == 200
    assert rep["alpha"] == 0.9
    assert rep["seed"] == 42
    assert rep["min_solvable_required"] == 160

    # frozen context: original velocity model, (uncalibrated) calibration
    # coordinate, processing parameters, stations and the associated picks
    frozen = rep["frozen"]
    assert frozen["model"]["name"] == "default"
    assert frozen["model"]["version"] == "1.0"
    assert isinstance(frozen["model"]["layers"], list) and frozen["model"]["layers"]
    assert frozen["calibration"] is None
    assert frozen["params"] == {"max_residual_s": 1.0, "min_stations": 4}
    assert len(frozen["stations"]) == 8
    frozen_pick_ids = {p["pick_id"] for p in frozen["picks"]}
    assert frozen_pick_ids == {p["pick_id"] for p in event["picks"]}
    for p in frozen["picks"]:
        assert p["phase"] in ("P", "S")
        assert p["weight"] > 0.0

    # baseline snapshot equals the stored original event solution
    baseline = rep["baseline"]
    assert baseline["event_id"] == event["event_id"]
    assert baseline["origin_time"] == event["origin_time"]
    assert baseline["latitude"] == event["latitude"]
    assert baseline["longitude"] == event["longitude"]
    assert baseline["depth_km"] == event["depth_km"]
    assert baseline["rms_residual_s"] == event["rms_residual_s"]
    assert baseline["station_count"] == event["station_count"]
    assert baseline["quality"] == event["quality"]
    assert baseline["stations"] == sorted({p["station"] for p in event["picks"]})

    # request is bound to the event/run and the bootstrap parameters
    assert rep["request"] == {
        "event_id": event["event_id"],
        "run_id": data["run_id"],
        "n_resamples": 200,
        "alpha": 0.9,
        "seed": 42,
    }


def test_assessment_freeze_includes_calibration_version(client):
    """A calibrated run's assessment freezes and re-applies the calibration."""
    _post_events(client, range(0, 5), "UC", P_BIAS, S_BIAS)
    _post_events(client, [7], "UC", P_BIAS, S_BIAS)
    created = client.post(
        "/v1/models/default/1.0/calibrations",
        json=_cal_request(version="cal-unc"),
    )
    assert created.status_code == 201, created.text
    activated = client.post("/v1/models/default/1.0/calibrations/cal-unc/activate")
    assert activated.status_code == 200

    window = _cal_event_window(11, "UC", P_BIAS, S_BIAS)
    data = client.post(
        "/v1/process", json={**window, "calibration": {"version": "cal-unc"}}
    ).json()
    event = _best(data)
    assert data["calibration"] == {"version": "cal-unc", "status": "active"}

    resp = _assess(client, event["event_id"])
    assert resp.status_code == 201, resp.text
    cal = resp.json()["frozen"]["calibration"]
    assert cal["version"] == "cal-unc"
    assert cal["status"] in ("active", "published")
    assert cal["terms"], "frozen calibration carries its per-station P/S terms"
    assert {(t["station"], t["phase"]) for t in cal["terms"]}


# --------------------------------------------------------------------------- #
# statistics: intervals, covariance, solvable rate
# --------------------------------------------------------------------------- #
def test_statistics_match_recomputation_from_samples(client):
    event = _best(_process(client, "US", n_stations=8))
    rep = _assess(client, event["event_id"], alpha=0.9, seed=7).json()
    assert rep["n_solved"] + rep["n_failed"] == rep["n_resamples"]
    assert rep["solvable_rate"] == round(rep["n_solved"] / rep["n_resamples"], 4)

    samples = _all_samples(client, rep["assessment_id"])
    assert len(samples) == rep["n_resamples"]
    solved = [s for s in samples if s["status"] == "solved"]
    failed = [s for s in samples if s["status"] == "failed"]
    assert len(solved) == rep["n_solved"]
    assert len(failed) == rep["n_failed"]
    assert {s["failure_reason"] for s in failed} <= {
        "insufficient_stations", "location_failed", "not_converged"
    }
    from collections import Counter

    assert Counter(s["failure_reason"] for s in failed) == rep["failure_reasons"]

    # nearest-rank two-sided (1-alpha)/2 intervals recomputed from the samples
    ci = rep["confidence_intervals"]
    assert ci["alpha"] == 0.9
    assert ci["method"] == "nearest_rank"
    assert ci["sample_count"] == len(solved)
    n = len(solved)
    tail = (1.0 - 0.9) / 2.0
    lo_rank = min(max(math.ceil(tail * n), 1), n) - 1
    hi_rank = min(max(math.ceil((1.0 - tail) * n), 1), n) - 1

    lats = sorted(s["solution"]["latitude"] for s in solved)
    lons = sorted(s["solution"]["longitude"] for s in solved)
    depths = sorted(s["solution"]["depth_km"] for s in solved)
    times = sorted(s["solution"]["origin_time"] for s in solved)
    assert ci["latitude"] == {"lower": lats[lo_rank], "upper": lats[hi_rank]}
    assert ci["longitude"] == {"lower": lons[lo_rank], "upper": lons[hi_rank]}
    assert ci["depth_km"] == {"lower": depths[lo_rank], "upper": depths[hi_rank]}
    assert ci["origin_time"] == {"lower": times[lo_rank], "upper": times[hi_rank]}
    for key in ("latitude", "longitude", "depth_km"):
        assert ci[key]["lower"] <= ci[key]["upper"]
    assert ci["origin_time"]["lower"] <= ci["origin_time"]["upper"]

    # sample covariance over the solvable resamples only (ddof=1)
    cov = rep["covariance"]
    assert cov["parameters"] == ["origin_time_s", "latitude", "longitude", "depth_km"]
    assert cov["sample_count"] == len(solved)
    arr = np.array(
        [
            [
                _epoch(s["solution"]["origin_time"]),
                s["solution"]["latitude"],
                s["solution"]["longitude"],
                s["solution"]["depth_km"],
            ]
            for s in solved
        ]
    )
    expected = np.cov(arr, rowvar=False, ddof=1)
    got = np.array(cov["matrix"])
    assert got.shape == (4, 4)
    assert np.allclose(got, expected, rtol=5e-2, atol=1e-12)
    assert np.allclose(got, got.T, atol=1e-12)  # symmetric
    assert all(got[i, i] > 0.0 for i in range(4))  # positive variances


def test_bootstrap_draws_are_deterministic_per_seed(client):
    """Same seed -> identical resamples; a different seed -> different draws."""
    event = _best(_process(client, "UD", n_stations=8))
    rep_a = _assess(client, event["event_id"], alpha=0.9, seed=123).json()
    # same seed, different confidence level: the resamples must be identical
    rep_b = _assess(client, event["event_id"], alpha=0.8, seed=123).json()
    assert rep_a["assessment_id"] != rep_b["assessment_id"]

    samples_a = _all_samples(client, rep_a["assessment_id"])
    samples_b = _all_samples(client, rep_b["assessment_id"])
    assert [s["stations"] for s in samples_a] == [s["stations"] for s in samples_b]
    assert [s["status"] for s in samples_a] == [s["status"] for s in samples_b]
    assert [s["solution"] for s in samples_a] == [s["solution"] for s in samples_b]

    rep_c = _assess(client, event["event_id"], alpha=0.9, seed=124).json()
    samples_c = _all_samples(client, rep_c["assessment_id"])
    assert [s["stations"] for s in samples_a] != [s["stations"] for s in samples_c]


def test_station_multiset_matches_original_station_count(client):
    event = _best(_process(client, "UM", n_stations=8))
    rep = _assess(client, event["event_id"]).json()
    codes = set(rep["baseline"]["stations"])
    assert len(codes) == 8
    for s in _all_samples(client, rep["assessment_id"]):
        assert sum(s["stations"].values()) == 8  # as many draws as stations
        assert set(s["stations"]) <= codes
        assert s["distinct_station_count"] == len(s["stations"])
        if s["status"] == "solved":
            sol = s["solution"]
            assert sol["origin_time"].endswith("Z")
            assert -90.0 <= sol["latitude"] <= 90.0
            assert -180.0 <= sol["longitude"] <= 180.0
            assert sol["depth_km"] >= 0.0
            assert sol["rms_residual_s"] >= 0.0
            assert sol["iterations"] >= 1
        else:
            assert s["solution"] is None
            assert s["failure_reason"] is not None


# --------------------------------------------------------------------------- #
# idempotency + concurrency
# --------------------------------------------------------------------------- #
def test_assessment_is_idempotent(client):
    event = _best(_process(client, "UI", n_stations=8))
    first = _assess(client, event["event_id"])
    assert first.status_code == 201, first.text
    second = _assess(client, event["event_id"])
    assert second.status_code == 200
    assert second.json()["created"] is False
    assert second.json()["assessment_id"] == first.json()["assessment_id"]
    # identical report content apart from the created flag
    a = {k: v for k, v in first.json().items() if k != "created"}
    b = {k: v for k, v in second.json().items() if k != "created"}
    assert a == b
    # exactly one persisted assessment for the event
    listed = client.get(f"/v1/location-uncertainties?event_id={event['event_id']}")
    assert len(listed.json()["assessments"]) == 1


def test_distinct_params_or_seed_are_distinct_assessments(client):
    event = _best(_process(client, "UP", n_stations=8))
    ids = set()
    for kwargs in (
        {"seed": 1, "alpha": 0.9, "n_resamples": 200},
        {"seed": 2, "alpha": 0.9, "n_resamples": 200},
        {"seed": 1, "alpha": 0.8, "n_resamples": 200},
        {"seed": 1, "alpha": 0.9, "n_resamples": 201},
    ):
        resp = _assess(client, event["event_id"], **kwargs)
        assert resp.status_code == 201, resp.text
        ids.add(resp.json()["assessment_id"])
    assert len(ids) == 4


def test_concurrent_identical_assessments_collapse(client):
    event = _best(_process(client, "UX", n_stations=8))
    results = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        r = _assess(client, event["event_id"], seed=99)
        results.append((r.status_code, r.json()["assessment_id"]))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start(); t2.start(); t1.join(); t2.join()

    statuses = sorted(s for s, _ in results)
    assert statuses == [200, 201], results
    assert results[0][1] == results[1][1]
    listed = client.get(f"/v1/location-uncertainties?event_id={event['event_id']}")
    assert len(listed.json()["assessments"]) == 1


# --------------------------------------------------------------------------- #
# explicit rejection when too few resamples are solvable
# --------------------------------------------------------------------------- #
def test_insufficient_solvable_samples_is_rejected(client):
    # min_stations == station count: any resample missing a station is
    # unresolvable, so the solvable rate collapses far below 80%.
    event = _best(_process(client, "UR", n_stations=8, min_stations=8))
    assert event["station_count"] == 8
    resp = _assess(client, event["event_id"], seed=5)
    assert resp.status_code == 422, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "insufficient_solvable_samples"
    assert "80%" in detail["message"]

    # the rejection is deterministic and persists nothing
    again = _assess(client, event["event_id"], seed=5)
    assert again.status_code == 422
    assert again.json()["detail"]["code"] == "insufficient_solvable_samples"
    listed = client.get(f"/v1/location-uncertainties?event_id={event['event_id']}")
    assert listed.json()["assessments"] == []


# --------------------------------------------------------------------------- #
# queries: pagination, filters, cursors
# --------------------------------------------------------------------------- #
def test_samples_pagination_is_stable(client):
    event = _best(_process(client, "UG", n_stations=8))
    rep = _assess(client, event["event_id"]).json()
    samples = _all_samples(client, rep["assessment_id"], page=37)
    assert [s["sample_index"] for s in samples] == list(range(rep["n_resamples"]))


def test_samples_status_filter_and_cursor_binding(client):
    event = _best(_process(client, "UK", n_stations=8))
    rep = _assess(client, event["event_id"], seed=42).json()
    assert rep["n_failed"] >= 1  # seed 42 draws < 4 distinct stations 3 times

    failed = _all_samples(client, rep["assessment_id"], status="failed")
    assert len(failed) == rep["n_failed"]
    assert all(s["status"] == "failed" for s in failed)
    assert all(s["solution"] is None for s in failed)
    assert {s["failure_reason"] for s in failed} == {"insufficient_stations"}
    assert all(s["distinct_station_count"] < 4 for s in failed)

    solved = _all_samples(client, rep["assessment_id"], status="solved")
    assert len(solved) == rep["n_solved"]
    assert all(s["status"] == "solved" for s in solved)

    # a cursor issued for one status filter is rejected by another
    first_page = client.get(
        f"/v1/location-uncertainties/{rep['assessment_id']}/samples?limit=10"
    ).json()
    assert first_page["next_cursor"]
    misuse = client.get(
        f"/v1/location-uncertainties/{rep['assessment_id']}/samples"
        f"?limit=10&status=failed&cursor={first_page['next_cursor']}"
    )
    assert misuse.status_code == 422
    assert misuse.json()["detail"]["code"] == "invalid_cursor"

    malformed = client.get(
        f"/v1/location-uncertainties/{rep['assessment_id']}/samples?cursor=@@@"
    )
    assert malformed.status_code == 422
    assert malformed.json()["detail"]["code"] == "invalid_cursor"


def test_list_assessments_filter_and_pagination(client):
    event_a = _best(_process(client, "UA", n_stations=8))
    event_b = _best(_process(client, "UB", n_stations=8))
    for seed in (1, 2, 3):
        assert _assess(client, event_a["event_id"], seed=seed).status_code == 201
    assert _assess(client, event_b["event_id"], seed=1).status_code == 201

    # paginate the unfiltered list two at a time
    seen, cursor = [], None
    while True:
        url = "/v1/location-uncertainties?limit=2"
        if cursor:
            url += f"&cursor={cursor}"
        body = client.get(url).json()
        seen.extend(body["assessments"])
        cursor = body["next_cursor"]
        if not cursor:
            break
    assert len(seen) == 4
    assert len({a["assessment_id"] for a in seen}) == 4

    by_event = client.get(
        f"/v1/location-uncertainties?event_id={event_a['event_id']}"
    ).json()
    assert len(by_event["assessments"]) == 3
    assert all(a["event_id"] == event_a["event_id"] for a in by_event["assessments"])

    run_id = client.get(f"/v1/events/{event_b['event_id']}").json()["run_id"]
    by_run = client.get(f"/v1/location-uncertainties?run_id={run_id}").json()
    assert len(by_run["assessments"]) == 1
    assert by_run["assessments"][0]["event_id"] == event_b["event_id"]

    # a cursor issued for one filter set is rejected by another
    page = client.get("/v1/location-uncertainties?limit=2").json()
    misuse = client.get(
        f"/v1/location-uncertainties?event_id={event_a['event_id']}"
        f"&limit=2&cursor={page['next_cursor']}"
    )
    assert misuse.status_code == 422
    assert misuse.json()["detail"]["code"] == "invalid_cursor"

    # unknown filter targets are 404
    assert client.get("/v1/location-uncertainties?event_id=nope").status_code == 404
    assert client.get("/v1/location-uncertainties?run_id=nope").status_code == 404


def test_get_assessment_and_not_found(client):
    event = _best(_process(client, "UT", n_stations=8))
    rep = _assess(client, event["event_id"]).json()
    got = client.get(f"/v1/location-uncertainties/{rep['assessment_id']}")
    assert got.status_code == 200
    body = got.json()
    assert body["created"] is False
    assert body["assessment_id"] == rep["assessment_id"]
    assert body["confidence_intervals"] == rep["confidence_intervals"]
    assert body["covariance"] == rep["covariance"]
    assert body["frozen"] == rep["frozen"]

    missing = client.get("/v1/location-uncertainties/does-not-exist")
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "location_uncertainty_not_found"
    missing_samples = client.get("/v1/location-uncertainties/does-not-exist/samples")
    assert missing_samples.status_code == 404


# --------------------------------------------------------------------------- #
# validation + read-only guarantees
# --------------------------------------------------------------------------- #
def test_unknown_event_returns_404(client):
    resp = _assess(client, "no-such-event")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "event_not_found"


@pytest.mark.parametrize(
    "override",
    [
        {"n_resamples": 199},
        {"n_resamples": 2001},
        {"alpha": 0.79},
        {"alpha": 0.991},
        {"seed": -1},
        {"seed": 2**31},
    ],
)
def test_parameter_bounds_are_validated(client, override):
    event = _best(_process(client, "UV", n_stations=8))
    body = {"event_id": event["event_id"], "n_resamples": 200, "alpha": 0.9, "seed": 1}
    body.update(override)
    resp = client.post("/v1/location-uncertainties", json=body)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "validation_error"


def test_assessment_does_not_mutate_event_run_or_catalog(client):
    data = _process(client, "UO", n_stations=8)
    event = _best(data)
    before_event = client.get(f"/v1/events/{event['event_id']}").json()
    before_run = client.get(f"/v1/runs/{data['run_id']}").json()
    before_catalog = client.get("/v1/catalog/entries?limit=200").json()

    resp = _assess(client, event["event_id"])
    assert resp.status_code == 201, resp.text

    assert client.get(f"/v1/events/{event['event_id']}").json() == before_event
    assert client.get(f"/v1/runs/{data['run_id']}").json() == before_run
    assert client.get("/v1/catalog/entries?limit=200").json() == before_catalog

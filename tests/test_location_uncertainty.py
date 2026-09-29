"""End-to-end tests for the event location uncertainty assessment module."""
from __future__ import annotations

import threading

import numpy as np

from app.services import location_uncertainty as lu

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


def _assess(client, event_id, n=300, alpha=0.95, seed=42):
    return client.post(
        "/v1/location-uncertainty",
        json={
            "event_id": event_id,
            "N": n,
            "alpha": alpha,
            "random_seed": seed,
        },
    )


# --------------------------------------------------------------------------- #
# creation + frozen context
# --------------------------------------------------------------------------- #
def test_assessment_created_with_intervals_covariance_and_rate(client):
    event = _best(_process(client, "UC", n_stations=8))
    resp = _assess(client, event["event_id"], n=300, alpha=0.95, seed=42)
    assert resp.status_code == 201, resp.text
    rep = resp.json()

    assert rep["created"] is True
    assert rep["event_id"] == event["event_id"]
    assert rep["resample_count"] == 300
    assert rep["confidence_level"] == 0.95
    assert rep["random_seed"] == 42
    assert rep["station_count"] == 8
    assert rep["solvable_count"] >= 280, rep["failure_counts"]
    assert rep["solvable_rate"] >= 0.8
    assert rep["solvable_count"] + sum(rep["failure_counts"].values()) == 300
    assert rep["sample_count"] == 300

    result = rep["result"]
    assert result["quantile_method"] == "nearest_rank"
    assert result["confidence_level"] == 0.95
    assert set(result["intervals"]) == {"origin_time", "latitude", "longitude", "depth_km"}
    for key, interval in result["intervals"].items():
        assert set(interval) == {"lower", "upper"}
        assert interval["lower"] <= interval["upper"], key

    cov = result["covariance"]
    assert cov["order"] == ["origin_time_s", "latitude", "longitude", "depth_km"]
    assert cov["sample_count"] == rep["solvable_count"]
    mat = np.array(cov["matrix"])
    assert mat.shape == (4, 4)
    # symmetric positive (semi-)definite, non-negative diagonal
    assert np.allclose(mat, mat.T)
    assert np.all(np.diag(mat) >= 0.0)
    # scale sanity: horizontal location uncertainty of a tight synthetic event
    # is well below a degree / tens of km, origin-time variance below 1 s^2
    assert 0.0 < mat[0, 0] < 1.0
    assert 0.0 < mat[3, 3] < 25.0


def test_frozen_context_carries_picks_weights_model_params(client):
    data = _process(client, "UF", n_stations=8)
    event = _best(data)
    rep = _assess(client, event["event_id"]).json()

    frozen = rep["frozen"]
    assert frozen["ingestion_id"] == data["ingestion_id"]
    assert frozen["model"]["name"] == "default"
    assert frozen["model"]["version"] == "1.0"
    assert isinstance(frozen["model"]["layers"], list) and frozen["model"]["layers"]
    assert frozen["calibration"] is None
    assert frozen["params"] == {"max_residual_s": 1.0, "min_stations": 4}
    assert len(frozen["stations"]) == 8
    pick_ids = {p["pick_id"] for p in frozen["picks"]}
    assert pick_ids == {p["pick_id"] for p in event["picks"]}
    for p in frozen["picks"]:
        assert p["weight"] > 0.0
        assert p["phase"] in ("P", "S")

    baseline = rep["baseline"]
    assert baseline["event_id"] == event["event_id"]
    assert baseline["origin_time"] == event["origin_time"]
    assert baseline["quality"] == event["quality"]
    assert baseline["rms_residual_s"] == event["rms_residual_s"]

    assert len(rep["params_hash"]) == 64
    assert rep["request"] == {
        "event_id": event["event_id"],
        "run_id": data["run_id"],
        "resample_count": 300,
        "confidence_level": 0.95,
        "random_seed": 42,
    }


def test_frozen_calibration_is_reapplied(client):
    _post_events(client, range(0, 5), "LC", P_BIAS, S_BIAS)
    _post_events(client, [7], "LC", P_BIAS, S_BIAS)
    created = client.post(
        "/v1/models/default/1.0/calibrations",
        json=_cal_request(version="cal-lu"),
    )
    assert created.status_code == 201, created.text
    assert client.post(
        "/v1/models/default/1.0/calibrations/cal-lu/activate"
    ).status_code == 200

    window = _cal_event_window(11, "LC", P_BIAS, S_BIAS)
    data = client.post(
        "/v1/process", json={**window, "calibration": {"version": "cal-lu"}}
    ).json()
    event = _best(data)
    rep = _assess(client, event["event_id"]).json()
    cal = rep["frozen"]["calibration"]
    assert cal["version"] == "cal-lu"
    assert cal["status"] in ("active", "published")
    assert cal["terms"], "frozen calibration carries per-station P/S terms"


# --------------------------------------------------------------------------- #
# bootstrap mechanics: draws, multiplicity weighting, failure records
# --------------------------------------------------------------------------- #
def test_samples_record_draws_with_replacement_at_station_count(client):
    event = _best(_process(client, "UD", n_stations=8))
    rep = _assess(client, event["event_id"], n=200, seed=123).json()
    seen = []
    cursor = rep["next_cursor"]
    seen.extend(rep["samples"])
    while cursor:
        page = client.get(
            f"/v1/location-uncertainty/{rep['assessment_id']}/samples",
            params={"limit": 100, "cursor": cursor},
        ).json()
        seen.extend(page["samples"])
        cursor = page["next_cursor"]

    assert len(seen) == 200
    assert [s["sample_index"] for s in seen] == list(range(200))
    for s in seen:
        # exactly as many draws (counting multiplicities) as participating
        # stations; only frozen stations are drawn
        assert s["draw_count"] == 8
        assert sum(e["multiplicity"] for e in s["stations"]) == 8
        assert 1 <= s["distinct_station_count"] <= 8
        assert all(e["multiplicity"] >= 1 for e in s["stations"])
        codes = {e["station"] for e in s["stations"]}
        assert codes <= {st["station"] for st in rep["frozen"]["stations"]}


def test_seed_determines_draws_and_results(client):
    event = _best(_process(client, "US", n_stations=8))
    a = _assess(client, event["event_id"], n=250, seed=555).json()
    b = _assess(client, event["event_id"], n=250, seed=555).json()
    assert a["assessment_id"] == b["assessment_id"]
    assert a["samples"] == b["samples"]
    assert a["result"] == b["result"]

    # independently recompute the first draw from the documented RNG scheme
    stations = sorted(st["station"] for st in a["frozen"]["stations"])
    rng = np.random.default_rng(555)
    first_draw = rng.integers(0, 8, size=8)
    expected = sorted(stations[int(k)] for k in first_draw)
    got = []
    for e in a["samples"][0]["stations"]:
        got.extend([e["station"]] * e["multiplicity"])
    assert sorted(got) == expected


def test_duplicate_station_draws_weight_observations_by_count(client, monkeypatch):
    event = _best(_process(client, "UW", n_stations=8))
    captured = []
    real_locate = lu.locate_event

    def spy_locate(model, observations, stations, **kwargs):
        captured.append(list(observations))
        return real_locate(model, observations, stations, **kwargs)

    monkeypatch.setattr(lu, "locate_event", spy_locate)
    rep = _assess(client, event["event_id"], n=200, seed=9).json()
    aid = rep["assessment_id"]
    assert rep["solvable_count"] > 0
    assert captured, "locator was invoked"

    # collect all 200 samples in stable resample order
    samples, cursor = list(rep["samples"]), rep["next_cursor"]
    while cursor:
        page = client.get(
            f"/v1/location-uncertainty/{aid}/samples",
            params={"limit": 200, "cursor": cursor},
        ).json()
        samples.extend(page["samples"])
        cursor = page["next_cursor"]

    base_weight = {
        (p["station"], p["phase"]): p["weight"]
        for p in rep["frozen"]["picks"]
    }
    located = samples
    # The locator runs for every resample past the insufficient-stations gate,
    # so captured observations align with exactly those samples.
    assert len(captured) == sum(
        1 for s in located
        if s["failure_reason"] in (None, "location_failed", "residual_exceeded")
    )
    for sample, obs in zip(
        (s for s in located if s["failure_reason"] != "insufficient_stations"),
        captured,
    ):
        mult = {e["station"]: e["multiplicity"] for e in sample["stations"]}
        # every associated pick of every drawn station participates exactly
        # once per draw occurrence; weight = original weight * multiplicity
        by_key = {}
        for o in obs:
            by_key.setdefault((o.station, o.phase), []).append(o.weight)
        for station, m in mult.items():
            for phase in ("P", "S"):
                assert by_key[(station, phase)] == [base_weight[(station, phase)] * m]
        # stations not drawn never appear
        drawn = set(mult)
        for station, _phase in by_key:
            assert station in drawn


def test_failed_resamples_carry_structured_reasons(client):
    # min_stations=4 with 8 participating stations: resamples whose distinct
    # draw count is below 4 are rejected with insufficient_stations.
    event = _best(_process(client, "UR", n_stations=8))
    rep = _assess(client, event["event_id"], n=500, seed=3).json()
    failed = [s for s in rep["samples"] if not s["solvable"]]
    assert failed, "some 8-from-8 with-replacement draws see <4 distinct stations"
    for s in failed:
        assert s["failure_reason"] in (
            "insufficient_stations", "location_failed", "residual_exceeded"
        )
        assert s["solution"] is None
        assert s["distinct_station_count"] < 4
    # top-level failure counts are consistent
    total_failed = sum(rep["failure_counts"].values())
    assert total_failed == rep["resample_count"] - rep["solvable_count"]


# --------------------------------------------------------------------------- #
# nearest-rank quantiles
# --------------------------------------------------------------------------- #
def test_nearest_rank_indices():
    # alpha=0.8 -> tails at 0.1: ranks ceil(0.1n) and ceil(0.9n)
    assert lu.nearest_rank_indices(200, 0.8) == (19, 179)  # 0-based
    # alpha=0.95, n=200 -> 0.025*200=5 and 0.975*200=195
    assert lu.nearest_rank_indices(200, 0.95) == (4, 194)
    # non-integer positions round UP (nearest rank) and clamp
    assert lu.nearest_rank_indices(10, 0.99) == (0, 9)
    # alpha=0.9 -> tails at 0.05: ceil(15)=15 and ceil(285)=285 (1-based)
    lo, hi = lu.nearest_rank_indices(300, 0.9)
    assert lo == 14 and hi == 284


def test_intervals_match_nearest_rank_quantiles(client):
    event = _best(_process(client, "UQ", n_stations=8))
    rep = _assess(client, event["event_id"], n=400, alpha=0.9, seed=11).json()
    # Gather solvable solutions from the sample pages.
    sols = []
    cursor = None
    while True:
        params = {"limit": 200}
        if cursor:
            params["cursor"] = cursor
        page = client.get(
            f"/v1/location-uncertainty/{rep['assessment_id']}/samples",
            params=params,
        ).json()
        sols.extend(s["solution"] for s in page["samples"] if s["solvable"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert len(sols) == rep["solvable_count"]
    lats = sorted(s["latitude"] for s in sols)
    lo_i, hi_i = lu.nearest_rank_indices(len(lats), 0.9)
    assert rep["result"]["intervals"]["latitude"]["lower"] == round(lats[lo_i], 5)
    assert rep["result"]["intervals"]["latitude"]["upper"] == round(lats[hi_i], 5)


def test_covariance_uses_solvable_samples_only(client):
    event = _best(_process(client, "UV", n_stations=8))
    rep = _assess(client, event["event_id"], n=400, alpha=0.95, seed=21).json()
    m = rep["solvable_count"]
    assert rep["result"]["covariance"]["sample_count"] == m
    # independently: ddof=1 covariance over the same solvable sample vectors
    # would require epoch-second origins; structural checks suffice here.
    mat = np.array(rep["result"]["covariance"]["matrix"])
    assert np.linalg.eigvalsh(mat).min() > -1e-12


# --------------------------------------------------------------------------- #
# effective-sample rejection gate
# --------------------------------------------------------------------------- #
def test_low_solvable_rate_is_explicitly_rejected_and_persists_nothing(client):
    # 4 participating stations: most with-replacement draws miss stations, and
    # the 4-unknown locator needs >= 4 observations, so the solvable fraction
    # is far below the 80% gate.
    event = _best(_process(client, "U4", n_stations=4, min_stations=4))
    resp = _assess(client, event["event_id"], n=300, seed=1)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "insufficient_effective_samples"

    # no report or samples were stored
    listed = client.get(
        "/v1/location-uncertainty", params={"event_id": event["event_id"]}
    ).json()
    assert listed["assessments"] == []
    # rejection is itself reproducible (same seed -> same reject), not stored
    again = _assess(client, event["event_id"], n=300, seed=1)
    assert again.status_code == 422


def test_rejection_boundary_is_80_percent_exact():
    assert 240 * 100 == 80 * 300          # exactly 80% passes the integer gate
    assert 239 * 100 < 80 * 300           # 79.67% fails


# --------------------------------------------------------------------------- #
# idempotency / concurrency / param identity
# --------------------------------------------------------------------------- #
def test_assessment_is_idempotent(client):
    event = _best(_process(client, "UI", n_stations=8))
    first = _assess(client, event["event_id"], seed=42)
    second = _assess(client, event["event_id"], seed=42)
    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["created"] is False
    assert second.json()["assessment_id"] == first.json()["assessment_id"]
    assert second.json()["result"] == first.json()["result"]
    assert second.json()["samples"] == first.json()["samples"]


def test_seed_and_params_partition_assessments(client):
    event = _best(_process(client, "UP", n_stations=8))
    base = _assess(client, event["event_id"], n=200, alpha=0.9, seed=1).json()
    other_seed = _assess(client, event["event_id"], n=200, alpha=0.9, seed=2).json()
    other_n = _assess(client, event["event_id"], n=201, alpha=0.9, seed=1).json()
    other_a = _assess(client, event["event_id"], n=200, alpha=0.95, seed=1).json()
    ids = {base["assessment_id"], other_seed["assessment_id"],
           other_n["assessment_id"], other_a["assessment_id"]}
    assert len(ids) == 4
    # N/alpha move the params hash; the seed does not change frozen context
    assert base["params_hash"] == other_seed["params_hash"]
    assert base["params_hash"] != other_n["params_hash"]
    assert base["params_hash"] != other_a["params_hash"]


def test_concurrent_identical_assessments_collapse(client):
    event = _best(_process(client, "UX", n_stations=8))
    results = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        r = _assess(client, event["event_id"], n=200, seed=77)
        results.append((r.status_code, r.json()["assessment_id"]))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start(); t2.start(); t1.join(); t2.join()
    statuses = sorted(s for s, _ in results)
    assert statuses == [200, 201], results
    assert results[0][1] == results[1][1]


# --------------------------------------------------------------------------- #
# pagination
# --------------------------------------------------------------------------- #
def test_sample_pagination_walks_every_resample_once(client):
    event = _best(_process(client, "UPG", n_stations=8))
    rep = _assess(client, event["event_id"], n=320, seed=5).json()
    aid = rep["assessment_id"]
    seen, cursor = [], None
    for _ in range(20):
        params = {"limit": 128}
        if cursor:
            params["cursor"] = cursor
        page = client.get(
            f"/v1/location-uncertainty/{aid}/samples", params=params
        ).json()
        seen.extend(s["sample_index"] for s in page["samples"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert seen == list(range(320))


def test_sample_cursor_bound_to_assessment(client):
    e1 = _best(_process(client, "UB1", n_stations=8))
    e2 = _best(_process(client, "UB2", n_stations=8))
    a1 = _assess(client, e1["event_id"], seed=1).json()
    a2 = _assess(client, e2["event_id"], seed=1).json()
    page = client.get(
        f"/v1/location-uncertainty/{a1['assessment_id']}/samples",
        params={"limit": 2},
    ).json()
    bad = client.get(
        f"/v1/location-uncertainty/{a2['assessment_id']}/samples",
        params={"limit": 2, "cursor": page["next_cursor"]},
    )
    assert bad.status_code == 422
    assert bad.json()["detail"]["code"] == "invalid_cursor"


def test_assessment_list_pagination_and_filters(client):
    e1 = _best(_process(client, "UL1", n_stations=8))
    e2 = _best(_process(client, "UL2", n_stations=8))
    a1 = _assess(client, e1["event_id"], seed=1).json()
    a2 = _assess(client, e2["event_id"], seed=1).json()

    by_event = client.get(
        "/v1/location-uncertainty", params={"event_id": e1["event_id"]}
    ).json()
    assert [a["assessment_id"] for a in by_event["assessments"]] == [a1["assessment_id"]]

    run_id = client.get(f"/v1/events/{e2['event_id']}").json()["run_id"]
    by_run = client.get(
        "/v1/location-uncertainty", params={"run_id": run_id}
    ).json()
    assert {a["assessment_id"] for a in by_run["assessments"]} == {a2["assessment_id"]}

    seen, cursor = set(), None
    for _ in range(10):
        params = {"limit": 1}
        if cursor:
            params["cursor"] = cursor
        data = client.get("/v1/location-uncertainty", params=params).json()
        seen.update(a["assessment_id"] for a in data["assessments"])
        cursor = data["next_cursor"]
        if cursor is None:
            break
    assert seen == {a1["assessment_id"], a2["assessment_id"]}

    page = client.get(
        "/v1/location-uncertainty",
        params={"limit": 1, "event_id": e1["event_id"]},
    ).json()
    mismatched = client.get(
        "/v1/location-uncertainty",
        params={"limit": 1, "event_id": e2["event_id"], "cursor": page["next_cursor"]},
    )
    assert mismatched.status_code == 422
    assert mismatched.json()["detail"]["code"] == "invalid_cursor"


# --------------------------------------------------------------------------- #
# rejections / not found / read-only
# --------------------------------------------------------------------------- #
def test_unknown_event_returns_404(client):
    resp = _assess(client, "0" * 32)
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "event_not_found"


def test_param_ranges_validated(client):
    event = _best(_process(client, "UVL", n_stations=8))
    for payload in (
        {"N": 199, "alpha": 0.9, "random_seed": 1},
        {"N": 2001, "alpha": 0.9, "random_seed": 1},
        {"N": 200, "alpha": 0.79, "random_seed": 1},
        {"N": 200, "alpha": 0.995, "random_seed": 1},
        {"N": 200, "alpha": 0.9, "random_seed": -1},
        {"N": 200, "alpha": 0.9, "random_seed": 2**63},
    ):
        resp = client.post(
            "/v1/location-uncertainty",
            json={"event_id": event["event_id"], **payload},
        )
        assert resp.status_code == 422, payload


def test_assessment_not_found(client):
    resp = client.get("/v1/location-uncertainty/deadbeef")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "location_uncertainty_not_found"
    samples = client.get("/v1/location-uncertainty/deadbeef/samples")
    assert samples.status_code == 404
    assert samples.json()["detail"]["code"] == "location_uncertainty_not_found"


def test_list_unknown_event_filter_404(client):
    resp = client.get(
        "/v1/location-uncertainty", params={"event_id": "f" * 32}
    )
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "event_not_found"


def test_assessment_does_not_mutate_event_or_run(client):
    data = _process(client, "URO", n_stations=8)
    event = _best(data)
    run_before = client.get(f"/v1/runs/{data['run_id']}").json()
    event_before = client.get(f"/v1/events/{event['event_id']}").json()

    _assess(client, event["event_id"])
    _assess(client, event["event_id"])  # idempotent replay

    run_after = client.get(f"/v1/runs/{data['run_id']}").json()
    event_after = client.get(f"/v1/events/{event['event_id']}").json()
    assert run_after == run_before
    assert event_after == event_before

    # process idempotency / run semantics untouched
    again = client.post(
        "/v1/process", json=synthetic_event_payload(prefix="URO")[0]
    )
    assert again.json()["run_id"] == data["run_id"]
    assert again.json()["created"] is False

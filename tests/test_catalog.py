"""End-to-end tests for the cross-run event catalog module."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tests.conftest import make_waveform_b64, station_ring

from app.config import DEFAULT_MODEL_LAYERS
from app.services import catalog as cat_svc
from app.services.association import distance_km
from app.services.traveltime import LayeredModel

SR = 100.0
DURATION = 120.0
ORIGIN_OFFSET = 30.0
MODEL = LayeredModel.from_dicts(DEFAULT_MODEL_LAYERS)
BASE_DT = datetime(2026, 9, 22, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _event_window(
    prefix: str,
    day: int,
    *,
    ev_lat: float = 35.05,
    ev_lon: float = -117.95,
    ev_depth: float = 12.0,
    origin_offset: float = ORIGIN_OFFSET,
    n_stations: int = 8,
    radius_deg: float = 0.25,
    start: datetime | None = None,
    duration: float = DURATION,
    min_stations: int = 4,
    max_residual_s: float = 1.0,
):
    """One /v1/process body: a synthetic event, fully controlled by the caller."""
    start = start or (BASE_DT + timedelta(days=day))
    center = (35.0, -118.0)
    waveforms = []
    for code, lat, lon, elev in station_ring(
        prefix, center=center, radius_deg=radius_deg, n=n_stations
    ):
        dist = distance_km(ev_lat, ev_lon, lat, lon)
        z = ev_depth + elev / 1000.0
        tp = MODEL.travel_time(z, dist, "P")[0]
        ts = MODEL.travel_time(z, dist, "S")[0]
        b64 = make_waveform_b64(
            code, SR, _iso(start),
            [(origin_offset + tp, 4.0), (origin_offset + ts, 6.0)],
            duration=duration,
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
        "params": {"max_residual_s": max_residual_s, "min_stations": min_stations},
    }


def _process(client, prefix, day, **kwargs):
    resp = client.post(
        "/v1/process", json=_event_window(prefix, day, **kwargs)
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["events"], f"no event located for {prefix} day {day}"
    return data


def _best(data):
    return max(data["events"], key=lambda e: e["station_count"])


SLOWER_MODEL = [
    {"thickness_km": 15.0, "vp_kms": 5.2, "vs_kms": 3.0},
    {"thickness_km": 20.0, "vp_kms": 5.9, "vs_kms": 3.4},
    {"thickness_km": None, "vp_kms": 7.2, "vs_kms": 4.2},
]


def _ensure_model_v2(client):
    r = client.post(
        "/v1/models",
        json={"name": "default", "version": "2.0", "layers": SLOWER_MODEL},
    )
    assert r.status_code in (200, 201), r.text


def _same_quake_runs(client, day: int = 0):
    """Three completed runs of the SAME physical earthquake.

    Same wall-clock origin and observing network; runs differ by waveform
    window duration (two ingestions, same model) and by model version (replay
    of the first ingestion), which is exactly the cross-run catalog scenario.
    """
    d1 = _process(client, "CC", day, duration=120.0)
    d2 = _process(client, "CC", day, duration=118.0)
    _ensure_model_v2(client)
    r3 = client.post(
        "/v1/runs",
        json={
            "ingestion_id": d1["ingestion_id"],
            "model": {"name": "default", "version": "2.0"},
            "params": {"max_residual_s": 1.0, "min_stations": 4},
        },
    )
    assert r3.status_code in (200, 201), r3.text
    d3 = r3.json()
    assert d3["events"]
    out = [
        {"run_id": d1["run_id"], "event": _best(d1)},
        {"run_id": d2["run_id"], "event": _best(d2)},
        {"run_id": d3["run_id"], "event": _best(d3)},
    ]
    return out


def _compile(client, **overrides):
    body = {
        "start_time": _iso(BASE_DT + timedelta(days=0)),
        "end_time": _iso(BASE_DT + timedelta(days=30)),
        "radius_km": 50.0,
        "origin_tolerance_s": 5.0,
        "depth_tolerance_km": 15.0,
        "min_quality": "C",
        "min_sources": 1,
        "min_shared_stations": 1,
    }
    body.update(overrides)
    return client.post("/v1/catalog/compilations", json=body)


# --------------------------------------------------------------------------- #
# compilation: deterministic clustering and provenance
# --------------------------------------------------------------------------- #
def test_compile_merges_same_earthquake_across_runs(client):
    sources = _same_quake_runs(client)

    resp = _compile(client)
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["created"] is True
    assert data["entry_count"] == 1
    entry = data["entries"][0]
    assert entry["status"] == "draft"
    assert entry["revision"] == 1
    assert entry["source_count"] == 3
    assert entry["distinct_run_count"] == 3

    ids = {(s["run_id"], s["event_id"]) for s in entry["sources"]}
    assert ids == {(s["run_id"], s["event"]["event_id"]) for s in sources}

    master = next(s for s in entry["sources"] if s["is_master"])
    assert (master["run_id"], master["event_id"]) == (
        entry["master"]["run_id"], entry["master"]["event_id"]
    )
    # every source chain link carries version coordinates and master deltas
    models = {s["model"]["version"] for s in entry["sources"]}
    assert models == {"1.0", "2.0"}
    for s in entry["sources"]:
        assert s["model"]["name"] == "default"
        if s["model"]["version"] == "2.0":
            assert s["calibration"] is None
        assert set(s["delta"]) == {
            "origin_time_s", "latitude", "longitude", "depth_km", "rms_s"
        }
    # shared station codes are reported: the v2.0 replay located 6 of the 8
    # stations, so the all-member intersection is 6 while the union is 8.
    assert len(entry["shared_stations"]) == 6
    assert sorted(entry["shared_stations"]) == entry["shared_stations"]
    assert len(entry["stations"]) == 8
    assert set(entry["stations"]) >= set(entry["shared_stations"])


def test_distinct_earthquakes_stay_separate(client):
    _same_quake_runs(client)
    # a second quake ~0.4 degrees away at the same wall-clock time: outside
    # the 50 km epicentral linkage radius, so time agreement cannot merge it
    _process(
        client, "OT", 5,
        ev_lat=35.05, ev_lon=-118.4, origin_offset=30.0,
    )
    resp = _compile(client)
    assert resp.status_code == 201, resp.text
    entries = resp.json()["entries"]
    assert len(entries) == 2
    by_n = sorted(e["source_count"] for e in entries)
    assert by_n == [1, 3]
    singleton = next(e for e in entries if e["source_count"] == 1)
    assert singleton["origin_time"].endswith("Z")


def test_min_sources_drops_singletons(client):
    _same_quake_runs(client)
    _process(client, "OT", 5, ev_lat=35.05, ev_lon=-118.4, origin_offset=30.0)
    resp = _compile(client, min_sources=2)
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["entry_count"] == 1
    assert data["entries"][0]["source_count"] == 3


def test_compilation_is_idempotent(client):
    _same_quake_runs(client)
    r1 = _compile(client)
    r2 = _compile(client)
    assert r1.status_code == 201 and r2.status_code == 200
    assert r1.json()["compilation_id"] == r2.json()["compilation_id"]
    assert r2.json()["created"] is False
    first_entries = [e["entry_id"] for e in r1.json()["entries"]]
    assert [e["entry_id"] for e in r2.json()["entries"]] == first_entries


def test_time_window_filters_events(client):
    _same_quake_runs(client, day=2)
    # unrelated event on a different day must not enter a tight window
    _process(client, "TW", 9, ev_lat=35.05, ev_lon=-118.4)
    origin = datetime(2026, 9, 24, 0, 0, 29, tzinfo=timezone.utc)
    resp = _compile(
        client,
        start_time=_iso(origin - timedelta(seconds=10)),
        end_time=_iso(origin + timedelta(seconds=10)),
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["entry_count"] == 1
    assert resp.json()["entries"][0]["source_count"] == 3


def test_center_radius_region_filter(client):
    _same_quake_runs(client)  # near (35.05, -117.95)
    _process(client, "RG", 6, ev_lat=35.05, ev_lon=-118.7, origin_offset=30.0)
    near = _compile(
        client,
        center_latitude=35.0, center_longitude=-118.0, region_radius_km=50.0,
    )
    assert near.status_code == 201, near.text
    assert near.json()["entry_count"] == 1
    assert near.json()["entries"][0]["source_count"] == 3

    far = _compile(
        client,
        center_latitude=35.05, center_longitude=-118.7, region_radius_km=50.0,
    )
    assert far.status_code == 201, far.text
    assert far.json()["entry_count"] == 1
    assert far.json()["entries"][0]["source_count"] == 1


def test_min_quality_gate(client):
    # Two runs (different window durations) of the same quake located with
    # only four stations: four stations can never reach the A/B station
    # thresholds, so both solutions are C-grade while sharing all evidence.
    _process(
        client, "CQ", 0, n_stations=4, radius_deg=0.3, min_stations=4,
        duration=120.0,
    )
    _process(
        client, "CQ", 0, n_stations=4, radius_deg=0.3, min_stations=4,
        duration=118.0,
    )
    all_rows = _compile(
        client, min_quality="C", radius_km=60.0, depth_tolerance_km=20.0
    )
    assert all_rows.status_code == 201, all_rows.text
    assert all_rows.json()["entry_count"] == 1
    entry = all_rows.json()["entries"][0]
    assert entry["source_count"] == 2
    assert entry["quality"] == "C"
    assert {s["quality"] for s in entry["sources"]} == {"C"}
    only_a = _compile(
        client, min_quality="A", radius_km=60.0, depth_tolerance_km=20.0
    )
    assert only_a.status_code == 201, only_a.text
    assert only_a.json()["entry_count"] == 0


def test_version_conditions_filter_runs(client):
    _same_quake_runs(client)
    # _same_quake_runs already created a v2.0 replay of the first ingestion;
    # verify each model-version allowlist isolates its own runs.
    v1 = _compile(
        client,
        version_conditions={
            "model_versions": [{"name": "default", "version": "1.0"}],
        },
    )
    assert v1.status_code == 201, v1.text
    assert v1.json()["entry_count"] == 1
    assert v1.json()["entries"][0]["source_count"] == 2
    assert {
        s["model"]["version"] for s in v1.json()["entries"][0]["sources"]
    } == {"1.0"}

    v2 = _compile(
        client,
        version_conditions={
            "model_versions": [{"name": "default", "version": "2.0"}],
        },
    )
    assert v2.json()["entry_count"] == 1
    assert v2.json()["entries"][0]["source_count"] == 1

    both = _compile(client)  # no version conditions: all three runs merge
    assert both.json()["entries"][0]["source_count"] == 3
    v2_source = v2.json()["entries"][0]["sources"][0]
    assert v2_source["model"]["version"] == "2.0"

    # unknown model version reference -> 404
    missing = _compile(
        client,
        version_conditions={
            "model_versions": [{"name": "default", "version": "9.9"}],
        },
    )
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "model_not_found"


def test_calibration_version_filter_excludes_baseline(client):
    # uncalibrated runs only: a named calibration allowlist that does not
    # exist is a 404, while an empty allowlist plus include_uncalibrated=false
    # excludes every baseline run.
    _same_quake_runs(client)
    missing = _compile(
        client,
        version_conditions={
            "calibration_versions": ["cal-never"],
            "include_uncalibrated": False,
        },
    )
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "calibration_not_found"

    empty = _compile(
        client,
        version_conditions={"calibration_versions": [], "include_uncalibrated": False},
    )
    assert empty.status_code == 201
    assert empty.json()["entry_count"] == 0

    # baseline runs remain visible with the default include_uncalibrated=true
    assert _compile(client).json()["entry_count"] == 1


def test_calibrated_run_provenance_chain(client):
    """An activated calibration run is source-tracked with version + status."""
    from tests.test_calibration import (  # noqa: E402
        P_BIAS,
        S_BIAS,
        _cal_request,
        _event_window as _cal_window,
        _post_events,
    )

    # five biased training events + one holdout, then create+activate cal-1
    _post_events(client, range(0, 5), "XA", P_BIAS, S_BIAS)
    _post_events(client, [7], "XA", P_BIAS, S_BIAS)
    created = client.post(
        "/v1/models/default/1.0/calibrations", json=_cal_request(version="cal-x")
    )
    assert created.status_code == 201, created.text
    act = client.post(
        "/v1/models/default/1.0/calibrations/cal-x/activate"
    )
    assert act.status_code == 200

    # the same physical quake, once on the baseline and once calibrated
    from tests.test_calibration import BASE_DT as CAL_BASE  # noqa: E402

    window = _cal_window(11, "XA", P_BIAS, S_BIAS)
    baseline = client.post("/v1/process", json=window).json()
    calibrated = client.post(
        "/v1/process", json={**window, "calibration": {"version": "cal-x"}}
    ).json()
    assert calibrated["calibration"] == {"version": "cal-x", "status": "active"}

    body = {
        "start_time": _iso(CAL_BASE + timedelta(days=11) - timedelta(seconds=60)),
        "end_time": _iso(CAL_BASE + timedelta(days=11) + timedelta(seconds=60)),
        "radius_km": 50.0,
        "origin_tolerance_s": 5.0,
        "depth_tolerance_km": 15.0,
        "min_quality": "C",
        "version_conditions": {
            "calibration_versions": ["cal-x"],
            "include_uncalibrated": True,
        },
    }
    merged = client.post("/v1/catalog/compilations", json=body)
    assert merged.status_code == 201, merged.text
    entries = merged.json()["entries"]
    assert len(entries) == 1
    sources = entries[0]["sources"]
    by_cal = {
        (s["calibration"] or {}).get("version"): s for s in sources
    }
    assert set(by_cal) == {"cal-x", None}
    cal_source = by_cal["cal-x"]
    assert cal_source["calibration"]["status"] == "active"
    assert cal_source["run_id"] == calibrated["run_id"]
    assert by_cal[None]["run_id"] == baseline["run_id"]
    # calibrated and baseline runs are distinct source links
    assert len({s["run_id"] for s in sources}) == 2

    # include_uncalibrated=false keeps only the calibrated run
    only_cal = client.post(
        "/v1/catalog/compilations",
        json={**body, "version_conditions": {
            "calibration_versions": ["cal-x"], "include_uncalibrated": False}},
    )
    assert only_cal.status_code == 201, only_cal.text
    only_sources = only_cal.json()["entries"][0]["sources"]
    assert len(only_sources) == 1
    assert only_sources[0]["calibration"]["version"] == "cal-x"


# --------------------------------------------------------------------------- #
# master solution selection: quality > RMS > stable ids
# --------------------------------------------------------------------------- #
def test_master_selection_prefers_quality_then_rms_then_ids():
    def ev(run, event, quality, rms, stations=("S1", "S2", "S3", "S4"),
           t=100.0, lat=35.0, lon=-118.0, depth=10.0):
        return cat_svc.EventEvidence(
            run_id=run, event_id=event, origin_epoch=t,
            latitude=lat, longitude=lon, depth_km=depth,
            rms_s=rms, station_count=len(stations), quality=quality,
            stations=frozenset(stations),
        )

    cfg = cat_svc.ClusterConfig(50.0, 5.0, 15.0, 1)
    # B beats A/C regardless of RMS: quality is the primary key
    clusters = cat_svc.cluster_events(
        [
            ev("run-z", "ev-z", "C", 0.01),
            ev("run-b", "ev-b", "B", 0.90),
            ev("run-a", "ev-a", "A", 0.40),
        ],
        cfg,
    )
    assert len(clusters) == 1
    master = clusters[0].master
    assert (master.quality, master.rms_s) == ("A", 0.40)

    # same quality -> smallest RMS
    clusters = cat_svc.cluster_events(
        [ev("run-z", "ev-z", "B", 0.90), ev("run-a", "ev-a", "B", 0.20)],
        cfg,
    )
    assert clusters[0].master.run_id == "run-a"

    # same quality and RMS -> (run_id, event_id) lexicographic tie-break
    clusters = cat_svc.cluster_events(
        [ev("run-z", "ev-z", "A", 0.30), ev("run-a", "ev-a", "A", 0.30)],
        cfg,
    )
    m = clusters[0].master
    assert (m.run_id, m.event_id) == ("run-a", "ev-a")

    # determinism: reverse input order yields identical membership/order
    members = [
        ev(f"run{i}", f"ev{i}", "B", 0.5, t=100.0 + 0.01 * i)
        for i in range(5)
    ]
    c1 = cat_svc.cluster_events(members, cfg)
    c2 = cat_svc.cluster_events(list(reversed(members)), cfg)
    assert [e.event_id for e in c1[0].members] == [e.event_id for e in c2[0].members]


def test_station_evidence_gate_blocks_unrelated_network(client):
    # same origin window/time, but disjoint observing stations and far apart:
    # time alone must not merge them
    cfg = cat_svc.ClusterConfig(5000.0, 30.0, 700.0, 1)
    a = cat_svc.EventEvidence(
        "r1", "e1", 100.0, 35.0, -118.0, 10.0, 0.2, 4, "B",
        frozenset({"N1", "N2", "N3", "N4"}),
    )
    b = cat_svc.EventEvidence(
        "r2", "e2", 100.5, 36.0, -117.0, 12.0, 0.3, 4, "B",
        frozenset({"X1", "X2", "X3", "X4"}),
    )
    clusters = cat_svc.cluster_events([a, b], cfg)
    assert len(clusters) == 2
    # gate disabled (min_shared_stations=0) -> time/space now link them
    cfg_open = cat_svc.ClusterConfig(5000.0, 30.0, 700.0, 0)
    assert len(cat_svc.cluster_events([a, b], cfg_open)) == 1


# --------------------------------------------------------------------------- #
# lifecycle: draft -> published -> withdrawn with optimistic concurrency
# --------------------------------------------------------------------------- #
def _draft_entry(client):
    _same_quake_runs(client)
    # tight window so only the day-0 cluster enters this compilation
    return _compile(
        client,
        start_time=_iso(BASE_DT - timedelta(days=1)),
        end_time=_iso(BASE_DT + timedelta(days=1)),
    ).json()["entries"][0]


def test_publish_with_expected_revision_and_idempotent(client):
    entry = _draft_entry(client)
    eid = entry["entry_id"]
    # stale revision rejected without changing state
    stale = client.post(
        f"/v1/catalog/entries/{eid}/publish", json={"expected_revision": 99}
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "revision_conflict"

    ok = client.post(
        f"/v1/catalog/entries/{eid}/publish", json={"expected_revision": 1}
    )
    assert ok.status_code == 200, ok.text
    published = ok.json()
    assert published["status"] == "published"
    assert published["revision"] == 2
    assert published["published_at"] is not None
    # content is the same immutable snapshot
    assert published["origin_time"] == entry["origin_time"]
    assert published["source_count"] == 3

    # repeat publish is idempotent and never bumps revision
    again = client.post(
        f"/v1/catalog/entries/{eid}/publish", json={"expected_revision": 99}
    )
    assert again.status_code == 200
    assert again.json()["revision"] == 2
    assert again.json()["status"] == "published"


def test_withdraw_keeps_published_content(client):
    entry = _draft_entry(client)
    eid = entry["entry_id"]
    client.post(f"/v1/catalog/entries/{eid}/publish", json={"expected_revision": 1})

    # cannot withdraw a draft-style with wrong revision
    bad = client.post(
        f"/v1/catalog/entries/{eid}/withdraw",
        json={"expected_revision": 1, "reason": "superseded"},
    )
    assert bad.status_code == 409
    assert bad.json()["detail"]["code"] == "revision_conflict"
    view = client.get(f"/v1/catalog/entries/{eid}").json()
    assert view["status"] == "published"

    ok = client.post(
        f"/v1/catalog/entries/{eid}/withdraw",
        json={"expected_revision": 2, "reason": "superseded by Q3 review"},
    )
    assert ok.status_code == 200, ok.text
    withdrawn = ok.json()
    assert withdrawn["status"] == "withdrawn"
    assert withdrawn["revision"] == 3
    assert withdrawn["withdraw_reason"] == "superseded by Q3 review"
    # published content is preserved verbatim
    for field in ("origin_time", "latitude", "longitude", "depth_km",
                  "rms_residual_s", "quality", "master"):
        assert withdrawn[field] == entry[field]
    assert withdrawn["source_count"] == 3
    assert withdrawn["published_at"] is not None

    # re-publishing a withdrawn entry is rejected
    republish = client.post(
        f"/v1/catalog/entries/{eid}/publish", json={"expected_revision": 3}
    )
    assert republish.status_code == 409
    assert republish.json()["detail"]["code"] == "catalog_entry_withdrawn"

    # repeat withdraw is idempotent (tombstone, no further revision bump)
    again = client.post(
        f"/v1/catalog/entries/{eid}/withdraw",
        json={"expected_revision": 3, "reason": "again"},
    )
    assert again.status_code == 200
    assert again.json()["revision"] == 3
    assert again.json()["withdraw_reason"] == "superseded by Q3 review"


def test_cannot_withdraw_draft(client):
    entry = _draft_entry(client)
    resp = client.post(
        f"/v1/catalog/entries/{entry['entry_id']}/withdraw",
        json={"expected_revision": 1, "reason": "nope"},
    )
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "catalog_entry_not_published"


# --------------------------------------------------------------------------- #
# queries: filters and stable cursor pagination
# --------------------------------------------------------------------------- #
def _four_quakes(client, prefix="PG"):
    # four distinct quakes, one per day and >50 km apart in longitude so
    # neither time nor space linkage can merge them
    lons = (-119.3, -118.6, -117.9, -117.2)
    for i, lon in enumerate(lons):
        _process(
            client, f"{prefix}{i}", day=10 + i,
            ev_lat=35.0, ev_lon=lon,
            origin_offset=30.0, radius_deg=0.2, n_stations=8,
        )
    resp = _compile(
        client,
        start_time=_iso(BASE_DT + timedelta(days=9)),
        end_time=_iso(BASE_DT + timedelta(days=20)),
        radius_km=50.0,
        min_quality="A",
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_list_pagination_stable_cursor(client):
    _four_quakes(client)
    seen = []
    cursor = None
    window = {
        "start_time": _iso(BASE_DT + timedelta(days=9)),
        "end_time": _iso(BASE_DT + timedelta(days=20)),
        "min_quality": "A",
    }
    for _ in range(10):
        params = {"limit": 2, **window}
        if cursor:
            params["cursor"] = cursor
        page = client.get("/v1/catalog/entries", params=params)
        assert page.status_code == 200, page.text
        data = page.json()
        ids = [e["entry_id"] for e in data["entries"]]
        seen.extend(ids)
        cursor = data["next_cursor"]
        if cursor is None:
            break
    assert len(seen) == 4
    assert len(set(seen)) == 4
    # order follows (origin_time, id)
    times = [
        client.get(f"/v1/catalog/entries/{eid}").json()["origin_time"]
        for eid in seen
    ]
    assert times == sorted(times)


def test_list_filters_time_region_quality_status(client):
    _four_quakes(client, prefix="QF")
    entry = _draft_entry(client)  # day 0-2 cluster
    client.post(
        f"/v1/catalog/entries/{entry['entry_id']}/publish",
        json={"expected_revision": 1},
    )
    # status filter: default listing allows all explicitly
    published = client.get(
        "/v1/catalog/entries", params={"status": "published", "min_quality": "A"}
    ).json()["entries"]
    assert entry["entry_id"] in {e["entry_id"] for e in published}
    drafts = client.get(
        "/v1/catalog/entries", params={"status": "draft", "min_quality": "A"}
    ).json()
    assert drafts["entries"], "the four later quakes are still drafts"
    assert entry["entry_id"] not in {e["entry_id"] for e in drafts["entries"]}

    # region bbox isolates the clustered quake near (35.05, -117.95); the
    # later singletons are centered on latitude 35.0 and farther east/west
    bbox = client.get(
        "/v1/catalog/entries",
        params={
            "min_latitude": 35.02, "max_latitude": 35.08,
            "min_longitude": -117.98, "max_longitude": -117.92,
            "min_quality": "A",
        },
    ).json()
    assert {e["entry_id"] for e in bbox["entries"]} == {entry["entry_id"]}

    # time window isolates later events
    later = client.get(
        "/v1/catalog/entries",
        params={
            "start_time": _iso(BASE_DT + timedelta(days=10)),
            "end_time": _iso(BASE_DT + timedelta(days=20)),
            "min_quality": "A",
        },
    ).json()
    assert len(later["entries"]) == 4

    # min_quality filter is already applied above; verify the draft cluster at
    # day 0 plus four later singletons are all A-grade in the wide window
    a_wide = client.get(
        "/v1/catalog/entries", params={"min_quality": "A"}
    ).json()
    assert len(a_wide["entries"]) == 5


def test_cursor_bound_to_query_fingerprint(client):
    _four_quakes(client)
    page = client.get("/v1/catalog/entries", params={"limit": 2}).json()
    cursor = page["next_cursor"]
    bad = client.get(
        "/v1/catalog/entries", params={"limit": 2, "status": "draft", "cursor": cursor}
    )
    assert bad.status_code == 422
    assert bad.json()["detail"]["code"] == "invalid_cursor"


# --------------------------------------------------------------------------- #
# compilation listing / lookup and explicit rejections
# --------------------------------------------------------------------------- #
def test_compilation_lookup_and_entry_not_found(client):
    _same_quake_runs(client)
    comp = _compile(client).json()
    got = client.get(f"/v1/catalog/compilations/{comp['compilation_id']}")
    assert got.status_code == 200
    assert got.json()["entry_count"] == comp["entry_count"]

    listing = client.get("/v1/catalog/compilations")
    assert listing.status_code == 200
    assert comp["compilation_id"] in {
        c["compilation_id"] for c in listing.json()["compilations"]
    }

    missing = client.get("/v1/catalog/entries/deadbeef")
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "catalog_entry_not_found"
    missing_c = client.get("/v1/catalog/compilations/deadbeef")
    assert missing_c.status_code == 404
    assert missing_c.json()["detail"]["code"] == "catalog_compilation_not_found"


def test_invalid_windows_and_spatial_filters(client):
    bad_window = _compile(
        client,
        start_time=_iso(BASE_DT + timedelta(days=10)),
        end_time=_iso(BASE_DT + timedelta(days=1)),
    )
    assert bad_window.status_code == 422
    assert bad_window.json()["detail"]["code"] == "invalid_catalog_time_window"

    partial_region = _compile(
        client, center_latitude=35.0, center_longitude=-118.0
    )
    assert partial_region.status_code == 422
    assert partial_region.json()["detail"]["code"] == "invalid_spatial_filter"

    bad_bbox = client.get(
        "/v1/catalog/entries",
        params={"min_latitude": 36.0, "max_latitude": 35.0,
                "min_longitude": -118.0, "max_longitude": -117.0},
    )
    assert bad_bbox.status_code == 422
    assert bad_bbox.json()["detail"]["code"] == "invalid_spatial_filter"

    one_sided_time = client.get(
        "/v1/catalog/entries", params={"start_time": _iso(BASE_DT)}
    )
    assert one_sided_time.status_code == 422
    assert one_sided_time.json()["detail"]["code"] == "invalid_catalog_time_window"


def test_empty_compilation_is_persisted_and_idempotent(client):
    resp = _compile(
        client,
        start_time=_iso(BASE_DT + timedelta(days=400)),
        end_time=_iso(BASE_DT + timedelta(days=401)),
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["entry_count"] == 0
    again = _compile(
        client,
        start_time=_iso(BASE_DT + timedelta(days=400)),
        end_time=_iso(BASE_DT + timedelta(days=401)),
    )
    assert again.status_code == 200
    assert again.json()["compilation_id"] == resp.json()["compilation_id"]


# --------------------------------------------------------------------------- #
# concurrency
# --------------------------------------------------------------------------- #
def test_concurrent_identical_compilations_collapse(client):
    _same_quake_runs(client)
    import threading

    body = {
        "start_time": _iso(BASE_DT - timedelta(days=1)),
        "end_time": _iso(BASE_DT + timedelta(days=1)),
        "radius_km": 50.0,
        "origin_tolerance_s": 5.0,
        "depth_tolerance_km": 15.0,
        "min_quality": "C",
        "min_sources": 1,
        "min_shared_stations": 1,
    }
    results = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        r = client.post("/v1/catalog/compilations", json=body)
        results.append((r.status_code, r.json()["compilation_id"]))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start(); t2.start(); t1.join(); t2.join()

    statuses = sorted(s for s, _ in results)
    assert statuses == [200, 201], results
    assert results[0][1] == results[1][1]  # single surviving compilation
    listing = client.get("/v1/catalog/compilations").json()["compilations"]
    assert sum(
        1 for c in listing if c["request"]["radius_km"] == 50.0
    ) == 1


def test_concurrent_publish_only_one_revision_bump(client):
    # Two clients racing on publish.  Both must end on the published state
    # with exactly one revision bump: either the loser hits the OCC 409 or,
    # when the storage serializes writers (SQLite), its retry-style request
    # collapses on the idempotent no-op path.  The final state is identical.
    entry = _draft_entry(client)
    eid = entry["entry_id"]
    import threading

    results = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        r = client.post(
            f"/v1/catalog/entries/{eid}/publish",
            json={"expected_revision": 1},
        )
        results.append(r.status_code)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start(); t2.start(); t1.join(); t2.join()

    assert set(results) <= {200, 409} and 200 in results
    final = client.get(f"/v1/catalog/entries/{eid}").json()
    assert final["status"] == "published"
    assert final["revision"] == 2  # exactly one revision bump

    # a genuinely stale revision against a moved entry is still a hard 409
    stale = client.post(
        f"/v1/catalog/entries/{eid}/withdraw",
        json={"expected_revision": 1, "reason": "late"},
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "revision_conflict"


# --------------------------------------------------------------------------- #
# existing semantics remain intact
# --------------------------------------------------------------------------- #
def test_catalog_compilation_does_not_mutate_runs_or_events(client):
    sources = _same_quake_runs(client)
    before = {
        sid: client.get(f"/v1/runs/{sid}").json()
        for sid in {s["run_id"] for s in sources}
    }
    entry = _compile(client).json()["entries"][0]
    client.post(
        f"/v1/catalog/entries/{entry['entry_id']}/publish",
        json={"expected_revision": 1},
    )
    client.post(
        f"/v1/catalog/entries/{entry['entry_id']}/withdraw",
        json={"expected_revision": 2, "reason": "x"},
    )
    for run_id, payload in before.items():
        after = client.get(f"/v1/runs/{run_id}").json()
        assert after["events"] == payload["events"]
        assert after["model"] == payload["model"]
        assert after["calibration"] == payload["calibration"]
    # compare endpoint still works unchanged
    run_ids = list(before)
    cmp_resp = client.get(
        "/v1/runs/compare", params={"run_a": run_ids[0], "run_b": run_ids[1]}
    )
    assert cmp_resp.status_code == 200
    assert cmp_resp.json()["matches"]

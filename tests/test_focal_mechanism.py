"""End-to-end tests for the P-wave first-motion focal mechanism inversion."""
from __future__ import annotations

import math
import threading

import pytest

from app.config import DEFAULT_MODEL_LAYERS
from app.services import focal_mechanism as fm
from app.services.association import distance_km
from app.services.traveltime import LayeredModel

from tests.conftest import (
    BASE_TIME,
    make_waveform_b64,
    station_ring,
    synthetic_event_payload,
)

# Grid mechanism whose station amplitudes are all well away from nodal for the
# default synthetic geometry (min |A| ~ 0.21, balanced 4+/4- polarities).
TRUE_MECHANISM = (340.0, 70.0, -170.0)


def _process(client, prefix, **kwargs):
    resp = client.post(
        "/v1/process", json=synthetic_event_payload(prefix=prefix, **kwargs)[0]
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _best(data):
    return max(data["events"], key=lambda e: e["station_count"])


def _p_picks(event):
    return [p for p in event["picks"] if p["phase"] == "P"]


def _polarities_for_mechanism(event, prefix, mechanism, n_stations=8):
    """Client-side polarity assignment for a known mechanism.

    Uses the same public geometry/radiation helpers the service exposes, over
    the event's reported hypocenter and the ring station coordinates.
    """
    model = LayeredModel.from_dicts(DEFAULT_MODEL_LAYERS)
    coords = {code: (lat, lon, elev) for code, lat, lon, elev in station_ring(prefix, n=n_stations)}
    normal, slip = fm.fault_vectors(*mechanism)
    observations = []
    for pick in _p_picks(event):
        lat, lon, elev = coords[pick["station"]]
        az, to = fm.ray_geometry(
            model,
            event["latitude"],
            event["longitude"],
            event["depth_km"],
            lat,
            lon,
            elev,
        )
        amplitude = fm.p_radiation(normal, slip, az, to)
        assert abs(amplitude) > 0.05, "test geometry too close to a nodal plane"
        observations.append(
            {
                "pick_id": pick["pick_id"],
                "polarity": "positive" if amplitude > 0 else "negative",
                "weight": 1.0,
            }
        )
    return observations


def _invert(client, event_id, observations):
    return client.post(
        "/v1/focal-mechanisms",
        json={"event_id": event_id, "observations": observations},
    )


def _invert_known(client, prefix="FM", mechanism=TRUE_MECHANISM, **kwargs):
    data = _process(client, prefix, **kwargs)
    event = _best(data)
    obs = _polarities_for_mechanism(event, prefix, mechanism)
    return data, event, obs, _invert(client, event["event_id"], obs)


def _solution_key(solution):
    p = solution["plane_1"]
    return fm.mechanism_key(p["strike_deg"], p["dip_deg"], p["rake_deg"])


def _mechanism_key_close(key_a, key_b, tol=1e-4):
    """Mechanism keys from display-rounded angles agree within rounding noise."""
    return all(abs(a - b) <= tol for a, b in zip(key_a, key_b))


def _all_solutions(client, inversion_id, limit=7):
    """Walk the solutions endpoint page by page and return every solution."""
    seen, cursor = [], None
    for _ in range(1000):
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        resp = client.get(f"/v1/focal-mechanisms/{inversion_id}/solutions", params=params)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        seen.extend(data["solutions"])
        cursor = data["next_cursor"]
        if cursor is None:
            return seen
    raise AssertionError("pagination did not terminate")


# --------------------------------------------------------------------------- #
# double-couple geometry (unit)
# --------------------------------------------------------------------------- #
def test_plane_angles_round_trip_preserves_mechanism():
    for angles in [
        (0, 90, 0),
        (30, 60, -90),
        (120, 45, 10),
        (350, 80, 170),
        (0, 90, 90),  # auxiliary plane is horizontal
        (200, 30, 45),
        (90, 10, -180),
    ]:
        normal, slip = fm.fault_vectors(*angles)
        recovered = fm.plane_angles(normal, slip)
        assert fm.mechanism_key(*recovered) == fm.mechanism_key(*angles)


def test_mechanism_key_invariant_under_nodal_plane_swap():
    normal, slip = fm.fault_vectors(30, 60, -90)
    aux = fm.plane_angles(slip, normal)  # auxiliary plane: normal <-> slip
    assert aux == pytest.approx((210.0, 30.0, -90.0), abs=1e-9)
    assert fm.mechanism_key(*aux) == fm.mechanism_key(30, 60, -90)


def test_mechanism_key_distinguishes_slip_reversed_tensor():
    # Reversing the slip (rake += 180) negates the moment tensor: the P
    # radiation pattern is the opposite mechanism and must NOT be merged.
    for strike, dip, rake in [
        (0, 45, 30),
        (180, 50, 60),
        (260, 30, 80),
        (340, 70, 120),
    ]:
        reversed_rake = ((rake - 180 + 180) % 360) - 180
        key = fm.mechanism_key(strike, dip, rake)
        key_rev = fm.mechanism_key(strike, dip, reversed_rake)
        assert key != key_rev, (strike, dip, rake)
        # the tensors are exact opposites component by component
        assert tuple(a for a in key) == tuple(-a for a in key_rev)
        # ...which is precisely what the legacy key wrongly collapsed
        n, u = fm.fault_vectors(strike, dip, rake)
        n2, u2 = fm.fault_vectors(strike, dip, reversed_rake)
        assert fm._legacy_tensor_key(n, u) == fm._legacy_tensor_key(n2, u2)


def test_slip_reversed_mechanism_flips_every_prediction():
    # Sanity: M and -M predict opposite first motions at the same ray, so
    # dropping one candidate can change the winning polarity pattern.
    normal, slip = fm.fault_vectors(30, 60, 90)
    amp = fm.p_radiation(normal, slip, 45.0, 120.0)
    normal_r, slip_r = fm.fault_vectors(30, 60, -90)
    amp_r = fm.p_radiation(normal_r, slip_r, 45.0, 120.0)
    assert amp * amp_r < 0.0
    assert fm._predicted_sign(amp) == -fm._predicted_sign(amp_r)


def test_grid_search_keeps_slip_reversed_candidate_and_finds_true_minimum():
    # Ring geometry + a polarity/weight pattern on which the legacy
    # sign-canonicalizing dedup overstated the minimum as 0.5: the corrected
    # search must evaluate both M and -M and recover a zero-misfit solution.
    model = LayeredModel.from_dicts(DEFAULT_MODEL_LAYERS)
    coords = {
        code: (lat, lon, elev)
        for code, lat, lon, elev in station_ring("SR", n=8)
    }
    stations = sorted(coords)
    polarities = {
        "SR00": (-1, 0.5), "SR01": (-1, 2.0), "SR02": (1, 1.0),
        "SR03": (1, 1.0), "SR04": (-1, 1.0), "SR05": (1, 2.0),
        "SR06": (-1, 0.5), "SR07": (-1, 1.0),
    }
    rays = {}
    for code in stations:
        lat, lon, elev = coords[code]
        az, to = fm.ray_geometry(model, 35.05, -117.95, 12.0, lat, lon, elev)
        rays[code] = fm._ray_vector(az, to)

    best, tied = fm.grid_search(rays, polarities)
    assert best == 0.0
    assert tied, "a perfect-fit mechanism must be among the tied solutions"

    # the same data under the legacy rule misses it
    legacy_best = _legacy_grid_search(rays, polarities)
    assert legacy_best == 0.5


def _legacy_grid_search(ray_vectors, polarities, grid_step_deg=fm.GRID_STEP_DEG):
    """Replicates the released-v1 search (M == -M merged) for regression tests."""
    step = int(round(grid_step_deg))
    seen, best = set(), None
    for strike in range(0, 360, step):
        for dip in range(step, 91, step):
            for rake in range(-180, 180, step):
                normal, slip = fm.fault_vectors(strike, dip, rake)
                key = fm._legacy_tensor_key(normal, slip)
                if key in seen:
                    continue
                seen.add(key)
                misfit = 0.0
                for station, x in ray_vectors.items():
                    sign, weight = polarities[station]
                    if (
                        fm._predicted_sign(fm._dot(x, normal) * fm._dot(x, slip))
                        != sign
                    ):
                        misfit += weight
                misfit = round(misfit, 6)
                best = misfit if best is None else min(best, misfit)
    return best if best is not None else 0.0


def test_nodal_prediction_counts_as_misfit():
    # N-striking vertical strike-slip: azimuth 0 sits exactly on a nodal plane.
    normal, slip = fm.fault_vectors(0, 90, 0)
    amplitude = fm.p_radiation(normal, slip, 0.0, 120.0)
    assert abs(amplitude) <= fm.NODAL_EPS
    assert fm._predicted_sign(amplitude) == 0

    rays = {"STA": fm._ray_vector(0.0, 120.0)}
    best, tied = fm.grid_search(rays, {"STA": (1, 1.0)})
    assert best == 0.0
    # the nodal mechanism misfits (zero amplitude counts as a misfit), so it
    # cannot be among the zero-misfit ties
    nodal_key = fm.mechanism_key(0, 90, 0)
    assert all(fm.mechanism_key(*t[:3]) != nodal_key for t in tied)


def test_grid_search_weights_select_heavier_match():
    # Two stations at the same takeoff direction with opposite polarities: any
    # mechanism misfits at least one; the lighter one must be sacrificed.
    direction = fm._ray_vector(45.0, 115.0)
    rays = {"A": direction, "B": direction}
    best, tied = fm.grid_search(rays, {"A": (1, 1.0), "B": (-1, 2.5)})
    assert best == 1.0
    assert tied, "expected tied solutions"
    for strike, dip, rake, misfit, count in tied:
        normal, slip = fm.fault_vectors(strike, dip, rake)
        assert fm._predicted_sign(fm.p_radiation(normal, slip, 45.0, 115.0)) == -1
        assert misfit == 1.0
        assert count == 1


# --------------------------------------------------------------------------- #
# happy path: known mechanism recovered, structure, dedupe
# --------------------------------------------------------------------------- #
def test_inversion_recovers_known_mechanism(client):
    data, event, obs, resp = _invert_known(client)
    assert resp.status_code == 201, resp.text
    rep = resp.json()

    assert rep["created"] is True
    assert rep["event_id"] == event["event_id"]
    assert rep["run_id"] == data["run_id"]
    assert rep["grid_step_deg"] == 10.0
    assert rep["station_count"] == 8
    assert rep["min_misfit_weight"] == 0.0
    assert rep["solution_count"] >= 1
    assert len(rep["solutions"]) == min(50, rep["solution_count"])

    # the true mechanism is among the tied best solutions
    keys = {_solution_key(s) for s in _all_solutions(client, rep["inversion_id"])}
    assert fm.mechanism_key(*TRUE_MECHANISM) in keys


def test_solution_structure_and_frozen_context(client):
    data, event, obs, resp = _invert_known(client, prefix="FS")
    assert resp.status_code == 201, resp.text
    rep = resp.json()

    frozen = rep["frozen"]
    assert frozen["model"]["name"] == "default"
    assert frozen["model"]["version"] == "1.0"
    assert frozen["model"]["layers"]
    hyp = frozen["hypocenter"]
    assert hyp["latitude"] == event["latitude"]
    assert hyp["longitude"] == event["longitude"]
    assert hyp["depth_km"] == event["depth_km"]
    assert len(frozen["stations"]) == 8
    for st in frozen["stations"]:
        assert 0.0 <= st["azimuth_deg"] < 360.0
        assert 90.0 < st["takeoff_deg"] <= 180.0  # upgoing rays to the surface
    assert frozen["azimuthal_gap_deg"] < 180.0
    # normalized polarity set: one row per station, sorted by station code
    assert [p["station"] for p in frozen["polarities"]] == sorted(
        p["station"] for p in frozen["polarities"]
    )
    assert {p["polarity"] for p in frozen["polarities"]} == {"positive", "negative"}
    assert rep["request"]["event_id"] == event["event_id"]
    assert rep["request"]["model"] == {"name": "default", "version": "1.0"}

    pick_station = {p["pick_id"]: p["station"] for p in event["picks"]}
    for page_solution in _all_solutions(client, rep["inversion_id"]):
        p1, p2 = page_solution["plane_1"], page_solution["plane_2"]
        for plane in (p1, p2):
            assert 0.0 <= plane["strike_deg"] < 360.0
            assert 0.0 < plane["dip_deg"] <= 90.0
            assert -180.0 <= plane["rake_deg"] < 180.0
        # the two planes are the same mechanism (swapped representation)
        key_1 = fm.mechanism_key(p1["strike_deg"], p1["dip_deg"], p1["rake_deg"])
        key_2 = fm.mechanism_key(p2["strike_deg"], p2["dip_deg"], p2["rake_deg"])
        assert _mechanism_key_close(key_1, key_2)
        # per-station observed/predicted rows are consistent with the misfit
        observations = page_solution["observations"]
        assert len(observations) == 8
        mismatches = [o for o in observations if not o["match"]]
        assert page_solution["misfit_count"] == len(mismatches)
        assert page_solution["misfit_weight"] == round(
            sum(o["weight"] for o in mismatches), 6
        )
        for o in observations:
            assert pick_station[o["pick_id"]] == o["station"]
            assert o["observed"] in ("positive", "negative")
            assert o["predicted"] in ("positive", "negative", "nodal")
            assert o["match"] == (o["observed"] == o["predicted"])


def test_equivalent_plane_swap_solutions_are_deduplicated(client):
    _, _, _, resp = _invert_known(client, prefix="DD")
    assert resp.status_code == 201, resp.text
    rep = resp.json()
    solutions = _all_solutions(client, rep["inversion_id"])
    assert len(solutions) == rep["solution_count"]
    keys = [_solution_key(s) for s in solutions]
    assert len(keys) == len(set(keys)), "swapped-plane duplicates must be removed"
    # stable presentation order: (strike, dip, rake) of plane 1
    order = [
        (s["plane_1"]["strike_deg"], s["plane_1"]["dip_deg"], s["plane_1"]["rake_deg"])
        for s in solutions
    ]
    assert order == sorted(order)
    assert [s["solution_index"] for s in solutions] == list(range(len(solutions)))


# --------------------------------------------------------------------------- #
# idempotency / concurrency
# --------------------------------------------------------------------------- #
def test_inversion_is_idempotent(client):
    _, event, obs, first = _invert_known(client, prefix="II")
    assert first.status_code == 201
    second = _invert(client, event["event_id"], obs)
    assert second.status_code == 200
    assert second.json()["created"] is False
    assert second.json()["inversion_id"] == first.json()["inversion_id"]
    assert second.json()["solutions"] == first.json()["solutions"]


def test_request_order_and_duplicates_normalize_to_same_inversion(client):
    _, event, obs, first = _invert_known(client, prefix="IN")
    assert first.status_code == 201
    # reversed order plus duplicated entries (same polarity) normalize away
    shuffled = list(reversed(obs)) + obs[:3]
    second = _invert(client, event["event_id"], shuffled)
    assert second.status_code == 200
    assert second.json()["inversion_id"] == first.json()["inversion_id"]


def test_different_polarity_set_is_a_distinct_inversion(client):
    _, event, obs, first = _invert_known(client, prefix="ID")
    assert first.status_code == 201

    heavier = [dict(o, weight=2.0) if i == 0 else dict(o) for i, o in enumerate(obs)]
    second = _invert(client, event["event_id"], heavier)
    assert second.status_code == 201
    assert second.json()["inversion_id"] != first.json()["inversion_id"]

    flipped = [dict(o) for o in obs]
    flipped[0] = dict(flipped[0], polarity="negative"
                      if flipped[0]["polarity"] == "positive" else "positive")
    third = _invert(client, event["event_id"], flipped)
    assert third.status_code == 201
    assert third.json()["inversion_id"] != first.json()["inversion_id"]
    assert third.json()["inversion_id"] != second.json()["inversion_id"]


def test_concurrent_identical_inversions_collapse(client):
    _, event, obs, _ = _invert_known(client, prefix="IC")
    # fresh inversion of the same event: drop one station so the request hash
    # differs from the one created above
    obs = obs[:-1]
    results = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        r = _invert(client, event["event_id"], obs)
        results.append((r.status_code, r.json()["inversion_id"]))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start(); t2.start(); t1.join(); t2.join()

    statuses = sorted(s for s, _ in results)
    assert statuses == [200, 201], results
    assert results[0][1] == results[1][1]


# --------------------------------------------------------------------------- #
# lazy repair of reports persisted under the legacy M == -M dedup rule
# --------------------------------------------------------------------------- #
def _ring_geometry_for_event(client, prefix="LR"):
    """Per-station ray vectors + coords of the standard 8-station ring event."""
    data = _process(client, prefix)
    event = _best(data)
    model = LayeredModel.from_dicts(DEFAULT_MODEL_LAYERS)
    coords = {
        code: (lat, lon, elev)
        for code, lat, lon, elev in station_ring(prefix, n=8)
    }
    rays = {}
    for code, (lat, lon, elev) in coords.items():
        az, to = fm.ray_geometry(
            model,
            event["latitude"], event["longitude"], event["depth_km"],
            lat, lon, elev,
        )
        rays[code] = fm._ray_vector(az, to)
    return data, event, rays


# Polarity/weight pattern for which the corrected search scores 0.0 while the
# legacy M == -M search overstates the minimum as 0.5 (see unit regression).
# Keyed by ring station index so the same pattern works for any test prefix.
_REPAIR_BY_INDEX = {
    0: (-1, 0.5), 1: (-1, 2.0), 2: (1, 1.0), 3: (1, 1.0),
    4: (-1, 1.0), 5: (1, 2.0), 6: (-1, 0.5), 7: (-1, 1.0),
}


def _repair_polarities(prefix: str, weight_scale: float = 1.0) -> dict:
    return {
        f"{prefix}{i:02d}": (sign, round(weight * weight_scale, 6))
        for i, (sign, weight) in _REPAIR_BY_INDEX.items()
    }


def _persist_legacy_report(client, data, event, prefix: str, weight_scale: float = 1.0):
    """Insert a report exactly as the v1 release would have stored it."""
    from app.database import SessionLocal
    from app.models import (
        Event, Run, VelocityModel,
        FocalMechanismInversion, FocalMechanismSolution,
    )

    repair_polarities = _repair_polarities(prefix, weight_scale)
    p_by_station = {p["station"]: p for p in _p_picks(event)}
    observations = [
        {
            "pick_id": p_by_station[st]["pick_id"],
            "polarity": "positive" if sign > 0 else "negative",
            "weight": weight,
        }
        for st, (sign, weight) in repair_polarities.items()
    ]

    db = SessionLocal()
    try:
        row = fm.create_inversion(
            db,
            type(
                "Req", (),
                {"event_id": event["event_id"], "observations": [
                    type("Obs", (), o)() for o in observations
                ]},
            )(),
        )[0]
        inversion_id = row["inversion_id"]

        # Rebuild the solution set with the legacy key and stamp the row v1.
        inversion = db.get(FocalMechanismInversion, inversion_id)
        stations = sorted(repair_polarities)
        run = db.get(Run, inversion.run_id)
        ev = db.get(Event, inversion.event_id)
        model_row = db.get(VelocityModel, run.model_id)
        geometry, coords, layers, _gap = fm._station_geometry(
            db, run, ev, model_row, stations
        )
        polarities = {s: repair_polarities[s] for s in stations}
        pick_ids = {
            row_["station"]: row_["pick_id"]
            for row_ in inversion.frozen["polarities"]
        }

        step = 10
        seen, best, tied = set(), None, []
        for strike in range(0, 360, step):
            for dip in range(step, 91, step):
                for rake in range(-180, 180, step):
                    normal, slip = fm.fault_vectors(strike, dip, rake)
                    key = fm._legacy_tensor_key(normal, slip)
                    if key in seen:
                        continue
                    seen.add(key)
                    misfit, mismatches = 0.0, 0
                    for station in stations:
                        sign, weight = polarities[station]
                        x = fm._ray_vector(*geometry[station])
                        if (
                            fm._predicted_sign(
                                fm._dot(x, normal) * fm._dot(x, slip)
                            )
                            != sign
                        ):
                            misfit += weight
                            mismatches += 1
                    misfit_key = round(misfit, 6)
                    entry = (strike, dip, rake, misfit, mismatches)
                    if best is None or misfit_key < best:
                        best, tied = misfit_key, [entry]
                    elif misfit_key == best:
                        tied.append(entry)
        tied.sort(key=lambda t: (t[0], t[1], t[2]))

        db.query(FocalMechanismSolution).filter_by(
            inversion_id=inversion.id
        ).delete(synchronize_session=False)
        fm._persist_solutions(
            db, inversion.id, tied, geometry, polarities, pick_ids
        )
        inversion.solution_count = len(tied)
        inversion.min_misfit_weight = round(best, 6)
        inversion.dedup_rule_version = fm.DEDUP_RULE_LEGACY
        db.commit()
        legacy = {
            "id": inversion.id,
            "request_hash": inversion.request_hash,
            "created_at": inversion.created_at,
            "solution_count": len(tied),
            "min_misfit_weight": round(best, 6),
        }
    finally:
        db.close()
    return observations, legacy


def _expected_solutions(rays, prefix: str, weight_scale: float = 1.0):
    """The corrected tied set, computed directly from the search primitive."""
    best, tied = fm.grid_search(rays, _repair_polarities(prefix, weight_scale))
    return best, tied


def _repair_observations(event, prefix: str, weight_scale: float = 1.0):
    return [
        {"pick_id": next(p for p in _p_picks(event)
                         if p["station"] == st)["pick_id"],
         "polarity": "positive" if sign > 0 else "negative",
         "weight": weight}
        for st, (sign, weight) in sorted(
            _repair_polarities(prefix, weight_scale).items()
        )
    ]


def test_legacy_report_is_repaired_on_get_by_id(client):
    data, event, rays = _ring_geometry_for_event(client)
    observations, legacy = _persist_legacy_report(client, data, event, "LR")
    assert legacy["min_misfit_weight"] == 0.5
    correct_best, correct_tied = _expected_solutions(rays, "LR")
    assert correct_best == 0.0

    got = client.get(f"/v1/focal-mechanisms/{legacy['id']}")
    assert got.status_code == 200, got.text
    rep = got.json()
    assert rep["inversion_id"] == legacy["id"]  # original id preserved
    assert rep["min_misfit_weight"] == 0.0
    assert rep["solution_count"] == len(correct_tied)

    # every surviving solution reports internally consistent weighted misfit
    all_rows = _all_solutions(client, legacy["id"])
    assert len(all_rows) == rep["solution_count"]
    for s in all_rows:
        mismatches = [o for o in s["observations"] if not o["match"]]
        assert s["misfit_count"] == len(mismatches)
        assert s["misfit_weight"] == round(
            sum(o["weight"] for o in mismatches), 6
        )
    # dedup now collapses plane swaps but keeps the slip-reversed candidates
    keys = [_solution_key(s) for s in all_rows]
    assert len(keys) == len(set(keys))


def test_legacy_report_repair_is_idempotent(client):
    data, event, rays = _ring_geometry_for_event(client, prefix="LI")
    _, legacy = _persist_legacy_report(client, data, event, "LI")

    first = client.get(f"/v1/focal-mechanisms/{legacy['id']}").json()
    # a repeat request of the same normalized polarity set returns same report
    repeat = _invert(client, event["event_id"], _repair_observations(event, "LI"))
    assert repeat.status_code == 200
    assert repeat.json()["inversion_id"] == legacy["id"]
    assert repeat.json()["created"] is False

    again = client.get(f"/v1/focal-mechanisms/{legacy['id']}").json()
    solutions_again = client.get(
        f"/v1/focal-mechanisms/{legacy['id']}/solutions", params={"limit": 200}
    ).json()["solutions"]

    assert again["solutions"] == first["solutions"]
    # idempotency key preserved (the summary list exposes request_hash)
    listed = client.get(
        "/v1/focal-mechanisms",
        params={"event_id": event["event_id"], "limit": 200},
    ).json()["inversions"]
    by_id = {i["inversion_id"]: i for i in listed}
    assert by_id[legacy["id"]]["request_hash"] == legacy["request_hash"]
    assert solutions_again == first["solutions"]


def test_legacy_report_repair_preserves_original_event_run_and_directory(client):
    data, event, rays = _ring_geometry_for_event(client, prefix="LO")
    _, legacy = _persist_legacy_report(client, data, event, "LO")

    run_before = client.get(f"/v1/runs/{data['run_id']}").json()
    event_before = client.get(f"/v1/events/{event['event_id']}").json()
    catalog_before = client.get("/v1/catalog/entries").json()

    client.get(f"/v1/focal-mechanisms/{legacy['id']}")
    client.get(f"/v1/focal-mechanisms/{legacy['id']}/solutions")
    _invert(client, event["event_id"], _repair_observations(event, "LO"))

    assert client.get(f"/v1/runs/{data['run_id']}").json() == run_before
    assert client.get(f"/v1/events/{event['event_id']}").json() == event_before
    assert client.get("/v1/catalog/entries").json() == catalog_before


def test_legacy_report_is_repaired_concurrently_without_duplicate_writes(client):
    data, event, rays = _ring_geometry_for_event(client, prefix="LC")
    _, legacy = _persist_legacy_report(client, data, event, "LC")
    correct_best, correct_tied = _expected_solutions(rays, "LC")

    payloads = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        r = client.get(f"/v1/focal-mechanisms/{legacy['id']}")
        payloads.append(r.json())

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start(); t2.start(); t1.join(); t2.join()

    for rep in payloads:
        assert rep["min_misfit_weight"] == 0.0
        assert rep["solution_count"] == len(correct_tied)

    from app.database import SessionLocal
    from app.models import FocalMechanismSolution

    db = SessionLocal()
    try:
        count = (
            db.query(FocalMechanismSolution)
            .filter_by(inversion_id=legacy["id"])
            .count()
        )
    finally:
        db.close()
    assert count == len(correct_tied)  # exactly one solution set, never doubled
    # sort_index stays a dense 0..n-1 pagination order
    all_rows = _all_solutions(client, legacy["id"], limit=13)
    assert [s["solution_index"] for s in all_rows] == list(range(len(all_rows)))


def test_legacy_report_is_repaired_through_solutions_endpoint(client):
    data, event, rays = _ring_geometry_for_event(client, prefix="LS")
    _, legacy = _persist_legacy_report(client, data, event, "LS")
    correct_best, correct_tied = _expected_solutions(rays, "LS")

    page = client.get(
        f"/v1/focal-mechanisms/{legacy['id']}/solutions", params={"limit": 3}
    )
    assert page.status_code == 200, page.text
    body = page.json()
    assert body["min_misfit_weight"] == 0.0
    assert body["solution_count"] == len(correct_tied)
    assert len(body["solutions"]) == 3


def test_legacy_report_repair_uses_full_precision_weights(client):
    # Weights with more than three decimals: frozen rows round to 3 decimals,
    # so recomputing from the frozen weights would visibly change misfits; the
    # repair must instead grid-search with the full-precision request weights.
    scale = 0.3337
    data, event, rays = _ring_geometry_for_event(client, prefix="LW")
    observations, legacy = _persist_legacy_report(
        client, data, event, "LW", weight_scale=scale
    )
    correct_best, correct_tied = _expected_solutions(rays, "LW", scale)
    assert correct_best == 0.0

    rep = client.get(f"/v1/focal-mechanisms/{legacy['id']}").json()
    assert rep["min_misfit_weight"] == 0.0
    assert rep["solution_count"] == len(correct_tied)

    # repeating the exact request returns the same (now repaired) report; the
    # id and request hash are preserved and the solutions are fully corrected
    repeat = _invert(client, event["event_id"], observations)
    assert repeat.status_code == 200
    assert repeat.json()["inversion_id"] == legacy["id"]
    assert repeat.json()["solutions"] == rep["solutions"]

    # a corrected grid search over the exact same inputs yields these rows,
    # so the repair matches what the corrected rule would produce fresh
    assert [
        (
            s["plane_1"]["strike_deg"],
            s["plane_1"]["dip_deg"],
            s["plane_1"]["rake_deg"],
        )
        for s in rep["solutions"]
    ] == [
        (float(a), float(b), float(c)) for a, b, c, *_ in correct_tied[:50]
    ]

    # ... while the frozen display weights still carry the 3-decimal rounding
    frozen_weights = {
        p["station"]: p["weight"] for p in rep["frozen"]["polarities"]
    }
    assert all(round(w, 3) == w for w in frozen_weights.values())


def test_migration_adds_marker_and_flags_existing_rows_legacy(tmp_path):
    # A database created by the v1 release has no dedup_rule_version column.
    from sqlalchemy import create_engine, inspect, text

    from app.database import run_lightweight_migrations

    legacy_db = tmp_path / "legacy.db"
    engine = create_engine(f"sqlite:///{legacy_db}")
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE focal_mechanism_inversions ("
            "id VARCHAR(36) PRIMARY KEY, request_hash VARCHAR(64) NOT NULL, "
            "request JSON NOT NULL, event_id VARCHAR(36) NOT NULL, "
            "run_id VARCHAR(36) NOT NULL, frozen JSON NOT NULL, "
            "grid_step_deg FLOAT NOT NULL, station_count INTEGER NOT NULL, "
            "solution_count INTEGER NOT NULL, min_misfit_weight FLOAT NOT NULL, "
            "created_at DATETIME NOT NULL)"
        ))
        conn.execute(text(
            "INSERT INTO focal_mechanism_inversions "
            "(id, request_hash, request, event_id, run_id, frozen, "
            "grid_step_deg, station_count, solution_count, min_misfit_weight, "
            "created_at) VALUES "
            "('a', 'h1', '{}', 'e', 'r', '{}', 10.0, 6, 3, 1.5, '2026-01-01 00:00:00')"
        ))

    run_lightweight_migrations(engine)
    cols = {c["name"]: c for c in inspect(engine).get_columns(
        "focal_mechanism_inversions"
    )}
    assert "dedup_rule_version" in cols
    with engine.begin() as conn:
        version = conn.execute(text(
            "SELECT dedup_rule_version FROM focal_mechanism_inversions WHERE id='a'"
        )).scalar()
    assert version == fm.DEDUP_RULE_LEGACY  # existing rows need lazy repair

    # the migration is idempotent
    run_lightweight_migrations(engine)
    engine.dispose()


def test_fresh_database_marker_defaults_to_current_rule(tmp_path):
    from datetime import datetime

    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session

    from app.database import Base, run_lightweight_migrations
    from app import models  # noqa: F401  register tables

    engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    Base.metadata.create_all(bind=engine)
    run_lightweight_migrations(engine)  # no-op on the current schema

    # an ORM-inserted row is stamped with the current rule without the caller
    # having to set the marker
    with Session(engine) as session:
        session.add(models.FocalMechanismInversion(
            request_hash="h", request={}, event_id="e", run_id="r",
            frozen={}, grid_step_deg=10.0, station_count=6,
            solution_count=0, min_misfit_weight=0.0,
            created_at=datetime(2026, 1, 1),
        ))
        session.commit()
        version = session.execute(text(
            "SELECT dedup_rule_version FROM focal_mechanism_inversions"
        )).scalar()
    assert version == fm.DEDUP_RULE_CURRENT
    engine.dispose()


# --------------------------------------------------------------------------- #
# explicit rejections
# --------------------------------------------------------------------------- #
def test_unknown_event_returns_404(client):
    resp = _invert(client, "0" * 32, [{"pick_id": "f" * 32, "polarity": "positive"}])
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "event_not_found"


def test_unknown_pick_returns_404(client):
    _, event, obs, _ = _invert_known(client, prefix="UP")
    obs[0] = {"pick_id": "f" * 32, "polarity": "positive", "weight": 1.0}
    resp = _invert(client, event["event_id"], obs)
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "pick_not_found"


def test_pick_of_another_event_is_rejected(client):
    _, event_a, obs_a, _ = _invert_known(client, prefix="PA")
    _, event_b, obs_b, _ = _invert_known(client, prefix="PB")
    # a valid P pick, but associated with the other event
    obs_a[0] = obs_b[0]
    resp = _invert(client, event_a["event_id"], obs_a)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "pick_not_associated"


def test_s_phase_pick_is_rejected(client):
    data = _process(client, "SP", n_stations=8)
    event = _best(data)
    obs = _polarities_for_mechanism(event, "SP", TRUE_MECHANISM)
    s_pick = next(p for p in event["picks"] if p["phase"] == "S")
    obs[0] = {"pick_id": s_pick["pick_id"], "polarity": "positive", "weight": 1.0}
    resp = _invert(client, event["event_id"], obs)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "pick_not_associated"


def test_unassociated_pick_is_rejected(client):
    body, _ = synthetic_event_payload(prefix="UA", n_stations=8)
    body["manual_picks"] = [
        {"station": "UA00", "phase": "P", "time": BASE_TIME, "weight": 1.0}
    ]
    data = client.post("/v1/process", json=body).json()
    event = _best(data)
    stray = next(p for p in data["unassociated_picks"] if p["station"] == "UA00")
    obs = _polarities_for_mechanism(event, "UA", TRUE_MECHANISM)
    obs[0] = {"pick_id": stray["pick_id"], "polarity": "positive", "weight": 1.0}
    resp = _invert(client, event["event_id"], obs)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "pick_not_associated"


def test_contradictory_polarity_at_one_station_is_rejected(client):
    _, event, obs, _ = _invert_known(client, prefix="CP")
    dup = dict(obs[0])
    dup["polarity"] = "negative" if obs[0]["polarity"] == "positive" else "positive"
    resp = _invert(client, event["event_id"], obs + [dup])
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "contradictory_polarity"


def test_fewer_than_six_stations_is_rejected(client):
    _, event, obs, _ = _invert_known(client, prefix="ST")
    resp = _invert(client, event["event_id"], obs[:5])
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "insufficient_polarity_stations"
    # exactly six independent stations is acceptable
    ok = _invert(client, event["event_id"], obs[:6])
    assert ok.status_code == 201, ok.text
    assert ok.json()["station_count"] == 6


def test_large_azimuthal_gap_is_rejected(client):
    """Stations clustered in a 70-degree wedge leave a ~290-degree gap."""
    ev_lat, ev_lon, ev_depth = 35.05, -117.95, 12.0
    model = LayeredModel.from_dicts(DEFAULT_MODEL_LAYERS)
    waveforms = []
    for i in range(8):
        az = 10.0 + i * 10.0  # azimuths 10..80 degrees
        lat = ev_lat + 0.25 * math.cos(math.radians(az))
        lon = ev_lon + 0.25 * math.sin(math.radians(az)) / math.cos(math.radians(ev_lat))
        code = f"GA{i:02d}"
        dist = distance_km(ev_lat, ev_lon, lat, lon)
        tp = model.travel_time(ev_depth, dist, "P")[0]
        ts = model.travel_time(ev_depth, dist, "S")[0]
        waveforms.append(
            {
                "station": code,
                "latitude": lat,
                "longitude": lon,
                "elevation_m": 0.0,
                "sampling_rate": 100.0,
                "start_time": BASE_TIME,
                "data_b64": make_waveform_b64(
                    code, 100.0, BASE_TIME, [(30.0 + tp, 4.0), (30.0 + ts, 6.0)]
                ),
            }
        )
    body = {
        "waveforms": waveforms,
        "manual_picks": [],
        "model": {"name": "default", "version": "1.0"},
        "params": {"max_residual_s": 1.0, "min_stations": 4},
    }
    data = client.post("/v1/process", json=body)
    assert data.status_code == 200, data.text
    event = _best(data.json())
    obs = [
        {"pick_id": p["pick_id"], "polarity": "positive", "weight": 1.0}
        for p in _p_picks(event)
    ]
    assert len(obs) >= 6
    resp = _invert(client, event["event_id"], obs)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "azimuthal_gap_too_large"


# --------------------------------------------------------------------------- #
# pagination
# --------------------------------------------------------------------------- #
def test_solutions_pagination_is_stable(client):
    # all-positive polarities leave many tied best solutions to page through
    data = _process(client, "PG", n_stations=8)
    event = _best(data)
    obs = [
        {"pick_id": p["pick_id"], "polarity": "positive", "weight": 1.0}
        for p in _p_picks(event)
    ]
    rep = _invert(client, event["event_id"], obs)
    assert rep.status_code == 201, rep.text
    inversion_id = rep.json()["inversion_id"]
    total = rep.json()["solution_count"]
    assert total > 20, "expected enough tied solutions to exercise pagination"

    solutions = _all_solutions(client, inversion_id, limit=7)
    assert len(solutions) == total
    assert [s["solution_index"] for s in solutions] == list(range(total))
    keys = [s["plane_1"]["strike_deg"] for s in solutions]
    assert keys == sorted(keys)


def test_solution_cursor_bound_to_inversion(client):
    _, event, obs, first = _invert_known(client, prefix="CB")
    assert first.status_code == 201
    inv1 = first.json()["inversion_id"]
    heavier = [dict(o, weight=2.0) if i == 0 else dict(o) for i, o in enumerate(obs)]
    second = _invert(client, event["event_id"], heavier)
    assert second.status_code == 201
    inv2 = second.json()["inversion_id"]

    page = client.get(f"/v1/focal-mechanisms/{inv1}/solutions", params={"limit": 2})
    assert page.status_code == 200
    cursor = page.json()["next_cursor"]
    assert cursor is not None
    bad = client.get(
        f"/v1/focal-mechanisms/{inv2}/solutions",
        params={"limit": 2, "cursor": cursor},
    )
    assert bad.status_code == 422
    assert bad.json()["detail"]["code"] == "invalid_cursor"


def test_list_inversions_filter_and_pagination(client):
    _, event1, obs1, r1 = _invert_known(client, prefix="L1")
    _, event2, _, r2 = _invert_known(client, prefix="L2")
    assert r1.status_code == 201 and r2.status_code == 201
    inv1, inv2 = r1.json()["inversion_id"], r2.json()["inversion_id"]
    # a second, distinct inversion of event1 (different weights)
    heavier = [dict(o, weight=2.0) if i == 0 else dict(o) for i, o in enumerate(obs1)]
    r1b = _invert(client, event1["event_id"], heavier)
    assert r1b.status_code == 201
    inv1b = r1b.json()["inversion_id"]

    by_event = client.get("/v1/focal-mechanisms", params={"event_id": event1["event_id"]})
    assert by_event.status_code == 200
    assert {i["inversion_id"] for i in by_event.json()["inversions"]} == {inv1, inv1b}

    # keyset pagination walks every inversion exactly once
    seen, cursor = set(), None
    for _ in range(10):
        params = {"limit": 1}
        if cursor:
            params["cursor"] = cursor
        data = client.get("/v1/focal-mechanisms", params=params).json()
        seen.update(i["inversion_id"] for i in data["inversions"])
        cursor = data["next_cursor"]
        if cursor is None:
            break
    assert seen == {inv1, inv1b, inv2}

    # list cursor is bound to the event filter fingerprint
    page = client.get(
        "/v1/focal-mechanisms", params={"limit": 1, "event_id": event1["event_id"]}
    ).json()
    assert page["next_cursor"] is not None
    mismatched = client.get(
        "/v1/focal-mechanisms",
        params={"limit": 1, "event_id": event2["event_id"],
                "cursor": page["next_cursor"]},
    )
    assert mismatched.status_code == 422
    assert mismatched.json()["detail"]["code"] == "invalid_cursor"


def test_get_inversion_and_not_found(client):
    _, event, obs, first = _invert_known(client, prefix="GN")
    assert first.status_code == 201
    inversion_id = first.json()["inversion_id"]

    got = client.get(f"/v1/focal-mechanisms/{inversion_id}")
    assert got.status_code == 200
    assert got.json()["created"] is False
    assert got.json()["inversion_id"] == inversion_id
    assert got.json()["solutions"] == first.json()["solutions"]

    missing = client.get("/v1/focal-mechanisms/" + "d" * 32)
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "focal_mechanism_not_found"
    missing = client.get("/v1/focal-mechanisms/" + "d" * 32 + "/solutions")
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "focal_mechanism_not_found"

    unknown_event = client.get("/v1/focal-mechanisms", params={"event_id": "e" * 32})
    assert unknown_event.status_code == 404
    assert unknown_event.json()["detail"]["code"] == "event_not_found"


# --------------------------------------------------------------------------- #
# read-only: original event, run and catalog content never change
# --------------------------------------------------------------------------- #
def test_inversion_does_not_mutate_event_or_run(client):
    data, event, obs, resp = _invert_known(client, prefix="RO")
    assert resp.status_code == 201
    run_before = client.get(f"/v1/runs/{data['run_id']}").json()
    event_before = client.get(f"/v1/events/{event['event_id']}").json()

    # repeat the inversion (idempotent replay) and list everything
    _invert(client, event["event_id"], obs)
    client.get("/v1/focal-mechanisms")
    client.get(f"/v1/focal-mechanisms/{resp.json()['inversion_id']}/solutions")

    assert client.get(f"/v1/runs/{data['run_id']}").json() == run_before
    assert client.get(f"/v1/events/{event['event_id']}").json() == event_before

    # process semantics are untouched: the same window maps to the same run
    again = client.post("/v1/process", json=synthetic_event_payload(prefix="RO")[0])
    assert again.json()["run_id"] == data["run_id"]
    assert again.json()["created"] is False

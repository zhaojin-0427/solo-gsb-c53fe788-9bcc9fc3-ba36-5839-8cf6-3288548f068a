"""P-wave first-motion focal mechanism inversion.

For one event of a completed run the client submits the event's associated P
picks together with their first-motion polarities (``positive``/``negative``)
and weights.  The service computes, for each participating station, the ray
azimuth and the P takeoff angle from the event hypocenter through the run's
layered velocity model, then grid-searches double-couple (strike, dip, rake)
on a 10-degree grid and keeps every mechanism whose weighted polarity misfit
equals the minimum.  A station whose predicted amplitude is zero (nodal)
always counts as a misfit.

Mechanisms that differ only by swapping the two nodal planes are the same
double couple; duplicates are removed via a canonical moment-tensor key that
identifies the two nodal-plane representations of one moment tensor only --
the overall sign-flipped tensor (the slip-reversed mechanism, whose predicted
polarities are all opposite) is a *different* candidate and is always kept --
and each surviving solution reports both nodal planes, its misfit weight and
the per-station observed/predicted polarities.  Tied best solutions are
presented in a stable (strike, dip, rake) order with keyset pagination.

Reports persisted by the previous release merged M with -M in the dedup key
and therefore miss slip-reversed candidates; such legacy reports are repaired
lazily (on repeat submission or any fetch by id) by recomputing the solution
set under the corrected rule.  Repair is idempotent and concurrency safe and
leaves the inversion id, request hash, frozen context and every original
ingestion/run/event/pick/catalog row untouched.

Requests are validated explicitly: every pick must be an associated P pick of
the event, one polarity per station (contradictions are rejected), at least
six independent stations, and the maximum azimuthal gap must stay below 180
degrees.  Inversions are idempotent per (event, model version, normalized
polarity set): identical concurrent requests collapse onto one persisted
report.  Like the catalog and the station audit this layer never modifies
ingestions/runs/events/picks or catalog content.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from datetime import datetime
from types import SimpleNamespace

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models import (
    Event,
    EventPick,
    FocalMechanismInversion,
    FocalMechanismSolution,
    Pick,
    Run,
    VelocityModel,
    Waveform,
)
from ..schemas import FocalMechanismCreate
from .association import distance_km
from .location import KM_PER_DEG_LAT
from .pipeline import ApiError, canonical_layers, iso_z
from .traveltime import LayeredModel

# Fixed 10-degree search grid (per product spec).
GRID_STEP_DEG = 10.0
# Acceptance gates (per product spec).
MIN_STATIONS = 6
MAX_AZIMUTHAL_GAP_DEG = 180.0
# |predicted amplitude| <= NODAL_EPS is nodal and counts as a misfit.
NODAL_EPS = 1e-6

DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 200

# Dedup-rule marker persisted on every inversion.  Version 1 was the original
# release, whose canonical moment-tensor key sign-canonicalized the tensor and
# therefore collapsed M and -M (the slip-reversed mechanism) into one
# candidate.  Version 2 normalizes the tensor magnitude only, so the key
# identifies the two nodal-plane representations of one moment tensor while
# the overall sign-flipped tensor survives as a distinct candidate.
DEDUP_RULE_LEGACY = 1
DEDUP_RULE_CURRENT = 2

_SIGN_LABEL = {1: "positive", -1: "negative", 0: "nodal"}


# --------------------------------------------------------------------------- #
# ray geometry (azimuth + takeoff through the run's layered model)
# --------------------------------------------------------------------------- #
def ray_geometry(
    model: LayeredModel,
    hyp_lat: float,
    hyp_lon: float,
    depth_km: float,
    sta_lat: float,
    sta_lon: float,
    elev_m: float,
) -> tuple[float, float]:
    """(azimuth_deg, takeoff_deg) of the P ray from hypocenter to station.

    Azimuth is clockwise from north; the takeoff angle is measured from the
    downward vertical (90 deg = horizontal, > 90 deg = upgoing ray), derived
    from the layered-model ray parameter via ``sin(i) = p * v_source``.
    """
    dist = distance_km(hyp_lat, hyp_lon, sta_lat, sta_lon)
    z_eff = depth_km + elev_m / 1000.0  # station elevation, as in the locator
    _, p = model.travel_time(z_eff, dist, "P")
    v_src = model.velocity_at_depth(z_eff, "P")
    sin_theta = min(1.0, max(0.0, p * v_src))
    takeoff = 180.0 - math.degrees(math.asin(sin_theta))
    d_north = (sta_lat - hyp_lat) * KM_PER_DEG_LAT
    d_east = (sta_lon - hyp_lon) * KM_PER_DEG_LAT * math.cos(
        math.radians(0.5 * (sta_lat + hyp_lat))
    )
    azimuth = math.degrees(math.atan2(d_east, d_north)) % 360.0
    return azimuth, takeoff


def _ray_vector(azimuth_deg: float, takeoff_deg: float) -> tuple[float, float, float]:
    """Unit takeoff direction (north, east, down) of the ray to a station."""
    az = math.radians(azimuth_deg)
    to = math.radians(takeoff_deg)
    return (
        math.sin(to) * math.cos(az),
        math.sin(to) * math.sin(az),
        math.cos(to),
    )


def _max_azimuthal_gap(azimuths: list[float]) -> float:
    ordered = sorted(azimuths)
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    gaps.append(ordered[0] + 360.0 - ordered[-1])
    return max(gaps)


# --------------------------------------------------------------------------- #
# double-couple geometry and the P radiation pattern
# --------------------------------------------------------------------------- #
def fault_vectors(
    strike_deg: float, dip_deg: float, rake_deg: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(normal, slip) unit vectors in (north, east, down) coordinates.

    Aki & Richards convention: strike clockwise from north, dip down from
    horizontal, rake in the fault plane from the strike direction.
    """
    phi = math.radians(strike_deg)
    delta = math.radians(dip_deg)
    lam = math.radians(rake_deg)
    sd, cd = math.sin(delta), math.cos(delta)
    ss, cs = math.sin(phi), math.cos(phi)
    sl, cl = math.sin(lam), math.cos(lam)
    normal = (-sd * ss, sd * cs, -cd)
    slip = (cl * cs + cd * sl * ss, cl * ss - cd * sl * cs, -sl * sd)
    return normal, slip


def plane_angles(
    normal: tuple[float, float, float], slip: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Recover (strike, dip, rake) from normal/slip unit vectors (NED).

    The pair is flipped to its equivalent (-n, -u) form whenever needed so
    the reported dip stays within (0, 90] degrees.
    """
    n, u = normal, slip
    if n[2] > 0.0:  # dip would exceed 90 deg: use the equivalent flipped pair
        n = tuple(-c for c in n)
        u = tuple(-c for c in u)
    dip = math.degrees(math.acos(min(1.0, max(-1.0, -n[2]))))
    sin_d = math.sin(math.radians(dip))
    if sin_d < 1e-9:
        # Horizontal plane: strike is unconstrained; report strike 0 and the
        # rake as the slip azimuth.
        strike = 0.0
        rake = math.degrees(math.atan2(u[1], u[0]))
    else:
        strike = math.degrees(math.atan2(-n[0], n[1])) % 360.0
        phi = math.radians(strike)
        cos_l = u[0] * math.cos(phi) + u[1] * math.sin(phi)
        sin_l = -u[2] / sin_d
        rake = math.degrees(math.atan2(sin_l, cos_l))
    rake = ((rake + 180.0) % 360.0) - 180.0
    return strike, dip, rake


def p_radiation(
    normal: tuple[float, float, float],
    slip: tuple[float, float, float],
    azimuth_deg: float,
    takeoff_deg: float,
) -> float:
    """Signed P first-motion amplitude (unit ray dotted with the couple)."""
    x = _ray_vector(azimuth_deg, takeoff_deg)
    return _dot(x, normal) * _dot(x, slip)


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _predicted_sign(amplitude: float) -> int:
    if amplitude > NODAL_EPS:
        return 1
    if amplitude < -NODAL_EPS:
        return -1
    return 0  # nodal: counts as a misfit whatever the observed polarity


def mechanism_key(strike_deg: float, dip_deg: float, rake_deg: float) -> tuple:
    """Canonical double-couple key, invariant under nodal-plane swap only.

    The symmetric moment tensor ``u n^T + n u^T`` is identical for the two
    nodal-plane representations of one mechanism; a magnitude-normalized
    (but never sign-canonicalized), rounded copy of it therefore dedupes
    equivalent grid solutions.  The key is *not* invariant under the overall
    sign flip ``(n, u) -> (-n, -u)``: that tensor is ``-M`` -- the
    slip-reversed mechanism -- whose P radiation pattern is the opposite of
    ``M``, so it must survive as a candidate of its own.
    """
    normal, slip = fault_vectors(strike_deg, dip_deg, rake_deg)
    return _tensor_key(normal, slip)


def _tensor_key(normal, slip) -> tuple:
    n, u = normal, slip
    m = [
        2.0 * n[0] * u[0],
        2.0 * n[1] * u[1],
        2.0 * n[2] * u[2],
        n[0] * u[1] + n[1] * u[0],
        n[0] * u[2] + n[2] * u[0],
        n[1] * u[2] + n[2] * u[1],
    ]
    # Magnitude normalization only: dividing by the largest component makes
    # the two nodal-plane representations of the same moment tensor collapse
    # onto one key, while M and -M retain opposite signs throughout and stay
    # distinct.
    scale = max(abs(c) for c in m)
    if scale <= 0.0:  # pragma: no cover - a unit couple never vanishes
        scale = 1.0
    return tuple(round(c / scale, 6) for c in m)


def _legacy_tensor_key(normal, slip) -> tuple:
    """The v1 dedup key: magnitude normalization *plus* sign canonicalization.

    Kept solely to recognize reports persisted under the old rule (M and -M
    merged); new grid searches use :func:`_tensor_key`.
    """
    m = list(_tensor_key(normal, slip))
    for c in m:
        if abs(c) > 1e-9:
            if c < 0.0:
                m = [-x for x in m]
            break
    return tuple(m)


# --------------------------------------------------------------------------- #
# grid search
# --------------------------------------------------------------------------- #
def grid_search(
    ray_vectors: dict[str, tuple[float, float, float]],
    polarities: dict[str, tuple[int, float]],
    grid_step_deg: float = GRID_STEP_DEG,
) -> tuple[float, list[tuple[int, int, int, float, int]]]:
    """Minimum weighted misfit and every deduplicated mechanism attaining it.

    ``ray_vectors``: station -> unit takeoff direction;
    ``polarities``: station -> (sign, weight) with sign in {+1, -1}.
    Returns ``(min_misfit_weight, tied)`` where each tied entry is
    ``(strike, dip, rake, misfit_weight, misfit_count)`` sorted by
    ``(strike, dip, rake)``.
    """
    step = int(round(grid_step_deg))
    seen: set[tuple] = set()
    best: float | None = None
    tied: list[tuple[int, int, int, float, int]] = []
    for strike in range(0, 360, step):
        for dip in range(step, 91, step):
            for rake in range(-180, 180, step):
                normal, slip = fault_vectors(strike, dip, rake)
                key = _tensor_key(normal, slip)
                if key in seen:
                    continue
                seen.add(key)
                misfit = 0.0
                mismatches = 0
                for station, x in ray_vectors.items():
                    sign, weight = polarities[station]
                    if _predicted_sign(_dot(x, normal) * _dot(x, slip)) != sign:
                        misfit += weight
                        mismatches += 1
                misfit_key = round(misfit, 6)
                if best is None or misfit_key < best:
                    best = misfit_key
                    tied = [(strike, dip, rake, misfit, mismatches)]
                elif misfit_key == best:
                    tied.append((strike, dip, rake, misfit, mismatches))
    tied.sort(key=lambda t: (t[0], t[1], t[2]))
    return best if best is not None else 0.0, tied


def _solution_detail(
    strike: int,
    dip: int,
    rake: int,
    misfit: float,
    mismatches: int,
    geometry: dict[str, tuple[float, float]],
    polarities: dict[str, tuple[int, float]],
    pick_ids: dict[str, str],
) -> dict:
    """Both nodal planes and the per-station observed/predicted polarities."""
    normal, slip = fault_vectors(strike, dip, rake)
    aux_strike, aux_dip, aux_rake = plane_angles(slip, normal)
    observations = []
    for station in sorted(geometry):
        azimuth, takeoff = geometry[station]
        sign, weight = polarities[station]
        amplitude = p_radiation(normal, slip, azimuth, takeoff)
        predicted = _predicted_sign(amplitude)
        observations.append(
            {
                "station": station,
                "pick_id": pick_ids[station],
                "azimuth_deg": round(azimuth, 2),
                "takeoff_deg": round(takeoff, 2),
                "observed": _SIGN_LABEL[sign],
                "weight": round(weight, 3),
                "predicted": _SIGN_LABEL[predicted],
                "amplitude": round(amplitude, 4),
                "match": predicted == sign,
            }
        )
    return {
        "plane_1": {
            "strike_deg": float(strike),
            "dip_deg": float(dip),
            "rake_deg": float(rake),
        },
        "plane_2": {
            "strike_deg": round(aux_strike, 4),
            "dip_deg": round(aux_dip, 4),
            "rake_deg": round(aux_rake, 4),
        },
        "misfit_weight": round(misfit, 6),
        "misfit_count": mismatches,
        "observations": observations,
    }


# --------------------------------------------------------------------------- #
# inversion creation (validate -> normalize -> grid search -> persist)
# --------------------------------------------------------------------------- #
def create_inversion(db: Session, req: FocalMechanismCreate) -> tuple[dict, bool]:
    event = db.get(Event, req.event_id)
    if event is None:
        raise ApiError(404, "event_not_found", f"event {req.event_id} not found")
    run = db.get(Run, event.run_id)
    if run is None:  # pragma: no cover - FK integrity makes this unreachable
        raise ApiError(404, "run_not_found", f"run {event.run_id} not found")
    model_row = db.get(VelocityModel, run.model_id)

    polarities, pick_ids = _normalize_observations(db, event, req.observations)
    stations = sorted(polarities)
    if len(stations) < MIN_STATIONS:
        raise ApiError(
            422,
            "insufficient_polarity_stations",
            f"need at least {MIN_STATIONS} independent stations with P "
            f"polarities, got {len(stations)}",
        )

    geometry, coords, layers, gap = _station_geometry(
        db, run, event, model_row, stations
    )
    if gap >= MAX_AZIMUTHAL_GAP_DEG:
        raise ApiError(
            422,
            "azimuthal_gap_too_large",
            f"maximum azimuthal gap {gap:.1f} deg is not below "
            f"{MAX_AZIMUTHAL_GAP_DEG:.0f} deg",
        )

    canon_request = {
        "event_id": event.id,
        "model": {"name": model_row.name, "version": model_row.version},
        "polarities": [
            {
                "station": s,
                "polarity": _SIGN_LABEL[polarities[s][0]],
                "weight": polarities[s][1],
            }
            for s in stations
        ],
    }
    request_hash = hashlib.sha256(
        json.dumps(canon_request, sort_keys=True).encode()
    ).hexdigest()
    existing = (
        db.query(FocalMechanismInversion)
        .filter_by(request_hash=request_hash)
        .one_or_none()
    )
    if existing is not None:
        # A repeat submission also serves as a repair trigger: reports saved
        # under the old M == -M dedup rule are recomputed before the response,
        # so the same request always returns the corrected same report.
        existing = _ensure_repaired(db, existing)
        db.commit()
        return inversion_payload(db, existing, created=False), False

    ray_vectors = {s: _ray_vector(*geometry[s]) for s in stations}
    min_misfit, tied = grid_search(ray_vectors, polarities)
    frozen = _build_frozen(
        model_row, layers, event, stations, coords, geometry, polarities,
        pick_ids, gap,
    )
    inversion = FocalMechanismInversion(
        request_hash=request_hash,
        request=canon_request,
        event_id=event.id,
        run_id=run.id,
        frozen=frozen,
        grid_step_deg=GRID_STEP_DEG,
        station_count=len(stations),
        solution_count=len(tied),
        min_misfit_weight=round(min_misfit, 6),
        dedup_rule_version=DEDUP_RULE_CURRENT,
    )
    db.add(inversion)
    try:
        db.flush()
    except IntegrityError:
        # Concurrent identical request: collapse onto the surviving inversion.
        db.rollback()
        winner = (
            db.query(FocalMechanismInversion)
            .filter_by(request_hash=request_hash)
            .one()
        )
        winner = _ensure_repaired(db, winner)
        db.commit()
        return inversion_payload(db, winner, created=False), False

    _persist_solutions(db, inversion.id, tied, geometry, polarities, pick_ids)
    db.commit()
    db.refresh(inversion)
    return inversion_payload(db, inversion, created=True), True


def _station_geometry(db: Session, run: Run, event: Event, model_row, stations):
    """Per-station ray geometry through the run's model, plus the inputs used.

    Returns ``(geometry, coords, layers, gap)`` where ``geometry`` maps station
    to ``(azimuth_deg, takeoff_deg)`` and ``coords`` maps station to
    ``(lat, lon, elev_m)``.
    """
    waveforms = db.query(Waveform).filter_by(ingestion_id=run.ingestion_id).all()
    coords = {w.station: (w.latitude, w.longitude, w.elevation_m) for w in waveforms}
    layers = canonical_layers(model_row.layers)
    model = LayeredModel.from_dicts(layers)
    geometry = {}
    for station in stations:
        lat, lon, elev = coords[station]
        geometry[station] = ray_geometry(
            model, event.latitude, event.longitude, event.depth_km, lat, lon, elev
        )
    gap = _max_azimuthal_gap([geometry[s][0] for s in stations])
    return geometry, coords, layers, gap


def _build_frozen(
    model_row, layers, event, stations, coords, geometry, polarities, pick_ids, gap
) -> dict:
    return {
        "model": {"name": model_row.name, "version": model_row.version,
                  "layers": layers},
        "hypocenter": {
            "origin_time": iso_z(event.origin_time),
            "latitude": round(event.latitude, 5),
            "longitude": round(event.longitude, 5),
            "depth_km": round(event.depth_km, 2),
        },
        "stations": [
            {
                "station": s,
                "latitude": coords[s][0],
                "longitude": coords[s][1],
                "elevation_m": coords[s][2],
                "azimuth_deg": round(geometry[s][0], 2),
                "takeoff_deg": round(geometry[s][1], 2),
            }
            for s in stations
        ],
        "polarities": [
            {
                "station": s,
                "pick_id": pick_ids[s],
                "polarity": _SIGN_LABEL[polarities[s][0]],
                "weight": round(polarities[s][1], 3),
            }
            for s in stations
        ],
        "azimuthal_gap_deg": round(gap, 2),
    }


def _persist_solutions(db, inversion_id, tied, geometry, polarities, pick_ids):
    for idx, (strike, dip, rake, misfit, mismatches) in enumerate(tied):
        db.add(
            FocalMechanismSolution(
                inversion_id=inversion_id,
                sort_index=idx,
                misfit_weight=round(misfit, 6),
                misfit_count=mismatches,
                detail=_solution_detail(
                    strike, dip, rake, misfit, mismatches,
                    geometry, polarities, pick_ids,
                ),
            )
        )


# --------------------------------------------------------------------------- #
# lazy repair of reports persisted under the legacy M == -M dedup rule
#
# The inversion id, request hash, frozen context (hypocenter/model/ray
# geometry/normalized polarity set) and every original ingestion/run/event/
# pick/catalog row stay untouched; only the derived solution set and its
# aggregate columns are recomputed.  The expensive recomputation happens
# before any write; a single conditional UPDATE atomically claims the row for
# one worker (a write lock on SQLite, a row lock on PostgreSQL), so concurrent
# repairs of the same report serialize: exactly one rewrites the solutions,
# every other worker simply reads the corrected report.  Re-running the
# repair is a no-op, hence idempotent.
# --------------------------------------------------------------------------- #
def _polarities_from_frozen(inversion: FocalMechanismInversion):
    """Rebuild (stations, polarities, pick_ids) for a recompute.

    Signs/weights come from the canonical ``request`` (full-precision weights,
    exactly what a fresh creation grid-searches with); the pick ids are only
    recorded in the frozen snapshot.
    """
    pick_ids = {
        row["station"]: row["pick_id"] for row in inversion.frozen["polarities"]
    }
    polarities = {}
    for row in inversion.request["polarities"]:
        sign = 1 if row["polarity"] == "positive" else -1
        polarities[row["station"]] = (sign, float(row["weight"]))
    return sorted(polarities), polarities, pick_ids


def _recompute_solutions(db: Session, inversion: FocalMechanismInversion):
    """Grid-search the report again under the current dedup rule.

    Ray geometry is recomputed from the immutable event/run/waveform rows and
    the velocity layers frozen in the report (rather than the rounded frozen
    geometry), and the frozen context is rebuilt from the same inputs, so the
    result is exactly what the same normalized request produces today while
    the inversion id, request hash and event/run links stay unchanged.
    """
    event = db.get(Event, inversion.event_id)
    run = db.get(Run, inversion.run_id)
    frozen_model = SimpleNamespace(
        name=inversion.frozen["model"]["name"],
        version=inversion.frozen["model"]["version"],
        layers=inversion.frozen["model"]["layers"],
    )

    stations, polarities, pick_ids = _polarities_from_frozen(inversion)
    geometry, coords, layers, gap = _station_geometry(
        db, run, event, frozen_model, stations
    )
    ray_vectors = {s: _ray_vector(*geometry[s]) for s in stations}
    min_misfit, tied = grid_search(ray_vectors, polarities)
    frozen = _build_frozen(
        frozen_model, layers, event, stations, coords, geometry, polarities,
        pick_ids, gap,
    )
    return frozen, min_misfit, tied, geometry, polarities, pick_ids


def _ensure_repaired(db: Session, inversion: FocalMechanismInversion):
    """Recompute a legacy-rule report in place; no-op for current reports.

    Returns the (possibly refreshed) ORM row ready to serialize; the caller
    owns the outer transaction (it commits after serializing).
    """
    if getattr(inversion, "dedup_rule_version", DEDUP_RULE_LEGACY) >= (
        DEDUP_RULE_CURRENT
    ):
        return inversion

    # Read-only recompute before taking any write lock.
    frozen, min_misfit, tied, geometry, polarities, pick_ids = (
        _recompute_solutions(db, inversion)
    )

    # Atomic claim: exactly one worker can flip the version marker.  Claim and
    # solution rewrite live in one transaction, so a non-claiming worker (which
    # blocks on the write/row lock) only ever sees the fully repaired report.
    result = db.execute(
        update(FocalMechanismInversion)
        .where(
            FocalMechanismInversion.id == inversion.id,
            FocalMechanismInversion.dedup_rule_version == DEDUP_RULE_LEGACY,
        )
        .values(dedup_rule_version=DEDUP_RULE_CURRENT)
    )
    claimed = (result.rowcount or 0) == 1
    if claimed:
        db.query(FocalMechanismSolution).filter_by(
            inversion_id=inversion.id
        ).delete(synchronize_session=False)
        db.query(FocalMechanismInversion).filter_by(id=inversion.id).update(
            {
                "solution_count": len(tied),
                "min_misfit_weight": round(min_misfit, 6),
                "frozen": frozen,
            },
            synchronize_session=False,
        )
        _persist_solutions(
            db, inversion.id, tied, geometry, polarities, pick_ids
        )
        db.commit()
    else:
        db.rollback()
    # Expire the local ORM object so the serialized view reflects either our
    # own rewrite or the concurrent worker's.
    db.expire_all()
    return db.get(FocalMechanismInversion, inversion.id)


def _normalize_observations(
    db: Session, event: Event, observations
) -> tuple[dict[str, tuple[int, float]], dict[str, str]]:
    """Validate picks and fold them into one (sign, weight) per station.

    Every pick must be an associated P pick of the event; a station submitted
    with conflicting polarities is rejected, while identical repeats merge
    (keeping the larger weight) so the normalized set is idempotency-safe.
    """
    rows = (
        db.query(EventPick, Pick)
        .join(Pick, EventPick.pick_id == Pick.id)
        .filter(EventPick.event_id == event.id, EventPick.phase == "P")
        .all()
    )
    p_picks = {ep.pick_id: p.station for ep, p in rows}

    polarities: dict[str, tuple[int, float]] = {}
    pick_ids: dict[str, str] = {}
    for obs in observations:
        pick = db.get(Pick, obs.pick_id)
        if pick is None:
            raise ApiError(404, "pick_not_found", f"pick {obs.pick_id} not found")
        station = p_picks.get(obs.pick_id)
        if station is None:
            raise ApiError(
                422,
                "pick_not_associated",
                f"pick {obs.pick_id} is not an associated P pick of event "
                f"{event.id}",
            )
        sign = 1 if obs.polarity == "positive" else -1
        weight = float(obs.weight)
        if station in polarities:
            prev_sign, prev_weight = polarities[station]
            if prev_sign != sign:
                raise ApiError(
                    422,
                    "contradictory_polarity",
                    f"station {station} submitted with both positive and "
                    "negative polarity",
                )
            polarities[station] = (sign, max(prev_weight, weight))
        else:
            polarities[station] = (sign, weight)
            pick_ids[station] = obs.pick_id
    return polarities, pick_ids


# --------------------------------------------------------------------------- #
# queries
# --------------------------------------------------------------------------- #
def get_inversion(db: Session, inversion_id: str) -> dict:
    row = db.get(FocalMechanismInversion, inversion_id)
    if row is None:
        raise ApiError(
            404,
            "focal_mechanism_not_found",
            f"focal mechanism inversion {inversion_id} not found",
        )
    # Fetching by the original id lazily repairs legacy (M == -M) reports.
    row = _ensure_repaired(db, row)
    db.commit()
    return inversion_payload(db, row, created=False)


def list_inversions(
    db: Session,
    event_id: str | None,
    limit: int,
    cursor: str | None,
) -> dict:
    if event_id is not None and db.get(Event, event_id) is None:
        raise ApiError(404, "event_not_found", f"event {event_id} not found")

    filters = {"event_id": event_id, "limit": limit}
    fingerprint = hashlib.sha256(
        json.dumps(filters, sort_keys=True).encode()
    ).hexdigest()[:16]

    cursor_at, cursor_id = None, None
    if cursor is not None:
        cursor_at, cursor_id = _decode_list_cursor(cursor, fingerprint)

    q = db.query(FocalMechanismInversion)
    if event_id is not None:
        q = q.filter(FocalMechanismInversion.event_id == event_id)
    if cursor_at is not None:
        q = q.filter(
            (FocalMechanismInversion.created_at > cursor_at)
            | (
                (FocalMechanismInversion.created_at == cursor_at)
                & (FocalMechanismInversion.id > cursor_id)
            )
        )
    q = q.order_by(FocalMechanismInversion.created_at, FocalMechanismInversion.id)
    rows = q.limit(limit + 1).all()

    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        last = page[-1]
        next_cursor = _encode_list_cursor(last.created_at, last.id, fingerprint)
    return {
        "inversions": [inversion_summary(r) for r in page],
        "next_cursor": next_cursor,
    }


def list_solutions(
    db: Session, inversion_id: str, limit: int, cursor: str | None
) -> dict:
    inversion = db.get(FocalMechanismInversion, inversion_id)
    if inversion is None:
        raise ApiError(
            404,
            "focal_mechanism_not_found",
            f"focal mechanism inversion {inversion_id} not found",
        )
    inversion = _ensure_repaired(db, inversion)
    db.commit()
    rows, next_cursor = _solutions_page(db, inversion, limit, cursor)
    return {
        "inversion_id": inversion.id,
        "event_id": inversion.event_id,
        "run_id": inversion.run_id,
        "solution_count": inversion.solution_count,
        "min_misfit_weight": round(inversion.min_misfit_weight, 6),
        "solutions": [solution_payload(r) for r in rows],
        "next_cursor": next_cursor,
    }


def _solutions_page(
    db: Session, inversion: FocalMechanismInversion, limit: int, cursor: str | None
):
    fingerprint = hashlib.sha256(inversion.id.encode()).hexdigest()[:16]
    cursor_index = None
    if cursor is not None:
        cursor_index = _decode_solution_cursor(cursor, fingerprint)

    q = db.query(FocalMechanismSolution).filter_by(inversion_id=inversion.id)
    if cursor_index is not None:
        q = q.filter(FocalMechanismSolution.sort_index > cursor_index)
    q = q.order_by(FocalMechanismSolution.sort_index).limit(limit + 1)
    rows = q.all()

    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        next_cursor = _encode_solution_cursor(page[-1].sort_index, fingerprint)
    return page, next_cursor


# --------------------------------------------------------------------------- #
# cursors
# --------------------------------------------------------------------------- #
def _encode_list_cursor(created_at, inversion_id: str, fingerprint: str) -> str:
    raw = json.dumps(
        {"t": created_at.strftime("%Y-%m-%dT%H:%M:%S.%f"), "id": inversion_id,
         "fp": fingerprint}
    ).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_list_cursor(cursor: str, fingerprint: str):
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
        token_at = datetime.strptime(payload["t"], "%Y-%m-%dT%H:%M:%S.%f")
        token_id = payload["id"]
        token_fp = payload["fp"]
    except Exception as exc:
        raise ApiError(422, "invalid_cursor", "malformed pagination cursor") from exc
    if token_fp != fingerprint:
        raise ApiError(
            422,
            "invalid_cursor",
            "cursor was issued for a different inversion query (event filter "
            "mismatch)",
        )
    return token_at, token_id


def _encode_solution_cursor(sort_index: int, fingerprint: str) -> str:
    raw = json.dumps({"i": sort_index, "fp": fingerprint}).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_solution_cursor(cursor: str, fingerprint: str) -> int:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
        token_i = int(payload["i"])
        token_fp = payload["fp"]
    except Exception as exc:
        raise ApiError(422, "invalid_cursor", "malformed pagination cursor") from exc
    if token_fp != fingerprint:
        raise ApiError(
            422,
            "invalid_cursor",
            "cursor was issued for a different focal mechanism inversion",
        )
    return token_i


# --------------------------------------------------------------------------- #
# serialization
# --------------------------------------------------------------------------- #
def solution_payload(row: FocalMechanismSolution) -> dict:
    d = row.detail
    return {
        "solution_index": row.sort_index,
        "plane_1": d["plane_1"],
        "plane_2": d["plane_2"],
        "misfit_weight": round(row.misfit_weight, 6),
        "misfit_count": row.misfit_count,
        "observations": d["observations"],
    }


def inversion_summary(row: FocalMechanismInversion) -> dict:
    return {
        "inversion_id": row.id,
        "event_id": row.event_id,
        "run_id": row.run_id,
        "request_hash": row.request_hash,
        "grid_step_deg": row.grid_step_deg,
        "station_count": row.station_count,
        "solution_count": row.solution_count,
        "min_misfit_weight": round(row.min_misfit_weight, 6),
        "created_at": iso_z(row.created_at),
    }


def inversion_payload(
    db: Session,
    inversion: FocalMechanismInversion,
    created: bool,
    solution_limit: int = DEFAULT_PAGE_LIMIT,
) -> dict:
    rows, next_cursor = _solutions_page(db, inversion, solution_limit, None)
    return {
        "inversion_id": inversion.id,
        "created": created,
        "event_id": inversion.event_id,
        "run_id": inversion.run_id,
        "request": inversion.request,
        "frozen": inversion.frozen,
        "grid_step_deg": inversion.grid_step_deg,
        "station_count": inversion.station_count,
        "solution_count": inversion.solution_count,
        "min_misfit_weight": round(inversion.min_misfit_weight, 6),
        "solutions": [solution_payload(r) for r in rows],
        "next_cursor": next_cursor,
        "created_at": iso_z(inversion.created_at),
    }

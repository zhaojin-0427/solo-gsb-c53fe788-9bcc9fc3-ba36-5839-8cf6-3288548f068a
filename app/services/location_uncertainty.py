"""Event location uncertainty assessment (station bootstrap).

For one event of a completed run the client submits a resample count N
(200-2000), a confidence level alpha (0.8-0.99) and a random seed.  The
service freezes the event's associated P/S picks (with their weights), the
velocity model, the calibration version (with its per-station P/S terms) and
the processing parameters, then performs N seed-driven bootstrap resamples:

* stations are drawn WITH REPLACEMENT, exactly as many draws as the event has
  participating stations (``numpy.random.default_rng(seed).integers``; draws
  are therefore reproducible byte-for-byte);
* a drawn station contributes ALL of its associated P/S picks, so every
  resample uses only observations the frozen event was associated with;
* a station drawn k times keeps its picks k-fold -- each observation's
  effective weight is its original pick weight multiplied by the draw
  multiplicity -- and each resample is relocated with the very same weighted
  least-squares locator used by ``process``/``replay`` (with the frozen
  calibration corrections applied first and the frozen ``max_residual_s``
  residual gate enforced afterwards);
* every resample records its solvable status and, when it fails, a structured
  reason (``insufficient_stations`` / ``location_failed`` /
  ``residual_exceeded``).

Nearest-rank (two-sided (1-alpha)/2 tail) quantiles of the solvable samples
give alpha confidence intervals for origin time, latitude, longitude and
depth; the sample covariance is computed over the solvable samples only, and
the solvable rate (M/N) is reported.  When fewer than 80% of the N resamples
are solvable the request is explicitly rejected (422) and nothing persists.

Like the catalog and the station impact audit this is a read-only analysis
layer: reports and samples are immutable once created, requests are
idempotent by (event, params, seed) -- concurrent identical requests collapse
to one report -- and nothing in ingestions/runs/events/picks or catalog
content is ever modified.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from collections import Counter
from datetime import datetime

import numpy as np
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models import (
    Event,
    LocationUncertaintyAssessment,
    LocationUncertaintySample,
    Run,
)
from . import station_audit
from .association import (
    REASON_INSUFFICIENT_STATIONS,
    REASON_LOCATION_FAILED,
    REASON_RESIDUAL_EXCEEDED,
    distance_km,
)
from .location import Observation, locate_event
from .pipeline import ApiError, from_epoch, iso_z
from .quality import quality_grade

DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 200

# A resample is rejected unless at least this fraction of the N resamples is
# solvable (strictly: M*100 >= 80*N); otherwise no report is persisted.
MIN_SOLVABLE_RATE = 0.8
_SOLVABLE_PCT = 80

# Origin-time vector entries are epoch seconds internally; iso8601 strings
# only appear in the serialized confidence intervals.
_COMPONENT_ORDER = ("origin_time_s", "latitude", "longitude", "depth_km")
_COMPONENT_LABEL = {
    "origin_time_s": "origin_time",
    "latitude": "latitude",
    "longitude": "longitude",
    "depth_km": "depth_km",
}


# --------------------------------------------------------------------------- #
# assessment creation (freeze -> N seeded resamples -> aggregate -> persist)
# --------------------------------------------------------------------------- #
def create_assessment(
    db: Session, event_id: str, resample_count: int, confidence_level: float,
    random_seed: int,
) -> tuple[dict, bool]:
    event = db.get(Event, event_id)
    if event is None:
        raise ApiError(404, "event_not_found", f"event {event_id} not found")
    run = db.get(Run, event.run_id)
    if run is None:  # pragma: no cover - FK integrity makes this unreachable
        raise ApiError(404, "run_not_found", f"run {event.run_id} not found")

    canon_request = {
        "event_id": event.id,
        "run_id": run.id,
        "resample_count": int(resample_count),
        "confidence_level": float(confidence_level),
        "random_seed": int(random_seed),
    }
    request_hash = hashlib.sha256(
        json.dumps(canon_request, sort_keys=True).encode()
    ).hexdigest()

    existing = (
        db.query(LocationUncertaintyAssessment)
        .filter_by(request_hash=request_hash)
        .one_or_none()
    )
    if existing is not None:
        db.commit()
        return assessment_payload(db, existing, created=False), False

    frozen, baseline = station_audit._freeze_context(db, event, run, int(run.params["min_stations"]))
    station_codes = sorted({p["station"] for p in frozen["picks"]})
    if not frozen["picks"]:
        raise ApiError(
            422, "uncertainty_event_has_no_picks", "event has no associated picks"
        )

    params_hash = _params_hash(run, frozen, resample_count, confidence_level)
    samples = _run_bootstrap(frozen, resample_count, random_seed)

    failure_counts = dict(sorted(Counter(
        s["failure_reason"] for s in samples if not s["solvable"]
    ).items()))
    solvable = [s for s in samples if s["solvable"]]
    solvable_count = len(solvable)
    solvable_rate = solvable_count / resample_count

    # Explicit rejection: effective (solvable) sample size below 80% of N.
    # Integer comparison keeps the boundary exact (M/N >= 0.8).
    if solvable_count * 100 < _SOLVABLE_PCT * resample_count:
        raise ApiError(
            422,
            "insufficient_effective_samples",
            f"only {solvable_count} of {resample_count} resamples were solvable "
            f"({solvable_rate:.3f}); at least {MIN_SOLVABLE_RATE:.0%} effective "
            "(solvable) samples are required, so no uncertainty report is stored",
        )

    result = _aggregate(solvable, confidence_level)

    assessment = LocationUncertaintyAssessment(
        request_hash=request_hash,
        request=canon_request,
        params_hash=params_hash,
        event_id=event.id,
        run_id=run.id,
        resample_count=resample_count,
        confidence_level=confidence_level,
        random_seed=random_seed,
        frozen=_frozen_json(frozen),
        baseline=_baseline_json(baseline),
        station_count=len(station_codes),
        solvable_count=solvable_count,
        solvable_rate=solvable_rate,
        failure_counts=failure_counts,
        result=result,
    )
    db.add(assessment)
    try:
        db.flush()
    except IntegrityError:
        # Concurrent identical request: collapse onto the surviving report.
        db.rollback()
        winner = (
            db.query(LocationUncertaintyAssessment)
            .filter_by(request_hash=request_hash)
            .one()
        )
        return assessment_payload(db, winner, created=False), False

    for s in samples:
        db.add(
            LocationUncertaintySample(
                assessment_id=assessment.id,
                sample_index=s["sample_index"],
                solvable=1 if s["solvable"] else 0,
                failure_reason=s["failure_reason"],
                detail=_sample_detail_json(s),
            )
        )
    db.commit()
    db.refresh(assessment)
    return assessment_payload(db, assessment, created=True), True


def _params_hash(
    run: Run, frozen: dict, resample_count: int, confidence_level: float
) -> str:
    """Canonical hash binding the report to event/run, frozen versions and N/alpha."""
    cal = frozen["calibration_snapshot"]
    blob = {
        "run_id": run.id,
        "model": {
            "name": frozen["model_snapshot"]["name"],
            "version": frozen["model_snapshot"]["version"],
        },
        "calibration": None if cal is None else cal["version"],
        "max_residual_s": frozen["params"]["max_residual_s"],
        "min_stations": frozen["params"]["min_stations"],
        "resample_count": int(resample_count),
        "confidence_level": float(confidence_level),
    }
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# seeded station bootstrap
# --------------------------------------------------------------------------- #
def _run_bootstrap(frozen: dict, resample_count: int, random_seed: int) -> list[dict]:
    """Run N deterministic station resamples through the production locator."""
    model = frozen["model"]
    stations = frozen["stations"]
    corrections = frozen["corrections"]
    max_residual_s = frozen["params"]["max_residual_s"]
    min_stations = frozen["params"]["min_stations"]

    picks = frozen["picks"]
    station_codes = sorted({p["station"] for p in picks})
    by_station: dict[str, list[dict]] = {}
    for p in picks:
        by_station.setdefault(p["station"], []).append(p)

    # One RNG for the whole assessment: draws k=0..N-1 are a deterministic
    # function of (station universe, seed, k).
    rng = np.random.default_rng(random_seed)
    n_stations = len(station_codes)

    samples = []
    for idx in range(resample_count):
        draw = rng.integers(0, n_stations, size=n_stations)
        mult = Counter(station_codes[int(k)] for k in draw)
        multiset = sorted(
            ({"station": code, "multiplicity": mult[code]} for code in mult),
            key=lambda e: e["station"],
        )

        observations = []
        for code, m in mult.items():
            for p in by_station[code]:
                # The frozen calibration correction is applied before
                # location, exactly as in process/replay and the audit.
                t_corrected = p["t"] - _correction(
                    corrections, p["station"], p["phase"]
                )
                # A station drawn m times keeps its picks m-fold: the
                # multiplicity multiplies the original pick weight (the
                # locator forms normal equations with sqrt(weight)).
                observations.append(
                    Observation(
                        station=p["station"],
                        phase=p["phase"],
                        t_obs=t_corrected,
                        weight=p["weight"] * m,
                    )
                )
        # Deterministic observation order (irrelevant to the normal equations
        # but keeps all downstream artifacts reproducible).
        observations.sort(key=lambda o: (o.station, o.phase, o.t_obs))

        sample = {
            "sample_index": idx,
            "solvable": False,
            "failure_reason": None,
            "stations": multiset,
            "distinct_station_count": len(mult),
            "solution": None,
            "draw_count": int(sum(e["multiplicity"] for e in multiset)),
        }

        # Structured failure gates, in the same order the association engine
        # escalates a pick's disposition.
        if len(mult) < min_stations:
            sample["failure_reason"] = REASON_INSUFFICIENT_STATIONS
            samples.append(sample)
            continue
        if len(observations) < 4:
            sample["failure_reason"] = REASON_LOCATION_FAILED
            samples.append(sample)
            continue

        loc = locate_event(model, observations, stations)
        if loc is None:
            sample["failure_reason"] = REASON_LOCATION_FAILED
            samples.append(sample)
            continue

        # Apply the frozen production residual gate (weighted observations):
        # an accepted resample must keep at least min_stations DISTINCT
        # stations within max_residual_s after relocation.
        kept = []
        for o in observations:
            lat, lon, elev_m = stations[o.station]
            dist = distance_km(loc.lat, loc.lon, lat, lon)
            tt, _ = model.travel_time(
                loc.depth_km + elev_m / 1000.0, dist, o.phase
            )
            residual = o.t_obs - (loc.t0 + tt)
            if abs(residual) <= max_residual_s:
                kept.append((o, residual))
        kept_stations = {o.station for o, _ in kept}
        if len(kept_stations) < min_stations:
            sample["failure_reason"] = REASON_RESIDUAL_EXCEEDED
            samples.append(sample)
            continue

        rms = math.sqrt(sum(r * r for _, r in kept) / len(kept))
        sample["solvable"] = True
        sample["solution"] = {
            "origin_time": iso_z(from_epoch(loc.t0)),
            "t0": loc.t0,
            "latitude": loc.lat,
            "longitude": loc.lon,
            "depth_km": loc.depth_km,
            "rms_residual_s": rms,
            "station_count": len(kept_stations),
            "quality": quality_grade(rms, len(kept_stations)),
        }
        samples.append(sample)
    return samples


def _correction(corrections, station: str, phase: str) -> float:
    if corrections is None:
        return 0.0
    return corrections.get((station, phase), 0.0)


# --------------------------------------------------------------------------- #
# aggregation: nearest-rank intervals + solvable-sample covariance
# --------------------------------------------------------------------------- #
def nearest_rank_indices(n: int, confidence_level: float) -> tuple[int, int]:
    """0-based inclusive (lo, hi) nearest-rank indices for the two-sided tail.

    Each tail carries (1-alpha)/2 mass; positions are 1-based nearest ranks
    clamped to [1, n] and converted to 0-based indices.  A small tolerance
    keeps an exact integer rank from rounding up due to float overshoot
    (e.g. 0.1*200 == 20.000000000000004).
    """
    tail = (1.0 - confidence_level) / 2.0
    lo_rank = max(1, math.ceil(tail * n - 1e-9))
    hi_rank = min(n, math.ceil((1.0 - tail) * n - 1e-9))
    if hi_rank < lo_rank:  # extremely narrow samples; interval degenerates to lo
        hi_rank = lo_rank
    return lo_rank - 1, hi_rank - 1


def _quantile_interval(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    lo_i, hi_i = nearest_rank_indices(len(values), confidence_level)
    ordered = np.sort(values)
    return float(ordered[lo_i]), float(ordered[hi_i])


def _aggregate(solvable: list[dict], confidence_level: float) -> dict:
    m = len(solvable)
    t0 = np.array([s["solution"]["t0"] for s in solvable], dtype=np.float64)
    lat = np.array([s["solution"]["latitude"] for s in solvable], dtype=np.float64)
    lon = np.array([s["solution"]["longitude"] for s in solvable], dtype=np.float64)
    depth = np.array([s["solution"]["depth_km"] for s in solvable], dtype=np.float64)
    vectors = {
        "origin_time_s": t0,
        "latitude": lat,
        "longitude": lon,
        "depth_km": depth,
    }

    intervals = {}
    for key in _COMPONENT_ORDER:
        lo, hi = _quantile_interval(vectors[key], confidence_level)
        if key == "origin_time_s":
            intervals[_COMPONENT_LABEL[key]] = {
                "lower": iso_z(from_epoch(lo)),
                "upper": iso_z(from_epoch(hi)),
            }
        else:
            intervals[_COMPONENT_LABEL[key]] = {
                "lower": lo,
                "upper": hi,
            }

    # Sample covariance over solvable samples only (ddof=1).  Always defined
    # here: the 80% gate with N >= 200 guarantees M >= 160 >= 2.
    matrix = np.vstack([t0, lat, lon, depth])
    cov = np.cov(matrix, ddof=1)
    covariance = {
        "order": ["origin_time_s", "latitude", "longitude", "depth_km"],
        "sample_count": m,
        "matrix": [[float(v) for v in row] for row in cov],
    }
    return {
        "confidence_level": float(confidence_level),
        "quantile_method": "nearest_rank",
        "intervals": intervals,
        "covariance": covariance,
    }


# --------------------------------------------------------------------------- #
# JSON snapshots
# --------------------------------------------------------------------------- #
def _frozen_json(frozen: dict) -> dict:
    payload = station_audit._frozen_json(frozen)
    # The assessment never replaces min_stations; the key belongs to the
    # original run only.
    payload.pop("min_keep_stations", None)
    payload["picks"] = sorted(
        (
            {
                "pick_id": q["pick_id"],
                "station": q["station"],
                "phase": q["phase"],
                "time": iso_z(q["time_utc"]),
                "source": q["source"],
                "residual_s": round(q["residual_s"], 4),
                "weight": round(q["weight"], 3),
            }
            for q in frozen["picks"]
        ),
        key=lambda p: (p["time"], p["station"], p["phase"]),
    )
    return payload


def _baseline_json(baseline: dict) -> dict:
    return station_audit._baseline_json(baseline)


def _sample_detail_json(sample: dict) -> dict:
    return {
        "stations": sample["stations"],
        "draw_count": sample["draw_count"],
        "distinct_station_count": sample["distinct_station_count"],
        "solution": _solution_json(sample["solution"]),
    }


def _solution_json(solution: dict | None) -> dict | None:
    if solution is None:
        return None
    return {
        "origin_time": solution["origin_time"],
        "latitude": round(solution["latitude"], 5),
        "longitude": round(solution["longitude"], 5),
        "depth_km": round(solution["depth_km"], 2),
        "rms_residual_s": round(solution["rms_residual_s"], 4),
        "station_count": solution["station_count"],
        "quality": solution["quality"],
    }


# --------------------------------------------------------------------------- #
# queries
# --------------------------------------------------------------------------- #
def get_assessment(db: Session, assessment_id: str) -> dict:
    row = db.get(LocationUncertaintyAssessment, assessment_id)
    if row is None:
        raise ApiError(
            404,
            "location_uncertainty_not_found",
            f"location uncertainty assessment {assessment_id} not found",
        )
    return assessment_payload(db, row, created=False)


def list_assessments(
    db: Session,
    event_id: str | None,
    run_id: str | None,
    limit: int,
    cursor: str | None,
) -> dict:
    if event_id is not None and db.get(Event, event_id) is None:
        raise ApiError(404, "event_not_found", f"event {event_id} not found")
    if run_id is not None and db.get(Run, run_id) is None:
        raise ApiError(404, "run_not_found", f"run {run_id} not found")

    filters = {"event_id": event_id, "run_id": run_id, "limit": limit}
    fingerprint = hashlib.sha256(
        json.dumps(filters, sort_keys=True).encode()
    ).hexdigest()[:16]

    cursor_at, cursor_id = None, None
    if cursor is not None:
        cursor_at, cursor_id = _decode_list_cursor(cursor, fingerprint)

    q = db.query(LocationUncertaintyAssessment)
    if event_id is not None:
        q = q.filter(LocationUncertaintyAssessment.event_id == event_id)
    if run_id is not None:
        q = q.filter(LocationUncertaintyAssessment.run_id == run_id)
    if cursor_at is not None:
        q = q.filter(
            (LocationUncertaintyAssessment.created_at > cursor_at)
            | (
                (LocationUncertaintyAssessment.created_at == cursor_at)
                & (LocationUncertaintyAssessment.id > cursor_id)
            )
        )
    q = q.order_by(
        LocationUncertaintyAssessment.created_at, LocationUncertaintyAssessment.id
    ).limit(limit + 1)
    rows = q.all()

    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        last = page[-1]
        next_cursor = _encode_list_cursor(last.created_at, last.id, fingerprint)
    return {
        "assessments": [assessment_summary(a) for a in page],
        "next_cursor": next_cursor,
    }


def list_samples(
    db: Session, assessment_id: str, limit: int, cursor: str | None
) -> dict:
    assessment = db.get(LocationUncertaintyAssessment, assessment_id)
    if assessment is None:
        raise ApiError(
            404,
            "location_uncertainty_not_found",
            f"location uncertainty assessment {assessment_id} not found",
        )
    fingerprint = hashlib.sha256(assessment.id.encode()).hexdigest()[:16]
    cursor_index = None
    if cursor is not None:
        cursor_index = _decode_sample_cursor(cursor, fingerprint)

    q = db.query(LocationUncertaintySample).filter_by(assessment_id=assessment.id)
    if cursor_index is not None:
        q = q.filter(LocationUncertaintySample.sample_index > cursor_index)
    q = q.order_by(LocationUncertaintySample.sample_index).limit(limit + 1)
    rows = q.all()

    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        next_cursor = _encode_sample_cursor(page[-1].sample_index, fingerprint)
    return {
        "assessment_id": assessment.id,
        "event_id": assessment.event_id,
        "resample_count": assessment.resample_count,
        "solvable_count": assessment.solvable_count,
        "sample_count": assessment.resample_count,
        "samples": [sample_payload(r) for r in page],
        "next_cursor": next_cursor,
    }


# --------------------------------------------------------------------------- #
# cursors
# --------------------------------------------------------------------------- #
def _encode_list_cursor(created_at: datetime, assessment_id: str, fingerprint: str) -> str:
    raw = json.dumps(
        {"t": created_at.strftime("%Y-%m-%dT%H:%M:%S.%f"), "id": assessment_id,
         "fp": fingerprint}
    ).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_list_cursor(cursor: str, fingerprint: str) -> tuple[datetime, str]:
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
            422, "invalid_cursor",
            "cursor was issued for a different assessment query (event/run "
            "filter mismatch)",
        )
    return token_at, token_id


def _encode_sample_cursor(sample_index: int, fingerprint: str) -> str:
    raw = json.dumps({"i": sample_index, "fp": fingerprint}).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_sample_cursor(cursor: str, fingerprint: str) -> int:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
        token_i = int(payload["i"])
        token_fp = payload["fp"]
    except Exception as exc:
        raise ApiError(422, "invalid_cursor", "malformed pagination cursor") from exc
    if token_fp != fingerprint:
        raise ApiError(
            422, "invalid_cursor",
            "cursor was issued for a different location uncertainty assessment",
        )
    return token_i


# --------------------------------------------------------------------------- #
# serialization
# --------------------------------------------------------------------------- #
def sample_payload(row: LocationUncertaintySample) -> dict:
    d = row.detail
    return {
        "sample_index": row.sample_index,
        "solvable": bool(row.solvable),
        "failure_reason": row.failure_reason,
        "stations": d.get("stations", []),
        "draw_count": d.get("draw_count"),
        "distinct_station_count": d.get("distinct_station_count"),
        "solution": _solution_json(d.get("solution")),
    }


def _baseline_payload(baseline: dict) -> dict:
    return {
        "event_id": baseline["event_id"],
        "run_id": baseline["run_id"],
        "origin_time": baseline["origin_time"],
        "latitude": round(baseline["latitude"], 5),
        "longitude": round(baseline["longitude"], 5),
        "depth_km": round(baseline["depth_km"], 2),
        "rms_residual_s": round(baseline["rms_s"], 4),
        "station_count": baseline["station_count"],
        "quality": baseline["quality"],
        "stations": list(baseline["stations"]),
        "picks": baseline.get("picks", []),
    }


def _result_payload(result: dict) -> dict:
    intervals = {
        key: dict(value) for key, value in result["intervals"].items()
    }
    for key in ("latitude", "longitude", "depth_km"):
        interval = intervals[key]
        interval["lower"] = round(interval["lower"], 5 if key != "depth_km" else 2)
        interval["upper"] = round(interval["upper"], 5 if key != "depth_km" else 2)
    return {
        "confidence_level": result["confidence_level"],
        "quantile_method": result["quantile_method"],
        "intervals": intervals,
        "covariance": result["covariance"],
    }


def assessment_summary(a: LocationUncertaintyAssessment) -> dict:
    return {
        "assessment_id": a.id,
        "event_id": a.event_id,
        "run_id": a.run_id,
        "params_hash": a.params_hash,
        "request": a.request,
        "resample_count": a.resample_count,
        "confidence_level": a.confidence_level,
        "random_seed": a.random_seed,
        "station_count": a.station_count,
        "solvable_count": a.solvable_count,
        "solvable_rate": round(a.solvable_rate, 6),
        "failure_counts": a.failure_counts,
        "created_at": iso_z(a.created_at),
    }


def assessment_payload(db: Session, a: LocationUncertaintyAssessment, created: bool) -> dict:
    sample_rows = (
        db.query(LocationUncertaintySample)
        .filter_by(assessment_id=a.id)
        .order_by(LocationUncertaintySample.sample_index)
        .limit(DEFAULT_PAGE_LIMIT)
        .all()
    )
    return {
        "assessment_id": a.id,
        "created": created,
        "event_id": a.event_id,
        "run_id": a.run_id,
        "ingestion_id": a.frozen["ingestion_id"],
        "params_hash": a.params_hash,
        "request": a.request,
        "frozen": a.frozen,
        "baseline": _baseline_payload(a.baseline),
        "resample_count": a.resample_count,
        "confidence_level": a.confidence_level,
        "random_seed": a.random_seed,
        "station_count": a.station_count,
        "solvable_count": a.solvable_count,
        "solvable_rate": round(a.solvable_rate, 6),
        "failure_counts": a.failure_counts,
        "result": _result_payload(a.result),
        "sample_count": a.resample_count,
        "samples": [sample_payload(r) for r in sample_rows],
        "next_cursor": (
            _encode_sample_cursor(
                sample_rows[-1].sample_index,
                hashlib.sha256(a.id.encode()).hexdigest()[:16],
            )
            if a.resample_count > len(sample_rows) and sample_rows
            else None
        ),
        "created_at": iso_z(a.created_at),
    }

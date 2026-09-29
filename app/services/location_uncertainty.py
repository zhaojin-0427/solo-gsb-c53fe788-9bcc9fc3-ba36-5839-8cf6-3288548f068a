"""Event location uncertainty assessment (station bootstrap).

For one event of a completed run the service freezes the original associated
P/S pick subset (with the weights used by the original location), the velocity
model, the calibration version (with its terms) and the processing parameters,
then re-locates the event ``n_resamples`` times with the very same weighted
least-squares locator used by ``process``/``replay``.  Each resample draws --
with replacement, seeded by the client-supplied ``seed`` -- as many stations
as the event originally had, keeps ALL of a drawn station's frozen associated
picks and weights repeated stations by their multiplicity.  Every resample
records its solvable status and, when it fails, a structured failure reason
(``insufficient_stations`` / ``location_failed`` / ``not_converged``).

From the solvable resamples the report derives, per origin time / latitude /
longitude / depth, the two-sided nearest-rank ``(1 - alpha) / 2`` confidence
interval, plus the sample covariance (solvable resamples only) and the
solvable rate.  Assessments whose solvable resamples fall below 80% of the
requested resamples are explicitly rejected and nothing is persisted.

Reports are immutable and idempotent (canonical hash of event, run, resample
count, confidence level and seed); identical concurrent requests collapse onto
one persisted report.  Both the report list and the per-assessment samples
support keyset pagination.  Like the cross-run catalog, the station audit and
the focal mechanism inversion this is a read-only analysis layer: it never
writes runs/events/picks and never touches catalog content.
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
    CalibrationTerm,
    Event,
    EventPick,
    LocationUncertaintyAssessment,
    LocationUncertaintySample,
    Pick,
    Run,
    StationCalibration,
    VelocityModel,
    Waveform,
)
from ..schemas import LocationUncertaintyCreate
from .association import (
    REASON_INSUFFICIENT_STATIONS,
    REASON_LOCATION_FAILED,
    correction_offset,
)
from .location import Observation, locate_event
from .pipeline import (
    ApiError,
    canonical_layers,
    from_epoch,
    iso_z,
    to_epoch,
)
from .traveltime import LayeredModel

STATUS_SOLVED = "solved"
STATUS_FAILED = "failed"

# A resample whose locator iterations did not converge is failed with this
# structured reason (the other two reasons are shared with the association
# engine's vocabulary).
REASON_NOT_CONVERGED = "not_converged"

# Assessments with fewer solvable resamples than this fraction of the
# requested resamples are explicitly rejected (422) and nothing is persisted.
MIN_SOLVABLE_FRACTION = 0.8

# Covariance parameter order (origin time in epoch seconds).
COV_PARAMETERS = ["origin_time_s", "latitude", "longitude", "depth_km"]

DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 200


# --------------------------------------------------------------------------- #
# assessment creation (freeze -> bootstrap -> persist, idempotent)
# --------------------------------------------------------------------------- #
def create_assessment(
    db: Session, req: LocationUncertaintyCreate
) -> tuple[dict, bool]:
    event = db.get(Event, req.event_id)
    if event is None:
        raise ApiError(404, "event_not_found", f"event {req.event_id} not found")
    run = db.get(Run, event.run_id)
    if run is None:  # pragma: no cover - FK integrity makes this unreachable
        raise ApiError(404, "run_not_found", f"run {event.run_id} not found")

    canon_request = {
        "event_id": event.id,
        "run_id": run.id,
        "n_resamples": int(req.n_resamples),
        "alpha": float(req.alpha),
        "seed": int(req.seed),
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
        return assessment_payload(existing, created=False), False

    frozen, baseline = _freeze_context(db, event, run)
    samples = _run_bootstrap(frozen, req.n_resamples, req.seed)

    solved = [s for s in samples if s["status"] == STATUS_SOLVED]
    n_solved = len(solved)
    min_required = math.ceil(MIN_SOLVABLE_FRACTION * req.n_resamples)
    if n_solved < min_required:
        reasons = Counter(
            s["failure_reason"] for s in samples if s["status"] == STATUS_FAILED
        )
        raise ApiError(
            422,
            "insufficient_solvable_samples",
            f"only {n_solved} of {req.n_resamples} bootstrap resamples are "
            f"solvable ({n_solved / req.n_resamples:.1%}); at least "
            f"{min_required} ({MIN_SOLVABLE_FRACTION:.0%}) are required "
            f"(failures: {dict(sorted(reasons.items())) or 'none'})",
        )

    intervals = _confidence_intervals(solved, req.alpha)
    covariance = _sample_covariance(solved)
    failure_reasons = Counter(
        s["failure_reason"] for s in samples if s["status"] == STATUS_FAILED
    )

    assessment = LocationUncertaintyAssessment(
        request_hash=request_hash,
        request=canon_request,
        event_id=event.id,
        run_id=run.id,
        n_resamples=req.n_resamples,
        alpha=req.alpha,
        seed=req.seed,
        frozen=_frozen_json(frozen),
        baseline=_baseline_json(baseline),
        n_solved=n_solved,
        n_failed=req.n_resamples - n_solved,
        solvable_rate=n_solved / req.n_resamples,
        failure_reasons=dict(sorted(failure_reasons.items())),
        intervals=intervals,
        covariance=covariance,
    )
    db.add(assessment)
    try:
        db.flush()
    except IntegrityError:
        # Concurrent identical request: collapse onto the surviving assessment.
        db.rollback()
        winner = (
            db.query(LocationUncertaintyAssessment)
            .filter_by(request_hash=request_hash)
            .one()
        )
        return assessment_payload(winner, created=False), False

    for s in samples:
        db.add(
            LocationUncertaintySample(
                assessment_id=assessment.id,
                sample_index=s["sample_index"],
                status=s["status"],
                failure_reason=s["failure_reason"],
                stations=s["stations"],
                solution=s["solution"],
            )
        )
    db.commit()
    db.refresh(assessment)
    return assessment_payload(assessment, created=True), True


def _freeze_context(db: Session, event: Event, run: Run) -> tuple[dict, dict]:
    """Snapshot everything the bootstrap resamples must depend on."""
    model_row = db.get(VelocityModel, run.model_id)
    layers = canonical_layers(model_row.layers)
    model = LayeredModel.from_dicts(layers)

    calibration = None
    corrections: dict[tuple[str, str], float] | None = None
    if run.calibration_id is not None:
        cal = db.get(StationCalibration, run.calibration_id)
        terms = db.query(CalibrationTerm).filter_by(calibration_id=cal.id).all()
        corrections = {(t.station, t.phase): t.offset_s for t in terms}
        calibration = {
            "calibration_id": cal.id,
            "version": cal.calibration_version,
            "status": cal.status,
            "terms": [
                {"station": t.station, "phase": t.phase, "offset_s": t.offset_s}
                for t in sorted(terms, key=lambda t: (t.phase, t.station))
            ],
        }

    waveforms = db.query(Waveform).filter_by(ingestion_id=run.ingestion_id).all()
    stations = {
        w.station: (w.latitude, w.longitude, w.elevation_m) for w in waveforms
    }

    rows = (
        db.query(EventPick, Pick)
        .join(Pick, EventPick.pick_id == Pick.id)
        .filter(EventPick.event_id == event.id)
        .all()
    )
    picks = [
        {
            "pick_id": ep.pick_id,
            "station": p.station,
            "phase": ep.phase,
            "time_utc": p.time_utc,
            "t": to_epoch(p.time_utc),
            "source": p.source,
            "weight": ep.weight,
            "residual_s": ep.residual_s,
        }
        for ep, p in rows
    ]
    picks.sort(key=lambda q: (q["t"], q["station"], q["phase"], q["pick_id"]))

    frozen = {
        "run_id": run.id,
        "ingestion_id": run.ingestion_id,
        "model": model,
        "model_snapshot": {"name": model_row.name, "version": model_row.version,
                           "layers": layers},
        "calibration_snapshot": calibration,
        "corrections": corrections,
        "params": {
            "max_residual_s": float(run.params["max_residual_s"]),
            "min_stations": int(run.params["min_stations"]),
        },
        "stations": stations,
        "station_codes": sorted({q["station"] for q in picks}),
        "picks": picks,
    }
    baseline = {
        "event_id": event.id,
        "run_id": run.id,
        "t0": to_epoch(event.origin_time),
        "origin_time": event.origin_time,
        "latitude": event.latitude,
        "longitude": event.longitude,
        "depth_km": event.depth_km,
        "rms_s": event.rms_s,
        "station_count": event.station_count,
        "quality": event.quality,
        "stations": sorted({q["station"] for q in picks}),
    }
    return frozen, baseline


# --------------------------------------------------------------------------- #
# bootstrap resampling + relocation
# --------------------------------------------------------------------------- #
def _run_bootstrap(frozen: dict, n_resamples: int, seed: int) -> list[dict]:
    """Re-locate the frozen event on ``n_resamples`` seeded station resamples.

    Each resample draws ``len(station_codes)`` stations with replacement from
    the frozen participating stations; a station drawn ``k`` times contributes
    all of its frozen associated picks with weights multiplied by ``k``.
    """
    model = frozen["model"]
    stations = frozen["stations"]
    corrections = frozen["corrections"]
    min_stations = frozen["params"]["min_stations"]
    codes = frozen["station_codes"]

    picks_by_station: dict[str, list[dict]] = {c: [] for c in codes}
    for q in frozen["picks"]:
        picks_by_station[q["station"]].append(q)

    rng = np.random.default_rng(seed)
    n_stations = len(codes)
    samples = []
    for i in range(n_resamples):
        draw = rng.integers(0, n_stations, size=n_stations)
        counts = Counter(codes[int(j)] for j in draw)
        multiset = {c: counts[c] for c in sorted(counts)}

        if len(counts) < min_stations:
            samples.append(
                _failed_sample(i, multiset, REASON_INSUFFICIENT_STATIONS)
            )
            continue

        observations = [
            Observation(
                code,
                q["phase"],
                q["t"] - correction_offset(corrections, code, q["phase"]),
                q["weight"] * counts[code],
            )
            for code in sorted(counts)
            for q in picks_by_station[code]
        ]
        loc = locate_event(model, observations, stations)
        if loc is None:
            samples.append(_failed_sample(i, multiset, REASON_LOCATION_FAILED))
        elif not loc.converged:
            samples.append(_failed_sample(i, multiset, REASON_NOT_CONVERGED))
        else:
            samples.append(
                {
                    "sample_index": i,
                    "status": STATUS_SOLVED,
                    "failure_reason": None,
                    "stations": multiset,
                    "solution": {
                        "t0": loc.t0,
                        "origin_time": iso_z(from_epoch(loc.t0)),
                        "latitude": loc.lat,
                        "longitude": loc.lon,
                        "depth_km": loc.depth_km,
                        "rms_s": loc.rms_s,
                        "iterations": loc.iterations,
                    },
                }
            )
    return samples


def _failed_sample(sample_index: int, multiset: dict, reason: str) -> dict:
    return {
        "sample_index": sample_index,
        "status": STATUS_FAILED,
        "failure_reason": reason,
        "stations": multiset,
        "solution": None,
    }


# --------------------------------------------------------------------------- #
# statistics (nearest-rank intervals + solvable-only sample covariance)
# --------------------------------------------------------------------------- #
def _nearest_rank(values: list[float], p: float) -> float:
    """Nearest-rank quantile: the value at rank ``ceil(p * n)`` (1-based)."""
    xs = sorted(values)
    n = len(xs)
    rank = min(max(math.ceil(p * n), 1), n)
    return xs[rank - 1]


def _interval(values: list[float], alpha: float) -> tuple[float, float]:
    tail = (1.0 - alpha) / 2.0
    return _nearest_rank(values, tail), _nearest_rank(values, 1.0 - tail)


def _confidence_intervals(solved: list[dict], alpha: float) -> dict:
    t0s = [s["solution"]["t0"] for s in solved]
    lats = [s["solution"]["latitude"] for s in solved]
    lons = [s["solution"]["longitude"] for s in solved]
    depths = [s["solution"]["depth_km"] for s in solved]
    t_lo, t_hi = _interval(t0s, alpha)
    lat_lo, lat_hi = _interval(lats, alpha)
    lon_lo, lon_hi = _interval(lons, alpha)
    z_lo, z_hi = _interval(depths, alpha)
    return {
        "alpha": float(alpha),
        "method": "nearest_rank",
        "sample_count": len(solved),
        "origin_time": {
            "lower": iso_z(from_epoch(t_lo)),
            "upper": iso_z(from_epoch(t_hi)),
        },
        "latitude": {"lower": lat_lo, "upper": lat_hi},
        "longitude": {"lower": lon_lo, "upper": lon_hi},
        "depth_km": {"lower": z_lo, "upper": z_hi},
    }


def _sample_covariance(solved: list[dict]) -> dict:
    """Unbiased (ddof=1) sample covariance over the solvable resamples only."""
    arr = np.array(
        [
            [
                s["solution"]["t0"],
                s["solution"]["latitude"],
                s["solution"]["longitude"],
                s["solution"]["depth_km"],
            ]
            for s in solved
        ],
        dtype=np.float64,
    )
    matrix = np.cov(arr, rowvar=False, ddof=1)
    return {
        "parameters": list(COV_PARAMETERS),
        "sample_count": len(solved),
        "matrix": [[float(v) for v in row] for row in matrix],
    }


# --------------------------------------------------------------------------- #
# JSON snapshots (frozen context is stored verbatim with the report)
# --------------------------------------------------------------------------- #
def _frozen_json(frozen: dict) -> dict:
    return {
        "ingestion_id": frozen["ingestion_id"],
        "model": frozen["model_snapshot"],
        "calibration": frozen["calibration_snapshot"],
        "params": frozen["params"],
        "stations": [
            {"station": code, "latitude": lat, "longitude": lon, "elevation_m": elev}
            for code, (lat, lon, elev) in sorted(frozen["stations"].items())
        ],
        "picks": [
            {
                "pick_id": q["pick_id"],
                "station": q["station"],
                "phase": q["phase"],
                "time": iso_z(q["time_utc"]),
                "source": q["source"],
                "weight": round(q["weight"], 3),
                "residual_s": round(q["residual_s"], 4),
            }
            for q in frozen["picks"]
        ],
    }


def _baseline_json(baseline: dict) -> dict:
    return {
        "event_id": baseline["event_id"],
        "run_id": baseline["run_id"],
        "origin_time": iso_z(baseline["origin_time"]),
        "latitude": baseline["latitude"],
        "longitude": baseline["longitude"],
        "depth_km": baseline["depth_km"],
        "rms_s": baseline["rms_s"],
        "station_count": baseline["station_count"],
        "quality": baseline["quality"],
        "stations": baseline["stations"],
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
    return assessment_payload(row, created=False)


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
    db: Session,
    assessment_id: str,
    status: str,
    limit: int,
    cursor: str | None,
) -> dict:
    assessment = db.get(LocationUncertaintyAssessment, assessment_id)
    if assessment is None:
        raise ApiError(
            404,
            "location_uncertainty_not_found",
            f"location uncertainty assessment {assessment_id} not found",
        )
    fingerprint = hashlib.sha256(
        json.dumps({"a": assessment_id, "status": status}, sort_keys=True).encode()
    ).hexdigest()[:16]
    cursor_index = None
    if cursor is not None:
        cursor_index = _decode_sample_cursor(cursor, fingerprint)

    q = db.query(LocationUncertaintySample).filter_by(assessment_id=assessment.id)
    if status != "all":
        q = q.filter(LocationUncertaintySample.status == status)
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
        "run_id": assessment.run_id,
        "n_resamples": assessment.n_resamples,
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
            "cursor was issued for a different location uncertainty assessment "
            "or status filter",
        )
    return token_i


# --------------------------------------------------------------------------- #
# serialization
# --------------------------------------------------------------------------- #
def _interval_payload(interval: dict) -> dict:
    return {"lower": interval["lower"], "upper": interval["upper"]}


def _intervals_payload(intervals: dict) -> dict:
    return {
        "alpha": intervals["alpha"],
        "method": intervals["method"],
        "sample_count": intervals["sample_count"],
        "origin_time": _interval_payload(intervals["origin_time"]),
        "latitude": {
            "lower": round(intervals["latitude"]["lower"], 5),
            "upper": round(intervals["latitude"]["upper"], 5),
        },
        "longitude": {
            "lower": round(intervals["longitude"]["lower"], 5),
            "upper": round(intervals["longitude"]["upper"], 5),
        },
        "depth_km": {
            "lower": round(intervals["depth_km"]["lower"], 2),
            "upper": round(intervals["depth_km"]["upper"], 2),
        },
    }


def _covariance_payload(covariance: dict) -> dict:
    return {
        "parameters": list(covariance["parameters"]),
        "sample_count": covariance["sample_count"],
        "matrix": [
            [round(v, 12) for v in row] for row in covariance["matrix"]
        ],
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
    }


def _solution_payload(solution: dict | None) -> dict | None:
    if solution is None:
        return None
    return {
        "origin_time": solution["origin_time"],
        "latitude": round(solution["latitude"], 5),
        "longitude": round(solution["longitude"], 5),
        "depth_km": round(solution["depth_km"], 2),
        "rms_residual_s": round(solution["rms_s"], 4),
        "iterations": solution["iterations"],
    }


def sample_payload(row: LocationUncertaintySample) -> dict:
    return {
        "sample_index": row.sample_index,
        "status": row.status,
        "failure_reason": row.failure_reason,
        "stations": dict(sorted(row.stations.items())),
        "distinct_station_count": len(row.stations),
        "solution": _solution_payload(row.solution),
    }


def assessment_summary(assessment: LocationUncertaintyAssessment) -> dict:
    return {
        "assessment_id": assessment.id,
        "event_id": assessment.event_id,
        "run_id": assessment.run_id,
        "request": assessment.request,
        "n_resamples": assessment.n_resamples,
        "alpha": assessment.alpha,
        "seed": assessment.seed,
        "n_solved": assessment.n_solved,
        "n_failed": assessment.n_failed,
        "solvable_rate": round(assessment.solvable_rate, 4),
        "created_at": iso_z(assessment.created_at),
    }


def assessment_payload(assessment: LocationUncertaintyAssessment, created: bool) -> dict:
    return {
        "assessment_id": assessment.id,
        "created": created,
        "event_id": assessment.event_id,
        "run_id": assessment.run_id,
        "ingestion_id": assessment.frozen["ingestion_id"],
        "request": assessment.request,
        "frozen": assessment.frozen,
        "baseline": _baseline_payload(assessment.baseline),
        "n_resamples": assessment.n_resamples,
        "alpha": assessment.alpha,
        "seed": assessment.seed,
        "n_solved": assessment.n_solved,
        "n_failed": assessment.n_failed,
        "solvable_rate": round(assessment.solvable_rate, 4),
        "min_solvable_required": math.ceil(
            MIN_SOLVABLE_FRACTION * assessment.n_resamples
        ),
        "failure_reasons": dict(sorted(assessment.failure_reasons.items())),
        "confidence_intervals": _intervals_payload(assessment.intervals),
        "covariance": _covariance_payload(assessment.covariance),
        "created_at": iso_z(assessment.created_at),
    }

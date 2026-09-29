"""Ingestion / association / replay orchestration and persistence."""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timedelta, timezone

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import config
from ..models import (
    CalibrationTerm,
    Event,
    EventPick,
    Ingestion,
    Pick,
    PickStatus,
    Run,
    StationCalibration,
    VelocityModel,
    Waveform,
)
from ..schemas import (
    AssociationParams,
    CalibrationCreate,
    ManualPickIn,
    ProcessRequest,
    ReplayRequest,
)
from .association import PickIn, run_association
from . import huber as huber_svc
from .huber import PhaseObservation, fit_phase, holdout_rms
from .picking import generate_picks
from .traveltime import LayeredModel
from .waveform import (
    WaveformError,
    check_trace_consistency,
    decode_miniseed,
    sha256_hex,
    validate_waveform_meta,
)

EPOCH = datetime(1970, 1, 1)
COMPARE_MATCH_TOLERANCE_S = 2.0

CALIBRATION_CANDIDATE = "candidate"
CALIBRATION_ACTIVE = "active"
CALIBRATION_PUBLISHED = "published"
_QUALITY_RANK = {"A": 0, "B": 1, "C": 2, "D": 3}

# Per-model activation locks.  PostgreSQL additionally serializes via the
# FOR UPDATE row lock below; SQLite serializes writers but cannot abort a
# stale read transaction, so this in-process lock closes that gap.
_ACTIVATION_LOCKS: dict[int, threading.Lock] = {}
_ACTIVATION_LOCKS_GUARD = threading.Lock()


def _activation_lock(model_id: int) -> threading.Lock:
    with _ACTIVATION_LOCKS_GUARD:
        lock = _ACTIVATION_LOCKS.get(model_id)
        if lock is None:
            lock = threading.Lock()
            _ACTIVATION_LOCKS[model_id] = lock
        return lock


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


# --------------------------------------------------------------------------- #
# time helpers (all database timestamps are naive UTC)
# --------------------------------------------------------------------------- #
def to_naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def to_epoch(dt: datetime) -> float:
    return (dt - EPOCH).total_seconds()


def from_epoch(t: float) -> datetime:
    return EPOCH + timedelta(seconds=t)


def iso_z(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# --------------------------------------------------------------------------- #
# velocity models
# --------------------------------------------------------------------------- #
def canonical_layers(layers) -> list[dict]:
    return [
        {
            "thickness_km": None if l["thickness_km"] is None else float(l["thickness_km"]),
            "vp_kms": float(l["vp_kms"]),
            "vs_kms": float(l["vs_kms"]),
        }
        for l in layers
    ]


def create_model(db: Session, name: str, version: str, layers: list[dict]) -> tuple[VelocityModel, bool]:
    """Create an immutable model version. Returns (model, created)."""
    canon = canonical_layers(layers)
    try:
        LayeredModel.from_dicts(canon)  # structural validation
    except ValueError as exc:
        raise ApiError(422, "invalid_model", str(exc))
    existing = (
        db.query(VelocityModel).filter_by(name=name, version=version).one_or_none()
    )
    if existing is not None:
        if canonical_layers(existing.layers) == canon:
            return existing, False
        raise ApiError(
            409,
            "model_version_conflict",
            f"model {name}@{version} already exists with different layers; "
            "model versions are immutable, choose a new version",
        )
    model = VelocityModel(name=name, version=version, layers=canon)
    db.add(model)
    db.commit()
    db.refresh(model)
    return model, True


def get_model(db: Session, name: str, version: str) -> VelocityModel:
    model = db.query(VelocityModel).filter_by(name=name, version=version).one_or_none()
    if model is None:
        raise ApiError(404, "model_not_found", f"velocity model {name}@{version} not found")
    return model


def seed_default_model(db: Session) -> None:
    if (
        db.query(VelocityModel)
        .filter_by(name=config.DEFAULT_MODEL_NAME, version=config.DEFAULT_MODEL_VERSION)
        .one_or_none()
        is None
    ):
        db.add(
            VelocityModel(
                name=config.DEFAULT_MODEL_NAME,
                version=config.DEFAULT_MODEL_VERSION,
                layers=canonical_layers(config.DEFAULT_MODEL_LAYERS),
            )
        )
        db.commit()


# --------------------------------------------------------------------------- #
# hashing / idempotency
# --------------------------------------------------------------------------- #
def _params_hash(params: AssociationParams, calibration_id: str | None = None) -> str:
    blob = json.dumps(
        {
            "max_residual_s": params.max_residual_s,
            "min_stations": params.min_stations,
            "calibration_id": calibration_id,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _window_hash(decoded: list[dict], manual_picks: list[ManualPickIn]) -> str:
    payload = {
        "waveforms": sorted(
            (
                {
                    "station": d["station"],
                    "sampling_rate": d["sampling_rate"],
                    "start_time": iso_z(d["start_time"]),
                    "npts": d["npts"],
                    "data_sha256": d["data_sha256"],
                }
                for d in decoded
            ),
            key=lambda d: d["station"],
        ),
        "manual_picks": sorted(
            (
                {
                    "station": p.station,
                    "phase": p.phase,
                    "time": iso_z(to_naive_utc(p.time)),
                    "weight": p.weight,
                }
                for p in manual_picks
            ),
            key=lambda d: (d["station"], d["phase"], d["time"]),
        ),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# main processing entry points
# --------------------------------------------------------------------------- #
def process(db: Session, req: ProcessRequest) -> dict:
    """Validate, ingest (idempotently) and associate one waveform window."""
    _validate_request(req)
    model_row = get_model(db, req.model.name, req.model.version)
    model = LayeredModel.from_dicts(model_row.layers)
    calibration = _resolve_calibration(db, req.calibration, model_row)
    corrections = _correction_map(db, calibration)

    decoded = _decode_all(req)
    ingestion = (
        db.query(Ingestion).filter_by(window_hash=_window_hash(decoded, req.manual_picks)).one_or_none()
    )
    if ingestion is None:
        ingestion = _store_ingestion(db, req, decoded)

    params_hash = _params_hash(req.params, calibration.id if calibration else None)
    run = (
        db.query(Run)
        .filter_by(
            ingestion_id=ingestion.id,
            model_id=model_row.id,
            params_hash=params_hash,
        )
        .one_or_none()
    )
    created = run is None
    if created:
        run = Run(
            ingestion_id=ingestion.id,
            model_id=model_row.id,
            calibration_id=calibration.id if calibration else None,
            params=req.params.model_dump(),
            params_hash=params_hash,
        )
        db.add(run)
        db.flush()
        _execute_run(db, run, ingestion, model, req.params, corrections)
    db.commit()
    return run_payload(db, run, created=created)


def replay(db: Session, req: ReplayRequest) -> dict:
    """Re-run an existing ingestion against a (possibly different) model version."""
    ingestion = db.get(Ingestion, req.ingestion_id)
    if ingestion is None:
        raise ApiError(404, "ingestion_not_found", f"ingestion {req.ingestion_id} not found")
    model_row = get_model(db, req.model.name, req.model.version)
    model = LayeredModel.from_dicts(model_row.layers)
    calibration = _resolve_calibration(db, req.calibration, model_row)
    corrections = _correction_map(db, calibration)

    params_hash = _params_hash(req.params, calibration.id if calibration else None)
    run = (
        db.query(Run)
        .filter_by(
            ingestion_id=ingestion.id,
            model_id=model_row.id,
            params_hash=params_hash,
        )
        .one_or_none()
    )
    created = run is None
    if created:
        run = Run(
            ingestion_id=ingestion.id,
            model_id=model_row.id,
            calibration_id=calibration.id if calibration else None,
            params=req.params.model_dump(),
            params_hash=params_hash,
        )
        db.add(run)
        db.flush()
        _execute_run(db, run, ingestion, model, req.params, corrections)
    db.commit()
    return run_payload(db, run, created=created)


def _resolve_calibration(
    db: Session, ref, model_row: VelocityModel
) -> StationCalibration | None:
    """Look up an optional calibration reference for a run request.

    Only ``active``/``published`` versions may be used, and they must belong to
    the same velocity model as the run.
    """
    if ref is None:
        return None
    row = (
        db.query(StationCalibration)
        .filter_by(model_id=model_row.id, calibration_version=ref.version)
        .one_or_none()
    )
    if row is None:
        raise ApiError(
            404,
            "calibration_not_found",
            f"calibration {ref.version} not found for model "
            f"{model_row.name}@{model_row.version}",
        )
    if row.status == CALIBRATION_CANDIDATE:
        raise ApiError(
            409,
            "calibration_not_active",
            f"calibration {ref.version} is still a candidate; activate it before use",
        )
    return row


def _correction_map(
    db: Session | None, calibration: StationCalibration | None
) -> dict[tuple[str, str], float] | None:
    if calibration is None:
        return None
    rows = db.query(CalibrationTerm).filter_by(calibration_id=calibration.id).all()
    terms = {(t.station, t.phase): t.offset_s for t in rows}
    return terms or None


# --------------------------------------------------------------------------- #
# station arrival-time calibration
# --------------------------------------------------------------------------- #
def _calibration_request_hash(
    model_name: str, model_version: str, req: CalibrationCreate
) -> str:
    payload = {
        "model": {"name": model_name, "version": model_version},
        "version": req.calibration_version,
        "train_start": iso_z(to_naive_utc(req.train_start)),
        "train_end": iso_z(to_naive_utc(req.train_end)),
        "holdout_start": iso_z(to_naive_utc(req.holdout_start)),
        "holdout_end": iso_z(to_naive_utc(req.holdout_end)),
        "min_quality": req.min_quality,
        "min_events": req.min_events,
        "min_span_days": req.min_span_days,
        "min_station_coverage": req.min_station_coverage,
        "huber_delta_s": req.huber_delta_s,
        "label": req.label,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def _baseline_events(
    db: Session, model_row: VelocityModel, start: datetime, end: datetime
) -> list[Event]:
    """Accepted uncalibrated events of one model located in [start, end)."""
    return (
        db.query(Event)
        .join(Run, Event.run_id == Run.id)
        .filter(
            Run.model_id == model_row.id,
            Run.calibration_id.is_(None),
            Event.origin_time >= start,
            Event.origin_time < end,
        )
        .order_by(Event.origin_time, Event.id)
        .all()
    )


def _event_observations(
    db: Session, events: list[Event], min_quality: str
) -> tuple[list[Event], dict[str, list[tuple[str, str, float, float]]], set[str]]:
    """Filter events by quality; gather their (station, phase, residual, weight).

    Returns (kept_events, residuals_by_event_id, station_universe).
    """
    worst = _QUALITY_RANK[min_quality]
    kept: list[Event] = []
    residuals: dict[str, list[tuple[str, str, float, float]]] = {}
    universe: set[str] = set()
    if not events:
        return kept, residuals, universe
    event_ids = [e.id for e in events]
    rows = (
        db.query(EventPick, Pick)
        .join(Pick, EventPick.pick_id == Pick.id)
        .filter(EventPick.event_id.in_(event_ids))
        .all()
    )
    by_event: dict[str, list[tuple[str, str, float, float]]] = {}
    for ep, p in rows:
        universe.add(p.station)
        by_event.setdefault(ep.event_id, []).append(
            (p.station, ep.phase, ep.residual_s, ep.weight)
        )
    for e in events:
        if _QUALITY_RANK[e.quality] > worst:
            continue
        ev_rows = by_event.get(e.id, [])
        if not ev_rows:
            continue
        kept.append(e)
        residuals[e.id] = sorted(ev_rows, key=lambda r: (r[0], r[1]))
    return kept, residuals, universe


def create_calibration(
    db: Session, model_name: str, model_version: str, req: CalibrationCreate
) -> tuple[dict, bool]:
    """Build a candidate calibration from historical residuals (idempotent).

    Explicitly rejects (422) when the request windows are malformed or any
    acceptance gate fails; nothing is persisted on rejection.
    """
    model_row = get_model(db, model_name, model_version)
    train_start, train_end = to_naive_utc(req.train_start), to_naive_utc(req.train_end)
    holdout_start, holdout_end = (
        to_naive_utc(req.holdout_start),
        to_naive_utc(req.holdout_end),
    )
    if train_start >= train_end:
        raise ApiError(422, "invalid_time_window", "train_start must precede train_end")
    if holdout_start >= holdout_end:
        raise ApiError(422, "invalid_time_window", "holdout_start must precede holdout_end")
    if train_end > holdout_start:
        raise ApiError(
            422,
            "invalid_time_window",
            "training window must end before the holdout window starts",
        )

    # Idempotency by exact request content and by (model, version) label.
    request_hash = _calibration_request_hash(model_name, model_version, req)
    existing = (
        db.query(StationCalibration)
        .filter(
            (StationCalibration.request_hash == request_hash)
            | (
                (StationCalibration.model_id == model_row.id)
                & (StationCalibration.calibration_version == req.calibration_version)
            )
        )
        .one_or_none()
    )
    if existing is not None:
        if (
            existing.calibration_version == req.calibration_version
            and existing.request_hash != request_hash
        ):
            raise ApiError(
                409,
                "calibration_version_conflict",
                f"calibration {req.calibration_version} already exists for model "
                f"{model_row.name}@{model_row.version} with different parameters",
            )
        db.commit()
        return calibration_payload(db, existing, created=False), False

    train_events_all = _baseline_events(db, model_row, train_start, train_end)
    holdout_events_all = _baseline_events(db, model_row, holdout_start, holdout_end)
    train_events, train_res, universe = _event_observations(
        db, train_events_all, req.min_quality
    )
    holdout_events, holdout_res, holdout_universe = _event_observations(
        db, holdout_events_all, req.min_quality
    )
    universe |= holdout_universe

    # ---- acceptance gates: explicit rejection, no partial state ---------- #
    if len(train_events) < req.min_events:
        raise ApiError(
            422,
            "insufficient_events",
            f"training window has {len(train_events)} qualifying events, "
            f"minimum is {req.min_events}",
        )
    if len(holdout_events) < 1:
        raise ApiError(
            422,
            "insufficient_holdout_events",
            "holdout window has no qualifying events to validate against",
        )
    span_days = (
        train_events[-1].origin_time - train_events[0].origin_time
    ).total_seconds() / 86400.0
    if span_days < req.min_span_days:
        raise ApiError(
            422,
            "time_span_too_short",
            f"training events span {span_days:.3f} days, minimum is {req.min_span_days}",
        )

    phase_fits: dict[str, huber_svc.PhaseFitResult] = {}
    for phase in ("P", "S"):
        obs = []
        for idx, ev in enumerate(train_events):
            for station, ph, residual, weight in train_res[ev.id]:
                if ph == phase:
                    obs.append(PhaseObservation(idx, station, residual, weight))
        fit = fit_phase(
            phase,
            obs,
            station_universe=sorted(universe),
            huber_delta_s=req.huber_delta_s,
        )
        phase_fits[phase] = fit

    coverage_phase = min(
        (f.n_stations / len(universe) if universe else 0.0) for f in phase_fits.values()
    )
    if coverage_phase < req.min_station_coverage:
        raise ApiError(
            422,
            "station_coverage_too_low",
            f"per-phase station coverage {coverage_phase:.3f} is below "
            f"minimum {req.min_station_coverage:.3f}",
        )

    # Holdout: strictly better RMS than the uncalibrated baseline on unseen
    # events, evaluated independently for P and S.
    phase_metrics = {}
    for phase, fit in phase_fits.items():
        terms = {t.station: t.offset_s for t in fit.terms}
        obs = []
        for idx, ev in enumerate(holdout_events):
            for station, ph, residual, weight in holdout_res[ev.id]:
                if ph == phase:
                    obs.append(PhaseObservation(idx, station, residual, weight))
        base_rms, cal_rms, n_eval = holdout_rms(obs, terms)
        phase_metrics[phase] = {
            "sample_count": fit.n_obs,
            "station_count": fit.n_stations,
            "coverage": fit.coverage,
            "train_rms_baseline_s": round(fit.rms_baseline_s, 6),
            "train_rms_calibrated_s": round(fit.rms_calibrated_s, 6),
            "holdout_sample_count": n_eval,
            "holdout_rms_baseline_s": round(base_rms, 6),
            "holdout_rms_calibrated_s": round(cal_rms, 6),
            "residual_scale_s": round(fit.residual_scale_s, 6),
            "converged": fit.converged,
            "iterations": fit.iterations,
        }

    # Persist candidate atomically; concurrent identical requests collapse.
    metrics = {
        "train_event_count": len(train_events),
        "holdout_event_count": len(holdout_events),
        "train_span_days": round(span_days, 6),
        "station_universe": sorted(universe),
        "min_quality": req.min_quality,
        "phases": phase_metrics,
        "label": req.label,
    }
    candidate = StationCalibration(
        model_id=model_row.id,
        calibration_version=req.calibration_version,
        status=CALIBRATION_CANDIDATE,
        request={
            "train_start": iso_z(train_start),
            "train_end": iso_z(train_end),
            "holdout_start": iso_z(holdout_start),
            "holdout_end": iso_z(holdout_end),
            "min_quality": req.min_quality,
            "min_events": req.min_events,
            "min_span_days": req.min_span_days,
            "min_station_coverage": req.min_station_coverage,
            "huber_delta_s": req.huber_delta_s,
        },
        request_hash=request_hash,
        metrics=metrics,
    )
    db.add(candidate)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        winner = (
            db.query(StationCalibration)
            .filter_by(request_hash=request_hash)
            .one_or_none()
        )
        if winner is None:
            winner = (
                db.query(StationCalibration)
                .filter_by(
                    model_id=model_row.id,
                    calibration_version=req.calibration_version,
                )
                .one()
            )
            if winner.request_hash != request_hash:
                raise ApiError(
                    409,
                    "calibration_version_conflict",
                    f"calibration {req.calibration_version} already exists for "
                    "this model with different parameters",
                )
        return calibration_payload(db, winner, created=False), False

    for phase, fit in phase_fits.items():
        for t in fit.terms:
            db.add(
                CalibrationTerm(
                    calibration_id=candidate.id,
                    station=t.station,
                    phase=phase,
                    offset_s=t.offset_s,
                    uncertainty_s=t.uncertainty_s,
                    sample_count=t.sample_count,
                )
            )
    db.commit()
    return calibration_payload(db, candidate, created=True), True


def calibration_payload(
    db: Session, calibration: StationCalibration, created: bool
) -> dict:
    model_row = db.get(VelocityModel, calibration.model_id)
    terms = (
        db.query(CalibrationTerm)
        .filter_by(calibration_id=calibration.id)
        .order_by(CalibrationTerm.phase, CalibrationTerm.station)
        .all()
    )
    stations = [
        {
            "station": t.station,
            "phase": t.phase,
            "sample_count": t.sample_count,
            "offset_s": round(t.offset_s, 6),
            "uncertainty_s": round(t.uncertainty_s, 6),
        }
        for t in terms
    ]
    # Per-station, per-phase view plus the per-phase summary metrics.
    return {
        "calibration_id": calibration.id,
        "model": {"name": model_row.name, "version": model_row.version},
        "calibration_version": calibration.calibration_version,
        "version": calibration.calibration_version,
        "status": calibration.status,
        "created": created,
        "request": calibration.request,
        "metrics": calibration.metrics,
        "stations": stations,
        "activated_at": iso_z(calibration.activated_at) if calibration.activated_at else None,
    }


def list_calibrations(db: Session, name: str, version: str) -> dict:
    model_row = get_model(db, name, version)
    rows = (
        db.query(StationCalibration)
        .filter_by(model_id=model_row.id)
        .order_by(StationCalibration.created_at, StationCalibration.calibration_version)
        .all()
    )
    return {
        "model": {"name": model_row.name, "version": model_row.version},
        "calibrations": [
            {
                "calibration_id": c.id,
                "version": c.calibration_version,
                "status": c.status,
                "metrics": c.metrics,
                "created_at": iso_z(c.created_at),
                "activated_at": iso_z(c.activated_at) if c.activated_at else None,
            }
            for c in rows
        ],
    }


def get_calibration(db: Session, name: str, version: str, cal_version: str) -> dict:
    model_row = get_model(db, name, version)
    row = (
        db.query(StationCalibration)
        .filter_by(model_id=model_row.id, calibration_version=cal_version)
        .one_or_none()
    )
    if row is None:
        raise ApiError(
            404,
            "calibration_not_found",
            f"calibration {cal_version} not found for model {name}@{version}",
        )
    return calibration_payload(db, row, created=False)


def activate_calibration(
    db: Session, name: str, version: str, cal_version: str
) -> dict:
    """Atomically promote a candidate to active (idempotent).

    A repeat activation of the same version is a no-op; activating any other
    version demotes the incumbent to ``published``.  Published versions are
    immutable: their terms and metrics never change.  The hold-out gate is
    re-verified from the stored metrics immediately before the switch.  An
    in-process per-model lock (plus a FOR UPDATE row lock on PostgreSQL and
    the partial unique index) makes concurrent activations atomic: exactly
    one wins, the loser gets 409, and no partial state survives.
    """
    model_row = get_model(db, name, version)
    with _activation_lock(model_row.id):
        # Row lock (Postgres); the in-process lock covers SQLite deployments.
        db.query(VelocityModel).filter_by(id=model_row.id).with_for_update().one_or_none()

        row = (
            db.query(StationCalibration)
            .filter_by(model_id=model_row.id, calibration_version=cal_version)
            .one_or_none()
        )
        if row is None:
            raise ApiError(
                404,
                "calibration_not_found",
                f"calibration {cal_version} not found for model {name}@{version}",
            )

        incumbent_id = db.get(VelocityModel, model_row.id).active_calibration_id
        incumbent = (
            db.get(StationCalibration, incumbent_id) if incumbent_id else None
        )

        if incumbent is not None and incumbent.id == row.id:
            db.commit()  # already active: idempotent no-op
            return calibration_payload(db, row, created=False)

        if row.status == CALIBRATION_PUBLISHED:
            db.commit()
            raise ApiError(
                409,
                "calibration_published_immutable",
                f"calibration {cal_version} is already published and cannot be "
                "re-activated; create a new calibration version",
            )

        # Gate re-check from the persisted candidate metrics; published
        # artifacts are never recomputed or mutated.
        for phase, m in row.metrics["phases"].items():
            if not (m["holdout_rms_calibrated_s"] < m["holdout_rms_baseline_s"]):
                db.commit()
                raise ApiError(
                    409,
                    "holdout_gate_failed",
                    f"{phase} holdout RMS {m['holdout_rms_calibrated_s']} is not "
                    f"strictly below baseline {m['holdout_rms_baseline_s']}",
                )

        if incumbent is not None:
            incumbent.status = CALIBRATION_PUBLISHED
            db.flush()  # demote before promoting: keep the partial index valid
        row.status = CALIBRATION_ACTIVE
        row.activated_at = datetime.utcnow()
        model_row.active_calibration_id = row.id
        try:
            db.flush()
            db.commit()
        except IntegrityError:
            # Defense in depth (partial unique index) across processes.
            db.rollback()
            raise ApiError(
                409,
                "calibration_activation_conflict",
                "another calibration was activated concurrently; retry against "
                "the currently active version",
            )
        return calibration_payload(db, row, created=False)


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _validate_request(req: ProcessRequest) -> None:
    seen: set[str] = set()
    for wf in req.waveforms:
        if wf.station in seen:
            raise ApiError(422, "duplicate_station", f"duplicate station {wf.station}")
        seen.add(wf.station)
        try:
            validate_waveform_meta(wf)
        except WaveformError as exc:
            raise ApiError(422, exc.code, exc.message)
    stations = {wf.station for wf in req.waveforms}
    for mp in req.manual_picks:
        if mp.station not in stations:
            raise ApiError(
                422,
                "unknown_pick_station",
                f"manual pick references station {mp.station} which has no waveform",
            )


def _decode_all(req: ProcessRequest) -> list[dict]:
    decoded = []
    for wf in req.waveforms:
        try:
            trace, raw = decode_miniseed(wf.data_b64)
            check_trace_consistency(trace, wf)
        except WaveformError as exc:
            raise ApiError(422, exc.code, exc.message)
        decoded.append(
            {
                "wf": wf,
                "trace": trace,
                "station": wf.station,
                "sampling_rate": float(trace.stats.sampling_rate),
                "start_time": trace.stats.starttime.datetime,
                "npts": int(trace.stats.npts),
                "data_sha256": sha256_hex(raw),
            }
        )
    return decoded


def _store_ingestion(db: Session, req: ProcessRequest, decoded: list[dict]) -> Ingestion:
    ingestion = Ingestion(
        window_hash=_window_hash(decoded, req.manual_picks),
        summary={
            "station_count": len(decoded),
            "stations": sorted(d["station"] for d in decoded),
            "manual_pick_count": len(req.manual_picks),
        },
    )
    db.add(ingestion)
    db.flush()
    for d in decoded:
        wf = d["wf"]
        db.add(
            Waveform(
                ingestion_id=ingestion.id,
                station=wf.station,
                sampling_rate=d["sampling_rate"],
                start_time=d["start_time"],
                npts=d["npts"],
                latitude=wf.latitude,
                longitude=wf.longitude,
                elevation_m=wf.elevation_m,
                data_sha256=d["data_sha256"],
            )
        )
        for pick in generate_picks(d["trace"]):
            db.add(
                Pick(
                    ingestion_id=ingestion.id,
                    station=wf.station,
                    phase=None,
                    time_utc=pick["time_utc"].datetime,
                    source="stalta",
                    weight=pick["weight"],
                    snr=pick["snr"],
                )
            )
    for mp in req.manual_picks:
        db.add(
            Pick(
                ingestion_id=ingestion.id,
                station=mp.station,
                phase=mp.phase,
                time_utc=to_naive_utc(mp.time),
                source="manual",
                weight=mp.weight,
                snr=None,
            )
        )
    db.flush()
    return ingestion


def _execute_run(
    db: Session,
    run: Run,
    ingestion: Ingestion,
    model: LayeredModel,
    params: AssociationParams,
    corrections: dict[tuple[str, str], float] | None = None,
) -> None:
    picks = db.query(Pick).filter_by(ingestion_id=ingestion.id).all()
    waveforms = db.query(Waveform).filter_by(ingestion_id=ingestion.id).all()
    stations = {w.station: (w.latitude, w.longitude, w.elevation_m) for w in waveforms}
    pick_ins = [
        PickIn(
            id=p.id,
            station=p.station,
            phase=p.phase,
            t=to_epoch(p.time_utc),
            weight=p.weight,
            source=p.source,
        )
        for p in picks
    ]
    events, reasons = run_association(
        pick_ins,
        stations,
        model,
        max_residual_s=params.max_residual_s,
        min_stations=params.min_stations,
        corrections=corrections,
    )
    weights = {p.id: p.weight for p in picks}
    for ev in events:
        event_row = Event(
            run_id=run.id,
            origin_time=from_epoch(ev.t0),
            latitude=ev.lat,
            longitude=ev.lon,
            depth_km=ev.depth_km,
            rms_s=ev.rms_s,
            station_count=ev.station_count,
            quality=ev.quality,
        )
        db.add(event_row)
        db.flush()
        for a in ev.assignments:
            db.add(
                EventPick(
                    event_id=event_row.id,
                    pick_id=a.pick_id,
                    phase=a.phase,
                    residual_s=a.residual,
                    weight=weights[a.pick_id],
                )
            )
            db.add(PickStatus(run_id=run.id, pick_id=a.pick_id, status="associated"))
    for pick_id, reason in reasons.items():
        db.add(PickStatus(run_id=run.id, pick_id=pick_id, status="unassociated", reason=reason))
    db.flush()


# --------------------------------------------------------------------------- #
# result serialization
# --------------------------------------------------------------------------- #
def _calibration_echo(db: Session, run: Run) -> dict | None:
    """Calibration version applied to a run (None = uncalibrated baseline)."""
    if run.calibration_id is None:
        return None
    cal = db.get(StationCalibration, run.calibration_id)
    if cal is None:
        return None
    return {"version": cal.calibration_version, "status": cal.status}


def run_payload(db: Session, run: Run, created: bool) -> dict:
    model_row = db.get(VelocityModel, run.model_id)
    events = (
        db.query(Event)
        .filter_by(run_id=run.id)
        .order_by(Event.origin_time, Event.id)
        .all()
    )
    event_ids = [e.id for e in events]
    event_picks = (
        db.query(EventPick).filter(EventPick.event_id.in_(event_ids)).all()
        if event_ids
        else []
    )
    pick_ids = [ep.pick_id for ep in event_picks]
    picks_by_id = (
        {p.id: p for p in db.query(Pick).filter(Pick.id.in_(pick_ids)).all()}
        if pick_ids
        else {}
    )
    eps_by_event: dict[str, list[EventPick]] = {}
    for ep in event_picks:
        eps_by_event.setdefault(ep.event_id, []).append(ep)

    events_out = []
    for e in events:
        rows = []
        for ep in eps_by_event.get(e.id, []):
            p = picks_by_id[ep.pick_id]
            rows.append(
                {
                    "pick_id": p.id,
                    "station": p.station,
                    "phase": ep.phase,
                    "time": iso_z(p.time_utc),
                    "source": p.source,
                    "residual_s": round(ep.residual_s, 4),
                    "weight": round(ep.weight, 3),
                }
            )
        rows.sort(key=lambda r: (r["time"], r["station"], r["phase"]))
        events_out.append(
            {
                "event_id": e.id,
                "origin_time": iso_z(e.origin_time),
                "latitude": round(e.latitude, 5),
                "longitude": round(e.longitude, 5),
                "depth_km": round(e.depth_km, 2),
                "rms_residual_s": round(e.rms_s, 4),
                "station_count": e.station_count,
                "quality": e.quality,
                "picks": rows,
            }
        )

    unassociated = (
        db.query(PickStatus, Pick)
        .join(Pick, PickStatus.pick_id == Pick.id)
        .filter(PickStatus.run_id == run.id, PickStatus.status == "unassociated")
        .all()
    )
    unassociated_out = sorted(
        (
            {
                "pick_id": p.id,
                "station": p.station,
                "phase": p.phase,
                "time": iso_z(p.time_utc),
                "source": p.source,
                "reason": st.reason,
            }
            for st, p in unassociated
        ),
        key=lambda r: (r["time"], r["station"]),
    )
    return {
        "run_id": run.id,
        "ingestion_id": run.ingestion_id,
        "created": created,
        "model": {"name": model_row.name, "version": model_row.version},
        "calibration": _calibration_echo(db, run),
        "params": run.params,
        "events": events_out,
        "unassociated_picks": unassociated_out,
    }


def event_payload(db: Session, event: Event) -> dict:
    event_picks = db.query(EventPick).filter_by(event_id=event.id).all()
    picks_by_id = {
        p.id: p
        for p in db.query(Pick).filter(Pick.id.in_([ep.pick_id for ep in event_picks])).all()
    }
    rows = [
        {
            "pick_id": ep.pick_id,
            "station": picks_by_id[ep.pick_id].station,
            "phase": ep.phase,
            "time": iso_z(picks_by_id[ep.pick_id].time_utc),
            "source": picks_by_id[ep.pick_id].source,
            "residual_s": round(ep.residual_s, 4),
            "weight": round(ep.weight, 3),
        }
        for ep in event_picks
    ]
    rows.sort(key=lambda r: (r["time"], r["station"], r["phase"]))
    run = db.get(Run, event.run_id)
    return {
        "event_id": event.id,
        "run_id": event.run_id,
        "calibration": _calibration_echo(db, run),
        "origin_time": iso_z(event.origin_time),
        "latitude": round(event.latitude, 5),
        "longitude": round(event.longitude, 5),
        "depth_km": round(event.depth_km, 2),
        "rms_residual_s": round(event.rms_s, 4),
        "station_count": event.station_count,
        "quality": event.quality,
        "picks": rows,
    }


def compare_runs(db: Session, run_a_id: str, run_b_id: str) -> dict:
    run_a = db.get(Run, run_a_id)
    run_b = db.get(Run, run_b_id)
    if run_a is None:
        raise ApiError(404, "run_not_found", f"run {run_a_id} not found")
    if run_b is None:
        raise ApiError(404, "run_not_found", f"run {run_b_id} not found")
    events_a = db.query(Event).filter_by(run_id=run_a.id).order_by(Event.origin_time).all()
    events_b = db.query(Event).filter_by(run_id=run_b.id).order_by(Event.origin_time).all()

    candidates = sorted(
        (
            abs((a.origin_time - b.origin_time).total_seconds()), a.id, b.id
        )
        for a in events_a
        for b in events_b
    )
    by_id_a = {e.id: e for e in events_a}
    by_id_b = {e.id: e for e in events_b}
    used_a: set[str] = set()
    used_b: set[str] = set()
    matches = []
    for delta, a_id, b_id in candidates:
        if delta > COMPARE_MATCH_TOLERANCE_S:
            break
        if a_id in used_a or b_id in used_b:
            continue
        used_a.add(a_id)
        used_b.add(b_id)
        a, b = by_id_a[a_id], by_id_b[b_id]
        matches.append(
            {
                "event_a": a_id,
                "event_b": b_id,
                "delta": {
                    "origin_time_s": round(
                        (b.origin_time - a.origin_time).total_seconds(), 4
                    ),
                    "latitude": round(b.latitude - a.latitude, 5),
                    "longitude": round(b.longitude - a.longitude, 5),
                    "depth_km": round(b.depth_km - a.depth_km, 2),
                    "rms_s": round(b.rms_s - a.rms_s, 4),
                    "station_count": b.station_count - a.station_count,
                },
                "quality_a": a.quality,
                "quality_b": b.quality,
            }
        )
    return {
        "run_a": run_a_id,
        "run_b": run_b_id,
        "model_a": db.get(VelocityModel, run_a.model_id).version,
        "model_b": db.get(VelocityModel, run_b.model_id).version,
        "matches": matches,
        "only_in_a": sorted(e.id for e in events_a if e.id not in used_a),
        "only_in_b": sorted(e.id for e in events_b if e.id not in used_b),
    }

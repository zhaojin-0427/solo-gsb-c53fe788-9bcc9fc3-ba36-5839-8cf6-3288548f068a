"""HTTP API surface."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Event, VelocityModel
from ..schemas import (
    AlertEvaluateRequest,
    AlertRuleCreate,
    AlertRuleLifecycleRequest,
    AlertTransitionRequest,
    CalibrationCreate,
    CatalogCompileRequest,
    CatalogPublishRequest,
    CatalogWithdrawRequest,
    FocalMechanismCreate,
    ModelCreate,
    ProcessRequest,
    ReplayRequest,
    StationAuditCreate,
)
from ..services import alerting, catalog_pipeline, focal_mechanism, pipeline, station_audit
from ..services.pipeline import ApiError

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/v1/models", status_code=201)
def create_model(body: ModelCreate, db: Session = Depends(get_db)):
    """Create an immutable velocity model version (idempotent)."""
    model, created = pipeline.create_model(
        db, body.name, body.version, [l.model_dump() for l in body.layers]
    )
    payload = {
        "name": model.name,
        "version": model.version,
        "layers": model.layers,
        "created": created,
    }
    return JSONResponse(payload, status_code=201 if created else 200)


@router.get("/v1/models")
def list_models(db: Session = Depends(get_db)):
    models = db.query(VelocityModel).order_by(VelocityModel.name, VelocityModel.version).all()
    return {
        "models": [
            {"name": m.name, "version": m.version, "layers": m.layers} for m in models
        ]
    }


@router.get("/v1/models/{name}/{version}")
def get_model(name: str, version: str, db: Session = Depends(get_db)):
    model = pipeline.get_model(db, name, version)
    return {"name": model.name, "version": model.version, "layers": model.layers}


# --------------------------------------------------------------------------- #
# station arrival-time calibrations
# --------------------------------------------------------------------------- #
@router.post("/v1/models/{name}/{version}/calibrations", status_code=201)
def create_calibration(name: str, version: str, body: CalibrationCreate, db: Session = Depends(get_db)):
    """Create (or idempotently return) a calibration candidate for a model."""
    payload, created = pipeline.create_calibration(db, name, version, body)
    return JSONResponse(payload, status_code=201 if created else 200)


@router.get("/v1/models/{name}/{version}/calibrations")
def list_calibrations(name: str, version: str, db: Session = Depends(get_db)):
    return pipeline.list_calibrations(db, name, version)


@router.get("/v1/models/{name}/{version}/calibrations/{cal_version}")
def get_calibration(name: str, version: str, cal_version: str, db: Session = Depends(get_db)):
    return pipeline.get_calibration(db, name, version, cal_version)


@router.post("/v1/models/{name}/{version}/calibrations/{cal_version}/activate")
def activate_calibration(name: str, version: str, cal_version: str, db: Session = Depends(get_db)):
    """Atomically promote a candidate to the model's single active calibration."""
    return pipeline.activate_calibration(db, name, version, cal_version)


@router.post("/v1/process")
def process(body: ProcessRequest, db: Session = Depends(get_db)):
    """Ingest a waveform window (idempotent) and run association + location."""
    return pipeline.process(db, body)


@router.post("/v1/runs", status_code=201)
def replay(body: ReplayRequest, db: Session = Depends(get_db)):
    """Replay a stored ingestion against a model version (idempotent)."""
    payload = pipeline.replay(db, body)
    return JSONResponse(payload, status_code=201 if payload["created"] else 200)


@router.get("/v1/runs/compare")
def compare_runs(
    run_a: str = Query(..., description="first run id"),
    run_b: str = Query(..., description="second run id"),
    db: Session = Depends(get_db),
):
    """Compare events of two runs (e.g. two model versions of one window)."""
    return pipeline.compare_runs(db, run_a, run_b)


@router.get("/v1/runs/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_db)):
    from ..models import Run

    run = db.get(Run, run_id)
    if run is None:
        raise ApiError(404, "run_not_found", f"run {run_id} not found")
    return pipeline.run_payload(db, run, created=False)


@router.get("/v1/events/{event_id}")
def get_event(event_id: str, db: Session = Depends(get_db)):
    event = db.get(Event, event_id)
    if event is None:
        raise ApiError(404, "event_not_found", f"event {event_id} not found")
    return pipeline.event_payload(db, event)


# --------------------------------------------------------------------------- #
# cross-run event catalog
# --------------------------------------------------------------------------- #
@router.post("/v1/catalog/compilations", status_code=201)
def compile_catalog(body: CatalogCompileRequest, db: Session = Depends(get_db)):
    """Deterministically cluster events of completed runs (idempotent)."""
    payload, created = catalog_pipeline.compile_catalog(db, body)
    return JSONResponse(payload, status_code=201 if created else 200)


@router.get("/v1/catalog/compilations")
def list_catalog_compilations(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    return catalog_pipeline.list_compilations(db, limit)


@router.get("/v1/catalog/compilations/{compilation_id}")
def get_catalog_compilation(compilation_id: str, db: Session = Depends(get_db)):
    return catalog_pipeline.get_compilation(db, compilation_id)


@router.post("/v1/catalog/entries/{entry_id}/publish")
def publish_catalog_entry(
    entry_id: str, body: CatalogPublishRequest, db: Session = Depends(get_db)
):
    """Publish a draft entry (optimistic concurrency via expected_revision)."""
    return catalog_pipeline.publish_entry(db, entry_id, body.expected_revision)


@router.post("/v1/catalog/entries/{entry_id}/withdraw")
def withdraw_catalog_entry(
    entry_id: str, body: CatalogWithdrawRequest, db: Session = Depends(get_db)
):
    """Withdraw a published entry; published content is never rewritten."""
    return catalog_pipeline.withdraw_entry(
        db, entry_id, body.expected_revision, body.reason
    )


@router.get("/v1/catalog/entries")
def list_catalog_entries(
    start_time: datetime | None = Query(default=None),
    end_time: datetime | None = Query(default=None),
    min_latitude: float | None = Query(default=None, ge=-90.0, le=90.0),
    max_latitude: float | None = Query(default=None, ge=-90.0, le=90.0),
    min_longitude: float | None = Query(default=None, ge=-180.0, le=180.0),
    max_longitude: float | None = Query(default=None, ge=-180.0, le=180.0),
    min_quality: str = Query(default="D", pattern="^[A-D]$"),
    status: str = Query(
        default="all", pattern="^(draft|published|withdrawn|all)$"
    ),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """List entries with stable keyset pagination (origin_time, entry_id)."""
    return catalog_pipeline.list_entries(
        db,
        start_time=start_time,
        end_time=end_time,
        min_latitude=min_latitude,
        max_latitude=max_latitude,
        min_longitude=min_longitude,
        max_longitude=max_longitude,
        min_quality=min_quality,
        status=status,
        limit=limit,
        cursor=cursor,
    )


@router.get("/v1/catalog/entries/{entry_id}")
def get_catalog_entry(entry_id: str, db: Session = Depends(get_db)):
    return catalog_pipeline.get_entry(db, entry_id)


# --------------------------------------------------------------------------- #
# station impact audit (leave-one-station-out)
# --------------------------------------------------------------------------- #
@router.post("/v1/station-audits", status_code=201)
def create_station_audit(body: StationAuditCreate, db: Session = Depends(get_db)):
    """Freeze a completed event and run leave-one-station-out audits (idempotent)."""
    payload, created = station_audit.create_audit(
        db, body.event_id, body.min_keep_stations
    )
    return JSONResponse(payload, status_code=201 if created else 200)


@router.get("/v1/station-audits")
def list_station_audits(
    event_id: str | None = Query(default=None, description="filter by event id"),
    run_id: str | None = Query(default=None, description="filter by run id"),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """List station audits with stable keyset pagination (created_at, audit_id)."""
    return station_audit.list_audits(db, event_id, run_id, limit, cursor)


@router.get("/v1/station-audits/{audit_id}")
def get_station_audit(audit_id: str, db: Session = Depends(get_db)):
    return station_audit.get_audit(db, audit_id)


@router.get("/v1/station-audits/{audit_id}/results")
def list_station_audit_results(
    audit_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Paginate the per-station audit results in their stable presentation order."""
    return station_audit.list_results(db, audit_id, limit, cursor)


# --------------------------------------------------------------------------- #
# P-wave first-motion focal mechanism inversion
# --------------------------------------------------------------------------- #
@router.post("/v1/focal-mechanisms", status_code=201)
def create_focal_mechanism(body: FocalMechanismCreate, db: Session = Depends(get_db)):
    """Invert P first-motion polarities of a completed event (idempotent)."""
    payload, created = focal_mechanism.create_inversion(db, body)
    return JSONResponse(payload, status_code=201 if created else 200)


@router.get("/v1/focal-mechanisms")
def list_focal_mechanisms(
    event_id: str | None = Query(default=None, description="filter by event id"),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """List inversions with stable keyset pagination (created_at, inversion_id)."""
    return focal_mechanism.list_inversions(db, event_id, limit, cursor)


@router.get("/v1/focal-mechanisms/{inversion_id}")
def get_focal_mechanism(inversion_id: str, db: Session = Depends(get_db)):
    return focal_mechanism.get_inversion(db, inversion_id)


@router.get("/v1/focal-mechanisms/{inversion_id}/solutions")
def list_focal_mechanism_solutions(
    inversion_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Paginate the tied best solutions in their stable (strike, dip, rake) order."""
    return focal_mechanism.list_solutions(db, inversion_id, limit, cursor)


# --------------------------------------------------------------------------- #
# catalog alert orchestration
# --------------------------------------------------------------------------- #
@router.post("/v1/alert-rules", status_code=201)
def create_alert_rule(body: AlertRuleCreate, db: Session = Depends(get_db)):
    """Define an immutable alert rule version (idempotent by name+version)."""
    payload, created = alerting.create_rule(db, body)
    return JSONResponse(payload, status_code=201 if created else 200)


@router.get("/v1/alert-rules")
def list_alert_rules(
    name: str | None = Query(default=None, description="filter by rule name"),
    status: str = Query(
        default="all", pattern="^(draft|published|retired|all)$"
    ),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """List alert rules with stable keyset pagination (created_at, rule_id)."""
    return alerting.list_rules(db, name, status, limit, cursor)


@router.get("/v1/alert-rules/{rule_id}")
def get_alert_rule(rule_id: str, db: Session = Depends(get_db)):
    return alerting.get_rule(db, rule_id)


@router.post("/v1/alert-rules/{rule_id}/publish")
def publish_alert_rule(
    rule_id: str, body: AlertRuleLifecycleRequest, db: Session = Depends(get_db)
):
    """Publish a draft rule (optimistic concurrency via expected_revision)."""
    return alerting.publish_rule(db, rule_id, body.expected_revision)


@router.post("/v1/alert-rules/{rule_id}/retire")
def retire_alert_rule(
    rule_id: str, body: AlertRuleLifecycleRequest, db: Session = Depends(get_db)
):
    """Retire a published rule; it stops taking part in evaluations."""
    return alerting.retire_rule(db, rule_id, body.expected_revision)


# NOTE: /v1/alerts/evaluations* must be registered before /v1/alerts/{alert_id}.
@router.post("/v1/alerts/evaluations", status_code=201)
def evaluate_alerts(body: AlertEvaluateRequest, db: Session = Depends(get_db)):
    """Evaluate a catalog compilation against alert rules (idempotent).

    The first call creates the evaluation (201); identical repeats run an
    idempotent re-evaluation (200) that records only previously unseen matches.
    """
    payload, created = alerting.evaluate_compilation(db, body)
    return JSONResponse(payload, status_code=201 if created else 200)


@router.get("/v1/alerts/evaluations")
def list_alert_evaluations(
    compilation_id: str | None = Query(
        default=None, description="filter by catalog compilation id"
    ),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """List evaluations with stable keyset pagination (created_at, evaluation_id)."""
    return alerting.list_evaluations(db, compilation_id, limit, cursor)


@router.get("/v1/alerts/evaluations/{evaluation_id}")
def get_alert_evaluation(evaluation_id: str, db: Session = Depends(get_db)):
    return alerting.get_evaluation(db, evaluation_id)


@router.get("/v1/alerts")
def list_alerts(
    rule_id: str | None = Query(default=None, description="filter by rule id"),
    status: str = Query(
        default="all", pattern="^(open|acknowledged|resolved|closed|all)$"
    ),
    compilation_id: str | None = Query(
        default=None, description="filter alerts with hits in this compilation"
    ),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """List alerts with stable keyset pagination (created_at, alert_id)."""
    return alerting.list_alerts(db, rule_id, status, compilation_id, limit, cursor)


@router.get("/v1/alerts/{alert_id}")
def get_alert(alert_id: str, db: Session = Depends(get_db)):
    """Full alert detail: hits with dispositions/reasons and the audit chain."""
    return alerting.get_alert(db, alert_id)


@router.post("/v1/alerts/{alert_id}/acknowledge")
def acknowledge_alert(
    alert_id: str,
    body: AlertTransitionRequest | None = None,
    db: Session = Depends(get_db),
):
    """Acknowledge an open alert (idempotent no-op when already acknowledged)."""
    return alerting.acknowledge_alert(db, alert_id, body.note if body else None)


@router.post("/v1/alerts/{alert_id}/resolve")
def resolve_alert(
    alert_id: str,
    body: AlertTransitionRequest | None = None,
    db: Session = Depends(get_db),
):
    """Resolve an open/acknowledged alert (idempotent no-op when resolved)."""
    return alerting.resolve_alert(db, alert_id, body.note if body else None)

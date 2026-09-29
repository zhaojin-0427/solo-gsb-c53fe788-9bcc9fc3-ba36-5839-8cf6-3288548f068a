"""Cross-run event catalog: compilation, lifecycle and query orchestration."""
from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models import (
    CatalogCompilation,
    CatalogEntry,
    CatalogEntrySource,
    Event,
    EventPick,
    Pick,
    Run,
    StationCalibration,
    VelocityModel,
)
from ..schemas import CatalogCompileRequest
from . import alerting
from . import catalog as cat
from .association import distance_km
from .pipeline import ApiError, get_model, iso_z, to_epoch, to_naive_utc

CATALOG_DRAFT = "draft"
CATALOG_PUBLISHED = "published"
CATALOG_WITHDRAWN = "withdrawn"

DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 200


# --------------------------------------------------------------------------- #
# request canonicalization / idempotency
# --------------------------------------------------------------------------- #
def canonical_request(req: CatalogCompileRequest) -> dict:
    vc = req.version_conditions
    return {
        "start_time": iso_z(to_naive_utc(req.start_time)),
        "end_time": iso_z(to_naive_utc(req.end_time)),
        "radius_km": float(req.radius_km),
        "origin_tolerance_s": float(req.origin_tolerance_s),
        "depth_tolerance_km": float(req.depth_tolerance_km),
        "min_quality": req.min_quality,
        "min_sources": int(req.min_sources),
        "min_shared_stations": int(req.min_shared_stations),
        "center_latitude": req.center_latitude,
        "center_longitude": req.center_longitude,
        "region_radius_km": req.region_radius_km,
        "version_conditions": None
        if vc is None
        else {
            "model_versions": sorted(
                ({"name": m.name, "version": m.version} for m in vc.model_versions),
                key=lambda m: (m["name"], m["version"]),
            ),
            "calibration_versions": sorted(vc.calibration_versions),
            "include_uncalibrated": bool(vc.include_uncalibrated),
        },
    }


def _request_hash(canon: dict) -> str:
    return hashlib.sha256(
        json.dumps(canon, sort_keys=True).encode()
    ).hexdigest()


# --------------------------------------------------------------------------- #
# compilation
# --------------------------------------------------------------------------- #
def compile_catalog(db: Session, req: CatalogCompileRequest) -> tuple[dict, bool]:
    """Cluster completed-run events into immutable catalog entries.

    Deterministic and idempotent: an identical request (by canonical content
    hash) returns the original compilation with ``created=False``.
    """
    start, end = to_naive_utc(req.start_time), to_naive_utc(req.end_time)
    if start >= end:
        raise ApiError(
            422, "invalid_catalog_time_window",
            "start_time must precede end_time",
        )
    center_fields = (req.center_latitude, req.center_longitude, req.region_radius_km)
    if any(v is not None for v in center_fields) and not all(
        v is not None for v in center_fields
    ):
        raise ApiError(
            422, "invalid_spatial_filter",
            "center_latitude, center_longitude and region_radius_km must be "
            "provided together",
        )

    canon = canonical_request(req)
    request_hash = _request_hash(canon)
    existing = (
        db.query(CatalogCompilation)
        .filter_by(request_hash=request_hash)
        .one_or_none()
    )
    if existing is not None:
        db.commit()
        return compilation_payload(db, existing, created=False), False

    model_ids, cal_ids = _resolve_version_conditions(db, req)
    evidences, run_by_id, model_by_id, cal_by_id = _collect_evidence(
        db, req, start, end, model_ids, cal_ids
    )

    clusters = cat.cluster_events(evidences, cat.ClusterConfig.from_request(req))
    clusters = [c for c in clusters if len(c.members) >= req.min_sources]
    # Compilation-wide stable order: earliest anchor event first.
    clusters.sort(key=lambda c: c.anchor.order_key)

    compilation = CatalogCompilation(
        request_hash=request_hash,
        request=canon,
        start_time=start,
        end_time=end,
        entry_count=len(clusters),
    )
    db.add(compilation)
    try:
        db.flush()
    except IntegrityError:
        # Concurrent identical request: collapse onto the winner.
        db.rollback()
        winner = (
            db.query(CatalogCompilation)
            .filter_by(request_hash=request_hash)
            .one()
        )
        return compilation_payload(db, winner, created=False), False

    for cluster in clusters:
        _persist_entry(db, compilation, cluster, run_by_id, model_by_id, cal_by_id)
    db.commit()
    db.refresh(compilation)
    return compilation_payload(db, compilation, created=True), True


def _resolve_version_conditions(
    db: Session, req: CatalogCompileRequest
) -> tuple[set[int] | None, set[str]]:
    """Resolve (allowed model ids, allowed calibration ids); None = no limit."""
    vc = req.version_conditions
    if vc is None:
        return None, set()
    model_ids: set[int] = set()
    for ref in vc.model_versions:
        model_ids.add(get_model(db, ref.name, ref.version).id)
    cal_ids: set[str] = set()
    if vc.calibration_versions:
        rows = (
            db.query(StationCalibration)
            .filter(
                StationCalibration.calibration_version.in_(vc.calibration_versions)
            )
            .all()
        )
        scoped: dict[str, list[StationCalibration]] = {}
        for row in rows:
            if not model_ids or row.model_id in model_ids:
                scoped.setdefault(row.calibration_version, []).append(row)
        for name in vc.calibration_versions:
            if name not in scoped:
                scope = (
                    " among the requested model versions" if model_ids else ""
                )
                raise ApiError(
                    404, "calibration_not_found",
                    f"calibration {name} not found{scope}",
                )
            cal_ids.update(r.id for r in scoped[name])
    return (model_ids or None), cal_ids


def _run_eligible(
    run: Run,
    model_ids: set[int] | None,
    cal_ids: set[str],
    include_uncalibrated: bool,
    calibration_gated: bool,
) -> bool:
    if model_ids is not None and run.model_id not in model_ids:
        return False
    if run.calibration_id is None:
        return include_uncalibrated
    if not calibration_gated:
        return True  # no calibration allowlist requested
    return run.calibration_id in cal_ids


def _collect_evidence(
    db: Session,
    req: CatalogCompileRequest,
    start: datetime,
    end: datetime,
    model_ids: set[int] | None,
    cal_ids: set[str],
):
    vc = req.version_conditions
    include_uncalibrated = True if vc is None else vc.include_uncalibrated
    calibration_gated = bool(vc is not None and vc.calibration_versions)

    rows = (
        db.query(Event, Run)
        .join(Run, Event.run_id == Run.id)
        .filter(Event.origin_time >= start, Event.origin_time < end)
        .order_by(Event.origin_time, Run.id, Event.id)
        .all()
    )
    events: list[Event] = []
    run_by_id: dict[str, Run] = {}
    for event, run in rows:
        run_by_id[run.id] = run
        if cat.QUALITY_RANK[event.quality] > cat.QUALITY_RANK[req.min_quality]:
            continue
        if not _run_eligible(
            run, model_ids, cal_ids, include_uncalibrated, calibration_gated
        ):
            continue
        if req.center_latitude is not None:
            dist = distance_km(
                req.center_latitude, req.center_longitude,
                event.latitude, event.longitude,
            )
            if dist > req.region_radius_km:
                continue
        events.append(event)

    stations_by_event: dict[str, set[str]] = {e.id: set() for e in events}
    if events:
        ep_rows = (
            db.query(EventPick.event_id, Pick.station)
            .join(Pick, EventPick.pick_id == Pick.id)
            .filter(EventPick.event_id.in_([e.id for e in events]))
            .all()
        )
        for event_id, station in ep_rows:
            stations_by_event[event_id].add(station)

    needed_model_ids = {run.model_id for run in run_by_id.values()}
    model_by_id = {
        m.id: m
        for m in db.query(VelocityModel)
        .filter(VelocityModel.id.in_(needed_model_ids))
        .all()
    }
    needed_cal_ids = {
        run.calibration_id
        for run in run_by_id.values()
        if run.calibration_id is not None
    }
    cal_by_id = {
        c.id: c
        for c in db.query(StationCalibration)
        .filter(StationCalibration.id.in_(needed_cal_ids))
        .all()
    } if needed_cal_ids else {}

    evidences = [
        cat.EventEvidence(
            run_id=e.run_id,
            event_id=e.id,
            origin_epoch=to_epoch(e.origin_time),
            latitude=e.latitude,
            longitude=e.longitude,
            depth_km=e.depth_km,
            rms_s=e.rms_s,
            station_count=e.station_count,
            quality=e.quality,
            stations=frozenset(stations_by_event.get(e.id, set())),
        )
        for e in events
    ]
    return evidences, run_by_id, model_by_id, cal_by_id


def _delta_against(source: cat.EventEvidence, master: cat.EventEvidence) -> dict:
    return {
        "origin_time_s": round(source.origin_epoch - master.origin_epoch, 4),
        "latitude": round(source.latitude - master.latitude, 6),
        "longitude": round(source.longitude - master.longitude, 6),
        "depth_km": round(source.depth_km - master.depth_km, 3),
        "rms_s": round(source.rms_s - master.rms_s, 4),
    }


def _persist_entry(
    db: Session,
    compilation: CatalogCompilation,
    cluster: cat.Cluster,
    run_by_id: dict[str, Run],
    model_by_id: dict[int, VelocityModel],
    cal_by_id: dict[str, StationCalibration],
) -> None:
    master = cluster.master
    # Sources: master first, then conflicting candidates in the same
    # (quality, RMS, run, event) order used for master selection.
    ordered = sorted(cluster.members, key=lambda e: e.master_key)
    entry = CatalogEntry(
        compilation_id=compilation.id,
        revision=1,
        status=CATALOG_DRAFT,
        origin_time=datetime.utcfromtimestamp(master.origin_epoch),
        latitude=master.latitude,
        longitude=master.longitude,
        depth_km=master.depth_km,
        rms_s=master.rms_s,
        station_count=master.station_count,
        quality=master.quality,
        master_run_id=master.run_id,
        master_event_id=master.event_id,
        stations=sorted(cluster.union_stations),
        shared_stations=sorted(cluster.shared_stations),
        source_count=len(ordered),
    )
    db.add(entry)
    db.flush()
    for ev in ordered:
        run = run_by_id[ev.run_id]
        model_row = model_by_id[run.model_id]
        cal_row = cal_by_id.get(run.calibration_id) if run.calibration_id else None
        db.add(
            CatalogEntrySource(
                entry_id=entry.id,
                run_id=ev.run_id,
                event_id=ev.event_id,
                is_master=1 if ev.event_id == master.event_id else 0,
                origin_time=datetime.utcfromtimestamp(ev.origin_epoch),
                latitude=ev.latitude,
                longitude=ev.longitude,
                depth_km=ev.depth_km,
                rms_s=ev.rms_s,
                station_count=ev.station_count,
                quality=ev.quality,
                model_name=model_row.name,
                model_version=model_row.version,
                calibration_version=cal_row.calibration_version if cal_row else None,
                calibration_status=cal_row.status if cal_row else None,
                delta=_delta_against(ev, master),
            )
        )
    db.flush()


# --------------------------------------------------------------------------- #
# lifecycle: publish / withdraw (optimistic concurrency)
# --------------------------------------------------------------------------- #
def publish_entry(db: Session, entry_id: str, expected_revision: int) -> dict:
    """draft -> published.  Repeating on an already published entry is idempotent."""
    entry = _get_entry_locked(db, entry_id)
    if entry.status == CATALOG_PUBLISHED:
        db.commit()  # idempotent replay: never mutate, never bump revision
        return entry_payload(db, entry)
    if entry.status == CATALOG_WITHDRAWN:
        db.commit()
        raise ApiError(
            409, "catalog_entry_withdrawn",
            f"catalog entry {entry_id} has been withdrawn and cannot be republished",
        )
    if entry.revision != expected_revision:
        db.commit()
        raise ApiError(
            409, "revision_conflict",
            f"expected revision {entry.revision}, client sent {expected_revision}",
        )
    entry.status = CATALOG_PUBLISHED
    entry.revision += 1
    entry.published_at = datetime.utcnow()
    db.commit()
    db.refresh(entry)
    return entry_payload(db, entry)


def withdraw_entry(
    db: Session, entry_id: str, expected_revision: int, reason: str
) -> dict:
    """published -> withdrawn; withdrawal marks state but never rewrites content."""
    # The alert critical section spans the whole withdrawal so a concurrent
    # alert evaluation either sees the entry still published (its alerts get
    # closed below) or already withdrawn (it never fires on the entry).
    with alerting.alert_operation_lock(db):
        entry = _get_entry_locked(db, entry_id)
        if entry.status == CATALOG_WITHDRAWN:
            db.commit()  # idempotent replay; published content stays untouched
            return entry_payload(db, entry)
        if entry.status == CATALOG_DRAFT:
            db.commit()
            raise ApiError(
                409, "catalog_entry_not_published",
                f"catalog entry {entry_id} is still a draft; publish before withdrawal",
            )
        if entry.revision != expected_revision:
            db.commit()
            raise ApiError(
                409, "revision_conflict",
                f"expected revision {entry.revision}, client sent {expected_revision}",
            )
        entry.status = CATALOG_WITHDRAWN
        entry.revision += 1
        entry.withdrawn_at = datetime.utcnow()
        entry.withdraw_reason = reason
        # Content columns (origin, master solution, sources) are deliberately not
        # touched: a withdrawal is a tombstone, not an edit.
        # Only unacknowledged (open) alerts that fired on this entry are
        # closed; acknowledged/resolved alerts and the audit chain remain.
        alerting.close_open_alerts_for_entry(db, entry.id, reason)
        db.commit()
        db.refresh(entry)
        return entry_payload(db, entry)


def _get_entry_locked(db: Session, entry_id: str) -> CatalogEntry:
    # Single locked read (no-op lock on SQLite, where writers serialize);
    # the lock makes the OCC revision re-check race-safe on PostgreSQL.
    entry = (
        db.query(CatalogEntry)
        .filter_by(id=entry_id)
        .with_for_update()
        .one_or_none()
    )
    if entry is None:
        raise ApiError(
            404, "catalog_entry_not_found",
            f"catalog entry {entry_id} not found",
        )
    return entry


# --------------------------------------------------------------------------- #
# queries
# --------------------------------------------------------------------------- #
def get_compilation(db: Session, compilation_id: str) -> dict:
    row = db.get(CatalogCompilation, compilation_id)
    if row is None:
        raise ApiError(
            404, "catalog_compilation_not_found",
            f"catalog compilation {compilation_id} not found",
        )
    return compilation_payload(db, row, created=False)


def list_compilations(db: Session, limit: int) -> dict:
    rows = (
        db.query(CatalogCompilation)
        .order_by(CatalogCompilation.created_at, CatalogCompilation.id)
        .limit(limit)
        .all()
    )
    return {
        "compilations": [
            {
                "compilation_id": c.id,
                "request": c.request,
                "start_time": iso_z(c.start_time),
                "end_time": iso_z(c.end_time),
                "entry_count": c.entry_count,
                "created_at": iso_z(c.created_at),
            }
            for c in rows
        ]
    }


def get_entry(db: Session, entry_id: str) -> dict:
    entry = db.get(CatalogEntry, entry_id)
    if entry is None:
        raise ApiError(
            404, "catalog_entry_not_found",
            f"catalog entry {entry_id} not found",
        )
    return entry_payload(db, entry)


def list_entries(
    db: Session,
    start_time: datetime | None,
    end_time: datetime | None,
    min_latitude: float | None,
    max_latitude: float | None,
    min_longitude: float | None,
    max_longitude: float | None,
    min_quality: str,
    status: str,
    limit: int,
    cursor: str | None,
) -> dict:
    if (start_time is None) != (end_time is None):
        raise ApiError(
            422, "invalid_catalog_time_window",
            "start_time and end_time must be provided together",
        )
    bbox = (min_latitude, max_latitude, min_longitude, max_longitude)
    if any(v is not None for v in bbox) and not all(v is not None for v in bbox):
        raise ApiError(
            422, "invalid_spatial_filter",
            "min_latitude, max_latitude, min_longitude and max_longitude must "
            "be provided together",
        )
    if all(v is not None for v in bbox) and (
        min_latitude > max_latitude or min_longitude > max_longitude
    ):
        raise ApiError(
            422, "invalid_spatial_filter",
            "minimum bounds must not exceed maximum bounds",
        )

    filters = {
        "start_time": iso_z(to_naive_utc(start_time)) if start_time else None,
        "end_time": iso_z(to_naive_utc(end_time)) if end_time else None,
        "min_latitude": min_latitude,
        "max_latitude": max_latitude,
        "min_longitude": min_longitude,
        "max_longitude": max_longitude,
        "min_quality": min_quality,
        "status": status,
        "limit": limit,
    }
    fingerprint = hashlib.sha256(
        json.dumps(filters, sort_keys=True).encode()
    ).hexdigest()[:16]

    cursor_time = cursor_id = None
    if cursor is not None:
        cursor_time, cursor_id = _decode_cursor(cursor, fingerprint)

    q = db.query(CatalogEntry)
    if start_time is not None:
        q = q.filter(
            CatalogEntry.origin_time >= to_naive_utc(start_time),
            CatalogEntry.origin_time < to_naive_utc(end_time),
        )
    if all(v is not None for v in bbox):
        q = q.filter(
            CatalogEntry.latitude >= min_latitude,
            CatalogEntry.latitude <= max_latitude,
            CatalogEntry.longitude >= min_longitude,
            CatalogEntry.longitude <= max_longitude,
        )
    allowed = [
        g for g, rank in cat.QUALITY_RANK.items()
        if rank <= cat.QUALITY_RANK[min_quality]
    ]
    q = q.filter(CatalogEntry.quality.in_(allowed))
    if status != "all":
        q = q.filter(CatalogEntry.status == status)
    if cursor_time is not None:
        # Explicit row-value comparison (works on both SQLite and PostgreSQL;
        # SQLAlchemy does not overload ">" for a column tuple vs a python tuple).
        q = q.filter(
            (CatalogEntry.origin_time > cursor_time)
            | (
                (CatalogEntry.origin_time == cursor_time)
                & (CatalogEntry.id > cursor_id)
            )
        )
    q = q.order_by(CatalogEntry.origin_time, CatalogEntry.id).limit(limit + 1)
    rows = q.all()

    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        last = page[-1]
        next_cursor = _encode_cursor(last.origin_time, last.id, fingerprint)

    return {
        "entries": [entry_payload(db, e) for e in page],
        "next_cursor": next_cursor,
    }


def _encode_cursor(origin_time: datetime, entry_id: str, fingerprint: str) -> str:
    # Exact microsecond timestamp: keyset comparison must not round entries
    # into the same millisecond bucket (that would overlap pages).
    raw = json.dumps(
        {"t": origin_time.strftime("%Y-%m-%dT%H:%M:%S.%f"), "id": entry_id,
         "fp": fingerprint}
    ).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str, fingerprint: str) -> tuple[datetime, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
        token_time = datetime.strptime(payload["t"], "%Y-%m-%dT%H:%M:%S.%f")
        token_id = payload["id"]
        token_fp = payload["fp"]
    except Exception as exc:  # malformed/undecodable cursor
        raise ApiError(422, "invalid_cursor", "malformed pagination cursor") from exc
    if token_fp != fingerprint:
        raise ApiError(
            422, "invalid_cursor",
            "cursor was issued for a different query (time/region/quality/"
            "status filter mismatch)",
        )
    return token_time, token_id


# --------------------------------------------------------------------------- #
# serialization
# --------------------------------------------------------------------------- #
def _source_payload(s: CatalogEntrySource) -> dict:
    return {
        "run_id": s.run_id,
        "event_id": s.event_id,
        "is_master": bool(s.is_master),
        "origin_time": iso_z(s.origin_time),
        "latitude": round(s.latitude, 5),
        "longitude": round(s.longitude, 5),
        "depth_km": round(s.depth_km, 2),
        "rms_residual_s": round(s.rms_s, 4),
        "station_count": s.station_count,
        "quality": s.quality,
        "model": {"name": s.model_name, "version": s.model_version},
        "calibration": (
            {"version": s.calibration_version, "status": s.calibration_status}
            if s.calibration_version
            else None
        ),
        "delta": s.delta,
    }


def entry_payload(db: Session, entry: CatalogEntry) -> dict:
    sources = (
        db.query(CatalogEntrySource)
        .filter_by(entry_id=entry.id)
        # master first, then candidates in their persisted (deterministic) order
        .order_by(CatalogEntrySource.is_master.desc(), CatalogEntrySource.id)
        .all()
    )
    return {
        "entry_id": entry.id,
        "compilation_id": entry.compilation_id,
        "revision": entry.revision,
        "status": entry.status,
        "origin_time": iso_z(entry.origin_time),
        "latitude": round(entry.latitude, 5),
        "longitude": round(entry.longitude, 5),
        "depth_km": round(entry.depth_km, 2),
        "rms_residual_s": round(entry.rms_s, 4),
        "station_count": entry.station_count,
        "quality": entry.quality,
        "master": {"run_id": entry.master_run_id, "event_id": entry.master_event_id},
        "stations": list(entry.stations),
        "shared_stations": list(entry.shared_stations),
        "distinct_run_count": len({s.run_id for s in sources}),
        "source_count": entry.source_count,
        "sources": [_source_payload(s) for s in sources],
        "created_at": iso_z(entry.created_at),
        "published_at": iso_z(entry.published_at) if entry.published_at else None,
        "withdrawn_at": iso_z(entry.withdrawn_at) if entry.withdrawn_at else None,
        "withdraw_reason": entry.withdraw_reason,
    }


def compilation_payload(
    db: Session, compilation: CatalogCompilation, created: bool
) -> dict:
    entries = (
        db.query(CatalogEntry)
        .filter_by(compilation_id=compilation.id)
        .order_by(CatalogEntry.origin_time, CatalogEntry.id)
        .all()
    )
    return {
        "compilation_id": compilation.id,
        "created": created,
        "request": compilation.request,
        "start_time": iso_z(compilation.start_time),
        "end_time": iso_z(compilation.end_time),
        "entry_count": compilation.entry_count,
        "entries": [entry_payload(db, e) for e in entries],
        "created_at": iso_z(compilation.created_at),
    }

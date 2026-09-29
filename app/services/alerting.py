"""Catalog alert orchestration over the published cross-run catalog.

Administrators define *immutable alert rule versions* (quality grade, station
count, RMS, origin-time window and circular-region conditions, plus a merge
interval) and publish them with an ``expected_revision`` CAS
(``draft -> published -> retired``).  Evaluating a catalog compilation against
the published rules deterministically records exactly one *hit* per matching
(rule, entry) pair -- the ``(rule_id, entry_id)`` unique constraint plus a
serialized evaluation section guarantee that concurrent evaluations can
neither duplicate nor lose a hit.

Consecutive hits of one rule whose origin-time gap stays within the rule's
merge interval collapse into a single alert: the extra hit is *suppressed*
(``merged`` / ``within_merge_interval``); when the target alert was already
acknowledged or resolved, the hit *escalates* it back to ``open``
(``escalated`` / ``new_hit_after_acknowledge`` / ``new_hit_after_resolve``).
Hits beyond the interval open a new alert.  Operators acknowledge and resolve
alerts (idempotent transitions); withdrawing a catalog entry closes only the
unacknowledged (``open``) alerts that fired on it -- acknowledged/resolved
alerts and the full audit chain (hits + transition events) are preserved.

Like the catalog and the station audit this module never modifies
ingestions/runs/events/picks; it only reads the catalog and writes its own
tables.  Evaluations are idempotent by canonical request hash: an identical
request returns the same evaluation and runs a re-pass that records only
previously unseen matches (e.g. entries published after the first pass).
"""
from __future__ import annotations

import base64
import hashlib
import json
import threading
from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models import (
    Alert,
    AlertEvaluation,
    AlertEvent,
    AlertHit,
    AlertRule,
    CatalogCompilation,
    CatalogEntry,
)
from ..schemas import AlertEvaluateRequest, AlertRuleCreate
from .association import distance_km
from .catalog import QUALITY_RANK
from .pipeline import ApiError, iso_z, to_naive_utc

# Mirror of catalog_pipeline.CATALOG_PUBLISHED (importing catalog_pipeline
# here would create a cycle: it calls this module on entry withdrawal).
_CATALOG_ENTRY_PUBLISHED = "published"

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
RULE_DRAFT = "draft"
RULE_PUBLISHED = "published"
RULE_RETIRED = "retired"

ALERT_OPEN = "open"
ALERT_ACKNOWLEDGED = "acknowledged"
ALERT_RESOLVED = "resolved"
ALERT_CLOSED = "closed"
ALERT_STATUSES = (ALERT_OPEN, ALERT_ACKNOWLEDGED, ALERT_RESOLVED, ALERT_CLOSED)

HIT_OPENED = "opened"
HIT_MERGED = "merged"
HIT_ESCALATED = "escalated"

REASON_WITHIN_MERGE_INTERVAL = "within_merge_interval"
REASON_HIT_AFTER_ACKNOWLEDGE = "new_hit_after_acknowledge"
REASON_HIT_AFTER_RESOLVE = "new_hit_after_resolve"
REASON_ENTRY_WITHDRAWN = "catalog_entry_withdrawn"

EVENT_OPENED = "opened"
EVENT_MERGED = "merged"
EVENT_ESCALATED = "escalated"
EVENT_ACKNOWLEDGED = "acknowledged"
EVENT_RESOLVED = "resolved"
EVENT_CLOSED = "closed"

DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 200


# --------------------------------------------------------------------------- #
# alert-state serialization
#
# Every mutation of alert state (evaluation passes, acknowledge/resolve and
# the withdrawal close) runs inside the same critical section: an in-process
# lock for SQLite (single-process tests/dev) and a PostgreSQL transaction
# advisory lock for multi-process deployments.  Combined with the
# (rule_id, entry_id) unique hit key this makes concurrent evaluations
# duplicate- and loss-free.
# --------------------------------------------------------------------------- #
_ALERT_LOCK = threading.Lock()


@contextmanager
def alert_operation_lock(db: Session):
    """Serialize alert-state mutations (in-process + PG advisory xact lock)."""
    _ALERT_LOCK.acquire()
    try:
        if db.get_bind().dialect.name == "postgresql":
            db.execute(
                text("SELECT pg_advisory_xact_lock(hashtext('seisnet_alerting'))")
            )
        yield
    finally:
        _ALERT_LOCK.release()


# --------------------------------------------------------------------------- #
# canonicalization helpers
# --------------------------------------------------------------------------- #
def _sha256_canonical(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _canonical_rule_content(
    name: str,
    rule_version: str,
    min_quality: str,
    min_stations: int | None,
    max_rms_s: float | None,
    window_start: datetime | None,
    window_end: datetime | None,
    center_latitude: float | None,
    center_longitude: float | None,
    region_radius_km: float | None,
    merge_interval_s: float,
    label: str | None,
) -> dict:
    return {
        "name": name,
        "rule_version": rule_version,
        "min_quality": min_quality,
        "min_stations": None if min_stations is None else int(min_stations),
        "max_rms_s": None if max_rms_s is None else float(max_rms_s),
        "window_start": iso_z(window_start) if window_start else None,
        "window_end": iso_z(window_end) if window_end else None,
        "center_latitude": center_latitude,
        "center_longitude": center_longitude,
        "region_radius_km": region_radius_km,
        "merge_interval_s": float(merge_interval_s),
        "label": label,
    }


# --------------------------------------------------------------------------- #
# rule lifecycle: create (idempotent) -> publish (CAS) -> retire (CAS)
# --------------------------------------------------------------------------- #
def create_rule(db: Session, body: AlertRuleCreate) -> tuple[dict, bool]:
    window_start = to_naive_utc(body.window_start) if body.window_start else None
    window_end = to_naive_utc(body.window_end) if body.window_end else None
    if (window_start is None) != (window_end is None):
        raise ApiError(
            422, "invalid_rule_time_window",
            "window_start and window_end must be provided together",
        )
    if window_start is not None and window_start >= window_end:
        raise ApiError(
            422, "invalid_rule_time_window",
            "window_start must precede window_end",
        )
    region = (body.center_latitude, body.center_longitude, body.region_radius_km)
    if any(v is not None for v in region) and not all(v is not None for v in region):
        raise ApiError(
            422, "invalid_spatial_filter",
            "center_latitude, center_longitude and region_radius_km must be "
            "provided together",
        )

    canon = _canonical_rule_content(
        body.name, body.rule_version, body.min_quality, body.min_stations,
        body.max_rms_s, window_start, window_end, body.center_latitude,
        body.center_longitude, body.region_radius_km, body.merge_interval_s,
        body.label,
    )
    content_hash = _sha256_canonical(canon)

    existing = (
        db.query(AlertRule)
        .filter_by(name=body.name, rule_version=body.rule_version)
        .one_or_none()
    )
    if existing is not None:
        if existing.content_hash == content_hash:
            db.commit()
            return rule_payload(existing, created=False), False
        raise ApiError(
            409, "alert_rule_version_conflict",
            f"alert rule {body.name}@{body.rule_version} already exists with "
            "different conditions; rule versions are immutable, choose a new "
            "version",
        )

    rule = AlertRule(
        name=body.name,
        rule_version=body.rule_version,
        revision=1,
        status=RULE_DRAFT,
        request=canon,
        content_hash=content_hash,
        min_quality=body.min_quality,
        min_stations=body.min_stations,
        max_rms_s=body.max_rms_s,
        window_start=window_start,
        window_end=window_end,
        center_latitude=body.center_latitude,
        center_longitude=body.center_longitude,
        region_radius_km=body.region_radius_km,
        merge_interval_s=float(body.merge_interval_s),
        label=body.label,
    )
    db.add(rule)
    try:
        db.flush()
    except IntegrityError:
        # Concurrent create of the same (name, version): collapse onto it.
        db.rollback()
        winner = (
            db.query(AlertRule)
            .filter_by(name=body.name, rule_version=body.rule_version)
            .one()
        )
        if winner.content_hash == content_hash:
            return rule_payload(winner, created=False), False
        raise ApiError(
            409, "alert_rule_version_conflict",
            f"alert rule {body.name}@{body.rule_version} already exists with "
            "different conditions; rule versions are immutable, choose a new "
            "version",
        )
    db.commit()
    db.refresh(rule)
    return rule_payload(rule, created=True), True


def publish_rule(db: Session, rule_id: str, expected_revision: int) -> dict:
    """draft -> published.  Repeating on a published rule is an idempotent no-op."""
    rule = _get_rule_locked(db, rule_id)
    if rule.status == RULE_PUBLISHED:
        db.commit()  # idempotent replay: never mutate, never bump revision
        return rule_payload(rule, created=False)
    if rule.status == RULE_RETIRED:
        db.commit()
        raise ApiError(
            409, "alert_rule_retired",
            f"alert rule {rule_id} has been retired and cannot be republished",
        )
    if rule.revision != expected_revision:
        db.commit()
        raise ApiError(
            409, "revision_conflict",
            f"expected revision {rule.revision}, client sent {expected_revision}",
        )
    rule.status = RULE_PUBLISHED
    rule.revision += 1
    rule.published_at = datetime.utcnow()
    db.commit()
    db.refresh(rule)
    return rule_payload(rule, created=False)


def retire_rule(db: Session, rule_id: str, expected_revision: int) -> dict:
    """published -> retired; retired rules no longer take part in evaluations."""
    rule = _get_rule_locked(db, rule_id)
    if rule.status == RULE_RETIRED:
        db.commit()  # idempotent replay
        return rule_payload(rule, created=False)
    if rule.status == RULE_DRAFT:
        db.commit()
        raise ApiError(
            409, "alert_rule_not_published",
            f"alert rule {rule_id} is still a draft; publish before retiring",
        )
    if rule.revision != expected_revision:
        db.commit()
        raise ApiError(
            409, "revision_conflict",
            f"expected revision {rule.revision}, client sent {expected_revision}",
        )
    rule.status = RULE_RETIRED
    rule.revision += 1
    rule.retired_at = datetime.utcnow()
    db.commit()
    db.refresh(rule)
    return rule_payload(rule, created=False)


def _get_rule_locked(db: Session, rule_id: str) -> AlertRule:
    # Single locked read (no-op lock on SQLite, where writers serialize); the
    # lock makes the OCC revision re-check race-safe on PostgreSQL.
    rule = (
        db.query(AlertRule)
        .filter_by(id=rule_id)
        .with_for_update()
        .one_or_none()
    )
    if rule is None:
        raise ApiError(404, "alert_rule_not_found", f"alert rule {rule_id} not found")
    return rule


# --------------------------------------------------------------------------- #
# rule matching (deterministic, pure)
# --------------------------------------------------------------------------- #
def rule_matches(rule: AlertRule, entry: CatalogEntry) -> bool:
    """Conjunctive rule conditions against one published catalog entry."""
    if QUALITY_RANK[entry.quality] > QUALITY_RANK[rule.min_quality]:
        return False
    if rule.min_stations is not None and entry.station_count < rule.min_stations:
        return False
    if rule.max_rms_s is not None and entry.rms_s > rule.max_rms_s:
        return False
    if rule.window_start is not None and not (
        rule.window_start <= entry.origin_time < rule.window_end
    ):
        return False
    if rule.center_latitude is not None:
        dist = distance_km(
            rule.center_latitude, rule.center_longitude,
            entry.latitude, entry.longitude,
        )
        if dist > rule.region_radius_km:
            return False
    return True


# --------------------------------------------------------------------------- #
# evaluation: deterministic hit recording with merge / suppression / escalation
# --------------------------------------------------------------------------- #
def evaluate_compilation(db: Session, req: AlertEvaluateRequest) -> tuple[dict, bool]:
    # The whole evaluation -- validation reads included -- runs inside the
    # alert critical section, so its view of rules/entries and the hits it
    # writes are linearizable against rule lifecycle changes, operator
    # transitions and catalog withdrawals on both backends.
    with alert_operation_lock(db):
        compilation = db.get(CatalogCompilation, req.compilation_id)
        if compilation is None:
            raise ApiError(
                404, "catalog_compilation_not_found",
                f"catalog compilation {req.compilation_id} not found",
            )

        explicit_ids: list[str] | None = None
        rules_by_id: dict[str, AlertRule] = {}
        if req.rule_ids is not None:
            explicit_ids = sorted(set(req.rule_ids))
            if explicit_ids:
                rows = (
                    db.query(AlertRule).filter(AlertRule.id.in_(explicit_ids)).all()
                )
                rules_by_id = {r.id: r for r in rows}
                for rid in explicit_ids:
                    if rid not in rules_by_id:
                        raise ApiError(
                            404, "alert_rule_not_found",
                            f"alert rule {rid} not found",
                        )

        canon = {"compilation_id": compilation.id, "rule_ids": explicit_ids}
        request_hash = _sha256_canonical(canon)

        evaluation = (
            db.query(AlertEvaluation)
            .filter_by(request_hash=request_hash)
            .one_or_none()
        )
        created = False
        if evaluation is None:
            # First pass: every explicitly referenced rule must be published.
            for rid in explicit_ids or ():
                _require_published(rules_by_id[rid])
            evaluation = AlertEvaluation(
                request_hash=request_hash,
                request=canon,
                compilation_id=compilation.id,
            )
            db.add(evaluation)
            try:
                db.flush()
            except IntegrityError:
                # Concurrent identical request: collapse onto the winner.
                db.rollback()
                evaluation = (
                    db.query(AlertEvaluation)
                    .filter_by(request_hash=request_hash)
                    .one()
                )
            else:
                created = True

        rules = _effective_rules(db, evaluation.request["rule_ids"])
        entries = (
            db.query(CatalogEntry)
            .filter_by(compilation_id=compilation.id, status=_CATALOG_ENTRY_PUBLISHED)
            .order_by(CatalogEntry.origin_time, CatalogEntry.id)
            .all()
        )
        new_hits = _run_pass(db, evaluation, rules, entries)
        evaluation.effective_rule_ids = [r.id for r in rules]
        evaluation.entry_count = len(entries)
        evaluation.hit_count += len(new_hits)
        evaluation.evaluation_count += 1
        evaluation.last_evaluated_at = datetime.utcnow()
        db.commit()
        db.refresh(evaluation)
    return evaluation_payload(db, evaluation, created, new_hits), created


def _require_published(rule: AlertRule) -> None:
    if rule.status == RULE_PUBLISHED:
        return
    if rule.status == RULE_RETIRED:
        raise ApiError(
            409, "alert_rule_retired",
            f"alert rule {rule.id} ({rule.name}@{rule.rule_version}) is retired",
        )
    raise ApiError(
        409, "alert_rule_not_published",
        f"alert rule {rule.id} ({rule.name}@{rule.rule_version}) is not "
        "published; publish it before evaluating",
    )


def _effective_rules(db: Session, requested_ids: list[str] | None) -> list[AlertRule]:
    """Rules applied in this pass, in a deterministic order.

    ``None`` evaluates every currently published rule; an explicit list is
    restricted to rules that are still published (a rule retired after the
    evaluation was created is skipped on re-passes, never an error).
    """
    q = db.query(AlertRule).filter(AlertRule.status == RULE_PUBLISHED)
    if requested_ids is not None:
        if not requested_ids:
            return []
        q = q.filter(AlertRule.id.in_(requested_ids))
    rules = q.all()
    rules.sort(key=lambda r: (r.name, r.rule_version, r.id))
    return rules


def _run_pass(
    db: Session,
    evaluation: AlertEvaluation,
    rules: list[AlertRule],
    entries: list[CatalogEntry],
) -> list[AlertHit]:
    """Record one hit per unseen (rule, entry) match; fully deterministic."""
    new_hits: list[AlertHit] = []
    for rule in rules:
        for entry in entries:  # entries arrive in (origin_time, id) order
            if not rule_matches(rule, entry):
                continue
            exists = (
                db.query(AlertHit.id)
                .filter_by(rule_id=rule.id, entry_id=entry.id)
                .first()
            )
            if exists is not None:
                continue  # idempotent: this match was recorded before
            new_hits.append(_record_hit(db, evaluation, rule, entry))
    db.flush()
    return new_hits


def _record_hit(
    db: Session, evaluation: AlertEvaluation, rule: AlertRule, entry: CatalogEntry
) -> AlertHit:
    """Merge into the closest recent non-closed alert or open a new one."""
    origin = entry.origin_time
    interval = float(rule.merge_interval_s)
    candidates = (
        db.query(Alert)
        .filter(Alert.rule_id == rule.id, Alert.status != ALERT_CLOSED)
        .all()
    )
    target: Alert | None = None
    target_key: tuple | None = None
    for alert in candidates:
        gap = abs((origin - alert.last_hit_origin_time).total_seconds())
        if gap <= interval:
            key = (alert.last_hit_origin_time, alert.id)
            if target_key is None or key > target_key:
                target, target_key = alert, key

    snapshot = _entry_snapshot(entry)
    if target is None:
        alert = Alert(
            rule_id=rule.id,
            status=ALERT_OPEN,
            first_hit_origin_time=origin,
            last_hit_origin_time=origin,
            hit_count=1,
            best_quality=entry.quality,
            max_station_count=entry.station_count,
            min_rms_s=entry.rms_s,
        )
        db.add(alert)
        db.flush()
        hit = AlertHit(
            rule_id=rule.id,
            alert_id=alert.id,
            evaluation_id=evaluation.id,
            compilation_id=evaluation.compilation_id,
            entry_id=entry.id,
            disposition=HIT_OPENED,
            reason=None,
            entry_snapshot=snapshot,
        )
        db.add(hit)
        db.flush()
        _add_event(db, alert, EVENT_OPENED, hit_id=hit.id, reason=None,
                   detail={"entry_id": entry.id})
        return hit

    if target.status == ALERT_OPEN:
        disposition, reason = HIT_MERGED, REASON_WITHIN_MERGE_INTERVAL
    elif target.status == ALERT_ACKNOWLEDGED:
        disposition, reason = HIT_ESCALATED, REASON_HIT_AFTER_ACKNOWLEDGE
    else:  # resolved
        disposition, reason = HIT_ESCALATED, REASON_HIT_AFTER_RESOLVE

    hit = AlertHit(
        rule_id=rule.id,
        alert_id=target.id,
        evaluation_id=evaluation.id,
        compilation_id=evaluation.compilation_id,
        entry_id=entry.id,
        disposition=disposition,
        reason=reason,
        entry_snapshot=snapshot,
    )
    db.add(hit)
    db.flush()
    target.hit_count += 1
    if origin < target.first_hit_origin_time:
        target.first_hit_origin_time = origin
    if origin > target.last_hit_origin_time:
        target.last_hit_origin_time = origin
    if QUALITY_RANK[entry.quality] < QUALITY_RANK[target.best_quality]:
        target.best_quality = entry.quality
    target.max_station_count = max(target.max_station_count, entry.station_count)
    target.min_rms_s = min(target.min_rms_s, entry.rms_s)
    target.updated_at = datetime.utcnow()
    if disposition == HIT_ESCALATED:
        target.status = ALERT_OPEN
    _add_event(db, target,
               EVENT_MERGED if disposition == HIT_MERGED else EVENT_ESCALATED,
               hit_id=hit.id, reason=reason, detail={"entry_id": entry.id})
    return hit


def _entry_snapshot(entry: CatalogEntry) -> dict:
    return {
        "origin_time": iso_z(entry.origin_time),
        "latitude": entry.latitude,
        "longitude": entry.longitude,
        "depth_km": entry.depth_km,
        "rms_residual_s": entry.rms_s,
        "station_count": entry.station_count,
        "quality": entry.quality,
    }


def _add_event(
    db: Session,
    alert: Alert,
    kind: str,
    hit_id: str | None = None,
    reason: str | None = None,
    detail: dict | None = None,
) -> AlertEvent:
    alert.event_count += 1
    event = AlertEvent(
        alert_id=alert.id,
        seq=alert.event_count,
        kind=kind,
        hit_id=hit_id,
        reason=reason,
        detail=detail or {},
    )
    db.add(event)
    return event


# --------------------------------------------------------------------------- #
# operator transitions: acknowledge / resolve (idempotent)
# --------------------------------------------------------------------------- #
def acknowledge_alert(db: Session, alert_id: str, note: str | None) -> dict:
    """open -> acknowledged.  Repeating on an acknowledged alert is a no-op."""
    with alert_operation_lock(db):
        alert = _get_alert_locked(db, alert_id)
        if alert.status == ALERT_ACKNOWLEDGED:
            db.commit()
            return alert_payload(db, alert)
        if alert.status == ALERT_RESOLVED:
            db.commit()
            raise ApiError(
                409, "alert_already_resolved",
                f"alert {alert_id} is resolved and cannot be acknowledged",
            )
        if alert.status == ALERT_CLOSED:
            db.commit()
            raise ApiError(
                409, "alert_closed",
                f"alert {alert_id} is closed and cannot be acknowledged",
            )
        alert.status = ALERT_ACKNOWLEDGED
        alert.updated_at = datetime.utcnow()
        _add_event(db, alert, EVENT_ACKNOWLEDGED, detail={"note": note} if note else {})
        db.commit()
        db.refresh(alert)
    return alert_payload(db, alert)


def resolve_alert(db: Session, alert_id: str, note: str | None) -> dict:
    """open/acknowledged -> resolved.  Repeating on a resolved alert is a no-op."""
    with alert_operation_lock(db):
        alert = _get_alert_locked(db, alert_id)
        if alert.status == ALERT_RESOLVED:
            db.commit()
            return alert_payload(db, alert)
        if alert.status == ALERT_CLOSED:
            db.commit()
            raise ApiError(
                409, "alert_closed",
                f"alert {alert_id} is closed and cannot be resolved",
            )
        alert.status = ALERT_RESOLVED
        alert.updated_at = datetime.utcnow()
        _add_event(db, alert, EVENT_RESOLVED, detail={"note": note} if note else {})
        db.commit()
        db.refresh(alert)
    return alert_payload(db, alert)


def _get_alert_locked(db: Session, alert_id: str) -> Alert:
    alert = (
        db.query(Alert)
        .filter_by(id=alert_id)
        .with_for_update()
        .one_or_none()
    )
    if alert is None:
        raise ApiError(404, "alert_not_found", f"alert {alert_id} not found")
    return alert


# --------------------------------------------------------------------------- #
# catalog withdrawal hook: close only unacknowledged alerts, keep the chain
# --------------------------------------------------------------------------- #
def close_open_alerts_for_entry(
    db: Session, entry_id: str, withdraw_reason: str | None
) -> int:
    """Close ``open`` alerts that fired on a withdrawn catalog entry.

    Called from the catalog withdrawal flow inside the same transaction (the
    caller already holds :func:`alert_operation_lock`).  Acknowledged and
    resolved alerts are deliberately left untouched; hits and the audit chain
    are never removed.
    """
    alert_ids = (
        select(AlertHit.alert_id)
        .where(AlertHit.entry_id == entry_id)
        .distinct()
        .scalar_subquery()
    )
    alerts = (
        db.query(Alert)
        .filter(Alert.id.in_(alert_ids), Alert.status == ALERT_OPEN)
        .with_for_update()
        .all()
    )
    now = datetime.utcnow()
    for alert in sorted(alerts, key=lambda a: a.id):
        alert.status = ALERT_CLOSED
        alert.updated_at = now
        _add_event(
            db, alert, EVENT_CLOSED,
            reason=REASON_ENTRY_WITHDRAWN,
            detail={"entry_id": entry_id, "withdraw_reason": withdraw_reason},
        )
    db.flush()
    return len(alerts)


# --------------------------------------------------------------------------- #
# queries
# --------------------------------------------------------------------------- #
def get_rule(db: Session, rule_id: str) -> dict:
    rule = db.get(AlertRule, rule_id)
    if rule is None:
        raise ApiError(404, "alert_rule_not_found", f"alert rule {rule_id} not found")
    return rule_payload(rule, created=False)


def list_rules(
    db: Session,
    name: str | None,
    status: str,
    limit: int,
    cursor: str | None,
) -> dict:
    filters = {"name": name, "status": status, "limit": limit}
    fingerprint = _fingerprint(filters)
    cursor_at, cursor_id = (None, None)
    if cursor is not None:
        cursor_at, cursor_id = _decode_cursor(cursor, fingerprint)

    q = db.query(AlertRule)
    if name is not None:
        q = q.filter(AlertRule.name == name)
    if status != "all":
        q = q.filter(AlertRule.status == status)
    if cursor_at is not None:
        q = q.filter(
            (AlertRule.created_at > cursor_at)
            | ((AlertRule.created_at == cursor_at) & (AlertRule.id > cursor_id))
        )
    q = q.order_by(AlertRule.created_at, AlertRule.id).limit(limit + 1)
    rows = q.all()
    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        last = page[-1]
        next_cursor = _encode_cursor(
            {"t": last.created_at.strftime("%Y-%m-%dT%H:%M:%S.%f"), "id": last.id},
            fingerprint,
        )
    return {
        "rules": [rule_payload(r, created=False) for r in page],
        "next_cursor": next_cursor,
    }


def get_evaluation(db: Session, evaluation_id: str) -> dict:
    row = db.get(AlertEvaluation, evaluation_id)
    if row is None:
        raise ApiError(
            404, "alert_evaluation_not_found",
            f"alert evaluation {evaluation_id} not found",
        )
    return evaluation_payload(db, row, created=False, new_hits=[])


def list_evaluations(
    db: Session,
    compilation_id: str | None,
    limit: int,
    cursor: str | None,
) -> dict:
    if compilation_id is not None and db.get(CatalogCompilation, compilation_id) is None:
        raise ApiError(
            404, "catalog_compilation_not_found",
            f"catalog compilation {compilation_id} not found",
        )
    filters = {"compilation_id": compilation_id, "limit": limit}
    fingerprint = _fingerprint(filters)
    cursor_at, cursor_id = (None, None)
    if cursor is not None:
        cursor_at, cursor_id = _decode_cursor(cursor, fingerprint)

    q = db.query(AlertEvaluation)
    if compilation_id is not None:
        q = q.filter(AlertEvaluation.compilation_id == compilation_id)
    if cursor_at is not None:
        q = q.filter(
            (AlertEvaluation.created_at > cursor_at)
            | (
                (AlertEvaluation.created_at == cursor_at)
                & (AlertEvaluation.id > cursor_id)
            )
        )
    q = q.order_by(AlertEvaluation.created_at, AlertEvaluation.id).limit(limit + 1)
    rows = q.all()
    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        last = page[-1]
        next_cursor = _encode_cursor(
            {"t": last.created_at.strftime("%Y-%m-%dT%H:%M:%S.%f"), "id": last.id},
            fingerprint,
        )
    return {
        "evaluations": [evaluation_summary(e) for e in page],
        "next_cursor": next_cursor,
    }


def get_alert(db: Session, alert_id: str) -> dict:
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise ApiError(404, "alert_not_found", f"alert {alert_id} not found")
    return alert_payload(db, alert)


def list_alerts(
    db: Session,
    rule_id: str | None,
    status: str,
    compilation_id: str | None,
    limit: int,
    cursor: str | None,
) -> dict:
    if rule_id is not None and db.get(AlertRule, rule_id) is None:
        raise ApiError(404, "alert_rule_not_found", f"alert rule {rule_id} not found")
    if compilation_id is not None and db.get(CatalogCompilation, compilation_id) is None:
        raise ApiError(
            404, "catalog_compilation_not_found",
            f"catalog compilation {compilation_id} not found",
        )
    filters = {
        "rule_id": rule_id,
        "status": status,
        "compilation_id": compilation_id,
        "limit": limit,
    }
    fingerprint = _fingerprint(filters)
    cursor_at, cursor_id = (None, None)
    if cursor is not None:
        cursor_at, cursor_id = _decode_cursor(cursor, fingerprint)

    q = db.query(Alert)
    if rule_id is not None:
        q = q.filter(Alert.rule_id == rule_id)
    if status != "all":
        q = q.filter(Alert.status == status)
    if compilation_id is not None:
        alert_ids = (
            select(AlertHit.alert_id)
            .where(AlertHit.compilation_id == compilation_id)
            .distinct()
            .scalar_subquery()
        )
        q = q.filter(Alert.id.in_(alert_ids))
    if cursor_at is not None:
        q = q.filter(
            (Alert.created_at > cursor_at)
            | ((Alert.created_at == cursor_at) & (Alert.id > cursor_id))
        )
    q = q.order_by(Alert.created_at, Alert.id).limit(limit + 1)
    rows = q.all()
    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        last = page[-1]
        next_cursor = _encode_cursor(
            {"t": last.created_at.strftime("%Y-%m-%dT%H:%M:%S.%f"), "id": last.id},
            fingerprint,
        )
    rules_by_id = {
        r.id: r
        for r in db.query(AlertRule)
        .filter(AlertRule.id.in_({a.rule_id for a in page}))
        .all()
    } if page else {}
    return {
        "alerts": [alert_summary(a, rules_by_id.get(a.rule_id)) for a in page],
        "next_cursor": next_cursor,
    }


# --------------------------------------------------------------------------- #
# cursors (keyset pagination with a query fingerprint, like the catalog)
# --------------------------------------------------------------------------- #
def _fingerprint(filters: dict) -> str:
    return hashlib.sha256(
        json.dumps(filters, sort_keys=True).encode()
    ).hexdigest()[:16]


def _encode_cursor(payload: dict, fingerprint: str) -> str:
    raw = json.dumps({**payload, "fp": fingerprint}).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str, fingerprint: str) -> tuple[datetime, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
        token_at = datetime.strptime(payload["t"], "%Y-%m-%dT%H:%M:%S.%f")
        token_id = payload["id"]
        token_fp = payload["fp"]
    except Exception as exc:  # malformed/undecodable cursor
        raise ApiError(422, "invalid_cursor", "malformed pagination cursor") from exc
    if token_fp != fingerprint:
        raise ApiError(
            422, "invalid_cursor",
            "cursor was issued for a different alert query (filter mismatch)",
        )
    return token_at, token_id


# --------------------------------------------------------------------------- #
# serialization
# --------------------------------------------------------------------------- #
def rule_payload(rule: AlertRule, created: bool) -> dict:
    return {
        "rule_id": rule.id,
        "created": created,
        "name": rule.name,
        "rule_version": rule.rule_version,
        "revision": rule.revision,
        "status": rule.status,
        "conditions": {
            "min_quality": rule.min_quality,
            "min_stations": rule.min_stations,
            "max_rms_s": rule.max_rms_s,
            "window_start": iso_z(rule.window_start) if rule.window_start else None,
            "window_end": iso_z(rule.window_end) if rule.window_end else None,
            "center_latitude": rule.center_latitude,
            "center_longitude": rule.center_longitude,
            "region_radius_km": rule.region_radius_km,
        },
        "merge_interval_s": rule.merge_interval_s,
        "label": rule.label,
        "created_at": iso_z(rule.created_at),
        "published_at": iso_z(rule.published_at) if rule.published_at else None,
        "retired_at": iso_z(rule.retired_at) if rule.retired_at else None,
    }


def _rule_summary(rule: AlertRule) -> dict:
    return {
        "rule_id": rule.id,
        "name": rule.name,
        "rule_version": rule.rule_version,
        "status": rule.status,
    }


def evaluation_summary(evaluation: AlertEvaluation) -> dict:
    return {
        "evaluation_id": evaluation.id,
        "compilation_id": evaluation.compilation_id,
        "request": evaluation.request,
        "effective_rule_ids": list(evaluation.effective_rule_ids),
        "entry_count": evaluation.entry_count,
        "hit_count": evaluation.hit_count,
        "evaluation_count": evaluation.evaluation_count,
        "created_at": iso_z(evaluation.created_at),
        "last_evaluated_at": (
            iso_z(evaluation.last_evaluated_at)
            if evaluation.last_evaluated_at
            else None
        ),
    }


def evaluation_payload(
    db: Session,
    evaluation: AlertEvaluation,
    created: bool,
    new_hits: list[AlertHit],
) -> dict:
    rule_ids = list(evaluation.effective_rule_ids)
    rules = (
        db.query(AlertRule).filter(AlertRule.id.in_(rule_ids)).all()
        if rule_ids
        else []
    )
    rules.sort(key=lambda r: (r.name, r.rule_version, r.id))
    hits: list[AlertHit] = []
    if rule_ids:
        hits = (
            db.query(AlertHit)
            .join(CatalogEntry, AlertHit.entry_id == CatalogEntry.id)
            .filter(
                AlertHit.compilation_id == evaluation.compilation_id,
                AlertHit.rule_id.in_(rule_ids),
            )
            .order_by(AlertHit.rule_id, CatalogEntry.origin_time, AlertHit.entry_id)
            .all()
        )
    payload = evaluation_summary(evaluation)
    payload.update(
        {
            "created": created,
            "rules": [_rule_summary(r) for r in rules],
            "matched_count": len(hits),
            "new_hit_count": len(new_hits),
            "hits": [hit_payload(h) for h in hits],
        }
    )
    return payload


def hit_payload(hit: AlertHit) -> dict:
    snap = hit.entry_snapshot
    return {
        "hit_id": hit.id,
        "rule_id": hit.rule_id,
        "alert_id": hit.alert_id,
        "evaluation_id": hit.evaluation_id,
        "compilation_id": hit.compilation_id,
        "entry_id": hit.entry_id,
        "disposition": hit.disposition,
        "reason": hit.reason,
        "entry": {
            "origin_time": snap["origin_time"],
            "latitude": round(snap["latitude"], 5),
            "longitude": round(snap["longitude"], 5),
            "depth_km": round(snap["depth_km"], 2),
            "rms_residual_s": round(snap["rms_residual_s"], 4),
            "station_count": snap["station_count"],
            "quality": snap["quality"],
        },
        "created_at": iso_z(hit.created_at),
    }


def alert_summary(alert: Alert, rule: AlertRule | None) -> dict:
    return {
        "alert_id": alert.id,
        "rule_id": alert.rule_id,
        "rule_name": rule.name if rule else None,
        "rule_version": rule.rule_version if rule else None,
        "status": alert.status,
        "hit_count": alert.hit_count,
        "best_quality": alert.best_quality,
        "max_station_count": alert.max_station_count,
        "min_rms_s": round(alert.min_rms_s, 4),
        "first_hit_at": iso_z(alert.first_hit_origin_time),
        "last_hit_at": iso_z(alert.last_hit_origin_time),
        "created_at": iso_z(alert.created_at),
        "updated_at": iso_z(alert.updated_at),
    }


def alert_payload(db: Session, alert: Alert) -> dict:
    rule = db.get(AlertRule, alert.rule_id)
    hits = (
        db.query(AlertHit)
        .filter_by(alert_id=alert.id)
        .order_by(AlertHit.created_at, AlertHit.id)
        .all()
    )
    events = (
        db.query(AlertEvent)
        .filter_by(alert_id=alert.id)
        .order_by(AlertEvent.seq)
        .all()
    )
    payload = alert_summary(alert, rule)
    payload.update(
        {
            "status_since": iso_z(events[-1].created_at) if events else None,
            "hits": [hit_payload(h) for h in hits],
            "events": [
                {
                    "seq": e.seq,
                    "kind": e.kind,
                    "reason": e.reason,
                    "hit_id": e.hit_id,
                    "detail": e.detail,
                    "created_at": iso_z(e.created_at),
                }
                for e in events
            ],
        }
    )
    return payload

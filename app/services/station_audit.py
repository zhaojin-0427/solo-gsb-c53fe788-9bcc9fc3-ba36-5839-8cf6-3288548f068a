"""Station impact audit (leave-one-station-out re-association).

For one event of a completed run the service freezes the original associated
pick subset, the velocity model, the calibration version (with its terms) and
the processing parameters, then -- once per participating station -- removes
ALL of that station's P/S picks and re-runs the very same deterministic
association + location used by ``process``/``replay``.  Each re-run is compared
against the frozen baseline event:

* whether the event survives at all (``event_lost`` vs ``survived``),
* origin time / latitude / longitude / depth / RMS deltas and quality grade,
* the associated pick set (retained vs dropped under the removed station),
* the unresolvable reason when the event disappears.

Results are presented in a stable order -- event disappearance, then quality
downgrade depth, then RMS increase, then station code -- and the stations whose
removal loses the event or degrades its grade are flagged as key stations.

Like the cross-run catalog this is a read-only analysis layer: it never writes
runs/events/picks and never touches catalog content.  Reports are immutable and
idempotent (canonical request hash), and both the report list and the per-audit
station results support keyset pagination.
"""
from __future__ import annotations

import base64
import hashlib
import json
from collections import Counter
from datetime import datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..models import (
    CalibrationTerm,
    Event,
    EventPick,
    Pick,
    Run,
    StationAudit,
    StationAuditResult,
    StationCalibration,
    VelocityModel,
    Waveform,
)
from . import catalog as cat
from .association import (
    REASON_INSUFFICIENT_STATIONS,
    REASON_ISOLATED,
    REASON_LOCATION_FAILED,
    REASON_RESIDUAL_EXCEEDED,
    PickIn,
    distance_km,
    run_association,
)
from .pipeline import (
    ApiError,
    canonical_layers,
    from_epoch,
    iso_z,
    to_epoch,
)
from .traveltime import LayeredModel

OUTCOME_SURVIVED = "survived"
OUTCOME_EVENT_LOST = "event_lost"

# Outcome presentation severity (lower sorts first).
_SEVERITY_LOST = 0
_SEVERITY_DOWNGRADED = 1
_SEVERITY_STABLE = 2

# Summary tie-break rank for unresolvable reasons (same order the association
# engine itself uses to escalate a pick's disposition).
_REASON_RANK = {
    REASON_ISOLATED: 0,
    REASON_INSUFFICIENT_STATIONS: 1,
    REASON_LOCATION_FAILED: 2,
    REASON_RESIDUAL_EXCEEDED: 3,
}

DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 200


# --------------------------------------------------------------------------- #
# audit creation (freeze -> per-station re-run -> persist, idempotent)
# --------------------------------------------------------------------------- #
def create_audit(db: Session, event_id: str, min_keep_stations: int) -> tuple[dict, bool]:
    event = db.get(Event, event_id)
    if event is None:
        raise ApiError(404, "event_not_found", f"event {event_id} not found")
    run = db.get(Run, event.run_id)
    if run is None:  # pragma: no cover - FK integrity makes this unreachable
        raise ApiError(404, "run_not_found", f"run {event.run_id} not found")

    canon_request = {
        "event_id": event.id,
        "run_id": run.id,
        "min_keep_stations": int(min_keep_stations),
    }
    request_hash = hashlib.sha256(
        json.dumps(canon_request, sort_keys=True).encode()
    ).hexdigest()

    existing = db.query(StationAudit).filter_by(request_hash=request_hash).one_or_none()
    if existing is not None:
        db.commit()
        return audit_payload(db, existing, created=False), False

    frozen, baseline = _freeze_context(db, event, run, min_keep_stations)
    station_codes = sorted({p["station"] for p in frozen["picks"]})
    # Removing one station must still leave enough stations for an acceptable
    # event; otherwise every leave-one-out re-run is trivially unresolvable.
    if len(station_codes) - 1 < min_keep_stations:
        raise ApiError(
            422,
            "min_keep_stations_too_high",
            f"event has {len(station_codes)} participating stations; "
            f"min_keep_stations must be <= {len(station_codes) - 1} so the event "
            "can survive removing one station",
        )
    if not frozen["picks"]:  # pragma: no cover - every event has picks
        raise ApiError(422, "audit_event_has_no_picks", "event has no associated picks")

    params_hash = _params_hash(run, frozen, min_keep_stations)
    results = _run_leave_one_out(frozen, baseline)
    ordered = sorted(
        results,
        key=lambda r: (
            r["_sort"][0],
            r["_sort"][1],
            r["_sort"][2],
            r["station"],
        ),
    )
    key_stations = [r["station"] for r in ordered if r["is_key_station"]]

    audit = StationAudit(
        request_hash=request_hash,
        request=canon_request,
        params_hash=params_hash,
        event_id=event.id,
        run_id=run.id,
        min_keep_stations=min_keep_stations,
        frozen=_frozen_json(frozen),
        baseline=_baseline_json(baseline),
        station_count=len(station_codes),
        result_count=len(ordered),
        key_station_count=len(key_stations),
    )
    db.add(audit)
    try:
        db.flush()
    except IntegrityError:
        # Concurrent identical request: collapse onto the surviving audit.
        db.rollback()
        winner = db.query(StationAudit).filter_by(request_hash=request_hash).one()
        return audit_payload(db, winner, created=False), False

    for idx, r in enumerate(ordered):
        db.add(
            StationAuditResult(
                audit_id=audit.id,
                station=r["station"],
                sort_index=idx,
                outcome=r["outcome"],
                is_key_station=1 if r["is_key_station"] else 0,
                detail=_result_detail_json(r),
            )
        )
    db.commit()
    db.refresh(audit)
    return audit_payload(db, audit, created=True), True


def _freeze_context(
    db: Session, event: Event, run: Run, min_keep_stations: int
) -> tuple[dict, dict]:
    """Snapshot everything the audit re-runs must depend on."""
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
            "weight": p.weight,
            "residual_s": ep.residual_s,
        }
        for ep, p in rows
    ]
    picks.sort(key=lambda q: (q["t"], q["station"], q["phase"], q["pick_id"]))

    max_residual_s = float(run.params["max_residual_s"])
    frozen = {
        "run_id": run.id,
        "ingestion_id": run.ingestion_id,
        "model_row": model_row,
        "model": model,
        "model_snapshot": {"name": model_row.name, "version": model_row.version,
                           "layers": layers},
        "calibration_snapshot": calibration,
        "corrections": corrections,
        "params": {
            "max_residual_s": max_residual_s,
            "min_stations": int(run.params["min_stations"]),
        },
        "min_keep_stations": int(min_keep_stations),
        "stations": stations,
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
        "pick_ids": [q["pick_id"] for q in picks],
        "stations": sorted({q["station"] for q in picks}),
        "_pick_rows": picks,
    }
    return frozen, baseline


def _params_hash(
    run: Run, frozen: dict, min_keep_stations: int
) -> str:
    """Canonical hash binding the report to event/run and the frozen parameters."""
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
        "min_keep_stations": int(min_keep_stations),
    }
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()


def _run_leave_one_out(frozen: dict, baseline: dict) -> list[dict]:
    """Re-associate/locate once per participating station."""
    model = frozen["model"]
    stations = frozen["stations"]
    corrections = frozen["corrections"]
    max_residual_s = frozen["params"]["max_residual_s"]
    min_keep = frozen["min_keep_stations"]
    base_rank = cat.QUALITY_RANK[baseline["quality"]]

    by_station: dict[str, list[dict]] = {}
    for q in frozen["picks"]:
        by_station.setdefault(q["station"], []).append(q)

    results = []
    for station in sorted(by_station):
        removed_ids = {q["pick_id"] for q in by_station[station]}
        remaining = [q for q in frozen["picks"] if q["pick_id"] not in removed_ids]
        pick_ins = [
            PickIn(
                id=q["pick_id"],
                station=q["station"],
                phase=q["phase"],
                t=q["t"],
                weight=q["weight"],
                source=q["source"],
            )
            for q in remaining
        ]
        events_out, reasons = run_association(
            pick_ins,
            stations,
            model,
            max_residual_s=max_residual_s,
            min_stations=min_keep,
            corrections=corrections,
        )
        if events_out:
            results.append(
                _survived_result(
                    station, removed_ids, remaining, events_out,
                    baseline, base_rank,
                )
            )
        else:
            results.append(
                _lost_result(station, removed_ids, remaining, reasons)
            )
    return results


def _match_survivor(events_out, baseline: dict):
    """Pick the re-located event corresponding to the frozen baseline event.

    The frozen subset is one tight event cluster, so normally exactly one event
    comes back; the deterministic selection (most stations, closest origin,
    closest hypocenter) makes the choice robust regardless.
    """
    def key(ev):
        return (
            -ev.station_count,
            abs(ev.t0 - baseline["t0"]),
            distance_km(ev.lat, ev.lon, baseline["latitude"], baseline["longitude"]),
            ev.t0,
            ev.lat,
            ev.lon,
        )

    return min(events_out, key=key)


def _survived_result(
    station, removed_ids, remaining, events_out, baseline, base_rank
) -> dict:
    ev = _match_survivor(events_out, baseline)
    associated_ids = {a.pick_id for a in ev.assignments}
    remaining_ids = {q["pick_id"] for q in remaining}
    retained_ids = remaining_ids & associated_ids
    dropped_ids = remaining_ids - associated_ids
    # Picks of other events (can only occur on the degenerate multi-event path)
    # are neither retained by the survivor nor "dropped" from the audit event.
    downgrade = cat.QUALITY_RANK[ev.quality] - base_rank
    rms_increase = ev.rms_s - baseline["rms_s"]
    severity = _SEVERITY_DOWNGRADED if downgrade > 0 else _SEVERITY_STABLE
    return {
        "station": station,
        "outcome": OUTCOME_SURVIVED,
        "is_key_station": downgrade > 0,
        "removed_pick_ids": sorted(removed_ids),
        "solution": {
            "origin_time": iso_z(from_epoch(ev.t0)),
            "latitude": ev.lat,
            "longitude": ev.lon,
            "depth_km": ev.depth_km,
            "rms_s": ev.rms_s,
            "station_count": ev.station_count,
            "quality": ev.quality,
        },
        "delta": {
            "origin_time_s": ev.t0 - baseline["t0"],
            "latitude": ev.lat - baseline["latitude"],
            "longitude": ev.lon - baseline["longitude"],
            "depth_km": ev.depth_km - baseline["depth_km"],
            "rms_s": ev.rms_s - baseline["rms_s"],
            "station_count": ev.station_count - baseline["station_count"],
        },
        "rms_increase_s": rms_increase,
        "quality_downgraded": downgrade > 0,
        "associated_pick_ids": sorted(associated_ids),
        "retained_pick_ids": sorted(retained_ids),
        "dropped_pick_ids": sorted(dropped_ids),
        "stations": sorted({q["station"] for q in remaining if q["pick_id"] in associated_ids}),
        "unresolvable_reason": None,
        "reason_counts": {},
        "_sort": (severity, -downgrade, -rms_increase),
    }


def _lost_result(station, removed_ids, remaining, reasons) -> dict:
    counts = Counter(reasons.values())
    if counts:
        reason = max(
            counts.items(),
            key=lambda kv: (kv[1], _REASON_RANK.get(kv[0], -1), kv[0]),
        )[0]
    else:  # pragma: no cover - a non-empty subset always carries dispositions
        reason = REASON_INSUFFICIENT_STATIONS
    remaining_ids = {q["pick_id"] for q in remaining}
    return {
        "station": station,
        "outcome": OUTCOME_EVENT_LOST,
        "is_key_station": True,
        "removed_pick_ids": sorted(removed_ids),
        "solution": None,
        "delta": None,
        "rms_increase_s": None,
        "quality_downgraded": True,
        "associated_pick_ids": [],
        "retained_pick_ids": [],
        "dropped_pick_ids": sorted(remaining_ids),
        "stations": [],
        "unresolvable_reason": reason,
        "reason_counts": dict(sorted(counts.items())),
        "_sort": (_SEVERITY_LOST, 0, 0.0),
    }


# --------------------------------------------------------------------------- #
# JSON snapshots (frozen context is stored verbatim with the report)
# --------------------------------------------------------------------------- #
def _result_detail_json(result: dict) -> dict:
    """Persistable per-station outcome (drop the in-memory sort tuple)."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


def _frozen_json(frozen: dict) -> dict:
    return {
        "ingestion_id": frozen["ingestion_id"],
        "model": frozen["model_snapshot"],
        "calibration": frozen["calibration_snapshot"],
        "params": frozen["params"],
        "min_keep_stations": frozen["min_keep_stations"],
        "stations": [
            {"station": code, "latitude": lat, "longitude": lon, "elevation_m": elev}
            for code, (lat, lon, elev) in sorted(frozen["stations"].items())
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
        "picks": sorted(
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
                for q in baseline["_pick_rows"]
            ),
            key=lambda p: (p["time"], p["station"], p["phase"]),
        ),
    }


# --------------------------------------------------------------------------- #
# queries
# --------------------------------------------------------------------------- #
def get_audit(db: Session, audit_id: str) -> dict:
    row = db.get(StationAudit, audit_id)
    if row is None:
        raise ApiError(
            404, "station_audit_not_found", f"station audit {audit_id} not found"
        )
    return audit_payload(db, row, created=False)


def list_audits(
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

    q = db.query(StationAudit)
    if event_id is not None:
        q = q.filter(StationAudit.event_id == event_id)
    if run_id is not None:
        q = q.filter(StationAudit.run_id == run_id)
    if cursor_at is not None:
        q = q.filter(
            (StationAudit.created_at > cursor_at)
            | ((StationAudit.created_at == cursor_at) & (StationAudit.id > cursor_id))
        )
    q = q.order_by(StationAudit.created_at, StationAudit.id).limit(limit + 1)
    rows = q.all()

    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        last = page[-1]
        next_cursor = _encode_list_cursor(last.created_at, last.id, fingerprint)
    return {
        "audits": [audit_summary(a) for a in page],
        "next_cursor": next_cursor,
    }


def list_results(
    db: Session, audit_id: str, limit: int, cursor: str | None
) -> dict:
    audit = db.get(StationAudit, audit_id)
    if audit is None:
        raise ApiError(
            404, "station_audit_not_found", f"station audit {audit_id} not found"
        )
    fingerprint = hashlib.sha256(audit_id.encode()).hexdigest()[:16]
    cursor_index = None
    if cursor is not None:
        cursor_index = _decode_result_cursor(cursor, fingerprint)

    q = db.query(StationAuditResult).filter_by(audit_id=audit.id)
    if cursor_index is not None:
        q = q.filter(StationAuditResult.sort_index > cursor_index)
    q = q.order_by(StationAuditResult.sort_index).limit(limit + 1)
    rows = q.all()

    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        next_cursor = _encode_result_cursor(page[-1].sort_index, fingerprint)
    return {
        "audit_id": audit.id,
        "event_id": audit.event_id,
        "run_id": audit.run_id,
        "result_count": audit.result_count,
        "results": [result_payload(r) for r in page],
        "next_cursor": next_cursor,
    }


# --------------------------------------------------------------------------- #
# cursors
# --------------------------------------------------------------------------- #
def _encode_list_cursor(created_at: datetime, audit_id: str, fingerprint: str) -> str:
    raw = json.dumps(
        {"t": created_at.strftime("%Y-%m-%dT%H:%M:%S.%f"), "id": audit_id,
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
            "cursor was issued for a different audit query (event/run filter "
            "mismatch)",
        )
    return token_at, token_id


def _encode_result_cursor(sort_index: int, fingerprint: str) -> str:
    raw = json.dumps({"i": sort_index, "fp": fingerprint}).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_result_cursor(cursor: str, fingerprint: str) -> int:
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
            "cursor was issued for a different station audit",
        )
    return token_i


# --------------------------------------------------------------------------- #
# serialization
# --------------------------------------------------------------------------- #
def _solution_payload(solution: dict | None) -> dict | None:
    if solution is None:
        return None
    return {
        "origin_time": solution["origin_time"],
        "latitude": round(solution["latitude"], 5),
        "longitude": round(solution["longitude"], 5),
        "depth_km": round(solution["depth_km"], 2),
        "rms_residual_s": round(solution["rms_s"], 4),
        "station_count": solution["station_count"],
        "quality": solution["quality"],
    }


def _delta_payload(delta: dict | None) -> dict | None:
    if delta is None:
        return None
    return {
        "origin_time_s": round(delta["origin_time_s"], 4),
        "latitude": round(delta["latitude"], 5),
        "longitude": round(delta["longitude"], 5),
        "depth_km": round(delta["depth_km"], 2),
        "rms_s": round(delta["rms_s"], 4),
        "station_count": delta["station_count"],
    }


def result_payload(row: StationAuditResult) -> dict:
    d = row.detail
    return {
        "station": row.station,
        "sort_index": row.sort_index,
        "outcome": row.outcome,
        "is_key_station": bool(row.is_key_station),
        "quality_downgraded": bool(d.get("quality_downgraded", False)),
        "rms_increase_s": (
            round(d["rms_increase_s"], 4) if d.get("rms_increase_s") is not None else None
        ),
        "solution": _solution_payload(d.get("solution")),
        "delta": _delta_payload(d.get("delta")),
        "removed_pick_ids": d.get("removed_pick_ids", []),
        "associated_pick_ids": d.get("associated_pick_ids", []),
        "retained_pick_ids": d.get("retained_pick_ids", []),
        "dropped_pick_ids": d.get("dropped_pick_ids", []),
        "stations": d.get("stations", []),
        "unresolvable_reason": d.get("unresolvable_reason"),
        "reason_counts": d.get("reason_counts", {}),
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


def audit_summary(audit: StationAudit) -> dict:
    return {
        "audit_id": audit.id,
        "event_id": audit.event_id,
        "run_id": audit.run_id,
        "params_hash": audit.params_hash,
        "request": audit.request,
        "min_keep_stations": audit.min_keep_stations,
        "station_count": audit.station_count,
        "result_count": audit.result_count,
        "key_station_count": audit.key_station_count,
        "created_at": iso_z(audit.created_at),
    }


def audit_payload(db: Session, audit: StationAudit, created: bool) -> dict:
    result_rows = (
        db.query(StationAuditResult)
        .filter_by(audit_id=audit.id)
        .order_by(StationAuditResult.sort_index)
        .all()
    )
    key_stations = [r.station for r in result_rows if r.is_key_station]
    return {
        "audit_id": audit.id,
        "created": created,
        "event_id": audit.event_id,
        "run_id": audit.run_id,
        "ingestion_id": audit.frozen["ingestion_id"],
        "params_hash": audit.params_hash,
        "request": audit.request,
        "frozen": audit.frozen,
        "baseline": _baseline_payload(audit.baseline),
        "min_keep_stations": audit.min_keep_stations,
        "station_count": audit.station_count,
        "result_count": audit.result_count,
        "key_station_count": audit.key_station_count,
        "key_stations": key_stations,
        "results": [result_payload(r) for r in result_rows],
        "created_at": iso_z(audit.created_at),
    }

"""Relational schema (SQLAlchemy)."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)

from .database import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.utcnow()


class VelocityModel(Base):
    """Immutable, versioned 1-D layered velocity model."""

    __tablename__ = "velocity_models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(64), nullable=False)
    version = Column(String(64), nullable=False)
    layers = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    # Points at the single currently active calibration.  Not a foreign key by
    # design (it would form a cycle with station_calibrations.model_id); the
    # id is a station_calibrations.id maintained atomically during activation.
    # The partial unique index on station_calibrations is the hard guarantee.
    active_calibration_id = Column(String(36), nullable=True)

    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_name_version"),)


class Ingestion(Base):
    """One uploaded waveform window (idempotent via window_hash)."""

    __tablename__ = "ingestions"

    id = Column(String(36), primary_key=True, default=_uuid)
    window_hash = Column(String(64), nullable=False, unique=True, index=True)
    summary = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=_utcnow)


class Waveform(Base):
    __tablename__ = "waveforms"

    id = Column(String(36), primary_key=True, default=_uuid)
    ingestion_id = Column(String(36), ForeignKey("ingestions.id"), nullable=False, index=True)
    station = Column(String(32), nullable=False)
    sampling_rate = Column(Float, nullable=False)
    start_time = Column(DateTime, nullable=False)  # naive UTC
    npts = Column(Integer, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation_m = Column(Float, nullable=False)
    data_sha256 = Column(String(64), nullable=False)


class Pick(Base):
    __tablename__ = "picks"

    id = Column(String(36), primary_key=True, default=_uuid)
    ingestion_id = Column(String(36), ForeignKey("ingestions.id"), nullable=False, index=True)
    station = Column(String(32), nullable=False)
    phase = Column(String(1), nullable=True)  # 'P'/'S', NULL for unlabeled STA/LTA picks
    time_utc = Column(DateTime, nullable=False)  # naive UTC
    source = Column(String(8), nullable=False)  # 'stalta' | 'manual'
    weight = Column(Float, nullable=False, default=1.0)
    snr = Column(Float, nullable=True)


class Run(Base):
    """One association+location pass over an ingestion with a model version."""

    __tablename__ = "runs"

    id = Column(String(36), primary_key=True, default=_uuid)
    ingestion_id = Column(String(36), ForeignKey("ingestions.id"), nullable=False, index=True)
    model_id = Column(Integer, ForeignKey("velocity_models.id"), nullable=False)
    # NULL = uncalibrated baseline run; otherwise arrivals were corrected with
    # this (immutable, published) calibration before association/location.
    calibration_id = Column(
        String(36), ForeignKey("station_calibrations.id"), nullable=True
    )
    params = Column(JSON, nullable=False)
    params_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        UniqueConstraint("ingestion_id", "model_id", "params_hash", name="uq_run_identity"),
    )


class StationCalibration(Base):
    """Per-station P/S arrival corrections for one velocity model.

    Lifecycle: ``candidate`` -> (atomic activation) -> ``active``; when another
    version activates the old one becomes ``published`` and is immutable.
    """

    __tablename__ = "station_calibrations"

    id = Column(String(36), primary_key=True, default=_uuid)
    model_id = Column(Integer, ForeignKey("velocity_models.id"), nullable=False, index=True)
    calibration_version = Column(String(64), nullable=False)
    status = Column(String(16), nullable=False, default="candidate", index=True)
    request = Column(JSON, nullable=False)
    request_hash = Column(String(64), nullable=False)
    # Pooled + per-phase sample counts, coverage, train/holdout RMS (baseline
    # and calibrated); produced by the Huber estimator at creation time.
    metrics = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    activated_at = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "model_id", "calibration_version", name="uq_calibration_model_version"
        ),
        UniqueConstraint("request_hash", name="uq_calibration_request"),
        # At most one active calibration per velocity model.
        Index(
            "uq_calibration_active_per_model",
            "model_id",
            unique=True,
            postgresql_where=text("status = 'active'"),
            sqlite_where=text("status = 'active'"),
        ),
    )


class CalibrationTerm(Base):
    """One (station, phase) Huber offset of a calibration version."""

    __tablename__ = "calibration_terms"

    id = Column(String(36), primary_key=True, default=_uuid)
    calibration_id = Column(
        String(36),
        ForeignKey("station_calibrations.id"),
        nullable=False,
        index=True,
    )
    station = Column(String(32), nullable=False)
    phase = Column(String(1), nullable=False)  # 'P' or 'S'
    offset_s = Column(Float, nullable=False)
    uncertainty_s = Column(Float, nullable=False)
    sample_count = Column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "calibration_id", "station", "phase", name="uq_term_calibration_station_phase"
        ),
    )


class Event(Base):
    __tablename__ = "events"

    id = Column(String(36), primary_key=True, default=_uuid)
    run_id = Column(String(36), ForeignKey("runs.id"), nullable=False, index=True)
    origin_time = Column(DateTime, nullable=False)  # naive UTC
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    depth_km = Column(Float, nullable=False)
    rms_s = Column(Float, nullable=False)
    station_count = Column(Integer, nullable=False)
    quality = Column(String(1), nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)


class EventPick(Base):
    __tablename__ = "event_picks"

    id = Column(String(36), primary_key=True, default=_uuid)
    event_id = Column(String(36), ForeignKey("events.id"), nullable=False, index=True)
    pick_id = Column(String(36), ForeignKey("picks.id"), nullable=False)
    phase = Column(String(1), nullable=False)
    residual_s = Column(Float, nullable=False)
    weight = Column(Float, nullable=False)


class PickStatus(Base):
    """Per-run disposition of every pick (associated / unassociated+reason)."""

    __tablename__ = "pick_statuses"

    id = Column(String(36), primary_key=True, default=_uuid)
    run_id = Column(String(36), ForeignKey("runs.id"), nullable=False, index=True)
    pick_id = Column(String(36), ForeignKey("picks.id"), nullable=False)
    status = Column(String(16), nullable=False)  # 'associated' | 'unassociated'
    reason = Column(String(32), nullable=True)

    __table_args__ = (UniqueConstraint("run_id", "pick_id", name="uq_run_pick"),)


# --------------------------------------------------------------------------- #
# cross-run event catalog
#
# A compilation deterministically clusters events of several completed runs
# (origin time / hypocenter / station evidence); every surviving cluster
# becomes one immutable catalog entry.  Entries only move through the
# draft -> published -> withdrawn lifecycle; their content never changes.
# --------------------------------------------------------------------------- #
class CatalogCompilation(Base):
    """One idempotent cross-run clustering request and its resulting entries."""

    __tablename__ = "catalog_compilations"

    id = Column(String(36), primary_key=True, default=_uuid)
    request_hash = Column(String(64), nullable=False, unique=True, index=True)
    request = Column(JSON, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    entry_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=_utcnow)


class CatalogEntry(Base):
    """One immutable cataloged earthquake; master solution + lifecycle status."""

    __tablename__ = "catalog_entries"

    id = Column(String(36), primary_key=True, default=_uuid)
    compilation_id = Column(
        String(36), ForeignKey("catalog_compilations.id"), nullable=False, index=True
    )
    # Lifecycle: draft -> published -> withdrawn.  Revision starts at 1 and
    # bumps on each actual state transition (optimistic concurrency token).
    revision = Column(Integer, nullable=False, default=1)
    status = Column(String(16), nullable=False, default="draft", index=True)

    # Master solution snapshot (highest quality, then smallest RMS, then
    # stable run/event id tie-break).  Content columns are never updated.
    origin_time = Column(DateTime, nullable=False)  # naive UTC
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    depth_km = Column(Float, nullable=False)
    rms_s = Column(Float, nullable=False)
    station_count = Column(Integer, nullable=False)
    quality = Column(String(1), nullable=False)
    master_run_id = Column(String(36), ForeignKey("runs.id"), nullable=False)
    master_event_id = Column(String(36), ForeignKey("events.id"), nullable=False)

    stations = Column(JSON, nullable=False, default=list)        # sorted union
    shared_stations = Column(JSON, nullable=False, default=list)  # intersection
    source_count = Column(Integer, nullable=False, default=1)

    created_at = Column(DateTime, nullable=False, default=_utcnow)
    published_at = Column(DateTime, nullable=True)
    withdrawn_at = Column(DateTime, nullable=True)
    withdraw_reason = Column(String(500), nullable=True)

    __table_args__ = (Index("ix_catalog_entries_time_id", "origin_time", "id"),)


class CatalogEntrySource(Base):
    """One source (run, event) merged into a catalog entry (provenance chain)."""

    __tablename__ = "catalog_entry_sources"

    id = Column(String(36), primary_key=True, default=_uuid)
    entry_id = Column(
        String(36), ForeignKey("catalog_entries.id"), nullable=False, index=True
    )
    run_id = Column(String(36), ForeignKey("runs.id"), nullable=False, index=True)
    event_id = Column(String(36), ForeignKey("events.id"), nullable=False, index=True)
    is_master = Column(Integer, nullable=False, default=0)  # 0/1 (SQLite portability)

    # Solution snapshot so the source chain survives any future change.
    origin_time = Column(DateTime, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    depth_km = Column(Float, nullable=False)
    rms_s = Column(Float, nullable=False)
    station_count = Column(Integer, nullable=False)
    quality = Column(String(1), nullable=False)
    # Snapshot of the producing run's version coordinates.
    model_name = Column(String(64), nullable=False)
    model_version = Column(String(64), nullable=False)
    calibration_version = Column(String(64), nullable=True)
    calibration_status = Column(String(16), nullable=True)
    # Difference against the master solution (origin_time_s / lat / lon / depth / rms).
    delta = Column(JSON, nullable=False)

    __table_args__ = (
        UniqueConstraint("entry_id", "run_id", "event_id", name="uq_entry_source"),
    )


# --------------------------------------------------------------------------- #
# station impact audit (leave-one-station-out)
#
# An audit freezes one completed event's original picks, velocity model,
# calibration version and processing parameters, then re-runs association and
# location once per participating station after removing ALL of that station's
# P/S picks from the frozen subset.  Nothing in ingestions/runs/events/picks or
# the catalog is ever modified: audits are an immutable, idempotent, read-only
# analysis layer, like the catalog.
# --------------------------------------------------------------------------- #
class StationAudit(Base):
    """One leave-one-station-out audit bound to an event/run/params hash."""

    __tablename__ = "station_audits"

    id = Column(String(36), primary_key=True, default=_uuid)
    # Idempotency key: canonical content hash of (event, run, min_keep_stations).
    request_hash = Column(String(64), nullable=False, unique=True, index=True)
    request = Column(JSON, nullable=False)
    params_hash = Column(String(64), nullable=False)  # canonical audit params hash
    event_id = Column(String(36), ForeignKey("events.id"), nullable=False, index=True)
    run_id = Column(String(36), ForeignKey("runs.id"), nullable=False, index=True)
    min_keep_stations = Column(Integer, nullable=False)

    # Frozen analysis context (model/calibration versions, processing params
    # and the original associated pick subset).  Snapshotted at creation time,
    # so audits are reproducible even if the surrounding data evolves.
    frozen = Column(JSON, nullable=False)
    # Snapshot of the original (baseline) event solution.
    baseline = Column(JSON, nullable=False)

    station_count = Column(Integer, nullable=False)
    result_count = Column(Integer, nullable=False, default=0)
    key_station_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        Index("ix_station_audits_event", "event_id"),
        Index("ix_station_audits_run", "run_id"),
        Index("ix_station_audits_created", "created_at", "id"),
    )


class StationAuditResult(Base):
    """Outcome for one removed station within a station impact audit.

    Rows are written once in their stable presentation order (outcome
    severity, quality downgrade, RMS increase, station code), so
    ``(audit_id, sort_index)`` doubles as the keyset-pagination order.
    """

    __tablename__ = "station_audit_results"

    id = Column(String(36), primary_key=True, default=_uuid)
    audit_id = Column(
        String(36), ForeignKey("station_audits.id"), nullable=False, index=True
    )
    station = Column(String(32), nullable=False)
    sort_index = Column(Integer, nullable=False)
    outcome = Column(String(32), nullable=False, index=True)  # survived / event_lost
    is_key_station = Column(Integer, nullable=False, default=0)  # 0/1
    # Full outcome detail: delta vs baseline, relocated solution (or null),
    # associated pick sets and the unresolvable reason when the event is lost.
    detail = Column(JSON, nullable=False)

    __table_args__ = (
        UniqueConstraint("audit_id", "station", name="uq_audit_result_station"),
        UniqueConstraint("audit_id", "sort_index", name="uq_audit_result_order"),
        Index("ix_audit_results_audit_sort", "audit_id", "sort_index"),
    )


# --------------------------------------------------------------------------- #
# P-wave first-motion focal mechanism inversion
#
# For one event of a completed run the client submits the event's associated P
# picks with first-motion polarities and weights.  The service computes ray
# azimuth/takeoff through the run's layered velocity model, grid-searches
# double-couple (strike, dip, rake) on a 10-degree grid and keeps every
# mechanism whose weighted polarity misfit equals the minimum (a nodal --
# zero-amplitude -- prediction counts as a misfit).  Mechanisms equivalent
# under swapping the two nodal planes are deduplicated; each surviving
# solution reports both planes, its misfit weight and the per-station
# observed/predicted polarities.  Like the catalog and the station audit this
# layer never modifies ingestions/runs/events/picks or catalog content.
# --------------------------------------------------------------------------- #
class FocalMechanismInversion(Base):
    """One idempotent first-motion inversion bound to an event/model/polarity set."""

    __tablename__ = "focal_mechanism_inversions"

    id = Column(String(36), primary_key=True, default=_uuid)
    # Idempotency key: canonical hash of (event, model version, normalized
    # polarity set).  Concurrent identical requests collapse on this index.
    request_hash = Column(String(64), nullable=False, unique=True, index=True)
    request = Column(JSON, nullable=False)
    event_id = Column(String(36), ForeignKey("events.id"), nullable=False, index=True)
    run_id = Column(String(36), ForeignKey("runs.id"), nullable=False, index=True)
    # Frozen inversion context: velocity model snapshot, hypocenter, the
    # normalized polarity set and per-station ray geometry (azimuth/takeoff).
    frozen = Column(JSON, nullable=False)
    grid_step_deg = Column(Float, nullable=False)
    # Dedup rule used to produce ``solutions``: 1 = legacy release (canonical
    # key sign-canonicalized the moment tensor, wrongly merging M with -M);
    # 2 = corrected rule (plane-swap only, slip-reversed mechanisms retained).
    # Legacy rows are lazily repaired (and bumped to 2) on repeat submission or
    # any fetch, without changing the id/request hash/frozen context.
    dedup_rule_version = Column(Integer, nullable=False, default=2)
    station_count = Column(Integer, nullable=False)
    solution_count = Column(Integer, nullable=False)
    min_misfit_weight = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        Index("ix_focal_inversions_created", "created_at", "id"),
    )


class FocalMechanismSolution(Base):
    """One tied best-fitting double-couple solution of an inversion.

    Rows are written once in their stable presentation order (strike, dip,
    rake), so ``(inversion_id, sort_index)`` doubles as the keyset-pagination
    order.
    """

    __tablename__ = "focal_mechanism_solutions"

    id = Column(String(36), primary_key=True, default=_uuid)
    inversion_id = Column(
        String(36),
        ForeignKey("focal_mechanism_inversions.id"),
        nullable=False,
        index=True,
    )
    sort_index = Column(Integer, nullable=False)
    misfit_weight = Column(Float, nullable=False)
    misfit_count = Column(Integer, nullable=False)
    # Both nodal planes plus per-station observed/predicted polarities.
    detail = Column(JSON, nullable=False)

    __table_args__ = (
        UniqueConstraint("inversion_id", "sort_index", name="uq_focal_solution_order"),
        Index("ix_focal_solutions_inversion_sort", "inversion_id", "sort_index"),
    )


# --------------------------------------------------------------------------- #
# event location uncertainty assessment (station bootstrap)
#
# An assessment freezes one completed event's associated P/S picks (with the
# weights used by the original location), the velocity model, the calibration
# version and the processing parameters, then re-locates the event N times on
# seeded with-replacement resamples of the participating stations (repeated
# stations weighted by multiplicity).  Each resample records its solvable
# status and a structured failure reason; the report carries nearest-rank
# two-sided (1-alpha)/2 confidence intervals for origin time/latitude/
# longitude/depth, the sample covariance over the solvable resamples only and
# the solvable rate.  Like the catalog, the station audit and the focal
# mechanism inversion this layer never modifies ingestions/runs/events/picks
# or catalog content: reports and samples are immutable and read-only.
# --------------------------------------------------------------------------- #
class LocationUncertaintyAssessment(Base):
    """One idempotent bootstrap uncertainty assessment bound to an event."""

    __tablename__ = "location_uncertainty_assessments"

    id = Column(String(36), primary_key=True, default=_uuid)
    # Idempotency key: canonical content hash of (event, run, N, alpha, seed).
    # Concurrent identical requests collapse on this index.
    request_hash = Column(String(64), nullable=False, unique=True, index=True)
    request = Column(JSON, nullable=False)
    event_id = Column(String(36), ForeignKey("events.id"), nullable=False, index=True)
    run_id = Column(String(36), ForeignKey("runs.id"), nullable=False, index=True)

    n_resamples = Column(Integer, nullable=False)
    alpha = Column(Float, nullable=False)
    seed = Column(Integer, nullable=False)

    # Frozen analysis context (model/calibration snapshots, processing params,
    # station coordinates and the original associated pick subset with
    # weights) plus the baseline event solution; snapshotted at creation time
    # so the assessment is reproducible even if the surrounding data evolves.
    frozen = Column(JSON, nullable=False)
    baseline = Column(JSON, nullable=False)

    n_solved = Column(Integer, nullable=False)
    n_failed = Column(Integer, nullable=False)
    solvable_rate = Column(Float, nullable=False)
    # Aggregate structured failure reasons over the failed resamples.
    failure_reasons = Column(JSON, nullable=False, default=dict)
    # Nearest-rank two-sided (1-alpha)/2 intervals + solvable-only covariance.
    intervals = Column(JSON, nullable=False)
    covariance = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        Index("ix_location_uncertainty_created", "created_at", "id"),
    )


class LocationUncertaintySample(Base):
    """Outcome of one bootstrap resample within an assessment.

    Rows are written once in resample order, so ``(assessment_id,
    sample_index)`` doubles as the keyset-pagination order.
    """

    __tablename__ = "location_uncertainty_samples"

    id = Column(String(36), primary_key=True, default=_uuid)
    assessment_id = Column(
        String(36),
        ForeignKey("location_uncertainty_assessments.id"),
        nullable=False,
        index=True,
    )
    sample_index = Column(Integer, nullable=False)
    status = Column(String(16), nullable=False, index=True)  # solved | failed
    failure_reason = Column(String(32), nullable=True)
    # Resampled station multiset {station_code: multiplicity} and, for solved
    # resamples, the re-located solution snapshot.
    stations = Column(JSON, nullable=False)
    solution = Column(JSON, nullable=True)

    __table_args__ = (
        UniqueConstraint("assessment_id", "sample_index", name="uq_uncertainty_sample_order"),
        Index("ix_uncertainty_samples_assessment", "assessment_id", "sample_index"),
    )


# --------------------------------------------------------------------------- #
# catalog alert orchestration
#
# Administrators define immutable alert rule versions over the published
# catalog (quality grade / station count / RMS / origin-time window / circular
# region) and publish them with an expected_revision CAS.  Evaluating a
# catalog compilation against the published rules deterministically records
# one hit per (rule, entry); consecutive hits of a rule within its merge
# interval collapse into a single alert (suppression), possibly escalating an
# acknowledged/resolved alert back to open.  Alerts are acknowledged /
# resolved by operators; withdrawing a catalog entry closes only the
# unacknowledged (open) alerts that fired on it.  Every hit and every status
# transition is preserved in a per-alert audit chain.
# --------------------------------------------------------------------------- #
class AlertRule(Base):
    """One immutable alert rule version (draft -> published -> retired)."""

    __tablename__ = "alert_rules"

    id = Column(String(36), primary_key=True, default=_uuid)
    name = Column(String(64), nullable=False)
    rule_version = Column(String(64), nullable=False)
    # Revision starts at 1 and bumps on each lifecycle transition; publishing
    # and retiring require the client to present the current revision (CAS).
    revision = Column(Integer, nullable=False, default=1)
    status = Column(String(16), nullable=False, default="draft", index=True)
    # Canonical content + hash: same (name, rule_version) with identical
    # content is an idempotent replay, different content is a 409 conflict.
    request = Column(JSON, nullable=False)
    content_hash = Column(String(64), nullable=False)

    # Match conditions (immutable once created).  NULL = condition disabled.
    min_quality = Column(String(1), nullable=False, default="D")
    min_stations = Column(Integer, nullable=True)
    max_rms_s = Column(Float, nullable=True)
    window_start = Column(DateTime, nullable=True)  # naive UTC, inclusive
    window_end = Column(DateTime, nullable=True)    # naive UTC, exclusive
    center_latitude = Column(Float, nullable=True)
    center_longitude = Column(Float, nullable=True)
    region_radius_km = Column(Float, nullable=True)
    # Consecutive hits of this rule whose origin-time gap stays within this
    # interval merge into one alert instead of raising new ones.
    merge_interval_s = Column(Float, nullable=False)
    label = Column(String(200), nullable=True)

    created_at = Column(DateTime, nullable=False, default=_utcnow)
    published_at = Column(DateTime, nullable=True)
    retired_at = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("name", "rule_version", name="uq_alert_rule_name_version"),
        Index("ix_alert_rules_created", "created_at", "id"),
    )


class AlertEvaluation(Base):
    """One idempotent evaluation of a catalog compilation against alert rules.

    Keyed by the canonical request hash: the first call creates the record
    (201), identical repeats run an idempotent re-pass (200) that only records
    hits for matches not seen before (e.g. entries published meanwhile).
    """

    __tablename__ = "alert_evaluations"

    id = Column(String(36), primary_key=True, default=_uuid)
    request_hash = Column(String(64), nullable=False, unique=True, index=True)
    request = Column(JSON, nullable=False)  # {compilation_id, rule_ids|null}
    compilation_id = Column(
        String(36), ForeignKey("catalog_compilations.id"), nullable=False, index=True
    )
    # Rules actually evaluated in the latest pass (resolved rule ids).
    effective_rule_ids = Column(JSON, nullable=False, default=list)
    entry_count = Column(Integer, nullable=False, default=0)  # published entries seen
    hit_count = Column(Integer, nullable=False, default=0)    # hits recorded by this evaluation
    evaluation_count = Column(Integer, nullable=False, default=0)  # passes run so far
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    last_evaluated_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_alert_evaluations_created", "created_at", "id"),
    )


class Alert(Base):
    """One alert instance: a merged group of consecutive hits of one rule."""

    __tablename__ = "alerts"

    id = Column(String(36), primary_key=True, default=_uuid)
    rule_id = Column(String(36), ForeignKey("alert_rules.id"), nullable=False, index=True)
    # open -> acknowledged -> resolved; open -> closed (entry withdrawn).
    # A new hit escalates an acknowledged/resolved alert back to open.
    status = Column(String(16), nullable=False, default="open", index=True)

    first_hit_origin_time = Column(DateTime, nullable=False)  # naive UTC
    last_hit_origin_time = Column(DateTime, nullable=False)   # naive UTC
    hit_count = Column(Integer, nullable=False, default=1)
    # Severity aggregates over all hits (deterministically maintained).
    best_quality = Column(String(1), nullable=False)
    max_station_count = Column(Integer, nullable=False)
    min_rms_s = Column(Float, nullable=False)
    # Allocates AlertEvent.seq for this alert's audit chain.
    event_count = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime, nullable=False, default=_utcnow)
    updated_at = Column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        Index("ix_alerts_created", "created_at", "id"),
    )


class AlertHit(Base):
    """One deterministic (rule, entry) match recorded during an evaluation.

    The (rule_id, entry_id) unique constraint is the hard guarantee that
    concurrent evaluations can neither duplicate nor lose a hit.
    """

    __tablename__ = "alert_hits"

    id = Column(String(36), primary_key=True, default=_uuid)
    rule_id = Column(String(36), ForeignKey("alert_rules.id"), nullable=False, index=True)
    alert_id = Column(String(36), ForeignKey("alerts.id"), nullable=False, index=True)
    evaluation_id = Column(
        String(36), ForeignKey("alert_evaluations.id"), nullable=False, index=True
    )
    compilation_id = Column(
        String(36), ForeignKey("catalog_compilations.id"), nullable=False, index=True
    )
    entry_id = Column(
        String(36), ForeignKey("catalog_entries.id"), nullable=False, index=True
    )
    # opened: raised a new alert; merged: suppressed into an open alert within
    # the merge interval; escalated: merged and bumped an acknowledged/resolved
    # alert back to open.
    disposition = Column(String(16), nullable=False)
    reason = Column(String(64), nullable=True)  # suppression/escalation reason
    # Immutable snapshot of the matched catalog entry.
    entry_snapshot = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        UniqueConstraint("rule_id", "entry_id", name="uq_alert_hit_rule_entry"),
    )


class AlertEvent(Base):
    """One entry of an alert's audit chain (hits and status transitions)."""

    __tablename__ = "alert_events"

    id = Column(String(36), primary_key=True, default=_uuid)
    alert_id = Column(String(36), ForeignKey("alerts.id"), nullable=False, index=True)
    seq = Column(Integer, nullable=False)  # per-alert monotonic, from Alert.event_count
    # opened / merged / escalated / acknowledged / resolved / closed
    kind = Column(String(16), nullable=False)
    hit_id = Column(String(36), ForeignKey("alert_hits.id"), nullable=True)
    reason = Column(String(64), nullable=True)
    detail = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=_utcnow)

    __table_args__ = (
        UniqueConstraint("alert_id", "seq", name="uq_alert_event_seq"),
        Index("ix_alert_events_alert_seq", "alert_id", "seq"),
    )
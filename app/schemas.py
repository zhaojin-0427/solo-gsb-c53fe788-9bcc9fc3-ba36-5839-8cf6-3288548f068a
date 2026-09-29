"""Pydantic request/response schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class WaveformIn(BaseModel):
    station: str = Field(min_length=1, max_length=32)
    latitude: float
    longitude: float
    elevation_m: float = 0.0
    # Optional on purpose: a missing sampling rate must be rejected with an
    # explicit, domain-specific error code instead of a generic 422.
    sampling_rate: Optional[float] = None
    start_time: datetime
    data_b64: str = Field(min_length=1)


class ManualPickIn(BaseModel):
    station: str = Field(min_length=1, max_length=32)
    phase: Literal["P", "S"]
    time: datetime
    weight: float = Field(default=1.0, gt=0.0, le=10.0)


class AssociationParams(BaseModel):
    max_residual_s: float = Field(default=1.0, gt=0.0, le=30.0)
    min_stations: int = Field(default=4, ge=2, le=200)


class ModelRef(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    version: str = Field(min_length=1, max_length=64)


class CalibrationRef(BaseModel):
    """Selects the active calibration version of the request's velocity model."""

    version: str = Field(min_length=1, max_length=64)


class CalibrationCreate(BaseModel):
    # The velocity model is taken from the URL path; this body only carries
    # the estimation request.
    calibration_version: str = Field(min_length=1, max_length=64, alias="version")
    train_start: datetime
    train_end: datetime
    holdout_start: datetime
    holdout_end: datetime
    min_quality: Literal["A", "B", "C"] = "B"
    min_events: int = Field(default=5, ge=1, le=100000)
    min_span_days: float = Field(default=1.0, ge=0.0, le=100000.0)
    min_station_coverage: float = Field(default=0.6, gt=0.0, le=1.0)
    huber_delta_s: float = Field(default=0.5, gt=0.0, le=30.0)
    label: Optional[str] = Field(default=None, max_length=200)

    model_config = {"populate_by_name": True}


class LayerIn(BaseModel):
    thickness_km: Optional[float] = Field(default=None, gt=0.0)  # null = half-space
    vp_kms: float = Field(gt=0.0)
    vs_kms: float = Field(gt=0.0)


class ModelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    version: str = Field(min_length=1, max_length=64)
    layers: list[LayerIn] = Field(min_length=1, max_length=64)


class ProcessRequest(BaseModel):
    waveforms: list[WaveformIn] = Field(min_length=1, max_length=512)
    manual_picks: list[ManualPickIn] = Field(default_factory=list, max_length=4096)
    model: ModelRef
    params: AssociationParams = Field(default_factory=AssociationParams)
    # Optional published/active calibration whose terms are applied to arrivals
    # before association/location.  Absent/None keeps the historical behavior.
    calibration: Optional[CalibrationRef] = None


class ReplayRequest(BaseModel):
    ingestion_id: str = Field(min_length=1, max_length=36)
    model: ModelRef
    params: AssociationParams = Field(default_factory=AssociationParams)
    calibration: Optional[CalibrationRef] = None


# --------------------------------------------------------------------------- #
# cross-run event catalog
# --------------------------------------------------------------------------- #
class VersionConditions(BaseModel):
    """Optional run-version filter for a catalog compilation.

    Runs must belong to one of ``model_versions`` (when given) and must either
    be uncalibrated (``include_uncalibrated``) or use one of
    ``calibration_versions`` (when given).
    """

    model_versions: list[ModelRef] = Field(default_factory=list, max_length=128)
    calibration_versions: list[str] = Field(default_factory=list, max_length=128)
    include_uncalibrated: bool = True


class CatalogCompileRequest(BaseModel):
    start_time: datetime
    end_time: datetime
    # Deterministic linkage thresholds: events merge when they agree within
    # all of these (and share enough station evidence).
    radius_km: float = Field(default=50.0, gt=0.0, le=5000.0)
    origin_tolerance_s: float = Field(default=5.0, gt=0.0, le=600.0)
    depth_tolerance_km: float = Field(default=15.0, gt=0.0, le=700.0)
    min_quality: Literal["A", "B", "C", "D"] = "C"
    min_sources: int = Field(default=1, ge=1, le=1024)
    # Minimum number of station codes common between a candidate and the
    # cluster anchor; 0 disables the station-evidence gate.
    min_shared_stations: int = Field(default=1, ge=0, le=200)
    # Optional center filter: only events within ``region_radius_km`` of the
    # point are eligible.
    center_latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    center_longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    region_radius_km: Optional[float] = Field(default=None, gt=0.0, le=20000.0)
    version_conditions: Optional[VersionConditions] = None


class CatalogPublishRequest(BaseModel):
    # Required optimistic-concurrency token: must equal the current revision.
    expected_revision: int = Field(ge=1)


class CatalogWithdrawRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    reason: str = Field(min_length=1, max_length=500)


# --------------------------------------------------------------------------- #
# station impact audit (leave-one-station-out)
# --------------------------------------------------------------------------- #
class StationAuditCreate(BaseModel):
    """Run a leave-one-station-out audit against a completed run's event.

    ``event_id`` selects the event of a completed run; ``min_keep_stations`` is
    the minimum surviving station count required for a re-located event to be
    accepted (it replaces the original run's ``min_stations`` during the audit
    re-runs).  The frozen velocity model / calibration / processing parameters
    are always the original run's.
    """

    event_id: str = Field(min_length=1, max_length=36)
    min_keep_stations: int = Field(default=4, ge=2, le=200)


# --------------------------------------------------------------------------- #
# P-wave first-motion focal mechanism inversion
# --------------------------------------------------------------------------- #
class PolarityObservationIn(BaseModel):
    """One associated P pick's first-motion polarity and weight."""

    pick_id: str = Field(min_length=1, max_length=36)
    polarity: Literal["positive", "negative"]
    weight: float = Field(default=1.0, gt=0.0, le=10.0)


class FocalMechanismCreate(BaseModel):
    """Invert P first-motion polarities of a completed run's event.

    ``event_id`` selects the event; every ``pick_id`` must be one of the
    event's associated P picks.  At most one polarity per station (conflicting
    polarities at one station are rejected); at least six independent stations
    and a maximum azimuthal gap below 180 degrees are required.
    """

    event_id: str = Field(min_length=1, max_length=36)
    observations: list[PolarityObservationIn] = Field(min_length=1, max_length=512)


# --------------------------------------------------------------------------- #
# event location uncertainty assessment (station bootstrap)
# --------------------------------------------------------------------------- #
class LocationUncertaintyCreate(BaseModel):
    """Bootstrap uncertainty assessment for a completed run's event.

    ``event_id`` selects the event of a completed run; the service freezes its
    associated P/S picks (with weights), the velocity model, the calibration
    version and the processing parameters, then re-locates the event
    ``n_resamples`` times on with-replacement station resamples drawn with
    ``seed`` (repeated stations weighted by multiplicity).  ``alpha`` is the
    confidence level of the reported nearest-rank two-sided intervals.
    """

    event_id: str = Field(min_length=1, max_length=36)
    n_resamples: int = Field(ge=200, le=2000)
    alpha: float = Field(ge=0.8, le=0.99)
    seed: int = Field(ge=0, le=2_147_483_647)


# --------------------------------------------------------------------------- #
# catalog alert orchestration
# --------------------------------------------------------------------------- #
class AlertRuleCreate(BaseModel):
    """Define an immutable alert rule version over the published catalog.

    All conditions are optional and combine conjunctively; a rule with no
    conditions matches every published catalog entry.  The origin-time window
    bounds must be given together (start < end); the circular region requires
    center + radius together.
    """

    name: str = Field(min_length=1, max_length=64)
    rule_version: str = Field(min_length=1, max_length=64, alias="version")
    min_quality: Literal["A", "B", "C", "D"] = "D"
    min_stations: Optional[int] = Field(default=None, ge=1, le=500)
    max_rms_s: Optional[float] = Field(default=None, gt=0.0, le=30.0)
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    center_latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    center_longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    region_radius_km: Optional[float] = Field(default=None, gt=0.0, le=20000.0)
    # Consecutive hits within this interval merge into one alert.
    merge_interval_s: float = Field(default=1800.0, gt=0.0, le=86400.0)
    label: Optional[str] = Field(default=None, max_length=200)

    model_config = {"populate_by_name": True}


class AlertRuleLifecycleRequest(BaseModel):
    # Required optimistic-concurrency token for publish/retire (CAS).
    expected_revision: int = Field(ge=1)


class AlertEvaluateRequest(BaseModel):
    """Evaluate one catalog compilation against alert rules (idempotent).

    ``rule_ids`` selects the rules to apply; ``null``/omitted evaluates all
    currently published rules at each pass.  Repeating an identical request
    runs an idempotent re-evaluation that only records new matches.
    """

    compilation_id: str = Field(min_length=1, max_length=36)
    rule_ids: Optional[list[str]] = Field(default=None, max_length=256)


class AlertTransitionRequest(BaseModel):
    """Operator note for acknowledge / resolve (body may be omitted)."""

    note: Optional[str] = Field(default=None, max_length=500)

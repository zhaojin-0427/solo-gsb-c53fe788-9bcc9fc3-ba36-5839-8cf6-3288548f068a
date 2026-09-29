"""Deterministic phase association.

Picks are scanned in time order; each unused pick seeds a candidate cluster
within a network-aperture time window.  A cluster is iteratively refined:
locate -> compute per-pick residuals (unlabeled picks take the better of
P/S) -> stable-sort candidates by (|residual|, station code) -> keep those
within the requested maximum residual -> re-locate.  An event is accepted
once the set stabilizes and still satisfies the requested minimum station
count.  Every pick ends up associated to at most one event; unassociated
picks carry an explicit reason.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .location import KM_PER_DEG_LAT, Observation, locate_event
from .quality import quality_grade

REASON_ISOLATED = "isolated"
REASON_INSUFFICIENT_STATIONS = "insufficient_stations"
REASON_LOCATION_FAILED = "location_failed"
REASON_RESIDUAL_EXCEEDED = "residual_exceeded"

_REASON_RANK = {
    REASON_ISOLATED: 0,
    REASON_INSUFFICIENT_STATIONS: 1,
    REASON_LOCATION_FAILED: 2,
    REASON_RESIDUAL_EXCEEDED: 3,
}

_MAX_CLUSTER_PICKS = 400
_MAX_REFINEMENT_ROUNDS = 10


@dataclass
class PickIn:
    id: str
    station: str
    phase: str | None  # None = unlabeled (STA/LTA) pick
    t: float  # epoch seconds
    weight: float
    source: str  # 'stalta' | 'manual'


@dataclass
class Assignment:
    pick_id: str
    phase: str
    residual: float


@dataclass
class EventResult:
    t0: float
    lat: float
    lon: float
    depth_km: float
    rms_s: float
    station_count: int
    quality: str
    assignments: list[Assignment] = field(default_factory=list)


def run_association(
    picks: list[PickIn],
    stations: dict[str, tuple[float, float, float]],
    model,
    max_residual_s: float,
    min_stations: int,
    depth0_km: float = 10.0,
    corrections: dict[tuple[str, str], float] | None = None,
) -> tuple[list[EventResult], dict[str, str]]:
    """Associate picks into events.

    Returns (events, unassociated_reasons) where unassociated_reasons maps
    pick_id -> reason for every pick that is not part of any event.

    ``corrections`` optionally maps (station_code, phase) -> offset_s that is
    subtracted from the arrival before location; residuals returned on the
    assignments are the corrected residuals.
    """
    ordered = sorted(picks, key=lambda p: (p.t, p.station, p.phase or "", p.id))
    window_s = _network_window_s(stations, model, max_residual_s)

    used: set[str] = set()
    reasons: dict[str, str] = {}
    events: list[EventResult] = []

    for seed in ordered:
        if seed.id in used:
            continue
        cluster = [
            p
            for p in ordered
            if p.id not in used and seed.t <= p.t <= seed.t + window_s
        ][:_MAX_CLUSTER_PICKS]
        if len(cluster) == 1:
            _note(reasons, seed.id, REASON_ISOLATED)
            continue
        event, notes = _grow_event(
            cluster, stations, model, max_residual_s, min_stations, depth0_km,
            corrections,
        )
        for pid, reason in notes.items():
            _note(reasons, pid, reason)
        if event is not None:
            events.append(event)
            used.update(a.pick_id for a in event.assignments)
        else:
            _note(reasons, seed.id, REASON_INSUFFICIENT_STATIONS)

    events.sort(key=lambda e: (e.t0, e.lat, e.lon))
    unassociated = {
        p.id: reasons.get(p.id, REASON_ISOLATED) for p in ordered if p.id not in used
    }
    return events, unassociated


def correction_offset(
    corrections: dict[tuple[str, str], float] | None, station: str, phase: str
) -> float:
    """Station/phase offset subtracted from an arrival (0.0 when uncalibrated)."""
    if corrections is None:
        return 0.0
    return corrections.get((station, phase), 0.0)


def _grow_event(
    cluster: list[PickIn],
    stations,
    model,
    max_residual_s: float,
    min_stations: int,
    depth0_km: float,
    corrections: dict[tuple[str, str], float] | None = None,
) -> tuple[EventResult | None, dict[str, str]]:
    """Iteratively locate and prune one candidate cluster."""
    if len({p.station for p in cluster}) < min_stations:
        return None, {p.id: REASON_INSUFFICIENT_STATIONS for p in cluster}

    # Bootstrap: the earliest pick of each station must be a P arrival, so an
    # all-P location on those picks seeds the refinement even when the cluster
    # mixes (unlabeled) P and S picks.
    first: dict[str, PickIn] = {}
    for p in sorted(cluster, key=lambda p: (p.t, p.station, p.id)):
        first.setdefault(p.station, p)
    boot = list(first.values())
    loc = None
    for _ in range(4):
        if len(boot) < 4:
            return None, {p.id: REASON_LOCATION_FAILED for p in cluster}
        loc = locate_event(
            model,
            [
                Observation(
                    p.station,
                    "P",
                    p.t - correction_offset(corrections, p.station, "P"),
                    p.weight,
                )
                for p in boot
            ],
            stations,
            depth0_km=depth0_km,
        )
        if loc is None:
            return None, {p.id: REASON_LOCATION_FAILED for p in cluster}
        scored = sorted(
            ((p, _residual(model, loc, stations, p, "P", corrections)) for p in boot),
            key=lambda r: (abs(r[1]), r[0].station, r[0].t, r[0].id),
        )
        kept = [p for p, r in scored if abs(r) <= max_residual_s]
        if len(kept) == len(boot):
            break
        boot = kept

    # Refinement: assign phases by residual, stable-sort candidates by
    # (|residual|, station code), filter by max residual, re-locate, repeat.
    accepted: list[tuple[PickIn, str, float]] = []
    stable_key: tuple | None = None
    for _ in range(_MAX_REFINEMENT_ROUNDS):
        candidates = []
        for p in cluster:
            if p.phase in ("P", "S"):
                candidates.append(
                    (p, p.phase, _residual(model, loc, stations, p, p.phase, corrections))
                )
            else:
                rp = _residual(model, loc, stations, p, "P", corrections)
                rs = _residual(model, loc, stations, p, "S", corrections)
                candidates.append((p, "P", rp) if abs(rp) <= abs(rs) else (p, "S", rs))
        candidates.sort(key=lambda c: (abs(c[2]), c[0].station, c[1], c[0].t, c[0].id))
        kept = [c for c in candidates if abs(c[2]) <= max_residual_s]
        # At most one pick per (station, phase); the sort keeps the best first.
        seen: set[tuple[str, str]] = set()
        accepted = []
        for c in kept:
            key = (c[0].station, c[1])
            if key not in seen:
                seen.add(key)
                accepted.append(c)
        key = tuple((c[0].id, c[1]) for c in accepted)
        if key == stable_key:
            break
        stable_key = key
        if len(accepted) < 4 or len({c[0].station for c in accepted}) < min_stations:
            break  # under-determined; the final check below rejects the event
        new_loc = locate_event(
            model,
            [
                Observation(
                    c[0].station,
                    c[1],
                    c[0].t - correction_offset(corrections, c[0].station, c[1]),
                    c[0].weight,
                )
                for c in accepted
            ],
            stations,
            depth0_km=depth0_km,
        )
        if new_loc is None:
            break
        loc = new_loc

    if not accepted:
        return None, {p.id: REASON_RESIDUAL_EXCEEDED for p in cluster}
    if len({c[0].station for c in accepted}) < min_stations:
        return None, {p.id: REASON_INSUFFICIENT_STATIONS for p in cluster}

    notes: dict[str, str] = {}
    kept_ids = {c[0].id for c in accepted}
    for p in cluster:
        if p.id not in kept_ids:
            _note(notes, p.id, REASON_RESIDUAL_EXCEEDED)

    # Final joint solution on the stabilized set.
    obs = [
        Observation(
            c[0].station,
            c[1],
            c[0].t - correction_offset(corrections, c[0].station, c[1]),
            c[0].weight,
        )
        for c in accepted
    ]
    loc = locate_event(model, obs, stations, depth0_km=depth0_km)
    if loc is None:
        return None, {p.id: REASON_LOCATION_FAILED for p, _, _ in accepted}

    rows = [
        (p, ph, _residual(model, loc, stations, p, ph, corrections))
        for p, ph, _ in accepted
    ]
    rows.sort(key=lambda r: (abs(r[2]), r[0].station, r[1], r[0].t, r[0].id))
    rows = [r for r in rows if abs(r[2]) <= max_residual_s]
    station_count = len({r[0].station for r in rows})
    if station_count < min_stations:
        return None, {p.id: REASON_RESIDUAL_EXCEEDED for p in cluster}

    rms = math.sqrt(sum(r[2] ** 2 for r in rows) / len(rows))
    event = EventResult(
        t0=loc.t0,
        lat=loc.lat,
        lon=loc.lon,
        depth_km=loc.depth_km,
        rms_s=rms,
        station_count=station_count,
        quality=quality_grade(rms, station_count),
        assignments=[Assignment(r[0].id, r[1], r[2]) for r in rows],
    )
    return event, notes


def _residual(model, loc, stations, pick: PickIn, phase: str, corrections=None) -> float:
    lat, lon, elev_m = stations[pick.station]
    dist = distance_km(loc.lat, loc.lon, lat, lon)
    tt, _ = model.travel_time(loc.depth_km + elev_m / 1000.0, dist, phase)
    return pick.t - correction_offset(corrections, pick.station, phase) - (loc.t0 + tt)


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Equirectangular great-circle approximation (km), consistent with the locator."""
    x = math.radians(lon2 - lon1) * math.cos(math.radians(0.5 * (lat1 + lat2)))
    y = math.radians(lat2 - lat1)
    return math.hypot(x, y) * (KM_PER_DEG_LAT * 180.0 / math.pi)


def _network_window_s(stations, model, max_residual_s: float) -> float:
    """Time window in which picks of one event can possibly fall."""
    codes = list(stations)
    max_dist = 0.0
    for i in range(len(codes)):
        la1, lo1, _ = stations[codes[i]]
        for j in range(i + 1, len(codes)):
            la2, lo2, _ = stations[codes[j]]
            max_dist = max(max_dist, distance_km(la1, lo1, la2, lo2))
    return max_dist / model.min_vs + 35.0 + 2.0 * max_residual_s


def _note(reasons: dict[str, str], pick_id: str, reason: str) -> None:
    current = reasons.get(pick_id)
    if current is None or _REASON_RANK[reason] > _REASON_RANK[current]:
        reasons[pick_id] = reason

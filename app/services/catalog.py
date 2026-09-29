"""Deterministic cross-run event clustering for the catalog.

Events from several completed runs are grouped into catalog entries for the
same physical earthquake.  Linkage uses three independent signals:

* origin time within a fixed tolerance,
* epicentral distance / depth within fixed radii,
* shared station evidence (observing-station codes).

The scan is fully deterministic: events are processed in a fixed total order
``(origin_time, run_id, event_id)``; each event joins the earliest cluster
whose anchor satisfies the station-evidence gate and whose members already
agree in time and space (single-link expansion, anchored evidence).  This
makes the result independent of insertion timing, run ids, and database
ordering.

Within a cluster the master solution is chosen by the total order
``(quality rank, RMS, run_id, event_id)``; all other members are kept as
conflicting candidates, including their deltas against the master.
"""
from __future__ import annotations

from dataclasses import dataclass

from .association import distance_km

QUALITY_RANK = {"A": 0, "B": 1, "C": 2, "D": 3}


@dataclass(frozen=True)
class EventEvidence:
    """One located event with the data needed for clustering/master choice."""

    run_id: str
    event_id: str
    origin_epoch: float
    latitude: float
    longitude: float
    depth_km: float
    rms_s: float
    station_count: int
    quality: str
    stations: frozenset[str]

    @property
    def order_key(self) -> tuple:
        """Total deterministic order (scan order and stable tie-break)."""
        return (self.origin_epoch, self.run_id, self.event_id)

    @property
    def master_key(self) -> tuple:
        """Best quality first, then smallest RMS, then stable id tie-break."""
        return (QUALITY_RANK[self.quality], self.rms_s, self.run_id, self.event_id)


@dataclass
class Cluster:
    members: list[EventEvidence]  # sorted by order_key; members[0] is anchor

    @property
    def anchor(self) -> EventEvidence:
        return self.members[0]

    @property
    def master(self) -> EventEvidence:
        return min(self.members, key=lambda e: e.master_key)

    @property
    def distinct_runs(self) -> int:
        return len({e.run_id for e in self.members})

    @property
    def union_stations(self) -> set[str]:
        out: set[str] = set()
        for m in self.members:
            out |= m.stations
        return out

    @property
    def shared_stations(self) -> set[str]:
        common = set(self.anchor.stations)
        for m in self.members[1:]:
            common &= m.stations
        return common


def _linked(candidate: EventEvidence, cluster: Cluster, cfg: "ClusterConfig") -> bool:
    """Station evidence is checked against the anchor; time/space use any member.

    Anchor gating keeps every member traceable to a common observing set and
    prevents chaining across unrelated network geometries; the single-link
    time/space test still admits genuinely matching solutions whose geometry
    drifts slightly from the anchor.
    """
    anchor = cluster.anchor
    if len(candidate.stations & anchor.stations) < cfg.min_shared_stations:
        return False
    for member in cluster.members:
        dt = abs(candidate.origin_epoch - member.origin_epoch)
        dxy = distance_km(
            candidate.latitude, candidate.longitude, member.latitude, member.longitude
        )
        dz = abs(candidate.depth_km - member.depth_km)
        if (
            dt <= cfg.origin_tolerance_s
            and dxy <= cfg.radius_km
            and dz <= cfg.depth_tolerance_km
        ):
            return True
    return False


@dataclass(frozen=True)
class ClusterConfig:
    radius_km: float
    origin_tolerance_s: float
    depth_tolerance_km: float
    min_shared_stations: int

    @classmethod
    def from_request(cls, req) -> "ClusterConfig":
        return cls(
            radius_km=float(req.radius_km),
            origin_tolerance_s=float(req.origin_tolerance_s),
            depth_tolerance_km=float(req.depth_tolerance_km),
            min_shared_stations=int(req.min_shared_stations),
        )


def cluster_events(events: list[EventEvidence], cfg: ClusterConfig) -> list[Cluster]:
    """Deterministically group events.  Result is ordered by cluster anchor."""
    clusters: list[Cluster] = []
    # Fixed scan order makes cluster membership independent of caller order.
    for candidate in sorted(events, key=lambda e: e.order_key):
        target: Cluster | None = None
        for existing in clusters:
            if _linked(candidate, existing, cfg):
                target = existing
                break  # earliest compatible cluster wins -> deterministic
        if target is None:
            clusters.append(Cluster(members=[candidate]))
        else:
            target.members.append(candidate)
            target.members.sort(key=lambda e: e.order_key)
    return clusters

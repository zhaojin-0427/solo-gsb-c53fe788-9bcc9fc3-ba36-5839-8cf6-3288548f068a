"""Association: two events + a spurious pick + an isolated pick."""
from __future__ import annotations

import pytest

from app.config import DEFAULT_MODEL_LAYERS
from app.services.association import (
    PickIn,
    distance_km,
    run_association,
)
from app.services.traveltime import LayeredModel

MODEL = LayeredModel.from_dicts(DEFAULT_MODEL_LAYERS)

STATIONS = {
    "A01": (35.20, -118.00, 0.0),
    "A02": (35.10, -117.80, 100.0),
    "A03": (34.90, -117.85, 0.0),
    "A04": (34.85, -118.10, 50.0),
    "A05": (35.00, -118.25, 0.0),
    "A06": (35.15, -118.15, 150.0),
    "A07": (34.80, -118.00, 0.0),
    "A08": (35.25, -118.20, 0.0),
}

EVENT_A = {"lat": 35.05, "lon": -117.95, "depth_km": 12.0, "t0": 100.0}
EVENT_B = {"lat": 34.90, "lon": -118.10, "depth_km": 8.0, "t0": 400.0}


def _event_picks(event, prefix, phases=("P", "S")) -> list[PickIn]:
    picks = []
    for code, (lat, lon, elev) in STATIONS.items():
        dist = distance_km(event["lat"], event["lon"], lat, lon)
        z = event["depth_km"] + elev / 1000.0
        for ph in phases:
            tt, _ = MODEL.travel_time(z, dist, ph)
            picks.append(
                PickIn(
                    id=f"{prefix}-{code}-{ph}",
                    station=code,
                    phase=None,  # unlabeled, like STA/LTA picks
                    t=event["t0"] + tt,
                    weight=0.8,
                    source="stalta",
                )
            )
    return picks


def _scenario():
    picks = _event_picks(EVENT_A, "evA") + _event_picks(EVENT_B, "evB", phases=("P",))
    # spurious pick inside event A's window that fits neither P nor S
    dist = distance_km(EVENT_A["lat"], EVENT_A["lon"], *STATIONS["A01"][:2])
    tp, _ = MODEL.travel_time(EVENT_A["depth_km"], dist, "P")
    spurious = PickIn("spur-A01", "A01", None, EVENT_A["t0"] + tp + 13.0, 0.5, "stalta")
    isolated = PickIn("iso-A02", "A02", None, 800.0, 0.5, "stalta")
    return picks + [spurious, isolated]


def test_two_events_found_and_located():
    events, reasons = run_association(
        _scenario(), STATIONS, MODEL, max_residual_s=1.0, min_stations=4
    )
    assert len(events) == 2
    first, second = events  # sorted by origin time
    assert first.t0 == pytest.approx(EVENT_A["t0"], abs=0.2)
    assert first.lat == pytest.approx(EVENT_A["lat"], abs=0.01)
    assert first.lon == pytest.approx(EVENT_A["lon"], abs=0.01)
    assert first.depth_km == pytest.approx(EVENT_A["depth_km"], abs=2.0)
    assert first.station_count == 8
    assert first.quality == "A"  # rms ~ 0 with 8 stations
    assert second.t0 == pytest.approx(EVENT_B["t0"], abs=0.2)
    assert second.station_count == 8


def test_unassociated_reasons():
    events, reasons = run_association(
        _scenario(), STATIONS, MODEL, max_residual_s=1.0, min_stations=4
    )
    assert reasons["spur-A01"] == "residual_exceeded"
    assert reasons["iso-A02"] == "isolated"
    associated_ids = {a.pick_id for e in events for a in e.assignments}
    assert set(reasons).isdisjoint(associated_ids)


def test_min_stations_filter():
    events, _ = run_association(
        _scenario(), STATIONS, MODEL, max_residual_s=1.0, min_stations=9
    )
    assert events == []  # 8 stations < required 9


def test_max_residual_filter():
    # a near-zero residual budget must reject everything
    events, _ = run_association(
        _scenario(), STATIONS, MODEL, max_residual_s=1e-6, min_stations=4
    )
    assert events == []


def test_deterministic_repeat():
    picks = _scenario()
    events1, reasons1 = run_association(picks, STATIONS, MODEL, 1.0, 4)
    events2, reasons2 = run_association(picks, STATIONS, MODEL, 1.0, 4)
    assert [(e.t0, e.lat, e.lon, e.depth_km) for e in events1] == [
        (e.t0, e.lat, e.lon, e.depth_km) for e in events2
    ]
    assert reasons1 == reasons2

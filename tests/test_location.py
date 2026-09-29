"""Weighted least-squares locator against synthetic observations."""
from __future__ import annotations

import pytest

from app.config import DEFAULT_MODEL_LAYERS
from app.services.association import distance_km
from app.services.location import Observation, locate_event
from app.services.traveltime import LayeredModel

MODEL = LayeredModel.from_dicts(DEFAULT_MODEL_LAYERS)

STATIONS = {
    "S01": (35.20, -118.00, 0.0),
    "S02": (35.10, -117.80, 100.0),
    "S03": (34.90, -117.85, 0.0),
    "S04": (34.85, -118.10, 50.0),
    "S05": (35.00, -118.25, 0.0),
    "S06": (35.15, -118.15, 150.0),
}

TRUE = {"lat": 35.05, "lon": -117.95, "depth_km": 12.0, "t0": 1_000.0}


def _observations(phases=("P", "S")) -> list[Observation]:
    obs = []
    for code, (lat, lon, elev) in STATIONS.items():
        dist = distance_km(TRUE["lat"], TRUE["lon"], lat, lon)
        z = TRUE["depth_km"] + elev / 1000.0
        for ph in phases:
            tt, _ = MODEL.travel_time(z, dist, ph)
            obs.append(Observation(code, ph, TRUE["t0"] + tt, weight=1.0))
    return obs


def test_recovers_hypocenter_p_and_s():
    result = locate_event(MODEL, _observations(), STATIONS)
    assert result is not None
    assert result.converged
    assert result.t0 == pytest.approx(TRUE["t0"], abs=0.05)
    assert result.lat == pytest.approx(TRUE["lat"], abs=0.005)
    assert result.lon == pytest.approx(TRUE["lon"], abs=0.005)
    assert result.depth_km == pytest.approx(TRUE["depth_km"], abs=1.0)
    assert result.rms_s < 0.05


def test_recovers_hypocenter_p_only():
    result = locate_event(MODEL, _observations(phases=("P",)), STATIONS)
    assert result is not None
    assert result.lat == pytest.approx(TRUE["lat"], abs=0.01)
    assert result.depth_km == pytest.approx(TRUE["depth_km"], abs=2.0)


def test_weights_downweight_bad_pick():
    obs = _observations()
    obs.append(Observation("S01", "P", obs[0].t_obs + 5.0, weight=0.01))  # outlier
    result = locate_event(MODEL, obs, STATIONS)
    assert result is not None
    assert result.lat == pytest.approx(TRUE["lat"], abs=0.02)


def test_underdetermined_returns_none():
    assert locate_event(MODEL, _observations()[:3], STATIONS) is None

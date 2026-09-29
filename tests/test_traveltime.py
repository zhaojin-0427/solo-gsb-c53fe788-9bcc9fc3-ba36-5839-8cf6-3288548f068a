"""Travel-time correctness for the flat-layered model."""
from __future__ import annotations

import math

import pytest

from app.services.traveltime import LayeredModel

HALFSPACE = LayeredModel.from_dicts(
    [{"thickness_km": None, "vp_kms": 6.0, "vs_kms": 3.0}]
)
TWO_LAYER = LayeredModel.from_dicts(
    [
        {"thickness_km": 10.0, "vp_kms": 5.0, "vs_kms": 2.9},
        {"thickness_km": None, "vp_kms": 6.0, "vs_kms": 3.5},
    ]
)


def test_vertical_ray_halfspace():
    t, p = HALFSPACE.travel_time(depth_km=10.0, distance_km=0.0, phase="P")
    assert t == pytest.approx(10.0 / 6.0, abs=1e-6)
    assert p == 0.0


def test_straight_ray_halfspace():
    # straight-line ray in a homogeneous half-space: t = hypot(d, x) / v
    t, _ = HALFSPACE.travel_time(depth_km=10.0, distance_km=10.0, phase="P")
    assert t == pytest.approx(math.hypot(10.0, 10.0) / 6.0, rel=1e-4)


def test_vertical_ray_two_layers():
    t, _ = TWO_LAYER.travel_time(depth_km=20.0, distance_km=0.0, phase="P")
    assert t == pytest.approx(10.0 / 5.0 + 10.0 / 6.0, abs=1e-6)


def test_s_slower_than_p():
    tp, _ = TWO_LAYER.travel_time(15.0, 40.0, "P")
    ts, _ = TWO_LAYER.travel_time(15.0, 40.0, "S")
    assert ts > tp


def test_monotonic_in_distance():
    times = [TWO_LAYER.travel_time(12.0, d, "P")[0] for d in (0, 5, 20, 50, 100)]
    assert all(b > a for a, b in zip(times, times[1:]))


def test_ray_parameter_matches_dt_dx():
    d = 55.0
    t, p = TWO_LAYER.travel_time(12.0, d, "P")
    eps = 0.05
    slope = (
        TWO_LAYER.travel_time(12.0, d + eps, "P")[0]
        - TWO_LAYER.travel_time(12.0, d - eps, "P")[0]
    ) / (2 * eps)
    assert slope == pytest.approx(p, rel=1e-3)


def test_invalid_model_rejected():
    with pytest.raises(ValueError):
        LayeredModel.from_dicts([])
    with pytest.raises(ValueError):
        LayeredModel.from_dicts(
            [
                {"thickness_km": None, "vp_kms": 5.0, "vs_kms": 3.0},
                {"thickness_km": 10.0, "vp_kms": 6.0, "vs_kms": 3.5},
            ]
        )

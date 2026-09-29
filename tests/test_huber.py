"""Unit tests for the zero-sum Huber station-term estimator."""
from __future__ import annotations

import math

import pytest

from app.services.huber import PhaseObservation, fit_phase, holdout_rms


def _synth(n_events: int, stations, biases, seed: int = 7, noise: float = 0.01,
           outlier_frac: float = 0.0):
    obs = []
    rng = __import__("numpy").random.default_rng(seed)
    k = 0
    for e in range(n_events):
        intercept = 0.05 * (e % 3 - 1)  # per-event origin freedom
        for s in stations:
            is_outlier = (k % max(1, int(1 / max(outlier_frac, 1e-9))) == 0) and outlier_frac
            gross = 5.0 if is_outlier else 0.0
            obs.append(
                PhaseObservation(
                    event_index=e,
                    station=s,
                    residual_s=intercept + biases[s] + gross + rng.normal(0, noise),
                    weight=1.0,
                )
            )
            k += 1
    return obs


def test_recovers_known_zero_sum_terms():
    stations = [f"S{i}" for i in range(6)]
    biases = {"S0": 0.30, "S1": -0.20, "S2": 0.15, "S3": -0.25, "S4": 0.10, "S5": -0.10}
    obs = _synth(20, stations, biases)
    fit = fit_phase("P", obs, station_universe=stations, huber_delta_s=0.3)
    assert fit.converged
    assert fit.n_obs == 120
    offsets = {t.station: t.offset_s for t in fit.terms}
    for s, b in biases.items():
        assert offsets[s] == pytest.approx(b, abs=0.03), s
    # hard constraint: terms sum to zero exactly
    assert sum(offsets.values()) == pytest.approx(0.0, abs=1e-9)
    # every term carries a positive uncertainty
    assert all(t.uncertainty_s > 0 for t in fit.terms)
    assert all(t.sample_count == 20 for t in fit.terms)
    assert fit.rms_calibrated_s < fit.rms_baseline_s


def test_huber_rejects_outliers():
    stations = [f"H{i}" for i in range(5)]
    biases = {s: 0.0 for s in stations}
    obs_clean = _synth(10, stations, biases, seed=1, noise=0.005)
    obs_bad = _synth(10, stations, biases, seed=1, noise=0.005, outlier_frac=0.15)
    fit_clean = fit_phase("S", obs_clean, stations, huber_delta_s=0.1)
    fit_bad = fit_phase("S", obs_bad, stations, huber_delta_s=0.1)
    # gross outliers must not leak into the station offsets
    for t in fit_bad.terms:
        assert abs(t.offset_s) < 0.05
    assert max(t.offset_s for t in fit_bad.terms) == pytest.approx(
        max(t.offset_s for t in fit_clean.terms), abs=0.05
    )
    assert fit_bad.residual_scale_s < 0.5  # robust scale ignores the 5 s spikes


def test_unbalanced_design_still_identifiable():
    # stations observed on different, overlapping subsets of events
    stations = ["A", "B", "C", "D"]
    biases = {"A": 0.2, "B": -0.1, "C": 0.05, "D": -0.15}
    obs = []
    schedule = {"A": range(0, 8), "B": range(0, 6), "C": range(2, 10), "D": range(1, 9)}
    for s in stations:
        for e in schedule[s]:
            obs.append(PhaseObservation(e, s, biases[s], 1.0))
    fit = fit_phase("P", obs, stations, huber_delta_s=0.5)
    offsets = {t.station: t.offset_s for t in fit.terms}
    for s, b in biases.items():
        assert offsets[s] == pytest.approx(b, abs=1e-6)
    assert sum(offsets.values()) == pytest.approx(0.0, abs=1e-9)


def test_holdout_rms_only_at_known_stations():
    train = [PhaseObservation(0, "A", 0.2, 1.0), PhaseObservation(0, "B", -0.2, 1.0)]
    fit = fit_phase("P", train, ["A", "B"], huber_delta_s=0.5)
    terms = {t.station: t.offset_s for t in fit.terms}
    holdout = [
        PhaseObservation(0, "A", 0.21, 1.0),
        PhaseObservation(0, "B", -0.19, 1.0),
        PhaseObservation(0, "Z", 1.5, 1.0),  # unseen station: excluded
    ]
    base, cal, n = holdout_rms(holdout, terms)
    assert n == 2
    assert cal < base
    assert not math.isnan(cal)


def test_empty_phase_is_degenerate_but_safe():
    fit = fit_phase("S", [], [], huber_delta_s=0.5)
    assert fit.n_obs == 0 and fit.terms == []
    assert holdout_rms([], {}) == (0.0, 0.0, 0)

"""Huber-robust per-station arrival-time terms with a zero-sum constraint.

For each phase independently, historical residuals are modelled as

    r_i = a_{event(i)} + b_{station(i)}

where ``a`` is a per-event nuisance intercept (origin-time freedom) and ``b``
is the station correction we want to publish.  The station terms are made
identifiable by the hard equality constraint

    sum_j b_j = 0,

enforced as a KKT block of the weighted normal equations rather than by
re-centering after the fact.  Huber robustness comes from IRLS weights

    h_i = 1                    if |r_i| <= delta
          delta / |r_i|        otherwise

combined with the stored pick weight.  The reduced (constrained) inverse
gives the per-station formal uncertainty, scaled by the robust residual
variance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_MAX_ITER = 50
_TOL = 1e-7


@dataclass
class PhaseObservation:
    event_index: int
    station: str
    residual_s: float
    weight: float


@dataclass
class StationTermResult:
    station: str
    offset_s: float
    uncertainty_s: float
    sample_count: int


@dataclass
class PhaseFitResult:
    phase: str
    n_obs: int
    n_events: int
    n_stations: int
    coverage: float
    rms_baseline_s: float
    rms_calibrated_s: float
    residual_scale_s: float
    converged: bool
    iterations: int
    terms: list[StationTermResult] = field(default_factory=list)
    event_intercepts: dict[int, float] = field(default_factory=dict)


def fit_phase(
    phase: str,
    observations: list[PhaseObservation],
    station_universe: list[str],
    huber_delta_s: float,
) -> PhaseFitResult:
    """Estimate zero-sum station terms for one phase.

    ``station_universe`` is the set of stations participating in the training
    events (used for the coverage ratio); stations with no observation for
    this phase simply get no term.
    """
    if not observations:
        return PhaseFitResult(
            phase=phase,
            n_obs=0,
            n_events=0,
            n_stations=0,
            coverage=0.0,
            rms_baseline_s=0.0,
            rms_calibrated_s=0.0,
            residual_scale_s=0.0,
            converged=True,
            iterations=0,
        )

    stations = sorted({o.station for o in observations})
    s_idx = {s: j for j, s in enumerate(stations)}
    n_events = max(o.event_index for o in observations) + 1
    n_stations = len(stations)
    n = len(observations)
    k = n_events + n_stations

    ev_col = np.array([o.event_index for o in observations], dtype=np.int64)
    st_col = np.array([s_idx[o.station] for o in observations], dtype=np.int64)
    r = np.array([o.residual_s for o in observations], dtype=np.float64)
    pick_w = np.array([max(1e-3, o.weight) for o in observations], dtype=np.float64)

    def solve(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Constrained weighted LS: min sum w r^2 s.t. sum(b) = 0."""
        # Build normal equations by scatter accumulation (events and stations
        # have disjoint design columns, one '1' per row in each block).
        A = np.zeros((k, k))
        g = np.zeros(k)
        np.add.at(A, (ev_col, ev_col), weights)
        np.add.at(A, (n_events + st_col, n_events + st_col), weights)
        np.add.at(A, (ev_col, n_events + st_col), weights)
        np.add.at(A, (n_events + st_col, ev_col), weights)
        np.add.at(g, ev_col, weights * r)
        np.add.at(g, n_events + st_col, weights * r)
        # Levenberg-style ridge on the data block only; the equality row stays
        # exact so sum(b) = 0 holds (up to floating point).
        ridge = 1e-9 * max(1.0, float(np.trace(A)) / k)
        A = A + ridge * np.eye(k)
        C = np.zeros((1, k))
        C[0, n_events:] = 1.0
        M = np.block([[A, C.T], [C, np.zeros((1, 1))]])
        rhs = np.concatenate([g, [0.0]])
        try:
            x = np.linalg.solve(M, rhs)[:k]
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(M, rhs, rcond=None)[0][:k]
        return x, M

    huber_w = np.ones(n)
    x = None
    iterations = 0
    converged = False
    for iterations in range(1, _MAX_ITER + 1):
        weights = pick_w * huber_w
        x_new, _ = solve(weights)
        fitted = x_new[ev_col] + x_new[n_events + st_col]
        resid = r - fitted
        scale = np.maximum(np.abs(resid), 1e-12)
        huber_w = np.minimum(1.0, huber_delta_s / scale)
        if x is not None and float(np.max(np.abs(x_new - x))) < _TOL:
            x = x_new
            converged = True
            break
        x = x_new

    assert x is not None
    event_vals = x[:n_events]
    station_vals = x[n_events:]
    fitted = event_vals[ev_col] + station_vals[st_col]
    resid = r - fitted
    final_w = pick_w * np.minimum(1.0, huber_delta_s / np.maximum(np.abs(resid), 1e-12))

    # Constrained covariance: top-left block of the inverse KKT system, scaled
    # by the robust residual variance.
    final_w = pick_w * np.minimum(1.0, huber_delta_s / np.maximum(np.abs(resid), 1e-12))
    _, M = solve(final_w)
    dof = n_events + max(0, n_stations - 1)
    denom = max(1, n - dof)
    sigma2 = float(np.sum(final_w * resid * resid) / denom) / max(
        float(np.mean(final_w)), 1e-300
    )
    try:
        kinv = np.linalg.inv(M)[:k, :k]
    except np.linalg.LinAlgError:
        kinv = np.zeros((k, k))
    diag = np.maximum(np.diag(kinv), 0.0)
    station_sigma = np.sqrt(sigma2 * diag[n_events:])
    # Floor at 1 ms: synthetic/perfect data can drive the formal variance to
    # zero, but a reported uncertainty must stay positive and finite.
    station_sigma = np.maximum(station_sigma, 1e-3)

    # Deployment residuals: the origin is re-estimated after correction, so it
    # absorbs a per-event mean.  Center both raw and corrected residuals per
    # event so the two RMS figures are directly comparable.
    corrected = r - station_vals[st_col]
    base_c = np.zeros(n)
    cal_c = np.zeros(n)
    for e in range(n_events):
        m = ev_col == e
        base_c[m] = r[m] - float(np.mean(r[m]))
        c = corrected[m]
        cal_c[m] = c - float(np.mean(c))
    counts = np.bincount(st_col, minlength=n_stations)
    terms = [
        StationTermResult(
            station=stations[j],
            offset_s=float(station_vals[j]),
            uncertainty_s=float(station_sigma[j]),
            sample_count=int(counts[j]),
        )
        for j in range(n_stations)
    ]
    terms.sort(key=lambda t: t.station)

    universe = set(station_universe) | set(stations)
    coverage = n_stations / len(universe) if universe else 0.0
    return PhaseFitResult(
        phase=phase,
        n_obs=n,
        n_events=n_events,
        n_stations=n_stations,
        coverage=coverage,
        rms_baseline_s=float(math.sqrt(float(np.mean(base_c * base_c)))),
        rms_calibrated_s=float(math.sqrt(float(np.mean(cal_c * cal_c)))),
        residual_scale_s=float(math.sqrt(max(sigma2, 0.0))),
        converged=converged,
        iterations=iterations,
        terms=terms,
        event_intercepts={int(ev_col[i]): float(event_vals[ev_col[i]]) for i in range(n)},
    )


def holdout_rms(
    observations: list[PhaseObservation],
    terms: dict[str, float],
) -> tuple[float, float, int]:
    """Return (baseline_rms, calibrated_rms, n_evaluable) on unseen events.

    Only observations at stations that carry a term are evaluable.  Residuals
    are centered per event before RMS pooling (the origin is re-estimated at
    deployment), mirroring the training metric.
    """
    by_event: dict[int, list[tuple[float, float]]] = {}
    for o in observations:
        b = terms.get(o.station)
        if b is None:
            continue
        by_event.setdefault(o.event_index, []).append((o.residual_s, o.residual_s - b))
    if not by_event:
        return 0.0, 0.0, 0
    base: list[float] = []
    cal: list[float] = []
    for rows in by_event.values():
        rb = np.array([x[0] for x in rows])
        rc = np.array([x[1] for x in rows])
        base.extend(rb - float(np.mean(rb)))
        cal.extend(rc - float(np.mean(rc)))
    arr_b = np.asarray(base)
    arr_c = np.asarray(cal)
    return (
        float(math.sqrt(float(np.mean(arr_b * arr_b)))),
        float(math.sqrt(float(np.mean(arr_c * arr_c)))),
        len(arr_b),
    )

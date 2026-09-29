"""Candidate pick generation: fixed 1-15 Hz bandpass + 0.5/5 s STA/LTA."""
from __future__ import annotations

import numpy as np
from obspy import Trace
from obspy.signal.trigger import trigger_onset

from .. import config


def sta_lta_cf(data: np.ndarray, nsta: int, nlta: int) -> tuple[np.ndarray, int]:
    """Classic STA/LTA characteristic function (cumsum implementation).

    Returns (cf, offset) where ``cf[k]`` is the ratio of the STA ending at
    sample ``offset + k`` to the LTA ending at the same sample.
    """
    x = np.asarray(data, dtype=np.float64)
    sq = x * x
    c = np.concatenate(([0.0], np.cumsum(sq)))
    idx = np.arange(nlta, len(x) + 1)
    sta = (c[idx] - c[idx - nsta]) / nsta
    lta = (c[idx] - c[idx - nlta]) / nlta
    return sta / np.maximum(lta, 1e-30), nlta


def generate_picks(trace: Trace) -> list[dict]:
    """Run the fixed processing chain on one trace.

    Returns a list of {"time_utc": UTCDateTime, "weight": float, "snr": float}.
    """
    tr = trace.copy()
    tr.detrend("linear")
    tr.taper(0.02)
    tr.filter(
        "bandpass",
        freqmin=config.FILTER_FREQMIN_HZ,
        freqmax=config.FILTER_FREQMAX_HZ,
        corners=4,
        zerophase=True,
    )
    sr = float(tr.stats.sampling_rate)
    data = np.asarray(tr.data, dtype=np.float64)
    nsta = max(1, int(round(config.STA_WINDOW_S * sr)))
    nlta = max(nsta + 1, int(round(config.LTA_WINDOW_S * sr)))
    cf, offset = sta_lta_cf(data, nsta, nlta)
    picks = []
    for on, off in trigger_onset(cf, config.TRIGGER_ON, config.TRIGGER_OFF):
        seg = cf[on : off + 1]
        peak = float(seg.max()) if seg.size else config.TRIGGER_ON
        onset_sample = offset + int(on)
        pick_sample = _refine_aic(data, onset_sample, nsta)
        picks.append(
            {
                "time_utc": tr.stats.starttime + pick_sample / sr,
                "weight": float(min(1.0, max(0.2, peak / (2.0 * config.TRIGGER_ON)))),
                "snr": peak,
            }
        )
        if len(picks) >= config.MAX_PICKS_PER_TRACE:
            break
    return picks


def _refine_aic(data: np.ndarray, onset_sample: int, nsta: int) -> int:
    """Refine the trigger sample with an AIC change-point picker."""
    lo = max(0, onset_sample - 2 * nsta)
    hi = min(len(data), onset_sample + 2 * nsta)
    window = data[lo:hi]
    if window.size < 16:
        return onset_sample
    return lo + int(np.argmin(_aic_simple(window)))


def _aic_simple(x: np.ndarray) -> np.ndarray:
    """AIC of the two-segment Gaussian model; minimum marks the onset."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    c = np.concatenate(([0.0], np.cumsum(x * x)))
    k = np.arange(1, n)
    var1 = c[k] / k
    var2 = (c[n] - c[k]) / (n - k)
    aic = np.full(n, np.inf)
    aic[1:] = k * np.log(np.maximum(var1, 1e-30)) + (n - k) * np.log(
        np.maximum(var2, 1e-30)
    )
    return aic

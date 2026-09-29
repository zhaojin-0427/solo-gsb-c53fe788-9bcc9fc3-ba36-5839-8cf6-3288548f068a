#!/usr/bin/env python3
"""Generate an example /v1/process request with synthetic miniSEED waveforms.

Usage:
    python scripts/make_example.py > example_request.json
    curl -s -X POST http://localhost:8000/v1/process \
        -H 'Content-Type: application/json' -d @example_request.json | python -m json.tool

The synthetic dataset contains one event (8 stations, P+S arrivals) plus a
few noise picks, so the response demonstrates associated events, per-pick
phases/residuals, quality grading and unassociated reasons.
"""
from __future__ import annotations

import base64
import io
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DEFAULT_MODEL_LAYERS  # noqa: E402
from app.services.association import distance_km  # noqa: E402
from app.services.traveltime import LayeredModel  # noqa: E402

BASE_TIME = "2026-09-22T00:00:00Z"
SR = 100.0
DURATION_S = 120.0
CENTER = (35.0, -118.0)
EVENT = {"lat": 35.05, "lon": -117.95, "depth_km": 12.0, "origin_offset_s": 30.0}


def ricker(freq: float, sr: float, dur: float = 1.0) -> np.ndarray:
    t = np.arange(-dur / 2, dur / 2, 1.0 / sr)
    a = (np.pi * freq * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def make_waveform_b64(station: str, arrivals: list[tuple[float, float]]) -> str:
    from obspy import Stream, Trace, UTCDateTime

    rng = np.random.default_rng(zlib.crc32(station.encode()))
    n = int(round(DURATION_S * SR))
    data = rng.normal(0.0, 0.02, n)
    wavelet = ricker(5.0, SR)
    for t_offset, amp in arrivals:
        idx = int(round(t_offset * SR))
        s = max(0, idx - len(wavelet) // 2)
        data[s : s + len(wavelet)] += amp * wavelet[: n - s]
    tr = Trace(data=data.astype(np.float64))
    tr.stats.network = "XX"
    tr.stats.station = station
    tr.stats.channel = "BHZ"
    tr.stats.sampling_rate = SR
    tr.stats.starttime = UTCDateTime(BASE_TIME)
    buf = io.BytesIO()
    Stream([tr]).write(buf, format="MSEED")
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    model = LayeredModel.from_dicts(DEFAULT_MODEL_LAYERS)
    waveforms = []
    for i in range(8):
        az = 2.0 * math.pi * i / 8
        lat = CENTER[0] + 0.25 * math.cos(az)
        lon = CENTER[1] + 0.25 * math.sin(az) / math.cos(math.radians(CENTER[0]))
        elev = float(50 * (i % 4))
        dist = distance_km(EVENT["lat"], EVENT["lon"], lat, lon)
        z = EVENT["depth_km"] + elev / 1000.0
        tp = model.travel_time(z, dist, "P")[0]
        ts = model.travel_time(z, dist, "S")[0]
        code = f"EX{i:02d}"
        arrivals = [
            (EVENT["origin_offset_s"] + tp, 4.0),
            (EVENT["origin_offset_s"] + ts, 6.0),
        ]
        # uncorrelated noise bursts on three stations only (below min_stations,
        # so they demonstrate the unassociated-pick reasons in the response)
        if i < 3:
            arrivals.append((95.0 + 6.0 * i, 0.8))
        waveforms.append(
            {
                "station": code,
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
                "elevation_m": elev,
                "sampling_rate": SR,
                "start_time": BASE_TIME,
                "data_b64": make_waveform_b64(code, arrivals),
            }
        )
    body = {
        "waveforms": waveforms,
        "manual_picks": [],
        "model": {"name": "default", "version": "1.0"},
        "params": {"max_residual_s": 1.0, "min_stations": 4},
    }
    json.dump(body, sys.stdout, indent=2)
    print(file=sys.stdout)


if __name__ == "__main__":
    main()

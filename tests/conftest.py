"""Shared fixtures: synthetic miniSEED waveforms with known hypocenters."""
from __future__ import annotations

import base64
import io
import os
import zlib

os.environ["DATABASE_URL"] = "sqlite:////tmp/seis_assoc_test.db"  # before app imports

import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

BASE_TIME = "2026-09-22T00:00:00Z"
CENTER = (35.0, -118.0)


def ricker(freq: float, sr: float, dur: float = 1.0) -> np.ndarray:
    t = np.arange(-dur / 2, dur / 2, 1.0 / sr)
    a = (np.pi * freq * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def make_waveform_b64(
    station: str,
    sr: float,
    start: str,
    arrivals: list[tuple[float, float]],
    duration: float = 120.0,
    noise: float = 0.02,
) -> str:
    """miniSEED (base64) with Ricker-wavelet pulses at the given offsets."""
    rng = np.random.default_rng(zlib.crc32(station.encode()))
    n = int(round(duration * sr))
    data = rng.normal(0.0, noise, n)
    wavelet = ricker(5.0, sr, 1.0)
    for t_offset, amp in arrivals:
        idx = int(round(t_offset * sr))
        s = max(0, idx - len(wavelet) // 2)
        e = min(n, s + len(wavelet))
        if e > s:
            data[s:e] += amp * wavelet[: e - s]
    tr = Trace(data=data.astype(np.float64))
    tr.stats.network = "XX"
    tr.stats.station = station
    tr.stats.channel = "BHZ"
    tr.stats.sampling_rate = sr
    tr.stats.starttime = UTCDateTime(start)
    buf = io.BytesIO()
    Stream([tr]).write(buf, format="MSEED")
    return base64.b64encode(buf.getvalue()).decode()


def station_ring(prefix: str, center=CENTER, radius_deg: float = 0.25, n: int = 8):
    """(code, lat, lon, elev_m) evenly spaced on a ring around center."""
    out = []
    for i in range(n):
        az = 2.0 * np.pi * i / n
        lat = center[0] + radius_deg * np.cos(az)
        lon = center[1] + radius_deg * np.sin(az) / np.cos(np.radians(center[0]))
        out.append((f"{prefix}{i:02d}", float(lat), float(lon), float(50 * (i % 4))))
    return out


def synthetic_event_payload(
    prefix: str = "T",
    n_stations: int = 8,
    ev_lat: float = 35.05,
    ev_lon: float = -117.95,
    ev_depth_km: float = 12.0,
    origin_offset_s: float = 30.0,
    sr: float = 100.0,
    duration: float = 120.0,
    min_stations: int = 4,
    max_residual_s: float = 1.0,
    model_name: str = "default",
    model_version: str = "1.0",
):
    """Full /v1/process request body + ground truth for assertions."""
    from app.config import DEFAULT_MODEL_LAYERS
    from app.services.association import distance_km
    from app.services.traveltime import LayeredModel

    model = LayeredModel.from_dicts(DEFAULT_MODEL_LAYERS)
    waveforms = []
    for code, lat, lon, elev in station_ring(prefix, n=n_stations):
        dist = distance_km(ev_lat, ev_lon, lat, lon)
        z = ev_depth_km + elev / 1000.0
        tp = model.travel_time(z, dist, "P")[0]
        ts = model.travel_time(z, dist, "S")[0]
        b64 = make_waveform_b64(
            code,
            sr,
            BASE_TIME,
            [(origin_offset_s + tp, 4.0), (origin_offset_s + ts, 6.0)],
            duration=duration,
        )
        waveforms.append(
            {
                "station": code,
                "latitude": lat,
                "longitude": lon,
                "elevation_m": elev,
                "sampling_rate": sr,
                "start_time": BASE_TIME,
                "data_b64": b64,
            }
        )
    body = {
        "waveforms": waveforms,
        "manual_picks": [],
        "model": {"name": model_name, "version": model_version},
        "params": {"max_residual_s": max_residual_s, "min_stations": min_stations},
    }
    truth = {
        "lat": ev_lat,
        "lon": ev_lon,
        "depth_km": ev_depth_km,
        "origin_offset_s": origin_offset_s,
    }
    return body, truth


@pytest.fixture()
def client(tmp_path):
    # Per-test database: calibrations mine *historical events of a model* from
    # the DB, so tests must not see each other's events.
    from sqlalchemy import create_engine

    import app.database as database
    from app import models  # noqa: F401  register tables on Base.metadata

    test_engine = create_engine(
        f"sqlite:///{tmp_path / 'seis_assoc_test.db'}",
        connect_args={"check_same_thread": False, "timeout": 30.0},
    )
    original_bind = database.SessionLocal.kw["bind"]
    # Rebind first, then create tables through a pooled connection the
    # lifespan session will share (separate fresh engines miss WAL frames).
    database.SessionLocal.configure(bind=test_engine)
    from app.database import Base

    Base.metadata.create_all(bind=test_engine)
    from app.database import run_lightweight_migrations

    run_lightweight_migrations(test_engine)
    from fastapi.testclient import TestClient

    from app.main import app

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        database.SessionLocal.configure(bind=original_bind)
        test_engine.dispose()

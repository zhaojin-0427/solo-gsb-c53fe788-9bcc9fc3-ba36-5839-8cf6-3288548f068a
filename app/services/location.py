"""Weighted least-squares (Geiger) hypocenter location.

Unknowns are (origin time, x, y, depth) in a local Cartesian frame anchored
at the centroid of the observing stations.  Horizontal partial derivatives
use the exact ray parameter (dT/dX = p); the depth derivative is a centered
finite difference of the layered-model travel time.  Each observation is
weighted by its pick quality (manual picks carry weight 1.0 by default).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_KM = 6371.0
KM_PER_DEG_LAT = 2.0 * math.pi * EARTH_RADIUS_KM / 360.0


@dataclass
class Observation:
    station: str
    phase: str  # 'P' or 'S'
    t_obs: float  # epoch seconds
    weight: float = 1.0


@dataclass
class LocationResult:
    t0: float  # epoch seconds
    lat: float
    lon: float
    depth_km: float
    rms_s: float
    residuals: list[float]
    converged: bool
    iterations: int


def locate_event(
    model,
    observations: list[Observation],
    stations: dict[str, tuple[float, float, float]],
    depth0_km: float = 10.0,
    max_iter: int = 60,
) -> LocationResult | None:
    """Jointly solve for origin time, latitude, longitude and depth.

    ``stations`` maps station code -> (latitude, longitude, elevation_m).
    Returns None when the system is under-determined or singular.
    """
    if len(observations) < 4:
        return None
    lat0 = sum(stations[o.station][0] for o in observations) / len(observations)
    lon0 = sum(stations[o.station][1] for o in observations) / len(observations)
    km_per_deg_lon = KM_PER_DEG_LAT * math.cos(math.radians(lat0))

    def to_xy(code: str) -> tuple[float, float, float]:
        lat, lon, elev_m = stations[code]
        return (
            (lon - lon0) * km_per_deg_lon,
            (lat - lat0) * KM_PER_DEG_LAT,
            elev_m / 1000.0,
        )

    sta = {o.station: to_xy(o.station) for o in observations}
    t_ref = min(o.t_obs for o in observations)

    # Initial guess: network centroid, fixed depth, origin from nearest station.
    cx = sum(v[0] for v in sta.values()) / len(sta)
    cy = sum(v[1] for v in sta.values()) / len(sta)
    d_near = min(math.hypot(x - cx, y - cy) for x, y, _ in sta.values())
    t0_init = t_ref - t_ref - model.travel_time(depth0_km, d_near, "P")[0]
    m = np.array([t0_init, cx, cy, depth0_km], dtype=np.float64)

    n = len(observations)
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        G = np.zeros((n, 4))
        r = np.zeros(n)
        sw = np.zeros(n)
        for i, o in enumerate(observations):
            x, y, ek = sta[o.station]
            dx, dy = x - m[1], y - m[2]
            dist = math.hypot(dx, dy)
            z_eff = m[3] + ek  # station elevation shifts the effective source depth
            tt, p = model.travel_time(z_eff, dist, o.phase)
            r[i] = (o.t_obs - t_ref) - (m[0] + tt)
            ex, ey = (dx / dist, dy / dist) if dist > 1e-6 else (0.0, 0.0)
            dz = 0.5
            z1 = max(0.05, z_eff - dz)
            z2 = z_eff + dz
            dtdz = (
                model.travel_time(z2, dist, o.phase)[0]
                - model.travel_time(z1, dist, o.phase)[0]
            ) / (z2 - z1)
            G[i] = (1.0, -p * ex, -p * ey, dtdz)
            sw[i] = math.sqrt(max(o.weight, 1e-3))
        Gw = G * sw[:, None]
        rw = r * sw
        A = Gw.T @ Gw
        b = Gw.T @ rw
        A += 1e-3 * np.eye(4) * max(1.0, float(np.trace(A)) / 4.0)  # Levenberg damping
        try:
            dm = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(dm)):
            return None
        m += dm
        m[3] = min(max(m[3], 0.0), 700.0)
        if abs(dm[0]) < 1e-3 and abs(dm[3]) < 1e-3 and math.hypot(dm[1], dm[2]) < 1e-3:
            converged = True
            break

    residuals = []
    for o in observations:
        x, y, ek = sta[o.station]
        dist = math.hypot(x - m[1], y - m[2])
        tt, _ = model.travel_time(m[3] + ek, dist, o.phase)
        residuals.append((o.t_obs - t_ref) - (m[0] + tt))
    rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    lat = lat0 + m[2] / KM_PER_DEG_LAT
    lon = lon0 + (m[1] / km_per_deg_lon if abs(km_per_deg_lon) > 1e-9 else 0.0)
    return LocationResult(
        t0=t_ref + float(m[0]),
        lat=lat,
        lon=lon,
        depth_km=float(m[3]),
        rms_s=rms,
        residuals=residuals,
        converged=converged,
        iterations=iterations,
    )

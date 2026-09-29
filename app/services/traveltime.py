"""Travel times through a 1-D flat-layered velocity model.

For a ray parameter ``p`` each traversed layer contributes

    x_i = h_i * p*v_i / sqrt(1 - (p*v_i)^2)      (horizontal offset)
    t_i = h_i / (v_i * sqrt(1 - (p*v_i)^2))      (travel time)

``X(p)`` is monotonically increasing on ``[0, 1/v_max)`` so the ray parameter
for a given epicentral distance is found by bisection.  The derivative
``dT/dX == p`` is used directly by the locator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Layer:
    thickness_km: Optional[float]  # None -> bottom half-space
    vp_kms: float
    vs_kms: float


class LayeredModel:
    def __init__(self, layers: list[Layer]):
        if not layers:
            raise ValueError("velocity model needs at least one layer")
        for i, layer in enumerate(layers):
            if layer.vp_kms <= 0 or layer.vs_kms <= 0:
                raise ValueError("layer velocities must be positive")
            if layer.thickness_km is None:
                if i != len(layers) - 1:
                    raise ValueError("only the last layer may be a half-space (thickness null)")
            elif layer.thickness_km <= 0:
                raise ValueError("layer thickness must be positive")
        self.layers = list(layers)

    @classmethod
    def from_dicts(cls, items: list[dict]) -> "LayeredModel":
        return cls(
            [
                Layer(
                    thickness_km=it.get("thickness_km"),
                    vp_kms=float(it["vp_kms"]),
                    vs_kms=float(it["vs_kms"]),
                )
                for it in items
            ]
        )

    def to_dicts(self) -> list[dict]:
        return [
            {"thickness_km": l.thickness_km, "vp_kms": l.vp_kms, "vs_kms": l.vs_kms}
            for l in self.layers
        ]

    @property
    def min_vs(self) -> float:
        return min(l.vs_kms for l in self.layers)

    def velocity_at_depth(self, depth_km: float, phase: str) -> float:
        """Velocity of the layer containing ``depth_km`` (for takeoff angles)."""
        if phase not in ("P", "S"):
            raise ValueError("phase must be 'P' or 'S'")
        depth_km = max(0.0, float(depth_km))
        z = 0.0
        for layer in self.layers:
            v = layer.vp_kms if phase == "P" else layer.vs_kms
            if layer.thickness_km is None or z + layer.thickness_km >= depth_km:
                return v
            z += layer.thickness_km
        return self.layers[-1].vp_kms if phase == "P" else self.layers[-1].vs_kms

    def _traversed(self, depth_km: float, phase: str) -> list[tuple[float, float]]:
        """(thickness, velocity) pairs from the surface down to the source."""
        depth_km = max(0.0, float(depth_km))
        out: list[tuple[float, float]] = []
        z = 0.0
        for layer in self.layers:
            v = layer.vp_kms if phase == "P" else layer.vs_kms
            if layer.thickness_km is None:
                out.append((depth_km - z, v))
                return out
            if z + layer.thickness_km >= depth_km:
                out.append((depth_km - z, v))
                return out
            out.append((layer.thickness_km, v))
            z += layer.thickness_km
        return out  # unreachable (last layer is a half-space)

    def travel_time(self, depth_km: float, distance_km: float, phase: str) -> tuple[float, float]:
        """Return (travel_time_s, ray_parameter_s_per_km)."""
        if phase not in ("P", "S"):
            raise ValueError("phase must be 'P' or 'S'")
        layers = self._traversed(depth_km, phase)
        distance_km = max(0.0, float(distance_km))
        if distance_km < 1e-9:
            return sum(h / v for h, v in layers), 0.0
        v_max = max(v for _, v in layers)
        lo, hi = 0.0, (1.0 - 1e-9) / v_max
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if _offset(mid, layers) < distance_km:
                lo = mid
            else:
                hi = mid
        p = 0.5 * (lo + hi)
        return _time(p, layers), p


def _offset(p: float, layers: list[tuple[float, float]]) -> float:
    x = 0.0
    for h, v in layers:
        pv = p * v
        x += h * pv / math.sqrt(1.0 - pv * pv)
    return x


def _time(p: float, layers: list[tuple[float, float]]) -> float:
    t = 0.0
    for h, v in layers:
        pv = p * v
        t += h / (v * math.sqrt(1.0 - pv * pv))
    return t

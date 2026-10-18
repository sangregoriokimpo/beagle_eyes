"""WGS84 helpers: local ENU (east, north, up; metres) <-> geodetic lat/lon/alt."""
import math

import numpy as np

_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    n = _A / math.sqrt(1.0 - _E2 * math.sin(lat) ** 2)
    return np.array(
        [
            (n + alt_m) * math.cos(lat) * math.cos(lon),
            (n + alt_m) * math.cos(lat) * math.sin(lon),
            (n * (1.0 - _E2) + alt_m) * math.sin(lat),
        ]
    )


def ecef_to_geodetic(xyz: np.ndarray):
    x, y, z = (float(v) for v in xyz)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - _E2))
    for _ in range(6):
        n = _A / math.sqrt(1.0 - _E2 * math.sin(lat) ** 2)
        alt = p / math.cos(lat) - n if abs(lat) < 1.4 else z / math.sin(lat) - n * (1.0 - _E2)
        lat = math.atan2(z, p * (1.0 - _E2 * n / (n + alt)))
    return math.degrees(lat), math.degrees(lon), alt


def _enu_basis(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl, so, co = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    return np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ]
    )


def enu_to_geodetic(enu, origin_lat: float, origin_lon: float, origin_alt: float = 0.0):
    ecef = geodetic_to_ecef(origin_lat, origin_lon, origin_alt) + _enu_basis(origin_lat, origin_lon).T @ np.asarray(enu, dtype=np.float64)
    return ecef_to_geodetic(ecef)


def geodetic_to_enu(lat: float, lon: float, alt: float, origin_lat: float, origin_lon: float, origin_alt: float = 0.0) -> np.ndarray:
    d = geodetic_to_ecef(lat, lon, alt) - geodetic_to_ecef(origin_lat, origin_lon, origin_alt)
    return _enu_basis(origin_lat, origin_lon) @ d

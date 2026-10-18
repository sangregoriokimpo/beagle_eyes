"""VO world (V) -> local ENU (E) -> lat/lon for a drone and glasses sharing one start point.

Frames
  V: DPVO world. Origin = first camera pose, axes = first camera frame (x right, y down,
     z forward), arbitrary scale.
  E: East/North/Up metres, origin = the shared start point (on the ground when the start
     height is known).

Everything is anchored by a configured origin lat/lon, so lat/lon are always derived from
ENU. The origin is a placeholder until the glasses have GPS: `set_origin()` swaps in a real
fix and all later output uses it. Each assumption is reported with a status field so
consumers can tell placeholder values from real ones.
"""
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .geodesy import enu_to_geodetic
from .ground import GroundPlane, fit_ground_plane

# Placeholder origin until the glasses report GPS (UCLA, matches the dashboard map center).
PLACEHOLDER_ORIGIN = (34.0722, -118.4452, 0.0)


@dataclass
class GpsTfConfig:
    origin_lat: float = PLACEHOLDER_ORIGIN[0]
    origin_lon: float = PLACEHOLDER_ORIGIN[1]
    origin_alt: float = PLACEHOLDER_ORIGIN[2]
    origin_source: str = "placeholder"  # placeholder | fixed (known test coordinate) | gps
    heading_deg: float = 0.0  # true bearing (clockwise from north) the first camera faces
    cam_pitch_deg: float = 0.0  # first camera pitch below horizontal; 0 = level, 90 = straight down
    scale: Optional[float] = None  # metres per VO unit; None = unknown (1.0, flagged "assumed")
    start_height_m: float = 0.0  # drone height above ground at start (also used by autocal)
    autocal: bool = False  # estimate scale + gravity from the ground plane in the point cloud
    autocal_updates: int = 150  # give up after this many VO updates without a lock
    drift_frac: float = 0.05  # assumed VO error as a fraction of distance travelled


def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def _bearing_deg(e: float, n: float) -> float:
    return math.degrees(math.atan2(e, n)) % 360.0


class GpsTransform:
    def __init__(self, config: Optional[GpsTfConfig] = None):
        self.cfg = config or GpsTfConfig()
        self._scale_status = "assumed" if self.cfg.scale is None else "configured"
        self._scale = float(self.cfg.scale) if self.cfg.scale is not None else 1.0
        self._up_v = self._assumed_up()
        self._level_status = "assumed_pitch"
        self._ground_offset = -self.cfg.start_height_m / self._scale  # plane n.x == offset, V units
        self._cal_updates = 0
        self._cal_history: List[tuple] = []
        self._build_basis()
        # Placeholder glasses: standing at the origin, heading unknown, until their GPS/compass is wired in.
        self.glasses: Dict[str, Any] = {
            "enu": [0.0, 0.0, 0.0],
            "heading_true_deg": None,
            "position_source": "assumed_origin",
            "heading_source": "none",
        }

    # ---- configuration -------------------------------------------------------
    def set_origin(self, lat: float, lon: float, alt: float = 0.0, source: str = "gps") -> None:
        """Replace the placeholder origin (e.g. first glasses GPS fix). Later output uses it."""
        self.cfg.origin_lat, self.cfg.origin_lon, self.cfg.origin_alt = lat, lon, alt
        self.cfg.origin_source = source

    def set_glasses(self, enu: Optional[Sequence[float]] = None, heading_true_deg: Optional[float] = None,
                    position_source: str = "gps", heading_source: str = "compass") -> None:
        if enu is not None:
            self.glasses["enu"] = [float(v) for v in enu]
            self.glasses["position_source"] = position_source
        if heading_true_deg is not None:
            self.glasses["heading_true_deg"] = float(heading_true_deg) % 360.0
            self.glasses["heading_source"] = heading_source

    # ---- frame construction --------------------------------------------------
    def _assumed_up(self) -> np.ndarray:
        # Camera frame: x right, y down, z forward. Pitched down by t: up = (0, -cos t, -sin t).
        t = math.radians(self.cfg.cam_pitch_deg)
        return np.array([0.0, -math.cos(t), -math.sin(t)])

    def _build_basis(self) -> None:
        u = self._up_v / np.linalg.norm(self._up_v)
        # Horizontal "forward" of the first camera; at/near nadir the image top is the heading.
        f = np.array([0.0, 0.0, 1.0]) - u * u[2]
        if np.linalg.norm(f) < 0.2:
            f = np.array([0.0, -1.0, 0.0]) - u * u[1]
        f /= np.linalg.norm(f)
        r = np.cross(f, u)  # right of heading
        psi = math.radians(self.cfg.heading_deg)
        east = math.sin(psi) * f + math.cos(psi) * r
        north = math.cos(psi) * f - math.sin(psi) * r
        self._R_ev = np.stack([east, north, u])  # rows: V -> (E, N, U)
        self._up_v = u

    # ---- ground-plane calibration -------------------------------------------
    def _autocalibrate(self, cam_pos_v: np.ndarray, pcl_xyz: Optional[np.ndarray]) -> None:
        if not self.cfg.autocal or self._scale_status == "ground_plane" or pcl_xyz is None:
            return
        if self.cfg.start_height_m <= 0 or self._cal_updates >= self.cfg.autocal_updates:
            return
        self._cal_updates += 1
        plane: Optional[GroundPlane] = fit_ground_plane(pcl_xyz, self._assumed_up(), cam_pos_v)
        if plane is None:
            return
        height_v = float(plane.normal @ cam_pos_v - plane.offset)
        if height_v <= 1e-6:
            return
        self._cal_history.append((self.cfg.start_height_m / height_v, plane.normal, plane.offset))
        self._cal_history = self._cal_history[-5:]
        if len(self._cal_history) < 5:
            return
        scales = np.array([h[0] for h in self._cal_history])
        if np.ptp(scales) > 0.1 * np.median(scales):
            return  # estimates not stable yet
        self._scale = float(np.median(scales))
        normal = np.mean([h[1] for h in self._cal_history], axis=0)
        self._up_v = normal / np.linalg.norm(normal)
        self._ground_offset = float(np.median([h[2] for h in self._cal_history]))
        self._scale_status = "ground_plane"
        self._level_status = "ground_plane"
        self._build_basis()

    # ---- conversions ---------------------------------------------------------
    def v_to_enu(self, p_v: Sequence[float]) -> np.ndarray:
        p = np.asarray(p_v, dtype=np.float64)
        enu = self._scale * (self._R_ev @ p)
        enu[2] = self._scale * (float(self._up_v @ p) - self._ground_offset)  # height above ground
        return enu

    def enu_to_lla(self, enu: Sequence[float]):
        return enu_to_geodetic(enu, self.cfg.origin_lat, self.cfg.origin_lon, self.cfg.origin_alt)

    def _sigma_m(self, path_len_m: float) -> Optional[float]:
        if self._scale_status == "assumed":
            return None  # metric scale unknown, so no meaningful metre error bar
        base = 2.0 if self._scale_status == "ground_plane" else 5.0
        return round(math.hypot(base, self.cfg.drift_frac * path_len_m), 2)

    # ---- per-frame output ----------------------------------------------------
    def annotate(
        self,
        pose_wc_7d: Sequence[float],
        detections_3d: Optional[List[Dict[str, Any]]] = None,
        pcl_xyz: Optional[np.ndarray] = None,
        path_len_v: float = 0.0,
    ) -> Dict[str, Any]:
        """Build the `geo` block to merge into a telemetry packet."""
        pose = np.asarray(pose_wc_7d, dtype=np.float64).ravel()
        cam_v = pose[:3]
        self._autocalibrate(cam_v, pcl_xyz)

        drone_enu = self.v_to_enu(cam_v)
        lat, lon, alt = self.enu_to_lla(drone_enu)
        fwd_v = quat_to_rotmat(*pose[3:7]) @ np.array([0.0, 0.0, 1.0]) if pose.size >= 7 else np.array([0.0, 0.0, 1.0])
        fwd_e = self._R_ev @ fwd_v
        sigma = self._sigma_m(path_len_v * self._scale)

        g_enu = np.asarray(self.glasses["enu"], dtype=np.float64)
        g_head = self.glasses["heading_true_deg"]
        dets_out = []
        for d in detections_3d or []:
            enu = self.v_to_enu(d["center"])
            d_lat, d_lon, d_alt = self.enu_to_lla(enu)
            delta = enu - g_enu
            bearing = _bearing_deg(delta[0], delta[1])
            dets_out.append(
                {
                    "id": d.get("id"),
                    "label": d.get("label", "person"),
                    "active": d.get("active", True),
                    "enu": [round(float(v), 2) for v in enu],
                    "lla": [round(d_lat, 7), round(d_lon, 7), round(d_alt, 2)],
                    "range_m": round(float(np.hypot(delta[0], delta[1])), 2),
                    "bearing_true_deg": round(bearing, 1),
                    "bearing_rel_deg": None if g_head is None else round(((bearing - g_head + 180.0) % 360.0) - 180.0, 1),
                    "sigma_m": sigma,
                }
            )

        return {
            "origin": {
                "lat": self.cfg.origin_lat,
                "lon": self.cfg.origin_lon,
                "alt": self.cfg.origin_alt,
                "source": self.cfg.origin_source,
            },
            "transform_v_to_e": {
                "scale_m_per_vo_unit": round(self._scale, 5),
                "scale_status": self._scale_status,  # assumed | configured | ground_plane
                "level_status": self._level_status,  # assumed_pitch | ground_plane
                "heading_deg": self.cfg.heading_deg,
                "calibrated": self._scale_status != "assumed",
            },
            "drone": {
                "enu": [round(float(v), 2) for v in drone_enu],
                "lla": [round(lat, 7), round(lon, 7), round(alt, 2)],
                "height_agl_m": round(float(drone_enu[2]), 2),
                "heading_true_deg": round(_bearing_deg(fwd_e[0], fwd_e[1]), 1),
                "sigma_m": sigma,
            },
            "glasses": dict(self.glasses),
            "detections": dets_out,
            "timestamp": round(time.time(), 3),
        }

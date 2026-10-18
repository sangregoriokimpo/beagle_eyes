"""Monocular scale + gravity from the ground plane in DPVO's sparse point cloud.

DPVO has no metric scale and no gravity direction, and the DJI Mini 5 Pro altitude is
locked. If the drone starts at a known height above flat ground, a plane fit to the
point cloud gives both: the plane normal is "up", and (known height) / (VO height above
the plane) is metres per VO unit.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GroundPlane:
    normal: np.ndarray  # unit, points up (towards the camera side), in VO world coords
    offset: float  # plane is {x : normal . x == offset}
    inliers: int
    total: int


def fit_ground_plane(
    pts: np.ndarray,
    up_prior: np.ndarray,
    cam_pos: np.ndarray,
    iters: int = 300,
    max_tilt_deg: float = 45.0,
    min_inliers: int = 150,
    seed: int = 0,
) -> Optional[GroundPlane]:
    """RANSAC plane fit, restricted to planes whose normal is near `up_prior` (rejects walls)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < min_inliers:
        return None
    up_prior = np.asarray(up_prior, dtype=np.float64)
    up_prior = up_prior / np.linalg.norm(up_prior)
    thresh = 0.03 * max(float(np.median(np.linalg.norm(pts - cam_pos, axis=1))), 1e-6)
    cos_max = np.cos(np.radians(max_tilt_deg))
    rng = np.random.default_rng(seed)

    best_mask, best_count = None, 0
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        if n @ up_prior < 0:
            n = -n
        if n @ up_prior < cos_max:
            continue
        mask = np.abs((pts - a) @ n) < thresh
        count = int(mask.sum())
        if count > best_count:
            best_mask, best_count = mask, count

    if best_mask is None or best_count < min_inliers:
        return None
    inl = pts[best_mask]
    centroid = inl.mean(axis=0)
    n = np.linalg.svd(inl - centroid)[2][-1]
    if n @ up_prior < 0:
        n = -n
    if n @ (cam_pos - centroid) <= 0:  # ground must be below the camera
        return None
    return GroundPlane(normal=n, offset=float(n @ centroid), inliers=best_count, total=len(pts))

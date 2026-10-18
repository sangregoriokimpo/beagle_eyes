"""
Real-time 2D Person Instance Segmentation + DPVO 3D Frustum Bounding Box Localization.

Lifts 2D person masks/boxes into DPVO's 3D world coordinate system by:
  1. Projecting DPVO's active 3D sparse map points (slam.pg.points_) into the current camera frame.
  2. Computing the robust median depth of DPVO points falling inside each person's eroded segmentation mask.
  3. Falling back to a DPVO-scale-anchored pinhole human size prior when < 3 sparse points hit the person.
  4. Tracking 3D person boxes across frames with EMA smoothing and persistent world memory.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Converts unit quaternion [qx, qy, qz, qw] to a 3x3 rotation matrix R."""
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(1.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return float(inter / (area_a + area_b - inter))


class PersonDetector3D:
    """
    Lightweight 2D Person Segmentation + 3D Frustum Bounding Box Estimator for DPVO.
    Uses Ultralytics YOLO-seg (FP16) if available, with automatic fallback to TorchVision Mask R-CNN.
    """

    def __init__(
        self,
        model_path: str = "checkpoints/yolo11n-seg.pt",
        conf_thres: float = 0.45,
        device: str = "cuda",
        memory_sec: float = 15.0,
        ema_alpha: float = 0.45,
    ):
        self.conf_thres = float(conf_thres)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.memory_sec = float(memory_sec)
        self.ema_alpha = float(ema_alpha)

        self.backend = "none"
        self.model: Any = None
        self._load_model(model_path)

        # Persistent 3D tracks: track_id -> dict
        self._tracks: Dict[int, Dict[str, Any]] = {}
        self._next_track_id: int = 1
        self.last_infer_ms: float = 0.0

    def _load_model(self, model_path: str) -> None:
        try:
            from ultralytics import YOLO  # type: ignore

            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            print(f"[Person3D] Loading Ultralytics segmentation model ({model_path}) on {self.device.upper()}...")
            self.model = YOLO(model_path)
            self.backend = "yolo"
            return
        except Exception as e:
            print(f"[Person3D] Ultralytics YOLO not found ({e}); using TorchVision Mask R-CNN FP16 fallback...")

        try:
            import torchvision  # type: ignore

            weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
            self.model.to(self.device).eval()
            if self.device == "cuda":
                self.model.half()
            self.backend = "torchvision"
            print(f"[Person3D] Loaded TorchVision Mask R-CNN on {self.device.upper()}.")
        except Exception as e:
            print(f"[Person3D] Warning: Could not load segmentation model: {e}")
            self.backend = "none"

    def _segment_2d(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Runs 2D person instance segmentation and returns list of {bbox, conf, mask}."""
        if self.backend == "none" or self.model is None:
            return []

        h_orig, w_orig = frame_bgr.shape[:2]
        t0 = time.perf_counter()
        results_2d: List[Dict[str, Any]] = []

        if self.backend == "yolo":
            preds = self.model.predict(
                source=frame_bgr,
                classes=[0],  # COCO class 0 = person
                conf=self.conf_thres,
                imgsz=640,
                half=(self.device == "cuda"),
                device=self.device,
                verbose=False,
            )
            if preds and len(preds) > 0:
                res = preds[0]
                boxes = res.boxes
                masks = res.masks
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.detach().cpu().numpy()
                    confs = boxes.conf.detach().cpu().numpy()
                    mask_arr = (
                        masks.data.detach().cpu().numpy()
                        if (masks is not None and masks.data is not None)
                        else None
                    )
                    for i in range(len(xyxy)):
                        b = [float(xyxy[i, 0]), float(xyxy[i, 1]), float(xyxy[i, 2]), float(xyxy[i, 3])]
                        m_bin = None
                        if mask_arr is not None and i < len(mask_arr):
                            m_raw = (mask_arr[i] > 0.5).astype(np.uint8)
                            if m_raw.shape[0] != h_orig or m_raw.shape[1] != w_orig:
                                m_bin = cv2.resize(m_raw, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                            else:
                                m_bin = m_raw
                        results_2d.append({"bbox": b, "conf": float(confs[i]), "mask": m_bin})

        elif self.backend == "torchvision":
            # Resize to width 640 for fast inference (~6ms in FP16)
            scale = 640.0 / float(w_orig)
            w_in = 640
            h_in = max(32, int(round(h_orig * scale)))
            small_rgb = cv2.cvtColor(cv2.resize(frame_bgr, (w_in, h_in)), cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(small_rgb).permute(2, 0, 1).to(self.device, non_blocking=True)
            tensor = tensor.half() / 255.0 if self.device == "cuda" else tensor.float() / 255.0

            with torch.no_grad():
                out = self.model([tensor])[0]

            labels = out["labels"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            boxes = out["boxes"].detach().cpu().numpy()
            masks = out["masks"].detach().cpu().numpy() if "masks" in out else None

            # COCO label 1 in TorchVision = person
            keep = (labels == 1) & (scores >= self.conf_thres)
            inv_s = 1.0 / scale
            for idx in np.where(keep)[0][:8]:
                bx = boxes[idx] * inv_s
                b = [float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])]
                m_bin = None
                if masks is not None:
                    m_small = (masks[idx, 0] > 0.5).astype(np.uint8)
                    m_bin = cv2.resize(m_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                results_2d.append({"bbox": b, "conf": float(scores[idx]), "mask": m_bin})

        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        return results_2d

    def update(
        self,
        frame_bgr: np.ndarray,
        pose_wc_7d: np.ndarray,
        intrinsics_base: Tuple[float, float, float, float],
        pcl_xyz: Optional[np.ndarray] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        """
        Segments people in 2D and lifts them to 3D bounding boxes in DPVO's world frame.

        Returns:
          - detections_3d: list of active & remembered 3D person box dicts
          - person_pcl_mask: boolean mask of shape (len(pcl_xyz),) indicating which DPVO 3D points hit a person
        """
        h_img, w_img = frame_bgr.shape[:2]
        fx, fy, cx, cy = intrinsics_base

        dets_2d = self._segment_2d(frame_bgr)

        pose_arr = np.asarray(pose_wc_7d, dtype=np.float64).ravel()
        t_wc = pose_arr[:3]
        if pose_arr.size >= 7:
            R_wc = quat_to_rotmat(pose_arr[3], pose_arr[4], pose_arr[5], pose_arr[6])
        else:
            R_wc = np.eye(3, dtype=np.float64)

        # Project DPVO 3D world points into current camera frame: P_cam = (P_world - t_wc) @ R_wc
        proj_u = np.empty(0, dtype=np.int32)
        proj_v = np.empty(0, dtype=np.int32)
        proj_z = np.empty(0, dtype=np.float64)
        proj_orig_idx = np.empty(0, dtype=np.int64)
        person_pcl_mask: Optional[np.ndarray] = None
        scene_median_z = 3.0

        if pcl_xyz is not None and len(pcl_xyz) > 0:
            pts_w = np.asarray(pcl_xyz, dtype=np.float64).reshape(-1, 3)
            person_pcl_mask = np.zeros(len(pts_w), dtype=bool)
            finite_mask = np.isfinite(pts_w).all(axis=1) & (np.linalg.norm(pts_w, axis=1) > 1e-3)
            if np.any(finite_mask):
                valid_indices = np.where(finite_mask)[0]
                pts_cam = (pts_w[valid_indices] - t_wc.reshape(1, 3)) @ R_wc
                in_front = (pts_cam[:, 2] > 0.2) & (pts_cam[:, 2] < 40.0)
                if np.any(in_front):
                    pts_cam = pts_cam[in_front]
                    valid_indices = valid_indices[in_front]
                    z_vals = pts_cam[:, 2]
                    scene_median_z = float(np.clip(np.median(z_vals), 0.8, 12.0))

                    u_vals = (fx * (pts_cam[:, 0] / z_vals) + cx).astype(np.int32)
                    v_vals = (fy * (pts_cam[:, 1] / z_vals) + cy).astype(np.int32)
                    in_img = (u_vals >= 0) & (u_vals < w_img) & (v_vals >= 0) & (v_vals < h_img)

                    proj_u = u_vals[in_img]
                    proj_v = v_vals[in_img]
                    proj_z = z_vals[in_img]
                    proj_orig_idx = valid_indices[in_img]

        now = time.time()
        measured_objects: List[Dict[str, Any]] = []

        for det in dets_2d:
            u1, v1, u2, v2 = det["bbox"]
            u1_c = max(0.0, min(float(w_img - 1), u1))
            v1_c = max(0.0, min(float(h_img - 1), v1))
            u2_c = max(u1_c + 4.0, min(float(w_img), u2))
            v2_c = max(v1_c + 4.0, min(float(h_img), v2))
            bw = u2_c - u1_c
            bh = v2_c - v1_c

            mask_2d = det["mask"]
            pts_hit = 0
            z_person: Optional[float] = None

            # Stage 1: Find DPVO 3D points projecting inside the person's segmentation mask / torso box
            if len(proj_z) > 0:
                in_box = (
                    (proj_u >= int(u1_c + 0.15 * bw))
                    & (proj_u <= int(u2_c - 0.15 * bw))
                    & (proj_v >= int(v1_c + 0.10 * bh))
                    & (proj_v <= int(v2_c - 0.10 * bh))
                )
                if mask_2d is not None:
                    in_seg = (mask_2d[proj_v, proj_u] > 0) & (proj_u >= int(u1_c)) & (proj_u <= int(u2_c))
                    hit_sel = in_seg if np.sum(in_seg) >= 3 else in_box
                else:
                    hit_sel = in_box

                pts_hit = int(np.sum(hit_sel))
                if pts_hit >= 3:
                    hit_depths = proj_z[hit_sel]
                    # Use 35th percentile (front surface of person, rejecting any leaked background points)
                    z_person = float(np.percentile(hit_depths, 35))
                    if person_pcl_mask is not None:
                        person_pcl_mask[proj_orig_idx[hit_sel]] = True

            # Stage 2 Fallback: Pinhole human prior anchored to DPVO's scene depth scale
            if z_person is None:
                # Standard standing/seated human height prior (~1.65m) & shoulder width prior (~0.52m)
                z_from_h = (fy * 1.65) / max(bh, 32.0)
                z_from_w = (fx * 0.52) / max(bw, 24.0)
                # Blend width and height priors (width is more reliable when person is close/cropped vertically)
                raw_prior_z = 0.55 * z_from_w + 0.45 * z_from_h
                # Anchor scale gently toward DPVO's active scene depth range
                scale_factor = np.clip(scene_median_z / 3.0, 0.4, 2.5)
                z_person = float(np.clip(raw_prior_z * scale_factor, 0.5, 20.0))

            # Unproject 2D box center & dimensions at z_person into 3D camera frame
            u_mid = 0.5 * (u1_c + u2_c)
            v_mid = 0.5 * (v1_c + v2_c)
            x_cam = (u_mid - cx) * z_person / fx
            y_cam = (v_mid - cy) * z_person / fy

            w_3d = float(np.clip((bw * z_person) / fx, 0.35, 1.10))
            h_3d = float(np.clip((bh * z_person) / fy, 0.50, 2.05))
            d_3d = float(np.clip(0.65 * w_3d, 0.35, 0.80))

            # Transform 3D center from camera frame to DPVO world frame: C_world = R_wc @ C_cam + t_wc
            center_cam = np.array([x_cam, y_cam, z_person], dtype=np.float64)
            center_world = (R_wc @ center_cam) + t_wc

            measured_objects.append(
                {
                    "bbox_2d": [round(u1_c, 1), round(v1_c, 1), round(u2_c, 1), round(v2_c, 1)],
                    "conf": round(det["conf"], 3),
                    "mask": mask_2d,
                    "center_world": center_world,
                    "size_3d": np.array([w_3d, h_3d, d_3d], dtype=np.float64),
                    "distance_m": round(float(z_person), 2),
                    "points_hit": pts_hit,
                }
            )

        # Match measured 3D objects to persistent tracks (greedy 3D distance + 2D IoU matching)
        for tid in list(self._tracks.keys()):
            if (now - self._tracks[tid]["last_seen"]) > self.memory_sec:
                del self._tracks[tid]
            else:
                self._tracks[tid]["active"] = False
                self._tracks[tid]["mask"] = None

        unmatched_tracks = set(self._tracks.keys())
        for obj in measured_objects:
            best_tid = None
            best_score = -1e9
            for tid in unmatched_tracks:
                tr = self._tracks[tid]
                dist_3d = float(np.linalg.norm(obj["center_world"] - tr["center_world"]))
                iou_2d = bbox_iou(obj["bbox_2d"], tr["bbox_2d"])
                if dist_3d < 1.8 or iou_2d > 0.25:
                    score = iou_2d * 2.0 - dist_3d
                    if score > best_score:
                        best_score = score
                        best_tid = tid

            if best_tid is not None:
                unmatched_tracks.remove(best_tid)
                tr = self._tracks[best_tid]
                a = self.ema_alpha
                tr["center_world"] = (1.0 - a) * tr["center_world"] + a * obj["center_world"]
                tr["size_3d"] = (1.0 - a) * tr["size_3d"] + a * obj["size_3d"]
                tr["bbox_2d"] = obj["bbox_2d"]
                tr["conf"] = obj["conf"]
                tr["distance_m"] = obj["distance_m"]
                tr["points_hit"] = obj["points_hit"]
                tr["mask"] = obj["mask"]
                tr["last_seen"] = now
                tr["active"] = True
            else:
                tid = self._next_track_id
                self._next_track_id += 1
                self._tracks[tid] = {
                    "id": tid,
                    "label": "person",
                    "center_world": obj["center_world"],
                    "size_3d": obj["size_3d"],
                    "bbox_2d": obj["bbox_2d"],
                    "conf": obj["conf"],
                    "distance_m": obj["distance_m"],
                    "points_hit": obj["points_hit"],
                    "mask": obj["mask"],
                    "last_seen": now,
                    "active": True,
                }

        # Format output list for TrajectoryMap and WebSocket broadcast
        detections_out: List[Dict[str, Any]] = []
        for tid, tr in sorted(self._tracks.items()):
            cw = tr["center_world"]
            sz = tr["size_3d"]
            detections_out.append(
                {
                    "id": int(tid),
                    "label": "person",
                    "conf": float(tr["conf"]),
                    "active": bool(tr["active"]),
                    "distance_m": float(tr["distance_m"]),
                    "points_hit": int(tr["points_hit"]),
                    "center": [round(float(cw[0]), 3), round(float(cw[1]), 3), round(float(cw[2]), 3)],
                    "size": [round(float(sz[0]), 3), round(float(sz[1]), 3), round(float(sz[2]), 3)],
                    "bbox_2d": tr["bbox_2d"],
                    "_mask": tr["mask"],
                }
            )

        return detections_out, person_pcl_mask

    @staticmethod
    def draw_2d_overlay(frame_bgr: np.ndarray, detections_3d: List[Dict[str, Any]]) -> np.ndarray:
        """Draws translucent 2D segmentation masks and 3D distance tags onto the camera frame."""
        if not detections_3d:
            return frame_bgr

        overlay = frame_bgr.copy()
        any_mask = False
        palette = [
            (80, 255, 120),   # Neon green
            (255, 200, 50),   # Cyan-blue
            (180, 105, 255),  # Pink-magenta
            (50, 220, 255),   # Gold-yellow
        ]

        for det in detections_3d:
            if not det.get("active", False):
                continue
            color = palette[(det["id"] - 1) % len(palette)]
            mask = det.get("_mask")
            if mask is not None and mask.shape[:2] == frame_bgr.shape[:2]:
                overlay[mask > 0] = color
                any_mask = True

        if any_mask:
            cv2.addWeighted(overlay, 0.28, frame_bgr, 0.72, 0.0, dst=frame_bgr)

        for det in detections_3d:
            if not det.get("active", False):
                continue
            color = palette[(det["id"] - 1) % len(palette)]
            x1, y1, x2, y2 = [int(v) for v in det["bbox_2d"]]
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            cx, cy, cz = det["center"]
            pts_hit = det["points_hit"]
            src_tag = f"{pts_hit}pts" if pts_hit >= 3 else "prior"
            label = f"P#{det['id']} {det['distance_m']:.1f}m [{cx:+.1f},{cz:+.1f}m] ({src_tag})"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            ty = max(y1 - 6, th + 8)
            cv2.rectangle(frame_bgr, (x1, ty - th - 6), (x1 + tw + 10, ty + 4), (15, 18, 24), -1)
            cv2.rectangle(frame_bgr, (x1, ty - th - 6), (x1 + tw + 10, ty + 4), color, 1)
            cv2.putText(
                frame_bgr,
                label,
                (x1 + 5, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                1,
                cv2.LINE_AA,
            )

        return frame_bgr

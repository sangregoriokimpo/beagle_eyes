import argparse
import cv2
import torch
import numpy as np
import time
import os
import sys

from pathlib import Path

BEAGLE_ROOT = Path(__file__).resolve().parent.parent
if str(BEAGLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BEAGLE_ROOT))
REPO_DIR = str(BEAGLE_ROOT / "vendor" / "Deep-Visual-Odometry-SLAM")
sys.path.insert(0, os.path.join(REPO_DIR, "methods/dpvo"))

from broadcast import TelemetryBroadcaster
from dpvo.dpvo import DPVO
from dpvo.config import cfg
from dpvo.lietorch import SE3
from survey import PersonDetector3D

class TrajectoryMap:
    """Renders real-time top-down bird's-eye path (X = horizontal, Z = forward/depth)."""
    def __init__(self, width=540, height=720):
        self.w = width
        self.h = height
        self.canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.history = []

    def update(self, x, z, detections_3d=None):
        self.history.append((float(x), float(z)))
        if len(self.history) > 1500:
            self.history.pop(0)

        self.canvas[:] = (24, 24, 24)
        for i in range(0, self.w, 80):
            cv2.line(self.canvas, (i, 0), (i, self.h), (40, 40, 40), 1)
        for j in range(0, self.h, 80):
            cv2.line(self.canvas, (0, j), (self.w, j), (40, 40, 40), 1)

        cx, cy = self.w // 2, self.h // 2
        cv2.line(self.canvas, (cx, 0), (cx, self.h), (60, 60, 60), 1)
        cv2.line(self.canvas, (0, cy), (self.w, cy), (60, 60, 60), 1)

        pts = np.array(self.history)
        max_dist = max(float(np.max(np.abs(pts))), 0.5)
        if detections_3d:
            for det in detections_3d:
                dc = det["center"]
                max_dist = max(max_dist, abs(float(dc[0])) * 1.15, abs(float(dc[2])) * 1.15)
        scale = (min(self.w, self.h) * 0.42) / max_dist

        pixel_pts = []
        for px, pz in self.history:
            u = int(cx + px * scale)
            v = int(cy - pz * scale)
            pixel_pts.append((u, v))

        for k in range(1, len(pixel_pts)):
            cv2.line(self.canvas, pixel_pts[k - 1], pixel_pts[k], (0, 220, 255), 2, cv2.LINE_AA)

        if pixel_pts:
            cv2.circle(self.canvas, pixel_pts[0], 6, (0, 255, 0), -1)
            cv2.circle(self.canvas, pixel_pts[-1], 8, (255, 0, 255), -1)

        if detections_3d:
            for det in detections_3d:
                bx, _, bz = det["center"]
                bw, _, bd = det["size"]
                color = (80, 255, 120) if det.get("active", False) else (70, 160, 190)
                u1 = int(cx + (bx - bw * 0.5) * scale)
                v1 = int(cy - (bz + bd * 0.5) * scale)
                u2 = int(cx + (bx + bw * 0.5) * scale)
                v2 = int(cy - (bz - bd * 0.5) * scale)
                cv2.rectangle(self.canvas, (u1, v1), (u2, v2), color, 2)
                cv2.putText(self.canvas, f"P#{det['id']} ({det['distance_m']:.1f}m)",
                            (u1, max(v1 - 6, 85)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        cv2.putText(self.canvas, "TOP-DOWN TRAJECTORY (X-Z)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(self.canvas, f"Window: +/-{max_dist:.1f}m | People: {len(detections_3d or [])}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
        cv2.putText(self.canvas, "Green: Start | Magenta: Current | Box: Person", (20, self.h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 180, 180), 1)

        return self.canvas


def main():
    parser = argparse.ArgumentParser(description="Webcam Visual Odometry with optional 3D Person Segmentation")
    parser.add_argument("--detect-people", action="store_true", help="Enable 2D person segmentation + 3D bounding boxes")
    parser.add_argument("--detect-conf", type=float, default=0.45, help="Confidence threshold for person segmentation")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    WEIGHTS = str(BEAGLE_ROOT / "checkpoints" / "dpvo.pth")
    CONFIG = os.path.join(REPO_DIR, "methods/dpvo/config/default.yaml")

    if not os.path.exists(WEIGHTS):
        print(f"Error: Weights file missing at {WEIGHTS}")
        sys.exit(1)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Could not open camera on /dev/video0.")
        sys.exit(1)

    cfg.merge_from_file(CONFIG)
    cfg.BUFFER_SIZE = 2048

    print("Initializing DPVO on RTX 5070...")
    slam = DPVO(cfg, WEIGHTS, ht=720, wd=1280, viz=False)

    fx = 640.0 / np.tan(np.radians(31.0))
    fy = fx
    cx, cy = 640.0, 360.0
    intrinsics = torch.tensor([fx, fy, cx, cy], dtype=torch.float32, device=DEVICE)
    intrinsics_tuple = (float(fx), float(fy), float(cx), float(cy))

    person_detector = None
    if args.detect_people:
        person_detector = PersonDetector3D(
            model_path=str(BEAGLE_ROOT / "checkpoints" / "yolo11n-seg.pt"),
            conf_thres=args.detect_conf,
            device=DEVICE,
        )

    traj_plotter = TrajectoryMap(width=540, height=720)
    broadcaster = TelemetryBroadcaster(port=8765).start()
    print("Ready. Move laptop to track VO. Press 'q' to exit.")

    t_idx = 0
    t_start = time.time()
    frames_processed = 0
    fps = 0.0
    detections_3d = []

    try:
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                if frame.shape[0] != 720 or frame.shape[1] != 1280:
                    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

                img_tensor = torch.from_numpy(frame).permute(2, 0, 1).to(DEVICE)

                t0 = time.perf_counter()
                slam(t_idx, img_tensor, intrinsics)
                infer_ms = (time.perf_counter() - t0) * 1000.0

                frames_processed += 1
                if time.time() - t_start >= 1.0:
                    fps = frames_processed / (time.time() - t_start)
                    frames_processed = 0
                    t_start = time.time()

                x, y, z = 0.0, 0.0, 0.0
                pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                pcl_xyz, pcl_bgr = None, None
                if slam.is_initialized and slam.n > 0:
                    pose = SE3(slam.pg.poses_[slam.n - 1]).inv().data.cpu().numpy()
                    x, y, z = pose[:3]
                    pcl_xyz = slam.pg.points_[: slam.m].cpu().numpy()
                    pcl_bgr = slam.pg.colors_.view(-1, 3)[: slam.m].cpu().numpy().copy()

                if person_detector is not None:
                    detections_3d, person_pcl_mask = person_detector.update(
                        frame, pose, intrinsics_tuple, pcl_xyz=pcl_xyz
                    )
                    if person_pcl_mask is not None and pcl_bgr is not None and np.any(person_pcl_mask):
                        pcl_bgr[person_pcl_mask] = np.array([80, 255, 100], dtype=np.uint8)

                traj_img = traj_plotter.update(x, z, detections_3d=detections_3d)

                feed_display = frame.copy()
                if person_detector is not None:
                    PersonDetector3D.draw_2d_overlay(feed_display, detections_3d)

                clean_dets = [{k: v for k, v in d.items() if not k.startswith("_")} for d in detections_3d]
                broadcaster.publish_frame(frame_bgr=feed_display)
                broadcaster.publish_telemetry(
                    pose=pose,
                    frame_id=t_idx,
                    fps=fps,
                    infer_ms=infer_ms,
                    initialized=bool(slam.is_initialized),
                    pcl_xyz=pcl_xyz,
                    pcl_bgr=pcl_bgr,
                    extra={"detections_3d": clean_dets},
                )

                cv2.rectangle(feed_display, (15, 15), (640, 110), (15, 15, 15), -1)
                cv2.rectangle(feed_display, (15, 15), (640, 110), (80, 80, 80), 1)

                if not slam.is_initialized:
                    cv2.putText(feed_display, f"INITIALIZING GRAPH ({slam.n}/8)...", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)
                else:
                    cv2.putText(feed_display, f"X: {x:+6.2f}m | Y(Alt): {y:+6.2f}m | Z: {z:+6.2f}m",
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                seg_str = f" | People: {len(detections_3d)}" if person_detector is not None else ""
                cv2.putText(feed_display, f"RTX 5070: {infer_ms:4.1f}ms ({fps:4.1f} FPS) | KF: {slam.n}{seg_str}",
                            (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

                feed_display = cv2.resize(feed_display, (960, 720))
                dashboard = np.hstack((feed_display, traj_img))

                cv2.imshow("Beagle Eyes - Visual Odometry Live Feed", dashboard)
                t_idx += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        broadcaster.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import cv2
import torch
import numpy as np
import time
import os
import sys

REPO_DIR = os.path.expanduser("~/beagle_eyes/vendor/Deep-Visual-Odometry-SLAM")
sys.path.append(os.path.join(REPO_DIR, "methods/dpvo"))

from dpvo.dpvo import DPVO
from dpvo.config import cfg
from dpvo.lietorch import SE3

class TrajectoryMap:
    """Renders real-time top-down bird's-eye path (X = horizontal, Z = forward/depth)."""
    def __init__(self, width=540, height=720):
        self.w = width
        self.h = height
        self.canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.history = []

    def update(self, x, z):
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
        max_dist = max(np.max(np.abs(pts)), 0.5)
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

        cv2.putText(self.canvas, "TOP-DOWN TRAJECTORY (X-Z)", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(self.canvas, f"Window: +/-{max_dist:.1f}m", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
        cv2.putText(self.canvas, "Green: Start | Magenta: Current", (20, self.h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        return self.canvas


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    WEIGHTS = os.path.expanduser("~/beagle_eyes/checkpoints/dpvo.pth")
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
    # Intrinsics shape: [4] (DPVO.pg.intrinsics_[n] expects a 1D tensor of 4 elements)
    intrinsics = torch.tensor([fx, fy, cx, cy], dtype=torch.float32, device=DEVICE)

    traj_plotter = TrajectoryMap(width=540, height=720)
    print("Ready. Move laptop to track VO. Press 'q' to exit.")

    t_idx = 0
    t_start = time.time()
    frames_processed = 0
    fps = 0.0

    try:
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                if frame.shape[0] != 720 or frame.shape[1] != 1280:
                    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

                # 3D Tensor: [C=3, H=720, W=1280] (DPVO.__call__ adds [None, None] internally)
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
                if slam.is_initialized and slam.n > 0:
                    # Convert latest keyframe pose from world-to-camera (T_cw) to camera-to-world (T_wc)
                    pose = SE3(slam.pg.poses_[slam.n - 1]).inv().data.cpu().numpy()
                    x, y, z = pose[:3]
                    traj_img = traj_plotter.update(x, z)
                else:
                    traj_img = traj_plotter.update(0.0, 0.0)

                feed_display = frame.copy()
                cv2.rectangle(feed_display, (15, 15), (580, 110), (15, 15, 15), -1)
                cv2.rectangle(feed_display, (15, 15), (580, 110), (80, 80, 80), 1)

                if not slam.is_initialized:
                    cv2.putText(feed_display, f"INITIALIZING GRAPH ({slam.n}/8)...", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)
                else:
                    cv2.putText(feed_display, f"X: {x:+6.2f}m | Y(Alt): {y:+6.2f}m | Z: {z:+6.2f}m",
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                cv2.putText(feed_display, f"RTX 5070: {infer_ms:4.1f}ms ({fps:4.1f} FPS) | Keyframes: {slam.n}",
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
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
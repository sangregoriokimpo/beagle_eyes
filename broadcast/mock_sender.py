"""
Standalone test transmitter for Beagle Eyes -> React Dashboard testing.

Run on Laptop A to test Laptop B's React dashboard without needing the DJI drone or GPU SLAM:
  python broadcast/mock_sender.py
  python broadcast/mock_sender.py --webcam   # Use real webcam video + simulated 3D trajectory/point cloud
"""

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BEAGLE_ROOT = Path(__file__).resolve().parent.parent
if str(BEAGLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BEAGLE_ROOT))

from broadcast.broadcaster import TelemetryBroadcaster


def generate_demo_point_cloud(num_points: int = 1800) -> tuple[np.ndarray, np.ndarray]:
    """Generates a structured 3D room/terrain point cloud with BGR colors for testing."""
    rng = np.random.default_rng(42)
    theta = rng.uniform(0, 2 * np.pi, num_points)
    radius = rng.uniform(1.5, 6.0, num_points)
    x = radius * np.cos(theta)
    z = radius * np.sin(theta)
    y = 0.6 * np.sin(x * 0.8) * np.cos(z * 0.8) + rng.normal(0, 0.08, num_points)

    xyz = np.stack([x, y, z], axis=1).astype(np.float32)

    norm_y = np.clip((y + 1.0) / 2.0, 0.0, 1.0)
    b = (255 * (1.0 - norm_y)).astype(np.uint8)
    g = (180 * norm_y + 60).astype(np.uint8)
    r = (255 * norm_y).astype(np.uint8)
    bgr = np.stack([b, g, r], axis=1)
    return xyz, bgr


def render_synthetic_frame(width: int, height: int, t: float, x: float, y: float, z: float, frame_id: int) -> np.ndarray:
    """Renders a clean synthetic HUD camera frame when no webcam is used."""
    img = np.full((height, width, 3), (20, 24, 32), dtype=np.uint8)

    horizon_y = int(height * 0.5 + math.sin(t * 0.9) * 45)
    cv2.line(img, (0, horizon_y), (width, horizon_y), (70, 90, 110), 2)
    for offset in range(-400, 401, 80):
        px = int(width // 2 + offset + math.cos(t * 0.7) * 60)
        cv2.line(img, (width // 2, horizon_y), (px, height), (45, 55, 70), 1)

    cv2.rectangle(img, (20, 20), (620, 120), (12, 15, 20), -1)
    cv2.rectangle(img, (20, 20), (620, 120), (0, 220, 255), 1)
    cv2.putText(
        img,
        "BEAGLE EYES MOCK TRANSMITTER (LIVE)",
        (36, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"Frame #{frame_id:05d} | X: {x:+5.2f}m  Y: {y:+5.2f}m  Z: {z:+5.2f}m",
        (36, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (100, 255, 160),
        2,
        cv2.LINE_AA,
    )
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Beagle Eyes mock telemetry + MJPEG sender")
    parser.add_argument("--port", type=int, default=8765, help="Port to serve HTTP MJPEG and WebSocket on")
    parser.add_argument("--fps", type=float, default=30.0, help="Target broadcast rate in Hz")
    parser.add_argument("--webcam", action="store_true", help="Stream real /dev/video0 webcam instead of synthetic HUD")
    args = parser.parse_args()

    cap = None
    if args.webcam:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            print("[Warn] Could not open /dev/video0, falling back to synthetic video.")
            cap = None

    broadcaster = TelemetryBroadcaster(port=args.port).start()
    pcl_xyz, pcl_bgr = generate_demo_point_cloud()

    print("Press Ctrl+C to stop broadcasting.")
    frame_id = 0
    t0 = time.time()
    dt = 1.0 / max(args.fps, 1.0)

    try:
        while True:
            loop_start = time.perf_counter()
            t = time.time() - t0

            x = 2.5 * math.sin(t * 0.6)
            z = 2.5 * math.sin(t * 0.6) * math.cos(t * 0.6)
            y = -0.8 + 0.3 * math.sin(t * 0.3)

            yaw = math.atan2(math.cos(t * 0.6), math.cos(2 * t * 0.6))
            qy = math.sin(yaw * 0.5)
            qw = math.cos(yaw * 0.5)
            pose_7d = [x, y, z, 0.0, qy, 0.0, qw]

            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    frame = render_synthetic_frame(1280, 720, t, x, y, z, frame_id)
            else:
                frame = render_synthetic_frame(1280, 720, t, x, y, z, frame_id)

            broadcaster.publish_frame(frame_bgr=frame)
            broadcaster.publish_telemetry(
                pose=pose_7d,
                frame_id=frame_id,
                fps=args.fps,
                vo_hz=args.fps,
                infer_ms=8.5,
                initialized=True,
                pcl_xyz=pcl_xyz,
                pcl_bgr=pcl_bgr,
            )

            frame_id += 1
            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print("\nStopping broadcaster...")
    finally:
        if cap is not None:
            cap.release()
        broadcaster.stop()


if __name__ == "__main__":
    main()


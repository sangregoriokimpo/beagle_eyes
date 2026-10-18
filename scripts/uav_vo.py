import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

REPO_DIR = os.path.expanduser("~/beagle_eyes/vendor/Deep-Visual-Odometry-SLAM")
sys.path.append(os.path.join(REPO_DIR, "methods/dpvo"))

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.lietorch import SE3


def detect_active_ipv4(fallback_ip: str) -> str:
    """Detects the laptop's current IPv4 address on the active network interface."""
    try:
        out = subprocess.check_output(["ip", "-4", "-br", "addr"], text=True, timeout=2.0)
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] != "lo" and not parts[0].startswith("docker"):
                if parts[1] == "UP":
                    return parts[2].split("/")[0]
    except Exception:
        pass
    return fallback_ip


class MediaMTXServer:
    """
    Manages the local MediaMTX RTMP server process and monitors its logs
    to detect incoming DJI RC 2 publisher connections and stream paths.
    """

    PUBLISH_REGEX = re.compile(r"is publishing to path '([^']+)'")

    def __init__(self, mtx_cfg: dict, rtmp_port: int = 1935, on_publish_path=None):
        self.auto_start = bool(mtx_cfg.get("auto_start", True))
        self.binary = os.path.expanduser(
            mtx_cfg.get("binary", "~/beagle_eyes/vendor/mediamtx/mediamtx")
        )
        self.config_file = os.path.expanduser(
            mtx_cfg.get("config_file", "~/beagle_eyes/config/mediamtx.yml")
        )
        self.port = int(rtmp_port)
        self.on_publish_path = on_publish_path
        self._proc = None
        self._log_thread = None

    def _is_port_open(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            return s.connect_ex(("127.0.0.1", self.port)) == 0

    def _kill_stale_mediamtx(self):
        try:
            subprocess.run(["pkill", "-f", "mediamtx"], check=False, timeout=2.0)
            time.sleep(0.3)
        except Exception:
            pass

    def _monitor_logs(self):
        if self._proc is None or self._proc.stdout is None:
            return
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            if "[RTMP]" in line or "ERR" in line or "WAR" in line:
                print(f"[MediaMTX] {line}")
            m = self.PUBLISH_REGEX.search(line)
            if m and self.on_publish_path is not None:
                published_path = m.group(1)
                self.on_publish_path(published_path)

    def start(self):
        if self._is_port_open():
            # Restart cleanly so we can monitor MediaMTX logs directly
            self._kill_stale_mediamtx()

        if not self.auto_start and not self._is_port_open():
            print(
                f"[MediaMTX] Warning: Port {self.port} is not open and mediamtx.auto_start is False."
            )
            return

        cmd = None
        if os.path.isfile(self.binary) and os.access(self.binary, os.X_OK):
            cmd = [self.binary, self.config_file]
        elif shutil.which("mediamtx"):
            cmd = [shutil.which("mediamtx"), self.config_file]
        elif shutil.which("docker"):
            cmd = [
                "docker",
                "run",
                "--rm",
                "--network",
                "host",
                "-v",
                f"{self.config_file}:/mediamtx.yml:ro",
                "bluenviron/mediamtx:latest",
            ]

        if cmd is None:
            print(
                "[MediaMTX] Error: MediaMTX binary not found.\n"
                "           Run './scripts/setup_uav_rtmp.sh' once to download and configure it."
            )
            sys.exit(1)

        print(f"[MediaMTX] Starting RTMP server on port {self.port}: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._log_thread = threading.Thread(target=self._monitor_logs, daemon=True)
        self._log_thread.start()

        for _ in range(30):
            if self._is_port_open():
                print(f"[MediaMTX] Listening on 0.0.0.0:{self.port}")
                return
            if self._proc.poll() is not None:
                break
            time.sleep(0.1)

        print(f"[MediaMTX] Warning: MediaMTX did not bind port {self.port} within timeout.")

    def stop(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None


class RTMPJpegStream:
    """
    Pulls an RTMP stream from MediaMTX (pushed by DJI Mini 5 Pro RC 2) via FFmpeg
    and transcodes it on-the-fly into a continuous pipe of JPEG frames (image2pipe + mjpeg).
    A background daemon thread parses SOI (0xFFD8) and EOI (0xFFD9) markers to always
    keep the newest JPEG frame ready with zero queue lag.
    """

    SOI_MARKER = b"\xff\xd8"
    EOI_MARKER = b"\xff\xd9"

    def __init__(self, rtmp_cfg: dict, static_ip: str):
        self.mode = rtmp_cfg.get("mode", "pull")
        self.pull_host = rtmp_cfg.get("pull_host", "127.0.0.1")
        self.bind_ip = rtmp_cfg.get("bind_ip", "0.0.0.0")
        self.static_ip = static_ip
        self.port = int(rtmp_cfg.get("port", 1935))
        self.app = rtmp_cfg.get("app", "live").strip("/")
        self.stream_key = rtmp_cfg.get("stream_key", "dji").strip("/")
        self.width = int(rtmp_cfg.get("width", 1280))
        self.height = int(rtmp_cfg.get("height", 720))
        self.fps = int(rtmp_cfg.get("fps", 30))
        self.jpeg_quality = int(rtmp_cfg.get("jpeg_quality", 2))
        self.fflags = rtmp_cfg.get("fflags", "nobuffer")
        self.flags = rtmp_cfg.get("flags", "low_delay")
        self.probesize = int(rtmp_cfg.get("probesize", 32768))
        self.analyzeduration = int(rtmp_cfg.get("analyzeduration", 0))
        self.reconnect_delay = float(rtmp_cfg.get("reconnect_delay_sec", 1.0))

        default_path = f"{self.app}/{self.stream_key}" if self.stream_key else self.app
        url_override = rtmp_cfg.get("url_override", "").strip()
        if url_override:
            self.input_url = url_override
            self.public_url = url_override
        else:
            host = self.bind_ip if self.mode == "listen" else self.pull_host
            self.input_url = f"rtmp://{host}:{self.port}/{default_path}"
            self.public_url = f"rtmp://{self.static_ip}:{self.port}/{default_path}"

        self.save_jpegs = bool(rtmp_cfg.get("save_jpegs", False))
        self.jpeg_output_dir = Path(
            os.path.expanduser(rtmp_cfg.get("jpeg_output_dir", "~/beagle_eyes/logs/uav_jpegs"))
        )
        if self.save_jpegs:
            self.jpeg_output_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._latest_frame = None
        self._latest_jpeg = None
        self._frame_seq = 0
        self._running = False
        self._proc = None
        self._thread = None

    def set_published_path(self, path: str):
        """Updates the local pull URL if DJI RC 2 publishes to a different stream path."""
        path = path.strip("/")
        new_input_url = f"rtmp://{self.pull_host}:{self.port}/{path}"
        if new_input_url != self.input_url:
            print(f"[RTMPJpegStream] Detected DJI stream on path '{path}' -> pulling {new_input_url}")
            self.input_url = new_input_url
            self._terminate_proc()

    def _build_ffmpeg_cmd(self):
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-fflags",
            self.fflags,
            "-flags",
            self.flags,
            "-probesize",
            str(self.probesize),
            "-analyzeduration",
            str(self.analyzeduration),
        ]
        if self.mode == "listen":
            cmd += ["-listen", "1"]

        cmd += [
            "-i",
            self.input_url,
            "-vf",
            f"scale={self.width}:{self.height}",
            "-r",
            str(self.fps),
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-q:v",
            str(self.jpeg_quality),
            "-",
        ]
        return cmd

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self):
        while self._running:
            cmd = self._build_ffmpeg_cmd()
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0,
                )
            except FileNotFoundError:
                print("Error: 'ffmpeg' binary not found. Install with: sudo apt install ffmpeg")
                self._running = False
                return

            buf = bytearray()
            while self._running and self._proc is not None and self._proc.poll() is None:
                chunk = self._proc.stdout.read(65536)
                if not chunk:
                    break
                buf.extend(chunk)

                # Extract all complete JPEG frames in buffer and keep the newest one
                latest_jpg = None
                while True:
                    start = buf.find(self.SOI_MARKER)
                    if start == -1:
                        buf.clear()
                        break
                    end = buf.find(self.EOI_MARKER, start + 2)
                    if end == -1:
                        if start > 0:
                            del buf[:start]
                        break
                    latest_jpg = bytes(buf[start : end + 2])
                    del buf[: end + 2]

                if latest_jpg is not None:
                    arr = np.frombuffer(latest_jpg, dtype=np.uint8)
                    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if bgr is not None:
                        with self._lock:
                            self._latest_jpeg = latest_jpg
                            self._latest_frame = bgr
                            self._frame_seq += 1
                            seq = self._frame_seq

                        if self.save_jpegs:
                            jpg_path = self.jpeg_output_dir / f"frame_{seq:06d}.jpg"
                            jpg_path.write_bytes(latest_jpg)

            self._terminate_proc()
            if self._running:
                time.sleep(self.reconnect_delay)

    def read_latest(self, last_seq: int):
        """Returns (new_seq, bgr_frame, jpeg_bytes) if a newer JPEG frame is available."""
        with self._lock:
            if self._latest_frame is None or self._frame_seq == last_seq:
                return last_seq, None, None
            return self._frame_seq, self._latest_frame.copy(), self._latest_jpeg

    def _terminate_proc(self):
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def stop(self):
        self._running = False
        self._terminate_proc()


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

        cv2.putText(
            self.canvas,
            "UAV TOP-DOWN TRAJECTORY (X-Z)",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            self.canvas,
            f"Window: +/-{max_dist:.1f}m",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (160, 160, 160),
            1,
        )
        cv2.putText(
            self.canvas,
            "Green: Home | Magenta: UAV",
            (20, self.h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
        )

        return self.canvas


def render_waiting_screen(width: int, height: int, public_url: str, active_ip: str):
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    cv2.rectangle(canvas, (40, 40), (width - 40, height - 40), (70, 70, 70), 1)
    cv2.putText(
        canvas,
        "MEDIAMTX ACTIVE - WAITING FOR DJI MINI 5 PRO (RC 2)...",
        (60, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 180, 255),
        2,
    )
    steps = [
        f"1. Ensure laptop & DJI RC 2 share the same Wi-Fi (Laptop IP: {active_ip})",
        "2. On DJI RC 2: Open DJI Fly -> Transmission -> Live Streaming -> RTMP",
        f"3. Enter RTMP URL: {public_url}",
        "4. Select 720p (1280x720) @ 30fps and tap Start Live Stream",
        "Press 'q' at any time to exit.",
    ]
    for idx, line in enumerate(steps):
        cv2.putText(
            canvas,
            line,
            (60, 200 + idx * 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 210, 210),
            1,
        )
    return canvas


def build_intrinsics(cam_cfg: dict, width: int, height: int, device: str) -> torch.Tensor:
    explicit = cam_cfg.get("intrinsics")
    if explicit and len(explicit) == 4:
        fx, fy, cx, cy = [float(v) for v in explicit]
    else:
        hfov_deg = float(cam_cfg.get("hfov_deg", 73.74))
        fx = (width / 2.0) / np.tan(np.radians(hfov_deg / 2.0))
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
    return torch.tensor([fx, fy, cx, cy], dtype=torch.float32, device=device)


def main():
    parser = argparse.ArgumentParser(
        description="DJI Mini 5 Pro (RC 2) RTMP -> MediaMTX -> JPEG Visual Odometry"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.expanduser("~/beagle_eyes/config/uav_config.yaml"),
        help="Path to UAV YAML configuration file",
    )
    parser.add_argument(
        "--setup-static-ip",
        action="store_true",
        help="Run scripts/setup_uav_rtmp.sh to install MediaMTX & configure static IP before starting VO",
    )
    args = parser.parse_args()

    config_path = os.path.expanduser(args.config)
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        uav_cfg = yaml.safe_load(f)

    net_cfg = uav_cfg.get("network", {})
    mtx_cfg = uav_cfg.get("mediamtx", {})
    rtmp_cfg = uav_cfg.get("rtmp", {})
    cam_cfg = uav_cfg.get("camera", {})
    dpvo_cfg = uav_cfg.get("dpvo", {})
    disp_cfg = uav_cfg.get("display", {})

    if args.setup_static_ip or net_cfg.get("auto_configure_static_ip", False):
        setup_script = os.path.expanduser("~/beagle_eyes/scripts/setup_uav_rtmp.sh")
        print(f"Running automated MediaMTX & static IP setup: {setup_script}")
        subprocess.run(["bash", setup_script, config_path], check=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_path = os.path.expanduser(dpvo_cfg.get("weights", "~/beagle_eyes/checkpoints/dpvo.pth"))
    dpvo_config_path = os.path.expanduser(
        dpvo_cfg.get("config", os.path.join(REPO_DIR, "methods/dpvo/config/default.yaml"))
    )

    if not os.path.exists(weights_path):
        print(f"Error: Weights file missing at {weights_path}")
        sys.exit(1)

    width = int(rtmp_cfg.get("width", 1280))
    height = int(rtmp_cfg.get("height", 720))
    # Ensure dimensions are multiples of 16 for DPVO feature pyramid
    width = width - (width % 16)
    height = height - (height % 16)

    cfg.merge_from_file(dpvo_config_path)
    cfg.BUFFER_SIZE = int(dpvo_cfg.get("buffer_size", 2048))
    cfg.PATCHES_PER_FRAME = int(dpvo_cfg.get("patches_per_frame", 96))
    cfg.MIXED_PRECISION = bool(dpvo_cfg.get("mixed_precision", True))

    configured_ip = net_cfg.get("static_ip", "172.20.10.2")
    active_ip = detect_active_ipv4(configured_ip)
    stream = RTMPJpegStream(rtmp_cfg, static_ip=active_ip)

    # Start MediaMTX server before starting FFmpeg pull reader
    mtx_server = MediaMTXServer(
        mtx_cfg,
        rtmp_port=int(rtmp_cfg.get("port", 1935)),
        on_publish_path=stream.set_published_path,
    )
    mtx_server.start()

    print(f"Initializing DPVO ({width}x{height}) on {device.upper()}...")
    slam = DPVO(cfg, weights_path, ht=height, wd=width, viz=False)
    intrinsics = build_intrinsics(cam_cfg, width, height, device)

    stream.start()

    feed_w = int(disp_cfg.get("feed_width", 960))
    feed_h = int(disp_cfg.get("feed_height", 720))
    map_w = int(disp_cfg.get("map_width", 540))
    map_h = int(disp_cfg.get("map_height", 720))
    window_title = disp_cfg.get("window_title", "Beagle Eyes - DJI Mini 5 Pro RTMP Visual Odometry")

    traj_plotter = TrajectoryMap(width=map_w, height=map_h)
    traj_img = traj_plotter.update(0.0, 0.0)

    print("============================================================")
    print(" MediaMTX + FFmpeg JPEG Stream Reader Active")
    print(f" Active Laptop IP  : {active_ip}")
    print(f" DJI RC 2 Push URL : {stream.public_url}")
    print(f" Local Pull URL    : {stream.input_url}")
    print(" Press 'q' in the OpenCV window to exit.")
    print("============================================================")

    t_idx = 0
    last_seq = 0
    t_start = time.time()
    frames_processed = 0
    fps = 0.0

    try:
        with torch.no_grad():
            while True:
                seq, frame, jpeg_bytes = stream.read_latest(last_seq)
                if frame is None:
                    wait_view = render_waiting_screen(feed_w, feed_h, stream.public_url, active_ip)
                    dashboard = np.hstack((wait_view, traj_img))
                    cv2.imshow(window_title, dashboard)
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        break
                    continue

                last_seq = seq
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

                # 3D Tensor: [C=3, H, W] (DPVO.__call__ adds [None, None] internally)
                img_tensor = torch.from_numpy(frame).permute(2, 0, 1).to(device)

                t0 = time.perf_counter()
                slam(t_idx, img_tensor, intrinsics)
                infer_ms = (time.perf_counter() - t0) * 1000.0

                frames_processed += 1
                elapsed = time.time() - t_start
                if elapsed >= 1.0:
                    fps = frames_processed / elapsed
                    frames_processed = 0
                    t_start = time.time()

                x, y, z = 0.0, 0.0, 0.0
                if slam.is_initialized and slam.n > 0:
                    pose = SE3(slam.pg.poses_[slam.n - 1]).inv().data.cpu().numpy()
                    x, y, z = pose[:3]
                    traj_img = traj_plotter.update(x, z)
                else:
                    traj_img = traj_plotter.update(0.0, 0.0)

                jpeg_kb = len(jpeg_bytes) / 1024.0 if jpeg_bytes else 0.0
                feed_display = frame.copy()
                cv2.rectangle(feed_display, (15, 15), (650, 115), (15, 15, 15), -1)
                cv2.rectangle(feed_display, (15, 15), (650, 115), (80, 80, 80), 1)

                if not slam.is_initialized:
                    cv2.putText(
                        feed_display,
                        f"INITIALIZING GRAPH ({slam.n}/8)...",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 165, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        feed_display,
                        f"X: {x:+6.2f}m | Y(Alt): {y:+6.2f}m | Z: {z:+6.2f}m",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )

                cv2.putText(
                    feed_display,
                    f"VO: {infer_ms:4.1f}ms ({fps:4.1f} FPS) | KF: {slam.n} | JPEG: {jpeg_kb:4.0f}KB",
                    (30, 88),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 220, 255),
                    1,
                )

                feed_display = cv2.resize(feed_display, (feed_w, feed_h))
                dashboard = np.hstack((feed_display, traj_img))

                cv2.imshow(window_title, dashboard)
                t_idx += 1

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        mtx_server.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

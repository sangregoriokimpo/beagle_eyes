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

BEAGLE_ROOT = Path(__file__).resolve().parent.parent
if str(BEAGLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BEAGLE_ROOT))
REPO_DIR = str(BEAGLE_ROOT / "vendor" / "Deep-Visual-Odometry-SLAM")
sys.path.insert(0, os.path.join(REPO_DIR, "methods/dpvo"))

from broadcast import TelemetryBroadcaster
from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.lietorch import SE3


def resolve_repo_path(path_str: str) -> str:
    """Resolves a path relative to BEAGLE_ROOT (or expands ~ / absolute paths)."""
    expanded = os.path.expanduser(path_str)
    if os.path.isabs(expanded):
        return expanded
    return str((BEAGLE_ROOT / expanded).resolve())


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
        self.binary = resolve_repo_path(
            mtx_cfg.get("binary", "vendor/mediamtx/mediamtx")
        )
        self.config_file = resolve_repo_path(
            mtx_cfg.get("config_file", "config/mediamtx.yml")
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
            resolve_repo_path(rtmp_cfg.get("jpeg_output_dir", "logs/uav_jpegs"))
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
    """
    Renders real-time UAV trajectory and DPVO 3D Sparse Point Cloud in either:
      - "2d": Top-down bird's-eye view (X = horizontal, Z = forward/depth)
      - "3d": 3D perspective view (X, Z, and Altitude = -Y) with DPVO's live colored
              3D point cloud (slam.pg.points_ & slam.pg.colors_), ground grid,
              3D axes, ground shadow projection, and altitude drop-pillars.
    Supports interactive mouse controls:
      - Left-click + drag  : Rotate 3D view (yaw / pitch)
      - Right-click + drag : Pan / drag map (2D & 3D)
      - Scroll wheel       : Zoom in / out
      - Double-click / 'r' : Reset view
      - 'v' key            : Toggle 2D / 3D
      - 'p' key            : Toggle 3D point cloud display
    """

    def __init__(
        self,
        width=540,
        height=720,
        mode="3d",
        yaw_deg=-35.0,
        pitch_deg=28.0,
        x_offset_on_window=960,
        show_pointcloud=True,
        max_points=4000,
    ):
        self.w = width
        self.h = height
        self.x_offset_on_window = x_offset_on_window
        self.mode = mode.lower() if mode.lower() in ("2d", "3d") else "3d"
        self.show_pointcloud = bool(show_pointcloud)
        self.max_points = int(max_points)
        self.default_yaw = float(yaw_deg)
        self.default_pitch = float(pitch_deg)
        self.yaw_deg = float(yaw_deg)
        self.pitch_deg = float(pitch_deg)
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom = 1.0

        # Mouse drag state
        self._left_dragging = False
        self._right_dragging = False
        self._last_mx = 0
        self._last_my = 0
        self.dirty = False

        self.canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.history = []  # list of (x, y, z)
        self.pcl_xyz = None   # (N, 3) numpy array in camera coords (x, y, z)
        self.pcl_bgr = None   # (N, 3) uint8 numpy array in BGR order

    def toggle_mode(self):
        self.mode = "2d" if self.mode == "3d" else "3d"
        self.dirty = False
        return self.render()

    def toggle_pointcloud(self):
        self.show_pointcloud = not self.show_pointcloud
        self.dirty = False
        return self.render()

    def reset_view(self):
        self.yaw_deg = self.default_yaw
        self.pitch_deg = self.default_pitch
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom = 1.0
        self.dirty = False
        return self.render()

    def on_mouse(self, event, x, y, flags, param):
        """OpenCV mouse callback for rotating (left-drag), panning (right-drag), and zooming."""
        in_map = x >= self.x_offset_on_window

        if event == cv2.EVENT_LBUTTONDOWN and in_map:
            self._left_dragging = True
            self._last_mx, self._last_my = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self._left_dragging = False

        elif event == cv2.EVENT_RBUTTONDOWN and in_map:
            self._right_dragging = True
            self._last_mx, self._last_my = x, y

        elif event == cv2.EVENT_RBUTTONUP:
            self._right_dragging = False

        elif event == cv2.EVENT_LBUTTONDBLCLK and in_map:
            self.reset_view()
            self.dirty = True

        elif event == cv2.EVENT_MOUSEMOVE:
            dx = x - self._last_mx
            dy = y - self._last_my
            if self._left_dragging and self.mode == "3d":
                self.yaw_deg = (self.yaw_deg + dx * 0.45) % 360.0
                self.pitch_deg = float(np.clip(self.pitch_deg + dy * 0.35, 5.0, 85.0))
                self._last_mx, self._last_my = x, y
                self.dirty = True
            elif self._right_dragging or (self._left_dragging and self.mode == "2d"):
                self.pan_x += dx
                self.pan_y += dy
                self._last_mx, self._last_my = x, y
                self.dirty = True

        elif event == cv2.EVENT_MOUSEWHEEL and in_map:
            if flags > 0:
                self.zoom = float(np.clip(self.zoom * 1.12, 0.25, 8.0))
            else:
                self.zoom = float(np.clip(self.zoom / 1.12, 0.25, 8.0))
            self.dirty = True

    def update(self, x, y=0.0, z=0.0, all_poses_xyz=None, pcl_xyz=None, pcl_bgr=None):
        if all_poses_xyz is not None and len(all_poses_xyz) > 0:
            self.history = [(float(p[0]), float(p[1]), float(p[2])) for p in all_poses_xyz[-1500:]]
        else:
            self.history.append((float(x), float(y), float(z)))
            if len(self.history) > 1500:
                self.history.pop(0)

        if pcl_xyz is not None and pcl_bgr is not None and len(pcl_xyz) > 0:
            if len(pcl_xyz) > self.max_points:
                idx = np.linspace(0, len(pcl_xyz) - 1, self.max_points, dtype=np.int64)
                pcl_xyz = pcl_xyz[idx]
                pcl_bgr = pcl_bgr[idx]
            self.pcl_xyz = pcl_xyz
            self.pcl_bgr = pcl_bgr

        self.dirty = False
        return self.render()

    def render(self):
        self.dirty = False
        if not self.history:
            self.history.append((0.0, 0.0, 0.0))
        if self.mode == "3d":
            return self._render_3d()
        return self._render_2d()

    def _render_2d(self):
        self.canvas[:] = (24, 24, 24)
        for i in range(0, self.w, 80):
            cv2.line(self.canvas, (i, 0), (i, self.h), (40, 40, 40), 1)
        for j in range(0, self.h, 80):
            cv2.line(self.canvas, (0, j), (self.w, j), (40, 40, 40), 1)

        cx = int(self.w // 2 + self.pan_x)
        cy = int(self.h // 2 + self.pan_y)
        cv2.line(self.canvas, (cx, 0), (cx, self.h), (60, 60, 60), 1)
        cv2.line(self.canvas, (0, cy), (self.w, cy), (60, 60, 60), 1)

        pts = np.array([[p[0], p[2]] for p in self.history], dtype=np.float32)
        max_dist = max(float(np.max(np.abs(pts))), 0.5) / self.zoom
        scale = (min(self.w, self.h) * 0.42) / max_dist

        pixel_pts = []
        for px, _, pz in self.history:
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
            "2D TOP-DOWN TRAJECTORY (X-Z)",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            self.canvas,
            f"Window: +/-{max_dist:.1f}m | Press 'v' for 3D",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (160, 160, 160),
            1,
        )
        cv2.putText(
            self.canvas,
            "Right-Drag: Pan | Scroll: Zoom | Dbl-Click: Reset",
            (20, self.h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (180, 180, 180),
            1,
        )

        return self.canvas

    def _project_3d(self, pts_xyz_alt: np.ndarray, scale: float, cx: int, cy: int) -> np.ndarray:
        """
        Projects 3D points [X, Z_forward, Alt_up] into 2D canvas coordinates (u, v)
        using yaw and pitch rotation matrices.
        """
        yaw = np.radians(self.yaw_deg)
        pitch = np.radians(self.pitch_deg)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)

        x = pts_xyz_alt[:, 0]
        z = pts_xyz_alt[:, 1]
        alt = pts_xyz_alt[:, 2]

        # Rotate around vertical (Alt) axis by yaw
        xr = x * cos_y + z * sin_y
        zr = -x * sin_y + z * cos_y

        # Tilt by pitch
        u = cx + xr * scale
        v = cy - (alt * cos_p + zr * sin_p) * scale
        return np.stack([u, v], axis=-1).astype(np.int32)

    def _render_3d(self):
        self.canvas[:] = (20, 20, 24)
        cx = int(self.w // 2 + self.pan_x)
        cy = int(self.h * 0.56 + self.pan_y)

        # Convert camera (x, y_down, z_forward) -> world (x, z_forward, alt_up = -y)
        pts_3d = np.array([[p[0], p[2], -p[1]] for p in self.history], dtype=np.float32)
        base_dist = max(float(np.max(np.abs(pts_3d))), 0.5)
        max_dist = base_dist / self.zoom
        scale = (min(self.w, self.h) * 0.36) / max_dist

        # 1. Draw 3D ground grid on the (X, Z, Alt=0) plane
        grid_steps = 4
        step_m = base_dist / grid_steps
        grid_vals = np.linspace(-base_dist, base_dist, 2 * grid_steps + 1)
        for gv in grid_vals:
            line_x = np.array([[gv, -base_dist, 0.0], [gv, base_dist, 0.0]], dtype=np.float32)
            line_z = np.array([[-base_dist, gv, 0.0], [base_dist, gv, 0.0]], dtype=np.float32)
            px = self._project_3d(line_x, scale, cx, cy)
            pz = self._project_3d(line_z, scale, cx, cy)
            color = (55, 55, 65) if abs(gv) < 1e-4 else (36, 36, 44)
            cv2.line(self.canvas, tuple(px[0]), tuple(px[1]), color, 1, cv2.LINE_AA)
            cv2.line(self.canvas, tuple(pz[0]), tuple(pz[1]), color, 1, cv2.LINE_AA)

        # 2. Render DPVO 3D Sparse RGB Point Cloud (slam.pg.points_ & slam.pg.colors_)
        num_pcl_drawn = 0
        if self.show_pointcloud and self.pcl_xyz is not None and self.pcl_bgr is not None:
            # Convert (x, y_down, z_forward) -> (x, z_forward, alt_up = -y)
            pcl_world = np.stack(
                [self.pcl_xyz[:, 0], self.pcl_xyz[:, 2], -self.pcl_xyz[:, 1]], axis=-1
            )
            # Filter out invalid/infinite depths and points far outside the scene radius
            clip_bound = max(base_dist * 3.5, 5.0)
            valid = np.isfinite(pcl_world).all(axis=1) & (np.abs(pcl_world) < clip_bound).all(axis=1)
            if np.any(valid):
                valid_pts = pcl_world[valid]
                valid_clr = self.pcl_bgr[valid]
                uv = self._project_3d(valid_pts, scale, cx, cy)
                in_bounds = (
                    (uv[:, 0] >= 1)
                    & (uv[:, 0] < self.w - 2)
                    & (uv[:, 1] >= 95)
                    & (uv[:, 1] < self.h - 30)
                )
                uv = uv[in_bounds]
                valid_clr = valid_clr[in_bounds]
                num_pcl_drawn = len(uv)
                if num_pcl_drawn > 0:
                    # Draw 2x2 splats for each 3D point with its RGB/BGR patch color
                    u0, v0 = uv[:, 0], uv[:, 1]
                    self.canvas[v0, u0] = valid_clr
                    self.canvas[v0 + 1, u0] = valid_clr
                    self.canvas[v0, u0 + 1] = valid_clr
                    self.canvas[v0 + 1, u0 + 1] = valid_clr

        # 3. Draw 3D coordinate axes at origin (X = Red, Z = Cyan/Blue, Alt = Green)
        axis_len = base_dist * 0.45
        axes_pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [axis_len, 0.0, 0.0],   # +X (Right)
                [0.0, axis_len, 0.0],   # +Z (Forward)
                [0.0, 0.0, axis_len],   # +Alt (Up)
            ],
            dtype=np.float32,
        )
        ap = self._project_3d(axes_pts, scale, cx, cy)
        origin = tuple(ap[0])
        cv2.arrowedLine(self.canvas, origin, tuple(ap[1]), (60, 80, 255), 2, cv2.LINE_AA, tipLength=0.15)
        cv2.arrowedLine(self.canvas, origin, tuple(ap[2]), (255, 180, 40), 2, cv2.LINE_AA, tipLength=0.15)
        cv2.arrowedLine(self.canvas, origin, tuple(ap[3]), (60, 230, 80), 2, cv2.LINE_AA, tipLength=0.15)
        cv2.putText(self.canvas, "+X", tuple(ap[1] + [5, -5]), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 100, 255), 1)
        cv2.putText(self.canvas, "+Z", tuple(ap[2] + [5, -5]), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 190, 60), 1)
        cv2.putText(self.canvas, "+Alt", tuple(ap[3] + [5, -5]), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 240, 100), 1)

        # 4. Project ground shadow (Alt=0) and 3D flight trajectory
        shadow_3d = pts_3d.copy()
        shadow_3d[:, 2] = 0.0
        shadow_px = self._project_3d(shadow_3d, scale, cx, cy)
        traj_px = self._project_3d(pts_3d, scale, cx, cy)

        # Draw ground shadow path
        for k in range(1, len(shadow_px)):
            cv2.line(self.canvas, tuple(shadow_px[k - 1]), tuple(shadow_px[k]), (65, 65, 75), 1, cv2.LINE_AA)

        # Draw vertical altitude drop-pillars every 25 points and at current pose
        pillar_stride = max(1, len(traj_px) // 20)
        for k in range(0, len(traj_px), pillar_stride):
            cv2.line(self.canvas, tuple(shadow_px[k]), tuple(traj_px[k]), (55, 85, 95), 1, cv2.LINE_AA)
        cv2.line(self.canvas, tuple(shadow_px[-1]), tuple(traj_px[-1]), (180, 80, 180), 1, cv2.LINE_AA)

        # Draw 3D trajectory path
        for k in range(1, len(traj_px)):
            cv2.line(self.canvas, tuple(traj_px[k - 1]), tuple(traj_px[k]), (0, 220, 255), 2, cv2.LINE_AA)

        # Draw start (Home) and current UAV markers
        cv2.circle(self.canvas, tuple(traj_px[0]), 5, (0, 255, 0), -1)
        cv2.circle(self.canvas, tuple(shadow_px[-1]), 4, (100, 100, 120), -1)
        cv2.circle(self.canvas, tuple(traj_px[-1]), 7, (255, 0, 255), -1)

        cur_x, cur_z, cur_alt = pts_3d[-1]
        cv2.putText(
            self.canvas,
            "3D ODOM + POINT CLOUD (X-Z-ALT)",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            self.canvas,
            f"Scale: +/-{max_dist:.1f}m | Alt: {cur_alt:+.2f}m | 3D Pts: {num_pcl_drawn}",
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (170, 210, 230),
            1,
        )
        cv2.putText(
            self.canvas,
            f"Yaw: {self.yaw_deg:.0f} deg | Pitch: {self.pitch_deg:.0f} deg | Zoom: {self.zoom:.1f}x",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (140, 140, 150),
            1,
        )
        cv2.putText(
            self.canvas,
            "L-Drag: Rotate | R-Drag: Pan | [p] Pts | [v] 2D/3D",
            (20, self.h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 180),
            1,
        )

        return self.canvas


def render_waiting_screen(width: int, height: int, public_url: str, active_ip: str, show_directions: bool = True):
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    if not show_directions:
        cv2.putText(
            canvas,
            "Waiting for video stream... (Press SPACE for setup instructions)",
            (60, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (130, 130, 130),
            1,
        )
        return canvas

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
        "Map: Left-Drag = Rotate | Right-Drag = Pan | 'p' = PointCloud | 'v' = 2D/3D",
        "Press SPACEBAR to hide this menu, or 'q' to exit.",
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


def build_intrinsics(
    cam_cfg: dict,
    base_width: int,
    base_height: int,
    vo_width: int,
    vo_height: int,
    device: str,
) -> torch.Tensor:
    explicit = cam_cfg.get("intrinsics")
    if explicit and len(explicit) == 4:
        fx, fy, cx, cy = [float(v) for v in explicit]
    else:
        hfov_deg = float(cam_cfg.get("hfov_deg", 73.74))
        fx = (base_width / 2.0) / np.tan(np.radians(hfov_deg / 2.0))
        fy = fx
        cx = base_width / 2.0
        cy = base_height / 2.0

    # Scale intrinsics to match the VO inference resolution
    sx = vo_width / float(base_width)
    sy = vo_height / float(base_height)
    return torch.tensor([fx * sx, fy * sy, cx * sx, cy * sy], dtype=torch.float32, device=device)


def handle_key(key: int, traj_plotter: TrajectoryMap, show_directions: bool):
    """Handles keyboard shortcuts for quitting, toggling 2D/3D, point cloud, directions, and resetting view."""
    if key == ord("q"):
        return True, None, show_directions
    if key == ord(" "):
        return False, None, not show_directions
    if key == ord("v"):
        return False, traj_plotter.toggle_mode(), show_directions
    if key == ord("p"):
        return False, traj_plotter.toggle_pointcloud(), show_directions
    if key == ord("r"):
        return False, traj_plotter.reset_view(), show_directions
    return False, None, show_directions


def main():
    parser = argparse.ArgumentParser(
        description="DJI Mini 5 Pro (RC 2) RTMP -> MediaMTX -> JPEG Visual Odometry"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(BEAGLE_ROOT / "config" / "uav_config.yaml"),
        help="Path to UAV YAML configuration file",
    )
    parser.add_argument(
        "--plot-mode",
        type=str,
        choices=["2d", "3d"],
        default=None,
        help="Trajectory visualization mode: '2d' (top-down X-Z) or '3d' (3D point cloud + X-Y-Z)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Override dpvo.frame_stride (compute VO every Nth frame while rendering video every frame)",
    )
    parser.add_argument(
        "--vo-scale",
        type=float,
        default=None,
        help="Override dpvo.vo_scale (e.g., 0.5 to run DPVO at half resolution while displaying full 720p)",
    )
    parser.add_argument(
        "--dpviewer",
        action="store_true",
        help="Also launch DPVO's native Pangolin OpenGL 3D Viewer (requires modules/DPViewer installed)",
    )
    parser.add_argument(
        "--setup-static-ip",
        action="store_true",
        help="Run scripts/setup_uav_rtmp.sh to install MediaMTX & configure static IP before starting VO",
    )
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config) if not os.path.exists(args.config) else os.path.abspath(args.config)
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
        setup_script = str(BEAGLE_ROOT / "scripts" / "setup_uav_rtmp.sh")
        print(f"Running automated MediaMTX & static IP setup: {setup_script}")
        subprocess.run(["bash", setup_script, config_path], check=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights_path = resolve_repo_path(dpvo_cfg.get("weights", "checkpoints/dpvo.pth"))
    dpvo_config_path = resolve_repo_path(
        dpvo_cfg.get("config", "vendor/Deep-Visual-Odometry-SLAM/methods/dpvo/config/default.yaml")
    )

    if not os.path.exists(weights_path):
        print(f"Error: Weights file missing at {weights_path}")
        sys.exit(1)

    base_width = int(rtmp_cfg.get("width", 1280))
    base_height = int(rtmp_cfg.get("height", 720))

    frame_stride = max(1, int(args.stride if args.stride is not None else dpvo_cfg.get("frame_stride", 2)))
    vo_scale = float(args.vo_scale if args.vo_scale is not None else dpvo_cfg.get("vo_scale", 1.0))
    plot_mode = (args.plot_mode if args.plot_mode is not None else disp_cfg.get("plot_mode", "3d")).lower()
    use_dpviewer = bool(args.dpviewer or dpvo_cfg.get("use_dpviewer", False))

    # Ensure DPVO dimensions are multiples of 16 for the feature pyramid
    vo_width = max(16, int(round(base_width * vo_scale)))
    vo_height = max(16, int(round(base_height * vo_scale)))
    vo_width = vo_width - (vo_width % 16)
    vo_height = vo_height - (vo_height % 16)

    cfg.merge_from_file(dpvo_config_path)
    cfg.BUFFER_SIZE = int(dpvo_cfg.get("buffer_size", 2048))
    cfg.PATCHES_PER_FRAME = int(dpvo_cfg.get("patches_per_frame", 80))
    cfg.OPTIMIZATION_WINDOW = int(dpvo_cfg.get("optimization_window", 10))
    cfg.LOOP_CLOSURE = bool(dpvo_cfg.get("loop_closure", False))
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

    print(
        f"Initializing DPVO (VO res: {vo_width}x{vo_height}, stride: {frame_stride}, "
        f"patches: {cfg.PATCHES_PER_FRAME}) on {device.upper()}..."
    )
    slam = DPVO(cfg, weights_path, ht=vo_height, wd=vo_width, viz=use_dpviewer)
    intrinsics = build_intrinsics(cam_cfg, base_width, base_height, vo_width, vo_height, device)

    stream.start()

    feed_w = int(disp_cfg.get("feed_width", 960))
    feed_h = int(disp_cfg.get("feed_height", 720))
    map_w = int(disp_cfg.get("map_width", 540))
    map_h = int(disp_cfg.get("map_height", 720))
    view_yaw = float(disp_cfg.get("view_yaw_deg", -35.0))
    view_pitch = float(disp_cfg.get("view_pitch_deg", 28.0))
    show_pcl = bool(disp_cfg.get("show_pointcloud", True))
    max_pcl = int(disp_cfg.get("max_pointcloud_points", 4000))
    window_title = disp_cfg.get("window_title", "Beagle Eyes - DJI Mini 5 Pro RTMP Visual Odometry")

    traj_plotter = TrajectoryMap(
        width=map_w,
        height=map_h,
        mode=plot_mode,
        yaw_deg=view_yaw,
        pitch_deg=view_pitch,
        x_offset_on_window=feed_w,
        show_pointcloud=show_pcl,
        max_points=max_pcl,
    )
    traj_img = traj_plotter.update(0.0, 0.0, 0.0)

    cv2.namedWindow(window_title)
    cv2.setMouseCallback(window_title, traj_plotter.on_mouse)

    print("============================================================")
    print(" MediaMTX + FFmpeg JPEG Stream Reader Active")
    print(f" Active Laptop IP  : {active_ip}")
    print(f" DJI RC 2 Push URL : {stream.public_url}")
    print(f" Local Pull URL    : {stream.input_url}")
    print(f" VO Optimization   : Every {frame_stride} frame(s) @ {vo_width}x{vo_height}")
    print(
        f" Trajectory Plot   : {plot_mode.upper()} + PointCloud={'ON' if show_pcl else 'OFF'} "
        "(L-Drag: Rotate | R-Drag: Pan | Scroll: Zoom | 'p': Pts | 'v': 2D/3D)"
    )
    print(" Press SPACE to hide/show instructions, or 'q' to exit.")
    print("============================================================")

    broadcaster = TelemetryBroadcaster(port=8765).start()

    frames_total = 0
    last_seq = 0
    t_start = time.time()
    vid_frames_window = 0
    vo_frames_window = 0
    vid_fps = 0.0
    vo_hz = 0.0
    infer_ms = 0.0
    x, y, z = 0.0, 0.0, 0.0
    last_feed_display = None
    show_directions = True

    try:
        with torch.no_grad():
            while True:
                if traj_plotter.dirty:
                    traj_img = traj_plotter.render()

                seq, frame, jpeg_bytes = stream.read_latest(last_seq)
                if frame is None:
                    # Keep displaying the latest camera frame between 30fps updates (no flashing!)
                    if last_feed_display is not None and not show_directions:
                        left_panel = last_feed_display
                    else:
                        left_panel = render_waiting_screen(
                            feed_w, feed_h, stream.public_url, active_ip, show_directions=show_directions
                        )
                    dashboard = np.hstack((left_panel, traj_img))
                    cv2.imshow(window_title, dashboard)
                    should_quit, updated_map, show_directions = handle_key(
                        cv2.waitKey(10) & 0xFF, traj_plotter, show_directions
                    )
                    if updated_map is not None:
                        traj_img = updated_map
                    if should_quit:
                        break
                    continue

                # Automatically dismiss directions once the first video frame arrives
                if frames_total == 0:
                    show_directions = False

                last_seq = seq
                frames_total += 1
                vid_frames_window += 1

                # Publish every incoming video frame immediately (reuses jpeg_bytes with zero re-encoding)
                broadcaster.publish_frame(frame_bgr=frame, jpeg_bytes=jpeg_bytes)

                # Run DPVO odometry every `frame_stride` frames
                if (frames_total - 1) % frame_stride == 0:
                    if frame.shape[0] != vo_height or frame.shape[1] != vo_width:
                        vo_frame = cv2.resize(frame, (vo_width, vo_height), interpolation=cv2.INTER_AREA)
                    else:
                        vo_frame = frame

                    # 3D Tensor: [C=3, H, W] (DPVO.__call__ adds [None, None] internally)
                    img_tensor = torch.from_numpy(vo_frame).permute(2, 0, 1).to(device)

                    t0 = time.perf_counter()
                    # Pass `seq` as timestamp so DPVO's damped linear motion model accounts for frame spacing
                    slam(seq, img_tensor, intrinsics)
                    infer_ms = (time.perf_counter() - t0) * 1000.0
                    vo_frames_window += 1

                    pose_7d = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                    all_poses = None
                    pcl_xyz, pcl_bgr = None, None
                    if slam.is_initialized and slam.n > 0:
                        # Extract all BA-refined keyframe poses in world coordinates (T_wc)
                        all_poses_7d = SE3(slam.pg.poses_[: slam.n]).inv().data.cpu().numpy()
                        pose_7d = all_poses_7d[-1]
                        all_poses = all_poses_7d[:, :3]
                        x, y, z = all_poses[-1]
                        # Extract DPVO's live 3D sparse point cloud and RGB/BGR patch colors
                        pcl_xyz = slam.pg.points_[: slam.m].cpu().numpy()
                        pcl_bgr = slam.pg.colors_.view(-1, 3)[: slam.m].cpu().numpy()
                        traj_img = traj_plotter.update(
                            x, y, z, all_poses_xyz=all_poses, pcl_xyz=pcl_xyz, pcl_bgr=pcl_bgr
                        )
                    else:
                        traj_img = traj_plotter.update(0.0, 0.0, 0.0)

                    broadcaster.publish_telemetry(
                        pose=pose_7d,
                        frame_id=seq,
                        fps=vid_fps,
                        vo_hz=vo_hz,
                        infer_ms=infer_ms,
                        initialized=bool(slam.is_initialized),
                        all_poses_xyz=all_poses,
                        pcl_xyz=pcl_xyz,
                        pcl_bgr=pcl_bgr,
                    )

                elapsed = time.time() - t_start
                if elapsed >= 1.0:
                    vid_fps = vid_frames_window / elapsed
                    vo_hz = vo_frames_window / elapsed
                    vid_frames_window = 0
                    vo_frames_window = 0
                    t_start = time.time()

                jpeg_kb = len(jpeg_bytes) / 1024.0 if jpeg_bytes else 0.0
                feed_display = frame.copy()
                cv2.rectangle(feed_display, (15, 15), (720, 115), (15, 15, 15), -1)
                cv2.rectangle(feed_display, (15, 15), (720, 115), (80, 80, 80), 1)

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
                        f"X: {x:+6.2f}m | Alt(-Y): {-y:+6.2f}m | Z: {z:+6.2f}m",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                    )

                cv2.putText(
                    feed_display,
                    f"Vid: {vid_fps:4.1f}FPS | VO(1/{frame_stride}): {infer_ms:4.1f}ms ({vo_hz:4.1f}Hz) | KF: {slam.n} | {jpeg_kb:3.0f}KB",
                    (30, 88),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 220, 255),
                    1,
                )

                if feed_display.shape[1] != feed_w or feed_display.shape[0] != feed_h:
                    feed_display = cv2.resize(feed_display, (feed_w, feed_h))
                last_feed_display = feed_display

                left_panel = (
                    render_waiting_screen(feed_w, feed_h, stream.public_url, active_ip, show_directions=True)
                    if show_directions
                    else last_feed_display
                )
                dashboard = np.hstack((left_panel, traj_img))

                cv2.imshow(window_title, dashboard)

                should_quit, updated_map, show_directions = handle_key(
                    cv2.waitKey(1) & 0xFF, traj_plotter, show_directions
                )
                if updated_map is not None:
                    traj_img = updated_map
                if should_quit:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        broadcaster.stop()
        stream.stop()
        mtx_server.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

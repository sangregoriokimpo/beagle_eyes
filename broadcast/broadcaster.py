import base64
import hashlib
import json
import queue
import socket
import struct
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def detect_lan_ip(fallback: str = "127.0.0.1") -> str:
    """Detects the active LAN/Wi-Fi IPv4 address of this laptop."""
    try:
        out = subprocess.check_output(["ip", "-4", "-br", "addr"], text=True, timeout=2.0)
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] != "lo" and not parts[0].startswith(("docker", "br-", "veth")):
                for token in parts[2:]:
                    ip = token.split("/")[0]
                    if not ip.startswith("127."):
                        return ip
    except Exception:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return fallback


def _encode_ws_frame(payload: bytes, opcode: int = 0x1) -> bytes:
    """Encodes a server-to-client RFC 6455 WebSocket frame (unmasked)."""
    first_byte = 0x80 | (opcode & 0x0F)
    length = len(payload)
    if length < 126:
        header = struct.pack("!BB", first_byte, length)
    elif length < 65536:
        header = struct.pack("!BBH", first_byte, 126, length)
    else:
        header = struct.pack("!BBQ", first_byte, 127, length)
    return header + payload


class _WSClient:
    """Non-blocking outbound queue for a single connected WebSocket client."""

    def __init__(self, sock: socket.socket, addr: Tuple[str, int]):
        self.sock = sock
        self.addr = addr
        # Small bounded queue with drop-oldest semantics so slow clients never lag SLAM
        self.queue: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=4)
        self.alive = True

    def enqueue(self, frame_bytes: bytes) -> None:
        if not self.alive:
            return
        try:
            self.queue.put_nowait(frame_bytes)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(frame_bytes)
            except queue.Full:
                pass

    def close(self) -> None:
        self.alive = False
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass


class _BroadcastHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: Tuple[str, int], RequestHandlerClass, broadcaster: "TelemetryBroadcaster"):
        super().__init__(server_address, RequestHandlerClass)
        self.broadcaster = broadcaster


class _BroadcastRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress per-frame HTTP access logs to keep terminal output clean
        return

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        b: TelemetryBroadcaster = self.server.broadcaster  # type: ignore[attr-defined]

        # Enable TCP_NODELAY so small telemetry & MJPEG chunks flush immediately without Nagle delay
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

        # 1. Check for WebSocket Upgrade request (supports ws://<ip>:<port>/ws, /telemetry, or /)
        upgrade = (self.headers.get("Upgrade") or "").lower()
        if upgrade == "websocket":
            self._handle_websocket(b)
            return

        path = self.path.split("?", 1)[0]

        # 2. Live MJPEG video stream
        if path in ("/video.mjpg", "/video_feed", "/stream.mjpg"):
            self._handle_mjpeg(b)
            return

        # 3. Single JPEG snapshot
        if path == "/snapshot.jpg":
            jpeg = b.get_latest_jpeg()
            if jpeg is None:
                self.send_response(503)
                self._send_cors_headers()
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)
            return

        # 4. JSON health/status endpoint
        if path in ("/status", "/health"):
            body = json.dumps(b.get_status()).encode("utf-8")
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # 5. Built-in browser diagnostic preview page at /
        html = b.render_preview_html().encode("utf-8")
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _handle_mjpeg(self, b: "TelemetryBroadcaster") -> None:
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        last_seq = -1
        try:
            while b.is_running:
                jpeg, seq = b.wait_for_jpeg(last_seq, timeout=1.0)
                if jpeg is None or seq == last_seq:
                    continue
                last_seq = seq
                header = (
                    f"--frame\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n"
                ).encode("ascii")
                self.wfile.write(header)
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _handle_websocket(self, b: "TelemetryBroadcaster") -> None:
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self.send_error(400, "Missing Sec-WebSocket-Key")
            return

        accept = base64.b64encode(hashlib.sha1((key.strip() + _WS_GUID).encode("ascii")).digest()).decode("ascii")
        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        ).encode("ascii")
        self.wfile.write(response)
        self.wfile.flush()

        client = _WSClient(self.connection, self.client_address)
        b._register_ws_client(client)

        # Monitor incoming ping/close frames in a small reader thread
        def _reader() -> None:
            try:
                while client.alive and b.is_running:
                    hdr = self.connection.recv(2)
                    if not hdr or len(hdr) < 2:
                        break
                    opcode = hdr[0] & 0x0F
                    masked = (hdr[1] & 0x80) != 0
                    length = hdr[1] & 0x7F
                    if length == 126:
                        ext = self.connection.recv(2)
                        length = struct.unpack("!H", ext)[0]
                    elif length == 127:
                        ext = self.connection.recv(8)
                        length = struct.unpack("!Q", ext)[0]
                    mask_key = self.connection.recv(4) if masked else b""
                    payload = self.connection.recv(length) if length > 0 else b""
                    if opcode == 0x8:  # Close frame
                        break
                    if opcode == 0x9:  # Ping -> send Pong
                        if masked and mask_key:
                            payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
                        client.enqueue(_encode_ws_frame(payload, opcode=0xA))
            except Exception:
                pass
            finally:
                client.close()

        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()

        try:
            # Immediately push latest telemetry snapshot if available so new clients don't wait
            latest_msg = b.get_latest_telemetry_bytes()
            if latest_msg is not None:
                self.connection.sendall(_encode_ws_frame(latest_msg, opcode=0x1))

            while client.alive and b.is_running:
                try:
                    frame_bytes = client.queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if frame_bytes is None:
                    break
                self.connection.sendall(frame_bytes)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            client.close()
            b._unregister_ws_client(client)


class TelemetryBroadcaster:
    """
    Low-latency HTTP MJPEG + WebSocket telemetry broadcaster for React dashboards.

    Serves on a single port (default 8765):
      - Video Stream (MJPEG) : http://<laptop-ip>:8765/video.mjpg
      - Telemetry WebSocket  : ws://<laptop-ip>:8765/ws
      - Snapshot JPEG        : http://<laptop-ip>:8765/snapshot.jpg
      - Health / Status JSON : http://<laptop-ip>:8765/status
      - Browser Test Page    : http://<laptop-ip>:8765/
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        jpeg_quality: int = 75,
        max_pcl_points: int = 2000,
        pcl_interval_sec: float = 0.20,
        max_trajectory_points: int = 300,
    ):
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self.max_pcl_points = int(max_pcl_points)
        self.pcl_interval_sec = float(pcl_interval_sec)
        self.max_trajectory_points = int(max_trajectory_points)

        self.lan_ip = detect_lan_ip()
        self.is_running = False

        self._jpeg_cond = threading.Condition()
        self._latest_jpeg: Optional[bytes] = None
        self._jpeg_seq: int = 0

        self._ws_lock = threading.Lock()
        self._ws_clients: List[_WSClient] = []
        self._latest_telemetry_bytes: Optional[bytes] = None
        self._last_pcl_sent_time: float = 0.0
        self._trajectory_history: List[List[float]] = []

        self._server: Optional[_BroadcastHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None

    @property
    def video_url(self) -> str:
        return f"http://{self.lan_ip}:{self.port}/video.mjpg"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.lan_ip}:{self.port}/ws"

    @property
    def status_url(self) -> str:
        return f"http://{self.lan_ip}:{self.port}/status"

    def start(self) -> "TelemetryBroadcaster":
        if self.is_running:
            return self
        self._server = _BroadcastHTTPServer((self.host, self.port), _BroadcastRequestHandler, self)
        self.is_running = True
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        print(f"[Broadcast] Server running on port {self.port}:")
        print(f"  -> Video (MJPEG) : {self.video_url}")
        print(f"  -> Telemetry (WS): {self.ws_url}")
        print(f"  -> Test Preview  : http://{self.lan_ip}:{self.port}/")
        return self

    def stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False
        with self._jpeg_cond:
            self._jpeg_cond.notify_all()
        with self._ws_lock:
            for c in list(self._ws_clients):
                c.close()
            self._ws_clients.clear()
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None

    def publish_frame(self, frame_bgr: Optional[np.ndarray] = None, jpeg_bytes: Optional[bytes] = None) -> None:
        """
        Publishes a video frame to all connected MJPEG clients.
        Pass `jpeg_bytes` directly if already encoded (e.g. inside uav_vo.py) to avoid re-encoding.
        """
        if not self.is_running:
            return
        if jpeg_bytes is None:
            if frame_bgr is None:
                return
            ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                return
            jpeg_bytes = buf.tobytes()

        with self._jpeg_cond:
            self._latest_jpeg = jpeg_bytes
            self._jpeg_seq += 1
            self._jpeg_cond.notify_all()

    def publish_telemetry(
        self,
        pose: Union[Sequence[float], np.ndarray],
        frame_id: int = 0,
        fps: float = 0.0,
        vo_hz: Optional[float] = None,
        infer_ms: Optional[float] = None,
        initialized: bool = True,
        all_poses_xyz: Optional[np.ndarray] = None,
        pcl_xyz: Optional[np.ndarray] = None,
        pcl_bgr: Optional[np.ndarray] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Publishes a low-latency JSON telemetry packet over WebSocket.
        - `pose` can be `[x, y, z]` or 7D `[x, y, z, qx, qy, qz, qw]`.
        - `pcl_xyz` / `pcl_bgr` are automatically filtered, converted BGR->RGB, downsampled,
          and throttled to `pcl_interval_sec` so pose updates stay fast.
        """
        if not self.is_running:
            return

        pose_arr = np.asarray(pose, dtype=np.float64).ravel()
        pos = [round(float(pose_arr[0]), 4), round(float(pose_arr[1]), 4), round(float(pose_arr[2]), 4)]
        if pose_arr.size >= 7:
            quat = [
                round(float(pose_arr[3]), 5),
                round(float(pose_arr[4]), 5),
                round(float(pose_arr[5]), 5),
                round(float(pose_arr[6]), 5),
            ]
        else:
            quat = [0.0, 0.0, 0.0, 1.0]

        if all_poses_xyz is not None and len(all_poses_xyz) > 0:
            pts = np.asarray(all_poses_xyz, dtype=np.float64)
            if len(pts) > self.max_trajectory_points:
                idx = np.linspace(0, len(pts) - 1, self.max_trajectory_points, dtype=int)
                pts = pts[idx]
            self._trajectory_history = np.round(pts[:, :3], 3).tolist()
        elif initialized:
            self._trajectory_history.append([round(pos[0], 3), round(pos[1], 3), round(pos[2], 3)])
            if len(self._trajectory_history) > self.max_trajectory_points:
                self._trajectory_history.pop(0)

        now = time.time()
        packet: Dict[str, Any] = {
            "type": "telemetry",
            "timestamp": round(now, 4),
            "frame_id": int(frame_id),
            "initialized": bool(initialized),
            "fps": round(float(fps), 1),
            "pose": {
                "position": pos,
                "quaternion": quat,
            },
            "trajectory": self._trajectory_history,
        }
        if vo_hz is not None:
            packet["vo_hz"] = round(float(vo_hz), 1)
        if infer_ms is not None:
            packet["infer_ms"] = round(float(infer_ms), 1)

        # Attach throttled 3D point cloud if provided
        if pcl_xyz is not None and (now - self._last_pcl_sent_time) >= self.pcl_interval_sec:
            pcl_payload = self._format_point_cloud(pcl_xyz, pcl_bgr, center_xyz=pose_arr[:3])
            if pcl_payload is not None:
                packet["point_cloud"] = pcl_payload
                self._last_pcl_sent_time = now

        if extra:
            packet.update(extra)

        raw_json = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        self._latest_telemetry_bytes = raw_json
        ws_frame = _encode_ws_frame(raw_json, opcode=0x1)

        with self._ws_lock:
            for client in self._ws_clients:
                client.enqueue(ws_frame)

    def _format_point_cloud(
        self,
        pcl_xyz: np.ndarray,
        pcl_bgr: Optional[np.ndarray],
        center_xyz: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        xyz = np.asarray(pcl_xyz, dtype=np.float32).reshape(-1, 3)
        if len(xyz) == 0:
            return None

        valid = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > 1e-3)
        if not np.any(valid):
            return None
        xyz = xyz[valid]

        rgb: Optional[np.ndarray] = None
        if pcl_bgr is not None:
            bgr = np.asarray(pcl_bgr, dtype=np.uint8).reshape(-1, 3)
            if len(bgr) == len(valid):
                bgr = bgr[valid]
                # Convert OpenCV BGR -> Web RGB
                rgb = bgr[:, ::-1]

        # Reject extreme outlier points (> 96th percentile distance from camera)
        dists = np.linalg.norm(xyz - center_xyz.reshape(1, 3), axis=1)
        if len(dists) > 16:
            clip_r = max(float(np.percentile(dists, 96)) * 1.5, 2.0)
            inlier = dists <= clip_r
            xyz = xyz[inlier]
            if rgb is not None:
                rgb = rgb[inlier]

        n_pts = len(xyz)
        if n_pts == 0:
            return None

        if n_pts > self.max_pcl_points:
            idx = np.linspace(0, n_pts - 1, self.max_pcl_points, dtype=int)
            xyz = xyz[idx]
            if rgb is not None:
                rgb = rgb[idx]

        # Flat arrays [x0,y0,z0, x1,y1,z1, ...] & [r0,g0,b0, ...] for zero-copy Float32Array in Three.js
        positions_flat = np.round(xyz.astype(np.float64), 3).ravel().tolist()
        if rgb is not None:
            colors_flat = rgb.ravel().tolist()
        else:
            colors_flat = [0, 220, 255] * len(xyz)

        return {
            "count": int(len(xyz)),
            "positions": positions_flat,
            "colors": colors_flat,
        }

    def wait_for_jpeg(self, last_seq: int, timeout: float = 1.0) -> Tuple[Optional[bytes], int]:
        with self._jpeg_cond:
            if self._jpeg_seq == last_seq:
                self._jpeg_cond.wait(timeout=timeout)
            return self._latest_jpeg, self._jpeg_seq

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._jpeg_cond:
            return self._latest_jpeg

    def get_latest_telemetry_bytes(self) -> Optional[bytes]:
        return self._latest_telemetry_bytes

    def _register_ws_client(self, client: _WSClient) -> None:
        with self._ws_lock:
            self._ws_clients.append(client)

    def _unregister_ws_client(self, client: _WSClient) -> None:
        with self._ws_lock:
            if client in self._ws_clients:
                self._ws_clients.remove(client)

    def get_status(self) -> Dict[str, Any]:
        with self._ws_lock:
            ws_count = len(self._ws_clients)
        return {
            "status": "ok" if self.is_running else "stopped",
            "lan_ip": self.lan_ip,
            "port": self.port,
            "video_url": self.video_url,
            "ws_url": self.ws_url,
            "ws_clients": ws_count,
            "frames_published": self._jpeg_seq,
        }

    def render_preview_html(self) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Beagle Eyes Broadcast Monitor</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #0f1117; color: #e2e8f0; display: flex; flex-direction: column; height: 100vh; }}
    header {{ padding: 12px 20px; background: #161922; border-bottom: 1px solid #262b38; display: flex; justify-content: space-between; align-items: center; }}
    .badge {{ background: #10b981; color: #052e16; font-weight: 700; padding: 3px 10px; border-radius: 999px; font-size: 12px; }}
    main {{ display: grid; grid-template-columns: 1.3fr 1fr; gap: 16px; padding: 16px; flex: 1; min-height: 0; }}
    .card {{ background: #161922; border: 1px solid #262b38; border-radius: 10px; padding: 14px; display: flex; flex-direction: column; overflow: hidden; }}
    .card h3 {{ margin: 0 0 10px 0; font-size: 14px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
    img {{ width: 100%; height: 100%; object-fit: contain; background: #090b0e; border-radius: 6px; }}
    pre {{ margin: 0; font-family: ui-monospace, monospace; font-size: 12px; color: #38bdf8; overflow: auto; flex: 1; background: #090b0e; padding: 12px; border-radius: 6px; }}
    code {{ color: #f8fafc; background: #1e2433; padding: 2px 6px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <div><strong>Beagle Eyes</strong> Live Broadcast Monitor &nbsp;|&nbsp; Video: <code>{self.video_url}</code> &nbsp;|&nbsp; WS: <code>{self.ws_url}</code></div>
    <span id="status" class="badge">CONNECTING...</span>
  </header>
  <main>
    <div class="card">
      <h3>Live HTTP MJPEG Stream (/video.mjpg)</h3>
      <img src="/video.mjpg" alt="Waiting for video frames..." />
    </div>
    <div class="card">
      <h3>Live WebSocket Telemetry (/ws)</h3>
      <pre id="telemetry">Waiting for telemetry packets...</pre>
    </div>
  </main>
  <script>
    const statusEl = document.getElementById("status");
    const preEl = document.getElementById("telemetry");
    let lastPclCount = 0;
    function connect() {{
      const ws = new WebSocket(`ws://${{location.host}}/ws`);
      ws.onopen = () => {{ statusEl.textContent = "LIVE"; statusEl.style.background = "#10b981"; }};
      ws.onclose = () => {{ statusEl.textContent = "RECONNECTING"; statusEl.style.background = "#f59e0b"; setTimeout(connect, 1000); }};
      ws.onmessage = (ev) => {{
        const data = JSON.parse(ev.data);
        if (data.point_cloud) {{
          lastPclCount = data.point_cloud.count;
          data.point_cloud = {{ count: data.point_cloud.count, note: "positions & colors arrays received" }};
        }}
        data.last_point_cloud_count = lastPclCount;
        preEl.textContent = JSON.stringify(data, null, 2);
      }};
    }}
    connect();
  </script>
</body>
</html>"""


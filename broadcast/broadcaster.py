import base64
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
import queue
import socket
import struct
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.request import Request, urlopen
import uuid

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

        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

        upgrade = (self.headers.get("Upgrade") or "").lower()
        if upgrade == "websocket":
            self._handle_websocket(b)
            return

        path = self.path.split("?", 1)[0]

        if path in ("/video.mjpg", "/video_feed", "/stream.mjpg"):
            self._handle_mjpeg(b)
            return

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

        if path in ("/snapshot", "/bundle"):
            export_token = os.getenv("BEAGLE_EXPORT_TOKEN", "")
            if export_token and not hmac.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + export_token
            ):
                self.send_response(401)
                self._send_cors_headers()
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            bundle = b.get_firesight_bundle()
            if bundle is None:
                self.send_response(204)
                self._send_cors_headers()
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(bundle)))
            self.end_headers()
            self.wfile.write(bundle)
            return

        if path in ("/status", "/health"):
            body = json.dumps(b.get_status()).encode("utf-8")
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

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
                    if opcode == 0x8:
                        break
                    if opcode == 0x9:
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
            latest_msg = b.get_latest_telemetry_bytes()
            if latest_msg is not None:
                self.connection.sendall(_encode_ws_frame(latest_msg, opcode=0x1))
            latest_jpeg = b.get_latest_jpeg()
            if latest_jpeg is not None:
                self.connection.sendall(_encode_ws_frame(latest_jpeg, opcode=0x2))

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
    Low-latency HTTP MJPEG + WebSocket telemetry broadcaster for React & FireSight dashboards.

    Serves on a single port (default 8765):
      - Video Stream (MJPEG) : http://<laptop-ip>:8765/video.mjpg
      - Telemetry WebSocket  : ws://<laptop-ip>:8765/ws
      - Snapshot JPEG        : http://<laptop-ip>:8765/snapshot.jpg
      - FireSight Bundle     : http://<laptop-ip>:8765/snapshot
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
        self.session_id = str(uuid.uuid4())

        self._jpeg_cond = threading.Condition()
        self._latest_jpeg: Optional[bytes] = None
        self._jpeg_seq: int = 0
        self._has_real_frame: bool = False
        self._last_real_frame_time: float = 0.0

        self._ws_lock = threading.Lock()
        self._ws_clients: List[_WSClient] = []
        self._latest_telemetry_bytes: Optional[bytes] = None
        self._latest_telemetry_dict: Optional[Dict[str, Any]] = None
        self._last_pcl_sent_time: float = 0.0
        self._trajectory_history: List[List[float]] = []

        self._bundle_lock = threading.Lock()
        self._latest_bundle_bytes: Optional[bytes] = None
        self._last_bundle_time: float = 0.0
        self._push_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=1)

        self._server: Optional[_BroadcastHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._push_thread: Optional[threading.Thread] = None

    @property
    def video_url(self) -> str:
        return f"http://{self.lan_ip}:{self.port}/video.mjpg"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.lan_ip}:{self.port}/ws"

    @property
    def status_url(self) -> str:
        return f"http://{self.lan_ip}:{self.port}/status"

    @property
    def snapshot_url(self) -> str:
        return f"http://{self.lan_ip}:{self.port}/snapshot"

    def start(self) -> "TelemetryBroadcaster":
        if self.is_running:
            return self
        try:
            self._server = _BroadcastHTTPServer((self.host, self.port), _BroadcastRequestHandler, self)
        except OSError:
            print(f"[Broadcast] Port {self.port} is busy — freeing stale broadcaster process...")
            subprocess.run(["fuser", "-k", f"{self.port}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.4)
            self._server = _BroadcastHTTPServer((self.host, self.port), _BroadcastRequestHandler, self)
        self.is_running = True
        if self._latest_jpeg is None:
            placeholder = np.full((360, 640, 3), (18, 22, 30), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "BEAGLE EYES BROADCASTER ONLINE",
                (75, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                placeholder,
                "Waiting for camera / drone stream...",
                (130, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (160, 175, 195),
                1,
                cv2.LINE_AA,
            )
            self.publish_frame(frame_bgr=placeholder, _is_placeholder=True)
            self.publish_telemetry(
                pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                frame_id=0,
                fps=0.0,
                initialized=False,
                _is_placeholder=True,
            )
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        if os.getenv("FIRESIGHT_URL"):
            self._push_thread = threading.Thread(target=self._push_worker, daemon=True)
            self._push_thread.start()
        print(f"[Broadcast] Server running on port {self.port}:")
        print(f"  -> Video (MJPEG)    : {self.video_url}")
        print(f"  -> Telemetry (WS)   : {self.ws_url}")
        print(f"  -> FireSight Bundle : {self.snapshot_url}")
        print(f"  -> Test Preview     : http://{self.lan_ip}:{self.port}/")
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

    def publish_frame(
        self,
        frame_bgr: Optional[np.ndarray] = None,
        jpeg_bytes: Optional[bytes] = None,
        _is_placeholder: bool = False,
    ) -> None:
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
            if not _is_placeholder:
                self._has_real_frame = True
                self._last_real_frame_time = time.time()
            self._jpeg_cond.notify_all()

        ws_bin_frame = _encode_ws_frame(jpeg_bytes, opcode=0x2)
        with self._ws_lock:
            for client in self._ws_clients:
                client.enqueue(ws_bin_frame)

        if not _is_placeholder:
            self._maybe_update_bundle()

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
        _is_placeholder: bool = False,
    ) -> None:
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

        if pcl_xyz is not None and (now - self._last_pcl_sent_time) >= self.pcl_interval_sec:
            pcl_payload = self._format_point_cloud(pcl_xyz, pcl_bgr, center_xyz=pose_arr[:3])
            if pcl_payload is not None:
                packet["point_cloud"] = pcl_payload
                self._last_pcl_sent_time = now

        if extra:
            packet.update(extra)

        raw_json = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        self._latest_telemetry_bytes = raw_json
        if not _is_placeholder:
            # Store telemetry copy (without bulky point_cloud array) for FireSight bundle metadata
            lite_packet = {k: v for k, v in packet.items() if k != "point_cloud"}
            self._latest_telemetry_dict = lite_packet
        ws_frame = _encode_ws_frame(raw_json, opcode=0x1)

        with self._ws_lock:
            for client in self._ws_clients:
                client.enqueue(ws_frame)

        if not _is_placeholder:
            self._maybe_update_bundle()

    def _maybe_update_bundle(self) -> None:
        now = time.time()
        if not self._has_real_frame or self._latest_telemetry_dict is None:
            return
        if now - self._last_bundle_time < 0.45:
            return
        with self._jpeg_cond:
            jpeg = self._latest_jpeg
            seq = self._jpeg_seq
        if not jpeg:
            return
        telemetry = dict(self._latest_telemetry_dict)
        # Ensure monotonically increasing source frame_id for deduplication
        source_frame_id = max(int(telemetry.get("frame_id", 0)), int(seq))
        telemetry["frame_id"] = source_frame_id
        observed = float(telemetry.get("timestamp") or now)
        sha256_hex = hashlib.sha256(jpeg).hexdigest()
        jpeg_b64 = base64.b64encode(jpeg).decode("ascii")
        frame_uuid = str(uuid.uuid5(uuid.UUID(self.session_id), str(source_frame_id)))
        pos = telemetry.get("pose", {}).get("position")
        initialized = bool(telemetry.get("initialized", False))
        context: Dict[str, Any] = {
            "producer": "beagle_eyes",
            "odometry_initialized": initialized,
        }
        if pos is not None:
            context["latest_keyframe_pose"] = {
                "translation": [float(v) for v in pos[:3]],
                "units": "meters_vo",
                "transform": "camera_to_world",
            }
        metadata = {
            "schema_version": "firesight.frame.v1",
            "frame_id": frame_uuid,
            "session_id": self.session_id,
            "source_frame_id": source_frame_id,
            "received_at": datetime.fromtimestamp(observed, timezone.utc).isoformat(),
            "image": {"sha256": sha256_hex},
            "telemetry": telemetry,
            "context": context,
        }
        bundle_bytes = json.dumps(
            {
                "session_id": self.session_id,
                "telemetry": telemetry,
                "jpeg_base64": jpeg_b64,
                "sha256": sha256_hex,
                "metadata": metadata,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        with self._bundle_lock:
            self._latest_bundle_bytes = bundle_bytes
            self._last_bundle_time = now
        if os.getenv("FIRESIGHT_URL"):
            try:
                self._push_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._push_queue.put_nowait(bundle_bytes)
            except queue.Full:
                pass

    def get_firesight_bundle(self) -> Optional[bytes]:
        with self._bundle_lock:
            if self._latest_bundle_bytes is None or (time.time() - self._last_bundle_time) > 5.0:
                return None
            return self._latest_bundle_bytes

    def _push_worker(self) -> None:
        base_url = os.getenv("FIRESIGHT_URL", "").rstrip("/")
        if not base_url:
            return
        ingest_url = base_url + "/ingest/beagle"
        while self.is_running:
            try:
                payload = self._push_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            token = os.getenv("INGEST_TOKEN", "")
            headers = {"Content-Type": "application/json"}
            if token:
                headers["Authorization"] = "Bearer " + token
            req = Request(ingest_url, data=payload, headers=headers)
            try:
                with urlopen(req, timeout=5.0) as resp:
                    resp.read()
            except Exception:
                pass
            time.sleep(0.5)

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
                rgb = bgr[:, ::-1]

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
      <h3>Live Video Stream</h3>
      <img id="video-feed" alt="Waiting for video frames..." />
    </div>
    <div class="card">
      <h3>Live WebSocket Telemetry (/ws)</h3>
      <pre id="telemetry">Waiting for telemetry packets...</pre>
    </div>
  </main>
  <script>
    const statusEl = document.getElementById("status");
    const preEl = document.getElementById("telemetry");
    const imgEl = document.getElementById("video-feed");
    let prevBlobUrl = null;
    let lastPclCount = 0;
    function connect() {{
      const wsProto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${{wsProto}}://${{location.host}}/ws`);
      ws.binaryType = "blob";
      ws.onopen = () => {{ statusEl.textContent = "LIVE"; statusEl.style.background = "#10b981"; }};
      ws.onclose = () => {{ statusEl.textContent = "RECONNECTING"; statusEl.style.background = "#f59e0b"; setTimeout(connect, 1000); }};
      ws.onmessage = (ev) => {{
        if (ev.data instanceof Blob) {{
          const nextUrl = URL.createObjectURL(ev.data);
          imgEl.src = nextUrl;
          if (prevBlobUrl) URL.revokeObjectURL(prevBlobUrl);
          prevBlobUrl = nextUrl;
          return;
        }}
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


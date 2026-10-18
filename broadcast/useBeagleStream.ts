import { useEffect, useRef, useState } from "react";

export interface BeaglePose {
  position: [number, number, number]; // [x, y, z] in meters
  quaternion: [number, number, number, number]; // [qx, qy, qz, qw]
}

export interface BeaglePointCloud {
  count: number;
  positions: Float32Array; // Flat [x0, y0, z0, x1, y1, z1, ...] ready for Three.js BufferAttribute
  colors: Uint8Array;      // Flat [r0, g0, b0, r1, g1, b1, ...] in 0..255 RGB
}

export interface BeagleTelemetry {
  timestamp: number;
  frame_id: number;
  initialized: boolean;
  fps: number;
  vo_hz?: number;
  infer_ms?: number;
  pose: BeaglePose;
  trajectory: [number, number, number][];
}

export function useBeagleStream(hostAndPort = "127.0.0.1:8765") {
  const [connected, setConnected] = useState(false);
  const [telemetry, setTelemetry] = useState<BeagleTelemetry | null>(null);
  const pointCloudRef = useRef<BeaglePointCloud | null>(null);

  const cleanHost = hostAndPort.replace(/^(https?|wss?):\/\//, "").replace(/\/.*$/, "");
  const videoUrl = `http://${cleanHost}/video.mjpg`;
  const wsUrl = `ws://${cleanHost}/ws`;

  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let unmounted = false;

    const connect = () => {
      if (unmounted) return;
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        if (!unmounted) setConnected(true);
      };

      ws.onclose = () => {
        if (!unmounted) {
          setConnected(false);
          reconnectTimer = setTimeout(connect, 1000);
        }
      };

      ws.onerror = () => {
        ws?.close();
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.point_cloud) {
            pointCloudRef.current = {
              count: msg.point_cloud.count,
              positions: new Float32Array(msg.point_cloud.positions),
              colors: new Uint8Array(msg.point_cloud.colors),
            };
          }
          setTelemetry({
            timestamp: msg.timestamp,
            frame_id: msg.frame_id,
            initialized: msg.initialized,
            fps: msg.fps,
            vo_hz: msg.vo_hz,
            infer_ms: msg.infer_ms,
            pose: msg.pose,
            trajectory: msg.trajectory ?? [],
          });
        } catch {
          // Ignore malformed packet
        }
      };
    };

    connect();

    return () => {
      unmounted = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, [wsUrl]);

  return {
    connected,
    videoUrl,
    wsUrl,
    telemetry,
    pointCloudRef,
  };
}


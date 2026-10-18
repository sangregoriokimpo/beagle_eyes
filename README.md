# Beagle Eyes — Real-Time Monocular Visual Odometry & 3D Point Cloud (DPVO)

Real-time monocular Visual Odometry (VO) and 3D sparse point cloud reconstruction using **DPVO** (`Deep-Visual-Odometry-SLAM`), supporting both:
- **DJI Mini 5 Pro (DJI RC 2)** live RTMP video streams via **MediaMTX** + low-latency FFmpeg JPEG (`image2pipe` / `mjpeg`) decoding (`scripts/uav_vo.py`)
- **Local USB / Laptop Webcam** (`scripts/webcam_vo.py`)

---

## 1. Environment Activation

```bash
conda activate visual_slam
cd ~/beagle_eyes
```

---

## 2. Automated MediaMTX & Static IP Setup (DJI RC 2)

Configure your network settings in [`config/uav_config.yaml`](config/uav_config.yaml) (`network.static_ip`, `network.subnet_prefix`, `network.gateway`), then run the setup script once to install MediaMTX (`~/beagle_eyes/vendor/mediamtx/mediamtx`), apply the static IP via NetworkManager (`nmcli`), and open TCP port `1935`:

```bash
# Run automated setup using default config/uav_config.yaml
./scripts/setup_uav_rtmp.sh

# Or specify a custom config path
./scripts/setup_uav_rtmp.sh config/uav_config.yaml
```

### Common Network Presets (`config/uav_config.yaml`)
- **iPhone Personal Hotspot (`172.20.10.0/28`)**:
  ```yaml
  network:
    static_ip: "172.20.10.2"
    subnet_prefix: 28
    gateway: "172.20.10.1"
  ```
- **Home / Router Wi-Fi (`192.168.7.0/24` or `192.168.1.0/24`)**:
  ```yaml
  network:
    static_ip: "192.168.7.223"
    subnet_prefix: 24
    gateway: "192.168.7.1"
  ```

---

## 3. Running UAV Visual Odometry (`scripts/uav_vo.py`)

`uav_vo.py` automatically starts **MediaMTX** (`config/mediamtx.yml`) on port `1935` if it is not already running, pulls the live RTMP stream via FFmpeg as a zero-lag JPEG stream, and runs DPVO on the GPU.

```bash
# Standard launch (uses settings from config/uav_config.yaml)
python scripts/uav_vo.py --config config/uav_config.yaml

# Run static IP + MediaMTX setup automatically before launching VO
python scripts/uav_vo.py --config config/uav_config.yaml --setup-static-ip

# Launch in 3D mode (3D trajectory + live RGB point cloud)
python scripts/uav_vo.py --plot-mode 3d

# Launch in 2D mode (top-down X-Z trajectory)
python scripts/uav_vo.py --plot-mode 2d

# Optimize speed: compute VO every 2nd or 3rd frame while keeping 30 FPS video display
python scripts/uav_vo.py --stride 2
python scripts/uav_vo.py --stride 3

# Optimize GPU compute (~4x faster CNN extraction): run VO at half resolution (640x352) while displaying 720p
python scripts/uav_vo.py --stride 2 --vo-scale 0.5

# Combine all flags
python scripts/uav_vo.py --config config/uav_config.yaml --plot-mode 3d --stride 2 --vo-scale 0.5
```

### DJI Mini 5 Pro (RC 2) Controller Settings
1. Connect the **DJI RC 2** and your laptop to the same Wi-Fi network.
2. In **DJI Fly**, go to **Settings (`...`) -> Transmission -> Live Streaming Platforms -> RTMP**.
3. Enter the RTMP URL shown in the terminal/waiting screen:
   ```text
   rtmp://<STATIC_IP>:1935/live/dji
   ```
   *(e.g., `rtmp://172.20.10.2:1935/live/dji`)*
4. Select **720p (1280x720) @ 30fps** and tap **Start Live Stream**.

---

## 4. Running Local Webcam Visual Odometry (`scripts/webcam_vo.py`)

To test DPVO directly using `/dev/video0` (local webcam):

```bash
python scripts/webcam_vo.py
```

---

## 5. Interactive Map & Keyboard Controls (`uav_vo.py`)

Interact directly with the trajectory / point cloud panel on the right side of the OpenCV window:

| Input | Action |
| :--- | :--- |
| **Left-Click + Drag** | Rotate the 3D map (horizontal = Yaw, vertical = Pitch) |
| **Right-Click + Drag** | Pan / drag the map across the canvas (works in 2D & 3D) |
| **Mouse Scroll Wheel** | Zoom in / out (`0.25x` – `8.0x`) |
| **Double Left-Click** or **`r`** | Reset rotation, pan, and zoom to default |
| **`Spacebar`** | Dismiss / toggle the setup directions menu |
| **`v`** | Toggle live between **2D** (top-down) and **3D** (perspective + point cloud) |
| **`p`** | Toggle live **3D RGB Point Cloud** overlay ON / OFF (in 3D mode) |
| **`q`** | Quit and cleanly stop FFmpeg & MediaMTX |

---

## 6. Manual MediaMTX Commands (Optional)

`uav_vo.py` starts and stops MediaMTX automatically when `mediamtx.auto_start: true` in `config/uav_config.yaml`. If you ever want to run or test MediaMTX standalone:

```bash
# Run MediaMTX manually in the foreground
~/beagle_eyes/vendor/mediamtx/mediamtx ~/beagle_eyes/config/mediamtx.yml

# Test pushing a synthetic 720p@30fps stream into MediaMTX
ffmpeg -re -f lavfi -i testsrc=size=1280x720:rate=30 \
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p \
  -f flv rtmp://127.0.0.1:1935/live/dji
```
# Beagle Eyes — Real-Time Monocular Visual Odometry & 3D Point Cloud (DPVO)

Real-time monocular Visual Odometry (VO) and 3D sparse point cloud reconstruction using **DPVO** (`Deep-Visual-Odometry-SLAM`), supporting both:
- **DJI Mini 5 Pro (DJI RC 2)** live RTMP video streams via **MediaMTX** + low-latency FFmpeg JPEG (`image2pipe` / `mjpeg`) decoding (`scripts/uav_vo.py`)
- **Local USB / Laptop Webcam** (`scripts/webcam_vo.py`)

---

## 1. Environment Activation

```bash
conda activate visual_slam
cd ~/beagle_eyes
```

---

## 2. Automated MediaMTX & Static IP Setup (DJI RC 2)

Configure your network settings in [`config/uav_config.yaml`](config/uav_config.yaml) (`network.static_ip`, `network.subnet_prefix`, `network.gateway`), then run the setup script once to install MediaMTX (`~/beagle_eyes/vendor/mediamtx/mediamtx`), apply the static IP via NetworkManager (`nmcli`), and open TCP port `1935`:

```bash
# Run automated setup using default config/uav_config.yaml
./scripts/setup_uav_rtmp.sh

# Or specify a custom config path
./scripts/setup_uav_rtmp.sh config/uav_config.yaml
```

### Common Network Presets (`config/uav_config.yaml`)
- **iPhone Personal Hotspot (`172.20.10.0/28`)**:
  ```yaml
  network:
    static_ip: "172.20.10.2"
    subnet_prefix: 28
    gateway: "172.20.10.1"
  ```
- **Home / Router Wi-Fi (`192.168.7.0/24` or `192.168.1.0/24`)**:
  ```yaml
  network:
    static_ip: "192.168.7.223"
    subnet_prefix: 24
    gateway: "192.168.7.1"
  ```

---

## 3. Running UAV Visual Odometry (`scripts/uav_vo.py`)

`uav_vo.py` automatically starts **MediaMTX** (`config/mediamtx.yml`) on port `1935` if it is not already running, pulls the live RTMP stream via FFmpeg as a zero-lag JPEG stream, and runs DPVO on the GPU.

```bash
# Standard launch (uses settings from config/uav_config.yaml)
python scripts/uav_vo.py --config config/uav_config.yaml

# Run static IP + MediaMTX setup automatically before launching VO
python scripts/uav_vo.py --config config/uav_config.yaml --setup-static-ip

# Launch in 3D mode (3D trajectory + live RGB point cloud)
python scripts/uav_vo.py --plot-mode 3d

# Launch in 2D mode (top-down X-Z trajectory)
python scripts/uav_vo.py --plot-mode 2d

# Optimize speed: compute VO every 2nd or 3rd frame while keeping 30 FPS video display
python scripts/uav_vo.py --stride 2
python scripts/uav_vo.py --stride 3

# Optimize GPU compute (~4x faster CNN extraction): run VO at half resolution (640x352) while displaying 720p
python scripts/uav_vo.py --stride 2 --vo-scale 0.5

# Combine all flags
python scripts/uav_vo.py --config config/uav_config.yaml --plot-mode 3d --stride 2 --vo-scale 0.5
```

### DJI Mini 5 Pro (RC 2) Controller Settings
1. Connect the **DJI RC 2** and your laptop to the same Wi-Fi network.
2. In **DJI Fly**, go to **Settings (`...`) -> Transmission -> Live Streaming Platforms -> RTMP**.
3. Enter the RTMP URL shown in the terminal/waiting screen:
   ```text
   rtmp://<STATIC_IP>:1935/live/dji
   ```
   *(e.g., `rtmp://172.20.10.2:1935/live/dji`)*
4. Select **720p (1280x720) @ 30fps** and tap **Start Live Stream**.

---

## 4. Running Local Webcam Visual Odometry (`scripts/webcam_vo.py`)

To test DPVO directly using `/dev/video0` (local webcam):

```bash
python scripts/webcam_vo.py
```

---

## 5. Interactive Map & Keyboard Controls (`uav_vo.py`)

Interact directly with the trajectory / point cloud panel on the right side of the OpenCV window:

| Input | Action |
| :--- | :--- |
| **Left-Click + Drag** | Rotate the 3D map (horizontal = Yaw, vertical = Pitch) |
| **Right-Click + Drag** | Pan / drag the map across the canvas (works in 2D & 3D) |
| **Mouse Scroll Wheel** | Zoom in / out (`0.25x` – `8.0x`) |
| **Double Left-Click** or **`r`** | Reset rotation, pan, and zoom to default |
| **`v`** | Toggle live between **2D** (top-down) and **3D** (perspective + point cloud) |
| **`p`** | Toggle live **3D RGB Point Cloud** overlay ON / OFF (in 3D mode) |
| **`q`** | Quit and cleanly stop FFmpeg & MediaMTX |

---

## 6. Manual MediaMTX Commands (Optional)

`uav_vo.py` starts and stops MediaMTX automatically when `mediamtx.auto_start: true` in `config/uav_config.yaml`. If you ever want to run or test MediaMTX standalone:

```bash
# Run MediaMTX manually in the foreground
~/beagle_eyes/vendor/mediamtx/mediamtx ~/beagle_eyes/config/mediamtx.yml

# Test pushing a synthetic 720p@30fps stream into MediaMTX
ffmpeg -re -f lavfi -i testsrc=size=1280x720:rate=30 \
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p \
  -f flv rtmp://127.0.0.1:1935/live/dji
```

---

## 7. Broadcasting Telemetry & Video to React Dashboard (Order of Operations)

`beagle_eyes` includes a built-in low-latency broadcaster (`broadcast/broadcaster.py`) on port **`8765`** that streams live video (HTTP MJPEG + binary WebSocket JPEG frames) and 30Hz telemetry + 3D point clouds over WebSocket (`/ws`).

### Step 0 (One-Time): Compile DPVO CUDA Extensions (If Needed)
If `cuda_corr`, `cuda_ba`, or `lietorch_backends` ever need to be rebuilt:
```bash
./scripts/setup_dpvo.sh
```

### Step 1 — Terminal 1 (Laptop A): Start the Broadcaster (Port `8765`)
Activate `visual_slam`, clean up any stale VO processes, and start **one** of the following broadcast sources:
```bash
conda activate visual_slam
cd ~/beagle_eyes

# Optional: kill any old background VO/mock processes holding port 8765
pkill -9 -f "uav_vo.py|webcam_vo.py|mock_sender.py" || true

# Option A — Mock 3D flight + point cloud (instant test, no drone or GPU needed)
python broadcast/mock_sender.py
# (Or stream real webcam video with mock 3D telemetry: python broadcast/mock_sender.py --webcam)

# Option B — Laptop webcam + live DPVO
python scripts/webcam_vo.py

# Option C — DJI Drone RTMP + live DPVO
python scripts/uav_vo.py --config config/uav_config.yaml --plot-mode 3d --stride 2 --vo-scale 0.5
```
Leave Terminal 1 running. You can verify the local stream immediately in a browser at `http://127.0.0.1:8765/`.

### Step 2 — Terminal 2 (Laptop A): Start Cloudflare Tunnel (Cross-Network)
If Laptop A and Laptop B are **not** on the same Wi-Fi/LAN, open a **second terminal** on Laptop A and run:
```bash
cd ~/beagle_eyes
./scripts/start_cloudflare_tunnel.sh
```
Leave Terminal 2 running and copy the public `https://xxxx-xxxx-xxxx.trycloudflare.com` URL printed in the box.

*(If both laptops are on the same Wi-Fi/LAN, you can skip Step 2 and use Laptop A's LAN IP printed by Terminal 1, e.g. `192.168.1.50:8765`.)*

### Step 3 — Laptop B (Dashboard / Browser): Connect to the Stream
1. **Browser Test Monitor**: Open `https://xxxx-xxxx-xxxx.trycloudflare.com` (or `http://<laptop-A-ip>:8765/`) in Chrome/Firefox to verify live video and WebSocket telemetry JSON.
2. **React Integration**: Copy [`broadcast/useBeagleStream.ts`](broadcast/useBeagleStream.ts) into your React app and pass either the Cloudflare URL or LAN IP:
   ```tsx
   import { useBeagleStream } from "./hooks/useBeagleStream";

   export function Dashboard() {
     const { connected, imgRef, videoUrl, telemetry, pointCloudRef } = useBeagleStream(
       "https://xxxx-xxxx-xxxx.trycloudflare.com" // or "192.168.1.50:8765" on same LAN
     );

     return (
       <div>
         {/* Live video feed (renders binary WebSocket frames over Cloudflare with 0 React re-renders) */}
         <img ref={imgRef} src={videoUrl} alt="Live Feed" style={{ width: 640, height: 360 }} />

         <div>Status: {connected ? "LIVE" : "RECONNECTING"}</div>
         {telemetry && (
           <div>
             X: {telemetry.pose.position[0].toFixed(2)}m |
             Y: {telemetry.pose.position[1].toFixed(2)}m |
             Z: {telemetry.pose.position[2].toFixed(2)}m |
             FPS: {telemetry.fps}
           </div>
         )}
       </div>
     );
   }
   ```
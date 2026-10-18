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
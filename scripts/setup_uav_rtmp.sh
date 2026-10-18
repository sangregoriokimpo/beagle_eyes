#!/usr/bin/env bash
# ============================================================================
# Automated Static IP & MediaMTX RTMP Setup for DJI Mini 5 Pro (RC 2)
# Reads ~/beagle_eyes/config/uav_config.yaml, installs MediaMTX if missing,
# and configures NetworkManager static IPv4 + firewall.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${1:-${REPO_ROOT}/config/uav_config.yaml}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Error: Config file not found at ${CONFIG_FILE}"
    exit 1
fi

# Parse YAML config via python3
read -r CFG_IFACE STATIC_IP PREFIX GATEWAY DNS PORT APP STREAM_KEY MTX_BIN MTX_VER < <(
    python3 - "${CONFIG_FILE}" <<'PY'
import os, sys, yaml
with open(sys.argv[1], "r") as f:
    cfg = yaml.safe_load(f)
net = cfg.get("network", {})
rtmp = cfg.get("rtmp", {})
mtx = cfg.get("mediamtx", {})
print(
    net.get("interface", "auto"),
    net.get("static_ip", "192.168.1.100"),
    net.get("subnet_prefix", 24),
    net.get("gateway", "192.168.1.1"),
    net.get("dns", "8.8.8.8,1.1.1.1"),
    rtmp.get("port", 1935),
    rtmp.get("app", "live"),
    rtmp.get("stream_key", "dji"),
    os.path.expanduser(mtx.get("binary", "~/beagle_eyes/vendor/mediamtx/mediamtx")),
    mtx.get("version", "v1.11.3"),
)
PY
)

echo "============================================================"
echo " Beagle Eyes - DJI Mini 5 Pro MediaMTX & Static IP Setup"
echo "============================================================"

# 1. Install MediaMTX binary if not already installed
if [[ -x "${MTX_BIN}" ]]; then
    echo "[1/4] MediaMTX already installed at ${MTX_BIN}"
elif command -v mediamtx >/dev/null 2>&1; then
    echo "[1/4] System MediaMTX found at $(command -v mediamtx)"
else
    echo "[1/4] Downloading MediaMTX ${MTX_VER} to $(dirname "${MTX_BIN}")..."
    mkdir -p "$(dirname "${MTX_BIN}")"
    ARCH="$(uname -m)"
    case "${ARCH}" in
        x86_64) MTX_ARCH="linux_amd64" ;;
        aarch64|arm64) MTX_ARCH="linux_arm64v8" ;;
        *) echo "Unsupported architecture: ${ARCH}"; exit 1 ;;
    esac
    TARBALL_URL="https://github.com/bluenviron/mediamtx/releases/download/${MTX_VER}/mediamtx_${MTX_VER}_${MTX_ARCH}.tar.gz"
    curl -fsSL "${TARBALL_URL}" | tar -xz -C "$(dirname "${MTX_BIN}")" mediamtx
    chmod +x "${MTX_BIN}"
    echo "      Installed MediaMTX to ${MTX_BIN}"
fi

# 2. Auto-detect active Wi-Fi or Ethernet interface if set to "auto"
if [[ "${CFG_IFACE}" == "auto" ]]; then
    IFACE="$(nmcli -t -f DEVICE,TYPE,STATE device status | awk -F: '$2=="wifi" && $3=="connected" {print $1; exit}')"
    if [[ -z "${IFACE}" ]]; then
        IFACE="$(nmcli -t -f DEVICE,TYPE,STATE device status | awk -F: '$2=="ethernet" && $3=="connected" {print $1; exit}')"
    fi
    if [[ -z "${IFACE}" ]]; then
        IFACE="$(ip route show default 2>/dev/null | awk '{print $5; exit}')"
    fi
else
    IFACE="${CFG_IFACE}"
fi

if [[ -z "${IFACE}" ]]; then
    echo "Error: Could not auto-detect an active network interface."
    echo "Please connect to the router/hotspot shared with DJI RC 2 or set network.interface in ${CONFIG_FILE}."
    exit 1
fi

CONN_NAME="$(nmcli -t -f NAME,DEVICE connection show --active | awk -F: -v dev="${IFACE}" '$2==dev {print $1; exit}')"
if [[ -z "${CONN_NAME}" ]]; then
    echo "Error: No active NetworkManager profile found for interface '${IFACE}'."
    exit 1
fi

echo " Interface       : ${IFACE} (${CONN_NAME})"
echo " Target Static IP: ${STATIC_IP}/${PREFIX} (Gateway: ${GATEWAY})"
echo " DJI RC 2 URL    : rtmp://${STATIC_IP}:${PORT}/${APP}/${STREAM_KEY}"

# 3. Apply static IP via NetworkManager
echo "[2/4] Configuring static IPv4 on '${CONN_NAME}'..."
sudo nmcli connection modify "${CONN_NAME}" \
    ipv4.addresses "${STATIC_IP}/${PREFIX}" \
    ipv4.gateway "${GATEWAY}" \
    ipv4.dns "${DNS}" \
    ipv4.method manual

echo "[3/4] Reactivating connection '${CONN_NAME}'..."
sudo nmcli connection up "${CONN_NAME}"

# 4. Allow RTMP port through UFW if active
echo "[4/4] Checking firewall for TCP port ${PORT}..."
if command -v ufw >/dev/null 2>&1 && sudo ufw status | grep -q "Status: active"; then
    sudo ufw allow "${PORT}/tcp"
fi

echo ""
echo "Setup complete! Current IP on ${IFACE}:"
ip -4 -br addr show dev "${IFACE}"
echo ""
echo "On your DJI RC 2 (DJI Fly -> Transmission -> Live Streaming -> RTMP):"
echo "  RTMP Address : rtmp://${STATIC_IP}:${PORT}/${APP}/${STREAM_KEY}"
echo "  Resolution   : 720p (1280x720) @ 30fps"
echo ""
echo "Then start visual odometry (MediaMTX starts automatically):"
echo "  python scripts/uav_vo.py --config config/uav_config.yaml"

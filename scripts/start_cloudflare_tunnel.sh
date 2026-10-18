#!/usr/bin/env bash
# ==============================================================================
# Beagle Eyes — Zero-Config Cloudflare Tunnel Launcher (Port 8765)
# Exposes both HTTP MJPEG (/video.mjpg) and WebSocket (/ws) over a single
# public https://*.trycloudflare.com URL (no Cloudflare account or same Wi-Fi needed).
# ==============================================================================
set -euo pipefail

PORT="${1:-8765}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEAGLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CF_DIR="${BEAGLE_ROOT}/vendor/cloudflared"
CF_BIN="${CF_DIR}/cloudflared"

if command -v cloudflared >/dev/null 2>&1; then
    CF_BIN="$(command -v cloudflared)"
elif [[ ! -x "${CF_BIN}" ]]; then
    echo "[Cloudflare] Downloading standalone cloudflared binary to ${CF_BIN}..."
    mkdir -p "${CF_DIR}"
    curl -fL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" -o "${CF_BIN}"
    chmod +x "${CF_BIN}"
fi

echo "============================================================"
echo " Beagle Eyes — Starting Cloudflare Tunnel -> 127.0.0.1:${PORT}"
echo " Copy the https://....trycloudflare.com URL printed below"
echo " into useBeagleStream(\"https://....trycloudflare.com\")"
echo "============================================================"

exec "${CF_BIN}" tunnel --no-autoupdate --url "http://127.0.0.1:${PORT}"


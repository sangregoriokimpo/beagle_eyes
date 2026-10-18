#!/usr/bin/env bash
# ==============================================================================
# Beagle Eyes — Automated DPVO Submodule & CUDA Extension Setup Script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEAGLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DPVO_DIR="${BEAGLE_ROOT}/vendor/Deep-Visual-Odometry-SLAM/methods/dpvo"

echo "============================================================"
echo " Beagle Eyes — DPVO & CUDA Extension Setup"
echo "============================================================"

# 1. Ensure the visual_slam conda environment is active
if [[ "${CONDA_DEFAULT_ENV:-}" != "visual_slam" ]]; then
    if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "${HOME}/anaconda3/etc/profile.d/conda.sh"
        conda activate visual_slam
    elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
        conda activate visual_slam
    else
        echo "[WARN] 'visual_slam' conda env is not active. Please run: conda activate visual_slam"
    fi
fi
echo "[1/4] Python environment: $(which python) (env: ${CONDA_DEFAULT_ENV:-unknown})"

# 2. Initialize vendor/Deep-Visual-Odometry-SLAM submodule (without --recursive to avoid broken nested Pangolin url)
if [[ ! -f "${DPVO_DIR}/setup.py" ]]; then
    echo "[2/4] Initializing vendor/Deep-Visual-Odometry-SLAM submodule..."
    git -C "${BEAGLE_ROOT}" submodule update --init vendor/Deep-Visual-Odometry-SLAM
else
    echo "[2/4] Submodule vendor/Deep-Visual-Odometry-SLAM is present."
fi

# 3. Detect PyTorch's compiled CUDA version and match it to /usr/local/cuda-<ver>
TORCH_CUDA_VER="$(python -c 'import torch; print(torch.version.cuda or "")')"
echo "[3/4] PyTorch expects CUDA version: ${TORCH_CUDA_VER}"

if [[ -n "${TORCH_CUDA_VER}" && -d "/usr/local/cuda-${TORCH_CUDA_VER}" ]]; then
    export CUDA_HOME="/usr/local/cuda-${TORCH_CUDA_VER}"
elif [[ -d "/usr/local/cuda-13.0" ]]; then
    export CUDA_HOME="/usr/local/cuda-13.0"
elif [[ -d "/usr/local/cuda" ]]; then
    export CUDA_HOME="/usr/local/cuda"
else
    echo "[ERROR] No /usr/local/cuda installation found!"
    exit 1
fi

export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

echo "      Using CUDA_HOME=${CUDA_HOME}"
echo "      nvcc: $(which nvcc) ($("${CUDA_HOME}/bin/nvcc" --version | tail -n 1))"

# 4. Build and install DPVO CUDA extensions (cuda_corr, cuda_ba, lietorch_backends)
echo "[4/4] Compiling DPVO CUDA extensions..."
pip install -e "${DPVO_DIR}" --no-build-isolation

echo "Verifying compiled extensions..."
python -c "import sys; sys.path.insert(0, '${DPVO_DIR}'); import torch, cuda_corr, cuda_ba, lietorch_backends; from dpvo.dpvo import DPVO; print('[OK] DPVO + cuda_corr, cuda_ba, and lietorch_backends imported successfully!')"

echo "============================================================"
echo " Setup Complete! You can now run:"
echo "   python scripts/uav_vo.py --config config/uav_config.yaml --plot-mode 3d --stride 2 --vo-scale 0.5"
echo "   python scripts/webcam_vo.py"
echo "============================================================"


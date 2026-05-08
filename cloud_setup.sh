#!/usr/bin/env bash
# One-shot bootstrap for a fresh GCP c3-standard-8 box (Ubuntu 22.04 LTS or
# Debian 12 bookworm; both tested). Idempotent: re-running on a partially-
# built box is safe.
#
# Usage on the GCP box:
#   git clone <repo-url> ~/flightmare
#   cd ~/flightmare
#   bash cloud_setup.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[cloud_setup] repo root: ${REPO_ROOT}"

# 1) System deps (mirrors flightlib's CMake requirements + conda toolchain).
#    libzmqpp-dev was dropped from Debian 12 bookworm — if apt can't find it
#    (e.g. on bookworm), build zmqpp from source against libzmq3-dev.
if ! command -v cmake >/dev/null; then
    echo "[cloud_setup] installing apt dependencies"
    sudo apt-get update -y
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates curl \
        libeigen3-dev libopencv-dev libyaml-cpp-dev \
        libzmq3-dev \
        tmux htop

    # Try the apt-packaged zmqpp first (Ubuntu 22.04 / Debian 11); fall back
    # to a from-source build on Debian 12 bookworm.
    if ! sudo apt-get install -y --no-install-recommends libzmqpp-dev 2>/dev/null; then
        echo "[cloud_setup] libzmqpp-dev not in apt — building from source"
        if [[ ! -d /tmp/zmqpp ]]; then
            git clone --depth 1 https://github.com/zeromq/zmqpp.git /tmp/zmqpp
        fi
        (cd /tmp/zmqpp && make -j"$(nproc)" && sudo make install && sudo ldconfig)
    fi
fi

# 2) Miniforge.
if [[ ! -d "${HOME}/miniforge3" ]]; then
    echo "[cloud_setup] installing miniforge"
    curl -fsSL -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /tmp/miniforge.sh -b -p "${HOME}/miniforge3"
    rm -f /tmp/miniforge.sh
fi
# shellcheck disable=SC1091
source "${HOME}/miniforge3/etc/profile.d/conda.sh"

# 3) Conda env.
if ! conda env list | grep -q '^flightmare\b'; then
    if [[ -f "${REPO_ROOT}/environment.yml" ]]; then
        echo "[cloud_setup] creating conda env from environment.yml"
        conda env create -f "${REPO_ROOT}/environment.yml"
    else
        echo "[cloud_setup] no environment.yml — creating minimal flightmare env"
        conda create -y -n flightmare python=3.8
        conda activate flightmare
        pip install --upgrade pip
        pip install "stable-baselines3[extra]==2.3.0" "gymnasium>=0.28" \
            "torch==2.2.*" numpy ruamel.yaml pyyaml tensorboard optuna
    fi
fi
conda activate flightmare

# 4) Build flightlib C++.
export FLIGHTMARE_PATH="${REPO_ROOT}"
echo "[cloud_setup] FLIGHTMARE_PATH=${FLIGHTMARE_PATH}"

if ! grep -q "FLIGHTMARE_PATH" "${HOME}/.bashrc"; then
    echo "export FLIGHTMARE_PATH=${REPO_ROOT}" >> "${HOME}/.bashrc"
fi

cd "${FLIGHTMARE_PATH}/flightlib"
echo "[cloud_setup] building flightlib"
python3 setup.py build_ext --inplace

# 5) Sanity: instantiate the env.
cd "${FLIGHTMARE_PATH}/flightrl/training"
echo "[cloud_setup] running diag.py"
python diag.py || {
    echo "[cloud_setup] diag.py FAILED — aborting before manifest" >&2
    exit 3
}

# 6) Reproducibility manifest.
cd "${FLIGHTMARE_PATH}"
SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
SO_HASH="$(sha256sum flightlib/flightgym*.so 2>/dev/null | head -n1 || echo unknown)"
{
    echo "# RUN_MANIFEST"
    echo
    echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git SHA:   ${SHA}"
    echo "host:      $(uname -a)"
    echo
    echo "## flightgym shared object"
    echo "${SO_HASH}"
    echo
    echo "## conda env (frozen)"
    echo '```yaml'
    conda env export -n flightmare --no-builds || true
    echo '```'
} > "${FLIGHTMARE_PATH}/RUN_MANIFEST.md"

echo "[cloud_setup] DONE. Manifest at ${FLIGHTMARE_PATH}/RUN_MANIFEST.md"
echo "[cloud_setup] Next: bash flightrl/training/experiment/run_all_experiments.sh"

#!/usr/bin/env bash
# Condition C — Automated random-track curriculum (IROS21).
# Per-env per-episode random track around circle_8 baseline, with adaptive
# sigma controller (§3). Locked §4 HPs. 100M env steps.
#
# Usage: run_condition_C.sh <SEED> [TOTAL_STEPS] [SIGMA_UPDATE_EVERY]

set -euo pipefail

SEED="${1:?seed required}"
TOTAL_STEPS="${2:-100000000}"
C_UPDATE_EVERY="${3:-100}"

: "${FLIGHTMARE_PATH:?must export FLIGHTMARE_PATH}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_run_lib.sh"

RUN_DIR="${RUNS_ROOT}/C_seed${SEED}"

# Reset YAML to the linear_8 baseline (c=0 layout). applyPerturbedGates
# mutates in-process from there, but the env still loads its initial gates
# from this YAML at __init__.
apply_track linear_8

if is_done "${RUN_DIR}"; then
    echo "[C] SKIP — already done: ${RUN_DIR}"
    exit 0
fi
mkdir -p "${RUN_DIR}"

(cd "${TRAINING_DIR}" && python -u "${SCRIPT_DIR}/random_track_train.py" \
    "${RUN_DIR}" "${HPARAMS_JSON}" \
    --total-timesteps "${TOTAL_STEPS}" \
    --seed "${SEED}" \
    --log-every 500000 \
    --c-update-every "${C_UPDATE_EVERY}") 2>&1 | tee -a "${RUN_DIR}/train.log"

if ! is_done "${RUN_DIR}"; then
    echo "[C] WARN — final_model.zip missing in ${RUN_DIR}" >&2
    exit 2
fi
echo "[C] done seed=${SEED}  -> ${RUN_DIR}"

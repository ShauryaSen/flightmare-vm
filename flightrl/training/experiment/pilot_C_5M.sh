#!/usr/bin/env bash
# Condition C 5M-step pilot — validates random-track sampler + adaptive
# sigma controller before committing 100M x 3 seeds.
#
# Per §9 risk register: if pilot seed only reaches a low gate count or
# sigma never adapts, abort the campaign and debug C before launching.
#
# Usage: pilot_C_5M.sh

set -euo pipefail

: "${FLIGHTMARE_PATH:?must export FLIGHTMARE_PATH}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PILOT_RUN_DIR="${FLIGHTMARE_PATH}/flightrl/training/experiment/runs/C_pilot_5M"
if [[ -f "${PILOT_RUN_DIR}/final_model.zip" ]]; then
    echo "[pilot] already complete: ${PILOT_RUN_DIR}"
    exit 0
fi

# Run with a faster sigma update cadence so the controller fires multiple
# times within 5M steps.
"${SCRIPT_DIR}/run_condition_C.sh" 0 5000000 25
echo "[pilot] check ${PILOT_RUN_DIR}/sigma_history.json before launching the full campaign"

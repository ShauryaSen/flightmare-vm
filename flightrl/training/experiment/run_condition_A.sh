#!/usr/bin/env bash
# Condition A — No curriculum.
# PPO from scratch on circle_8 for 100M env steps. Locked §4 HPs.
#
# Usage: run_condition_A.sh <SEED> [TOTAL_STEPS]

set -euo pipefail

SEED="${1:?seed required}"
TOTAL_STEPS="${2:-100000000}"

: "${FLIGHTMARE_PATH:?must export FLIGHTMARE_PATH}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_run_lib.sh"

RUN_DIR="${RUNS_ROOT}/A_seed${SEED}"

apply_track circle_8
train_rung "${RUN_DIR}" "${TOTAL_STEPS}" "${SEED}"

echo "[A] done seed=${SEED}  -> ${RUN_DIR}"

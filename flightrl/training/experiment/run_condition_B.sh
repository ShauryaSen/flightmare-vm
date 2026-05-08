#!/usr/bin/env bash
# Condition B — Hand-designed 6-rung structural curriculum.
# Rungs: straight_2 → straight_3 → bent_3_15 → bent_3_30 → bent_3_45 → circle_8
# Steps: 10M / 10M / 12M / 13M / 15M / 40M  (= 100M total)
# Locked §4 HPs across every rung. Carries forward both PPO weights and
# VecNormalize running stats between rungs.
#
# Usage: run_condition_B.sh <SEED> [SCALE]
#   SCALE — optional float multiplier for per-rung budgets (e.g. 0.001 for smoke).

set -euo pipefail

SEED="${1:?seed required}"
SCALE="${2:-1.0}"

: "${FLIGHTMARE_PATH:?must export FLIGHTMARE_PATH}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_run_lib.sh"

ROOT="${RUNS_ROOT}/B_seed${SEED}"

# Per-rung step budgets (full; SCALE multiplier applied below for smoke runs).
declare -a RUNG_NAMES=(A_straight_2  B_straight_3  C_bent_3_15  D_bent_3_30  E_bent_3_45  F_circle_8)
declare -a RUNG_TRACKS=(straight_2   straight_3    bent_3_15    bent_3_30    bent_3_45    circle_8)
declare -a RUNG_STEPS_FULL=(10000000 10000000      12000000     13000000     15000000     40000000)

apply_steps() {
    python3 -c "import sys; print(max(int(round(int(sys.argv[1]) * float(sys.argv[2]))), 50000))" "$1" "${SCALE}"
}

prev_model=""
prev_vecnorm=""

for i in "${!RUNG_NAMES[@]}"; do
    name="${RUNG_NAMES[$i]}"
    track="${RUNG_TRACKS[$i]}"
    steps_full="${RUNG_STEPS_FULL[$i]}"
    steps="$(apply_steps "${steps_full}")"
    rung_dir="${ROOT}/${name}"

    apply_track "${track}"
    train_rung "${rung_dir}" "${steps}" "${SEED}" "${prev_model}" "${prev_vecnorm}"

    prev_model="${rung_dir}/final_model.zip"
    prev_vecnorm="${rung_dir}/final_vec_normalize.pkl"
done

echo "[B] done seed=${SEED}  -> ${ROOT}"

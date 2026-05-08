#!/usr/bin/env bash
# Top-level launcher for the EXPERIMENT_PLAN.md campaign.
# Run inside tmux on the GCP box:
#
#   tmux new -s train
#   cd ~/flightmare/flightrl/training/experiment
#   bash run_all_experiments.sh
#
# Sequence:
#   1) Condition C 5M-step pilot (1 seed)  -> sanity-check sigma adapter
#   2) Condition A x 3 seeds (100M each)
#   3) Condition B x 3 seeds (6 rungs, 100M each)
#   4) Condition C x 3 seeds (100M each)
#   5) Eval grid across all 9 final checkpoints
#
# Each step is idempotent; a re-run skips already-done seeds/rungs.

set -euo pipefail

: "${FLIGHTMARE_PATH:?must export FLIGHTMARE_PATH}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate flightmare
export FLIGHTMARE_PATH

cd "${SCRIPT_DIR}"

echo "============================================================"
echo "  STEP 1/5 : Condition C 5M pilot"
echo "============================================================"
bash "${SCRIPT_DIR}/pilot_C_5M.sh"

echo "============================================================"
echo "  STEP 2/5 : Condition A x {0,1,2}"
echo "============================================================"
for s in 0 1 2; do
    bash "${SCRIPT_DIR}/run_condition_A.sh" "${s}"
done

echo "============================================================"
echo "  STEP 3/5 : Condition B x {0,1,2}"
echo "============================================================"
for s in 0 1 2; do
    bash "${SCRIPT_DIR}/run_condition_B.sh" "${s}"
done

echo "============================================================"
echo "  STEP 4/5 : Condition C x {0,1,2}"
echo "============================================================"
for s in 0 1 2; do
    bash "${SCRIPT_DIR}/run_condition_C.sh" "${s}"
done

echo "============================================================"
echo "  STEP 5/5 : Eval grid"
echo "============================================================"
python "${SCRIPT_DIR}/run_eval_grid.py" \
    --runs-dir "${SCRIPT_DIR}/runs/" \
    --out-dir  "${SCRIPT_DIR}/runs/eval_grid/"

echo "============================================================"
echo "  ALL DONE — see runs/eval_grid/eval_grid_results.json"
echo "============================================================"

#!/usr/bin/env bash
# Usage: curriculum_eval.sh <rung_dir> <min_gates> <success_count>
# Runs deterministic eval (20 episodes from start_pos) and prints a single
# summary line: PASS/FAIL  mean_gates  success_count.
set -e
RUNG_DIR="$1"
MIN_GATES="${2:-1}"
SUCCESS_COUNT="${3:-18}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate flightmare
export FLIGHTMARE_PATH=/home/ssen/flightmare

cd /home/ssen/flightmare/flightrl/training

OUT=$(python -u gate_racing_iros.py --train 0 \
  --model-path "$RUNG_DIR/best_model.zip" \
  --vecnorm-path "$RUNG_DIR/vec_normalize.pkl" \
  --n-eval-episodes 20 \
  --min-gates "$MIN_GATES" \
  --success-count "$SUCCESS_COUNT" 2>&1)

echo "$OUT" > "$RUNG_DIR/eval.log"
echo "$OUT" | tail -40
echo "----"
echo "$OUT" | grep -E "Mean gates_passed|Mean reward|Mean ep length|Success criterion|Episodes with gates"

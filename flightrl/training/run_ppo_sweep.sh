#!/usr/bin/env bash
# PPO HP sweep on figure_8, starting from F_figure_8_v4 best (5/8 baseline).
# Each trial: 5M steps fine-tune + 20-ep deterministic eval. Sequential to keep the VM single-job.

set -u
cd "$(dirname "$0")"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate flightmare
export FLIGHTMARE_PATH=~/flightmare

BASE_MODEL="saved/curriculum_v2/F_figure_8_v4/best_model.zip"
BASE_VECNORM="saved/curriculum_v2/F_figure_8_v4/vec_normalize.pkl"
SWEEP_DIR="saved/sweep_figure_8"
SWEEP_LOG="logs/sweep_figure_8_summary.log"

mkdir -p "$SWEEP_DIR" "logs"
echo "=== PPO sweep on figure_8, 5M each, base = F_figure_8_v4 (5/8) ===" > "$SWEEP_LOG"
echo "started: $(date -u)" >> "$SWEEP_LOG"

# Trials: (id, lr, ent_coef)
TRIALS=(
  "01_baseline:3e-4:0.005"
  "02_higher_lr:5e-4:0.005"
  "03_high_ent:3e-4:0.05"
  "04_low_lr:1e-4:0.005"
  "05_compress:3e-4:0.0001"
  "06_mid_ent:3e-4:0.02"
)

for t in "${TRIALS[@]}"; do
  IFS=':' read -r ID LR ENT <<< "$t"
  SAVE="$SWEEP_DIR/$ID"
  TB="saved/tb_logs/sweep_figure_8/$ID"
  TLOG="logs/sweep_${ID}_train.log"
  ELOG_BEST="logs/sweep_${ID}_eval_best.log"
  ELOG_FINAL="logs/sweep_${ID}_eval_final.log"

  echo "" | tee -a "$SWEEP_LOG"
  echo "=== TRIAL $ID  lr=$LR  ent=$ENT  ($(date -u)) ===" | tee -a "$SWEEP_LOG"

  python -u gate_racing_iros.py --train 1 --total-timesteps 5000000 \
    --learning-rate "$LR" --ent-coef "$ENT" \
    --load-model "$BASE_MODEL" --load-vecnorm "$BASE_VECNORM" \
    --save-path "$SAVE" --tb-log "$TB" > "$TLOG" 2>&1

  STOCH_R=$(grep "ep_rew_mean" "$TLOG" | tail -1 | awk -F'|' '{print $3}' | tr -d ' ')
  STOCH_LEN=$(grep "ep_len_mean" "$TLOG" | tail -1 | awk -F'|' '{print $3}' | tr -d ' ')
  STD=$(grep "std" "$TLOG" | grep -v "p_len" | tail -1 | awk -F'|' '{print $3}' | tr -d ' ')

  for M in best final; do
    OUT="logs/sweep_${ID}_eval_${M}.log"
    python -u gate_racing_iros.py --train 0 \
      --model-path "$SAVE/${M}_model.zip" --vecnorm-path "$SAVE/vec_normalize.pkl" \
      --n-eval-episodes 20 --min-gates 8 > "$OUT" 2>&1
    GATES=$(grep "Mean gates_passed" "$OUT" | head -1 | awk '{print $3}')
    DET_R=$(grep "Mean reward" "$OUT" | head -1 | awk '{print $3}')
    DET_LEN=$(grep "Mean ep length" "$OUT" | head -1 | awk '{print $4}')
    echo "  $M_model: gates=$GATES R=$DET_R ep_len=$DET_LEN" | tee -a "$SWEEP_LOG"
  done
  echo "  stoch: R=$STOCH_R len=$STOCH_LEN std=$STD" | tee -a "$SWEEP_LOG"
done

echo "" | tee -a "$SWEEP_LOG"
echo "=== sweep finished: $(date -u) ===" | tee -a "$SWEEP_LOG"

#!/usr/bin/env bash
# Run the circle_8 v4 best model (stochastic) on every curriculum track.
# Generates per-track 3D/topdown plots + GIF of best episode.
#
# Usage:
#   bash run_multi_track_eval.sh [out_dir]

set -e
cd /home/ssen/flightmare/flightrl/training
source ~/miniforge3/etc/profile.d/conda.sh
conda activate flightmare
export FLIGHTMARE_PATH=/home/ssen/flightmare

MODEL=saved/curriculum/rung06_circle_8_v4/best_model.zip
VECNORM=saved/curriculum/rung06_circle_8_v4/vec_normalize.pkl
OUT_ROOT="${1:-viz/multi_track}"
N_EPS=10

TRACKS=(
  single_1
  straight_2
  straight_3
  bent_3_15
  bent_3_30
  bent_3_45
  circle_4
  circle_6
  circle_8
  oval_8
  vertical_3
  helix_8
  figure_8
  alphapilot
)

mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/summary.txt"
echo "Multi-track evaluation — $(date)" > "$SUMMARY"
echo "Model: $MODEL" >> "$SUMMARY"
echo "Stochastic policy, $N_EPS episodes per track" >> "$SUMMARY"
echo "" >> "$SUMMARY"
printf "%-15s  %-8s  %-6s  %-6s  %s\n" "track" "best_R" "gates" "ep_len" "outcome" >> "$SUMMARY"
echo "----------------------------------------------------------------" >> "$SUMMARY"

for TRACK in "${TRACKS[@]}"; do
  echo ""
  echo "=========================================="
  echo "  $TRACK"
  echo "=========================================="
  python track_loader.py apply "$TRACK" >/dev/null 2>&1 || {
    echo "  apply failed, skipping"
    printf "%-15s  %-8s  %-6s  %-6s  %s\n" "$TRACK" "-" "-" "-" "track_apply_failed" >> "$SUMMARY"
    continue
  }
  TRACK_OUT="$OUT_ROOT/$TRACK"
  mkdir -p "$TRACK_OUT"

  OUT=$(python -u visualize_policy.py \
    --model "$MODEL" --vecnorm "$VECNORM" \
    -n "$N_EPS" --animate --animate-n 1 --speedup 1 \
    --stochastic --out-dir "$TRACK_OUT" 2>&1) || {
    echo "  viz crashed"
    printf "%-15s  %-8s  %-6s  %-6s  %s\n" "$TRACK" "-" "-" "-" "viz_crashed" >> "$SUMMARY"
    continue
  }
  echo "$OUT" > "$TRACK_OUT/run.log"

  # Parse the "Top 5 episodes" first line: "    #1: R=  XX.XX  len= NNN  gates=N  ..."
  TOP=$(echo "$OUT" | grep -E "^\s*#1:" | head -1 | sed -E 's/^\s+//')
  if [ -z "$TOP" ]; then
    BEST_R="?"; GATES="?"; LEN="?"; OUTCOME="parse_failed"
  else
    BEST_R=$(echo "$TOP" | grep -oE "R=\s*-?[0-9]+\.[0-9]+" | awk '{print $2}')
    LEN=$(echo "$TOP" | grep -oE "len=\s*[0-9]+" | awk '{print $2}')
    GATES=$(echo "$TOP" | grep -oE "gates=[0-9]+" | awk -F= '{print $2}')
    OUTCOME=$(echo "$TOP" | awk '{print $NF}')
  fi
  printf "%-15s  %-8s  %-6s  %-6s  %s\n" "$TRACK" "$BEST_R" "$GATES" "$LEN" "$OUTCOME" >> "$SUMMARY"

  GIF=$(find "$TRACK_OUT" -name "*.gif" | head -1)
  echo "  best episode: R=$BEST_R gates=$GATES len=$LEN ($OUTCOME)"
  echo "  gif: $GIF"
done

echo ""
echo "=========================================="
echo "  Summary written to $SUMMARY"
echo "=========================================="
cat "$SUMMARY"

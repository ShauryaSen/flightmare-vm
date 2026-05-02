#!/usr/bin/env bash
# Launch the Optuna PPO hyperparameter sweep under nohup.
# Usage: bash launch_hp_sweep.sh [--max-trials N] [--max-hours H]
#                                [--total-timesteps S]
#                                [--load-model PATH] [--load-vecnorm PATH]
#
# Run from anywhere; cd's to flightrl/training/.
set -e
THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINING_DIR="$(dirname "$THIS_DIR")"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate flightmare
export FLIGHTMARE_PATH="$HOME/flightmare"

cd "$TRAINING_DIR"

mkdir -p hp_sweep/logs
LOG="hp_sweep/logs/hp_sweep_$(date +%Y%m%d_%H%M%S).log"

nohup python -u hp_sweep/hp_sweep_optuna.py "$@" > "$LOG" 2>&1 &
PID=$!
echo "Launched HP sweep (PID $PID)"
echo "Log: $LOG"
echo "Tail with: tail -f $LOG"

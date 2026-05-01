#!/bin/bash
# Overnight sweep launcher.
#   bash sweep/launch.sh                 # full sweep, 25M/condition
#   bash sweep/launch.sh --total-timesteps 100000   # smoke test override
#   bash sweep/launch.sh --only 00_baseline,03_gate_bonus_on
#
# Usage notes:
#   - Run from ~/flightmare/flightrl/training
#   - Mac side: `caffeinate -i -d -s &` to block sleep
#   - Tail: `tail -f sweep/logs/sweep_*.log`
#   - Stop: `pkill -f sweep_overnight.py` (kills orchestrator + child train)

set -euo pipefail

cd "$(dirname "$0")/.."   # cd to flightrl/training

# Conda activation
if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "FATAL: no conda.sh found in ~/miniforge3 or ~/miniconda3" >&2
    exit 1
fi
conda activate flightmare
export FLIGHTMARE_PATH="$HOME/flightmare"

mkdir -p sweep/logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="sweep/logs/sweep_${TS}.log"

# Pre-flight summary
echo "================================================================" | tee -a "$LOG"
echo "  Overnight sweep launcher" | tee -a "$LOG"
echo "  Started: $(date)" | tee -a "$LOG"
echo "  Cores:   $(nproc)" | tee -a "$LOG"
echo "  Memory:  $(free -h | awk '/^Mem:/ {print $2 " total, " $7 " avail"}')" | tee -a "$LOG"
echo "  Log:     $LOG" | tee -a "$LOG"
echo "  Args:    $*" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"

# Launch under nohup. setsid + new pgid lets us kill the whole tree with
# `kill -- -<pgid>` if needed.
nohup setsid python -u sweep/sweep_overnight.py "$@" \
    >> "$LOG" 2>&1 &

PID=$!
echo "$PID" > sweep/logs/sweep.pid
echo
echo "Sweep PID: $PID  (saved to sweep/logs/sweep.pid)"
echo "Watch:     tail -f $LOG"
echo "Stop:      kill -- -\$(ps -o pgid= $PID | tr -d ' ')"
echo "Or:        pkill -f sweep_overnight.py"
echo
echo "On the Mac (host), run before bed:"
echo "  caffeinate -i -d -s &"
echo

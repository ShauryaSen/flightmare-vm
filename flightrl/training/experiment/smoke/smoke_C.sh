#!/usr/bin/env bash
# Smoke test for Condition C — 200K steps with complexity update every 10
# episodes so the adaptive controller actually fires inside the smoke budget.
set -euo pipefail
: "${FLIGHTMARE_PATH:?must export FLIGHTMARE_PATH}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "${SCRIPT_DIR}")"

RUNS_ROOT="${EXPERIMENT_DIR}/smoke/runs" \
  "${EXPERIMENT_DIR}/run_condition_C.sh" "${1:-0}" 200000 10

echo "[smoke_C] OK — inspect ${EXPERIMENT_DIR}/smoke/runs/C_seed${1:-0}/complexity_history.json"

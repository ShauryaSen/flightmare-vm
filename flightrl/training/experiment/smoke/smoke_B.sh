#!/usr/bin/env bash
# Smoke test for Condition B — 6 rungs, ~100K steps total to exercise warm-start
# handoff (SCALE applied to the per-rung budgets in run_condition_B.sh).
set -euo pipefail
: "${FLIGHTMARE_PATH:?must export FLIGHTMARE_PATH}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "${SCRIPT_DIR}")"

# SCALE = 0.0001 -> floor of 50K/rung x 6 rungs = ~300K total
RUNS_ROOT="${EXPERIMENT_DIR}/smoke/runs" \
  "${EXPERIMENT_DIR}/run_condition_B.sh" "${1:-0}" 0.0001

echo "[smoke_B] OK"

#!/usr/bin/env bash
# Smoke test for Condition A — 100K steps on circle_8.
set -euo pipefail
: "${FLIGHTMARE_PATH:?must export FLIGHTMARE_PATH}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "${SCRIPT_DIR}")"

# Use a separate runs root so smoke output doesn't pollute real runs/.
RUNS_ROOT="${EXPERIMENT_DIR}/smoke/runs" \
  "${EXPERIMENT_DIR}/run_condition_A.sh" "${1:-0}" 100000

echo "[smoke_A] OK"

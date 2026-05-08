# Shared helpers for the EXPERIMENT_PLAN.md campaign.
# Source this from each run_condition_*.sh.
#
# Required env vars set by the caller:
#   FLIGHTMARE_PATH    — repo root
#
# Conventions:
#   EXPERIMENT_DIR     — flightrl/training/experiment/
#   TRAINING_DIR       — flightrl/training/
#   HPARAMS_JSON       — experiment_hparams.json (locked §4 HPs)
#   RUNS_ROOT          — experiment/runs/

set -u

EXPERIMENT_DIR="${EXPERIMENT_DIR:-${FLIGHTMARE_PATH}/flightrl/training/experiment}"
TRAINING_DIR="${TRAINING_DIR:-${FLIGHTMARE_PATH}/flightrl/training}"
HPARAMS_JSON="${HPARAMS_JSON:-${EXPERIMENT_DIR}/experiment_hparams.json}"
RUNS_ROOT="${RUNS_ROOT:-${EXPERIMENT_DIR}/runs}"
TRAIN_PY="${TRAINING_DIR}/hp_sweep/_train_one_hp.py"
TRACK_LOADER="${TRAINING_DIR}/track_loader.py"

apply_track() {
    local track="$1"
    echo "[run_lib] applying track '${track}'"
    (cd "${TRAINING_DIR}" && python "${TRACK_LOADER}" apply "${track}" --force) || {
        echo "[run_lib] FAILED to apply track ${track}" >&2
        return 1
    }
}

is_done() {
    local run_dir="$1"
    [[ -f "${run_dir}/final_model.zip" ]]
}

# Train one rung / run. Args:
#   $1 run_dir
#   $2 total_timesteps
#   $3 seed
#   $4 (optional) --load-model path
#   $5 (optional) --load-vecnorm path
train_rung() {
    local run_dir="$1"
    local total_steps="$2"
    local seed="$3"
    local load_model="${4:-}"
    local load_vecnorm="${5:-}"

    if is_done "${run_dir}"; then
        echo "[run_lib] SKIP — already done: ${run_dir}"
        return 0
    fi

    mkdir -p "${run_dir}"
    local extra=()
    [[ -n "${load_model}" ]]    && extra+=(--load-model "${load_model}")
    [[ -n "${load_vecnorm}" ]]  && extra+=(--load-vecnorm "${load_vecnorm}")

    echo "[run_lib] train_rung ${run_dir}  steps=${total_steps}  seed=${seed}"
    [[ -n "${load_model}" ]] && echo "[run_lib]   warm-start model=${load_model}"

    (cd "${TRAINING_DIR}" && python -u "${TRAIN_PY}" \
        "${run_dir}" "${HPARAMS_JSON}" \
        --total-timesteps "${total_steps}" \
        --seed "${seed}" \
        --log-every 500000 \
        "${extra[@]}") 2>&1 | tee -a "${run_dir}/train.log"

    if ! is_done "${run_dir}"; then
        echo "[run_lib] WARN — final_model.zip missing in ${run_dir} after train" >&2
        return 2
    fi
}

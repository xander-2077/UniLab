#!/usr/bin/env bash

set -euo pipefail

TASK="go2_footstand/mujoco"
TASK_NAME="${TASK_NAME:-Go2FootStand}"
LOG_ROOT="${LOG_ROOT:-logs/rsl_rl_ppo/${TASK_NAME}}"
ENERGY_THRESHOLDS="${ENERGY_THRESHOLDS:-400 300 200}"
NO_PLAY="${NO_PLAY:-true}"
START_LOAD_RUN="${START_LOAD_RUN:-}"
FIRST_STAGE_MAX_ITERATIONS="${FIRST_STAGE_MAX_ITERATIONS:-12000}"
FINETUNE_STAGE_MAX_ITERATIONS="${FINETUNE_STAGE_MAX_ITERATIONS:-8000}"

read -r -a thresholds <<< "${ENERGY_THRESHOLDS}"
if [ "${#thresholds[@]}" -eq 0 ]; then
  echo "[go2_footstand_energy_curriculum] ENERGY_THRESHOLDS must contain at least one value."
  exit 1
fi

latest_run_dir() {
  if [ ! -d "${LOG_ROOT}" ]; then
    return 1
  fi
  find "${LOG_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

prev_load_run="${START_LOAD_RUN}"
stage_idx=0
for threshold in "${thresholds[@]}"; do
  stage_idx=$((stage_idx + 1))
  if [ "${stage_idx}" -eq 1 ]; then
    max_iterations="${FIRST_STAGE_MAX_ITERATIONS}"
  else
    max_iterations="${FINETUNE_STAGE_MAX_ITERATIONS}"
  fi
  echo "[go2_footstand_energy_curriculum] stage ${stage_idx}/${#thresholds[@]}: energy_termination_threshold=${threshold}"
  echo "[go2_footstand_energy_curriculum] stage ${stage_idx}/${#thresholds[@]}: algo.max_iterations=${max_iterations}"

  cmd=(
    uv run scripts/train_rsl_rl.py
    "task=${TASK}"
    "training.no_play=${NO_PLAY}"
    "env.energy_termination_threshold=${threshold}"
    "algo.max_iterations=${max_iterations}"
  )
  if [ -n "${prev_load_run}" ]; then
    cmd+=("algo.load_run=${prev_load_run}")
  fi
  if [ "$#" -gt 0 ]; then
    cmd+=("$@")
  fi

  printf '[go2_footstand_energy_curriculum] command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"

  new_run="$(latest_run_dir || true)"
  if [ -z "${new_run}" ]; then
    echo "[go2_footstand_energy_curriculum] no run directory found under ${LOG_ROOT}"
    exit 1
  fi
  prev_load_run="$(realpath "${new_run}")"
  echo "[go2_footstand_energy_curriculum] stage ${stage_idx} run: ${prev_load_run}"
done

echo "[go2_footstand_energy_curriculum] completed. Final run: ${prev_load_run}"

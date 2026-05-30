#!/usr/bin/env bash

set -euo pipefail

SOURCE_RUN="${SOURCE_RUN:-logs/rsl_rl_ppo/Go2FootStand/2026-05-21_00-44-24_mujoco-best}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-9999}"
STAGE1_THRESHOLD="${STAGE1_THRESHOLD:-300}"
STAGE2_THRESHOLD="${STAGE2_THRESHOLD:-200}"
STAGE1_ITERS="${STAGE1_ITERS:-2000}"
STAGE2_ITERS="${STAGE2_ITERS:-5000}"
TASK_LOG_ROOT="${TASK_LOG_ROOT:-logs/rsl_rl_ppo/Go2FootStand}"

common_overrides=(
  task=go2_footstand/mujoco
  training.no_play=true
)

extra_overrides=("$@")

latest_run_dir() {
  local latest=""
  local dir
  for dir in "${TASK_LOG_ROOT}"/*; do
    if [ -d "${dir}" ]; then
      latest="$(basename "${dir}")"
    fi
  done

  if [ -n "${latest}" ]; then
    printf '%s\n' "${latest}"
  fi
}

if [ ! -d "${SOURCE_RUN}" ]; then
  echo "[go2_footstand_curriculum] source run not found: ${SOURCE_RUN}" >&2
  exit 1
fi

if [ ! -f "${SOURCE_RUN}/model_${SOURCE_CHECKPOINT}.pt" ]; then
  echo "[go2_footstand_curriculum] checkpoint not found: ${SOURCE_RUN}/model_${SOURCE_CHECKPOINT}.pt" >&2
  exit 1
fi

echo "[go2_footstand_curriculum] stage 1: resume=${SOURCE_RUN}/model_${SOURCE_CHECKPOINT}.pt threshold=${STAGE1_THRESHOLD} iterations=${STAGE1_ITERS}"
uv run scripts/train_rsl_rl.py \
  "${common_overrides[@]}" \
  "algo.load_run=${SOURCE_RUN}" \
  "algo.checkpoint=${SOURCE_CHECKPOINT}" \
  "env.energy_termination_threshold=${STAGE1_THRESHOLD}" \
  "algo.max_iterations=${STAGE1_ITERS}" \
  "${extra_overrides[@]}"

stage1_run="$(latest_run_dir)"
if [ -z "${stage1_run}" ]; then
  echo "[go2_footstand_curriculum] could not resolve stage 1 run under ${TASK_LOG_ROOT}" >&2
  exit 1
fi

echo "[go2_footstand_curriculum] stage 2: resume=${stage1_run} threshold=${STAGE2_THRESHOLD} iterations=${STAGE2_ITERS}"
uv run scripts/train_rsl_rl.py \
  "${common_overrides[@]}" \
  "algo.load_run=${stage1_run}" \
  "env.energy_termination_threshold=${STAGE2_THRESHOLD}" \
  "algo.max_iterations=${STAGE2_ITERS}" \
  "${extra_overrides[@]}"

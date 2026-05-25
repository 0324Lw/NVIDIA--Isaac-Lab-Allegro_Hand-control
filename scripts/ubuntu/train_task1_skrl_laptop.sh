#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

DATASET_PATH="${ALLEGRO_TASK1_DATASET:-${PROJECT_ROOT}/assets/motions/task1_target_poses.pt}"

if [ ! -f "${DATASET_PATH}" ]; then
  echo "[INFO] Dataset not found. Generating dataset first..."
  bash scripts/ubuntu/generate_task1_dataset.sh
fi

python src/allegro_rl/tasks/task1/task1_train.py \
  --num-envs 2048 \
  --total-env-steps 1000000000 \
  --rollouts 64 \
  --learning-epochs 5 \
  --mini-batches 8 \
  --lr 3e-4 \
  --min-lr 1e-5 \
  --max-lr 5e-4 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-range 0.2 \
  --target-kl 0.015 \
  --hard-kl-stop 0.08 \
  --entropy-coef 0.001 \
  --value-coef 2.0 \
  --grad-clip 1.0 \
  --init-log-std -1.0 \
  --frame-stack 5 \
  --summary-interval 100 \
  --save-freq-env-steps 20000000 \
  --dataset-path "${DATASET_PATH}" \
  --headless \
  --device cuda:0

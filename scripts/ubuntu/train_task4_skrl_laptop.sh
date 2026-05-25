#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

python src/allegro_rl/tasks/task4/task4_train.py \
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
  --entropy-coef 0.0025 \
  --value-coef 2.0 \
  --grad-clip 0.8 \
  --init-log-std 0.0 \
  --frame-stack 5 \
  --summary-interval 10 \
  --save-freq-env-steps 20000000 \
  --headless \
  --device cuda:0

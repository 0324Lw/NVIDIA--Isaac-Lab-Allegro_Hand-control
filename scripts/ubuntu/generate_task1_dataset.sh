#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

python src/allegro_rl/data/generate_task1_pose_dataset.py \
  --output "${PROJECT_ROOT}/assets/motions/task1_target_poses.pt" \
  --num-easy 10000 \
  --num-hard 20000 \
  --seed 42

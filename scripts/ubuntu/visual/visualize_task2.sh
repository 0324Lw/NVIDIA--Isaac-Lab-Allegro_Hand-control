#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage:"
  echo "  bash scripts/ubuntu/visual/visualize_task2.sh /path/to/checkpoint_or_final_checkpoint_dir [start_k]"
  echo ""
  echo "Examples:"
  echo "  bash scripts/ubuntu/visual/visualize_task2.sh logs/task2/<run>/final_checkpoint 1.0"
  echo "  bash scripts/ubuntu/visual/visualize_task2.sh logs/task2/<run>/final_checkpoint/allegro_task2_model.pt 1.0"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

CKPT="$1"
START_K="${2:-1.0}"

echo "============================================================"
echo "Allegro Hand Task2 TRUE skrl PPO GUI visualization"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CHECKPOINT=${CKPT}"
echo "START_K=${START_K}"
echo "PYTHON=$(which python)"
echo "============================================================"

python src/allegro_rl/tasks/task2/task2_model_test.py \
  --checkpoint "${CKPT}" \
  --num-envs 4 \
  --steps 2000 \
  --start-k "${START_K}" \
  --print-interval 20 \
  --frame-stack 5 \
  --visualize \
  --device cuda:0

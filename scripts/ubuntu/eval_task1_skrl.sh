#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage:"
  echo "  bash scripts/ubuntu/eval_task1_skrl.sh /path/to/checkpoint_or_final_checkpoint_dir [start_k]"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

CKPT="$1"
START_K="${2:-1.0}"
DATASET_PATH="${ALLEGRO_TASK1_DATASET:-${PROJECT_ROOT}/assets/motions/task1_target_poses.pt}"

echo "============================================================"
echo "Allegro Hand Task1 TRUE skrl PPO model evaluation"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CHECKPOINT=${CKPT}"
echo "START_K=${START_K}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "PYTHON=$(which python)"
echo "============================================================"

python - <<'PY'
import sys
print("[CHECK] Python:", sys.executable)
import torch
print("[CHECK] torch:", torch.__version__)
print("[CHECK] cuda:", torch.cuda.is_available())
import isaaclab
print("[CHECK] isaaclab: ok")
import skrl
print("[CHECK] skrl:", getattr(skrl, "__version__", "unknown"))
PY

if [ ! -f "${DATASET_PATH}" ]; then
  echo "[INFO] Dataset not found. Generating dataset first..."
  bash scripts/ubuntu/generate_task1_dataset.sh
fi

python src/allegro_rl/tasks/task1/task1_model_test.py \
  --checkpoint "${CKPT}" \
  --num-envs 4 \
  --steps 200 \
  --start-k "${START_K}" \
  --print-interval 20 \
  --dataset-path "${DATASET_PATH}" \
  --frame-stack 5 \
  --headless \
  --device cuda:0

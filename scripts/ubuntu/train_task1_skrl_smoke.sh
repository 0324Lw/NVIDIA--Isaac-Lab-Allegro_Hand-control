#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

DATASET_PATH="${ALLEGRO_TASK1_DATASET:-${PROJECT_ROOT}/assets/motions/task1_target_poses.pt}"

echo "============================================================"
echo "Allegro Hand Task1 TRUE skrl PPO smoke training"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHON=$(which python)"
echo "DATASET_PATH=${DATASET_PATH}"
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

python src/allegro_rl/tasks/task1/task1_train.py \
  --num-envs 512 \
  --total-env-steps 5120 \
  --rollouts 4 \
  --learning-epochs 2 \
  --mini-batches 2 \
  --summary-interval 1 \
  --save-freq-env-steps 5120 \
  --dataset-path "${DATASET_PATH}" \
  --headless \
  --device cuda:0

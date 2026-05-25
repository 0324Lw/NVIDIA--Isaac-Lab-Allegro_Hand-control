#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

DATASET_PATH="${ALLEGRO_TASK1_DATASET:-${PROJECT_ROOT}/assets/motions/task1_target_poses.pt}"

echo "============================================================"
echo "Allegro Hand Task1 Env Test"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHON=$(which python)"
echo "DATASET_PATH=${DATASET_PATH}"
echo "============================================================"

python - <<'PY'
import sys
print("[CHECK] Python:", sys.executable)

try:
    import torch
    print("[CHECK] torch:", torch.__version__)
    print("[CHECK] cuda available:", torch.cuda.is_available())
except Exception as e:
    raise RuntimeError("Current Python cannot import torch. Please activate conda env: isaaclab") from e

try:
    import isaaclab
    print("[CHECK] isaaclab: ok")
except Exception as e:
    raise RuntimeError("Current Python cannot import isaaclab. Please activate IsaacLab conda env.") from e
PY

if [ ! -f "${DATASET_PATH}" ]; then
  echo "[INFO] Dataset not found. Generating dataset first..."
  bash scripts/ubuntu/generate_task1_dataset.sh
fi

python tests/task1/task1_env_test.py \
  --num-envs 512 \
  --steps 5000 \
  --collect-interval 500 \
  --dataset-path "${DATASET_PATH}" \
  --headless \
  --test-device cuda:0

#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage:"
  echo "  bash scripts/ubuntu/eval_task4_skrl.sh /path/to/checkpoint_or_final_checkpoint_dir [start_k]"
  echo ""
  echo "Examples:"
  echo "  bash scripts/ubuntu/eval_task4_skrl.sh logs/task4/<run>/final_checkpoint 1.0"
  echo "  bash scripts/ubuntu/eval_task4_skrl.sh logs/task4/<run>/final_checkpoint/allegro_task4_teacher_model.pt 1.0"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

CKPT="$1"
START_K="${2:-1.0}"

echo "============================================================"
echo "Allegro Hand Task4 Teacher TRUE skrl PPO model evaluation"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CHECKPOINT=${CKPT}"
echo "START_K=${START_K}"
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

python src/allegro_rl/tasks/task4/task4_model_test.py \
  --checkpoint "${CKPT}" \
  --num-envs 4 \
  --steps 200 \
  --start-k "${START_K}" \
  --print-interval 20 \
  --frame-stack 5 \
  --headless \
  --device cuda:0

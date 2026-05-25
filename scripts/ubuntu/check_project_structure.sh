#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"

required=(
  "assets/README.md"
  "assets/motions/README.md"
  "configs/task1_pose_tracking.yaml"
  "docs/project_overview.md"
  "scripts/ubuntu/generate_task1_dataset.sh"
  "src/allegro_rl/common/paths.py"
  "src/allegro_rl/data/generate_task1_pose_dataset.py"
  "src/allegro_rl/tasks/task1/task1_config.py"
  "tests/task1"
  "README.md"
  "LICENSE"
  "pyproject.toml"
)

for f in "${required[@]}"; do
  if [ ! -e "$f" ]; then
    echo "[MISSING] $f"
    exit 1
  fi
  echo "[OK] $f"
done

echo "[PASS] Allegro Hand project scaffold looks valid."

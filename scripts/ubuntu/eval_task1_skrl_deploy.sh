#!/usr/bin/env bash
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
echo "[INFO] Allegro Task1 deploy-style evaluation currently aliases standard skrl evaluation."
exec bash scripts/ubuntu/eval_task1_skrl.sh "$@"

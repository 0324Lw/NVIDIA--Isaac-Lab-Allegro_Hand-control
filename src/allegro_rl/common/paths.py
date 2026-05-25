from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def project_root() -> Path:
    return PROJECT_ROOT


def asset_path(relative: str) -> Path:
    return PROJECT_ROOT / "assets" / relative


def config_path(relative: str) -> Path:
    return PROJECT_ROOT / "configs" / relative


def default_asset_path(relative: str) -> str:
    return str(asset_path(relative))


def path_from_env(env_name: str, fallback: str | Path) -> str:
    value = os.environ.get(env_name, "")
    if value:
        return value
    return str(fallback)

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import read_yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_paths_config(path: str | Path) -> dict[str, Any]:
    cfg = read_yaml(resolve_repo_path(path))
    return cfg


def get_output_path(paths_config: dict[str, Any], key: str) -> Path:
    return resolve_repo_path(paths_config["outputs"][key])

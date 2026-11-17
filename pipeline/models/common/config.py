from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: Path | str) -> dict[str, Any]:
    data = Path(path).read_text(encoding="utf-8")
    loaded = yaml.safe_load(data)
    return loaded if loaded is not None else {}


def write_yaml(path: Path | str, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

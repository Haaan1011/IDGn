from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class _Line:
    indent: int
    text: str
    line_no: int


def _parse_scalar(value: str) -> Any:
    token = value.strip()
    if token == "":
        return ""
    lowered = token.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if token.startswith(("'", '"')) and token.endswith(("'", '"')) and len(token) >= 2:
        return token[1:-1]
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def _split_mapping(text: str) -> tuple[str, str | None]:
    if ":" not in text:
        raise ValueError(f"Invalid mapping line: {text}")
    key, rest = text.split(":", 1)
    key = key.strip()
    rest = rest.strip()
    return key, (rest if rest != "" else None)


def _prepare_lines(text: str) -> list[_Line]:
    prepared: list[_Line] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        if raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        prepared.append(_Line(indent=indent, text=raw.strip(), line_no=line_no))
    return prepared


def _parse_block(lines: list[_Line], start: int, indent: int) -> tuple[int, Any]:
    if start >= len(lines):
        return start, {}

    if lines[start].text.startswith("- "):
        items: list[Any] = []
        idx = start
        while idx < len(lines):
            line = lines[idx]
            if line.indent != indent or not line.text.startswith("- "):
                break
            content = line.text[2:].strip()
            idx += 1

            if not content:
                if idx < len(lines) and lines[idx].indent > indent:
                    idx, item = _parse_block(lines, idx, lines[idx].indent)
                else:
                    item = None
                items.append(item)
                continue

            if ":" in content:
                key, rest = _split_mapping(content)
                item: Any = {key: _parse_scalar(rest)} if rest is not None else {key: {}}
                if rest is None and idx < len(lines) and lines[idx].indent > indent:
                    idx, nested = _parse_block(lines, idx, lines[idx].indent)
                    item[key] = nested
                elif idx < len(lines) and lines[idx].indent > indent:
                    idx, nested = _parse_block(lines, idx, lines[idx].indent)
                    if isinstance(nested, dict):
                        item.update(nested)
                items.append(item)
                continue

            items.append(_parse_scalar(content))
        return idx, items

    mapping: dict[str, Any] = {}
    idx = start
    while idx < len(lines):
        line = lines[idx]
        if line.indent != indent or line.text.startswith("- "):
            break
        key, rest = _split_mapping(line.text)
        idx += 1
        if rest is not None:
            mapping[key] = _parse_scalar(rest)
            continue
        if idx < len(lines) and lines[idx].indent > indent:
            idx, nested = _parse_block(lines, idx, lines[idx].indent)
            mapping[key] = nested
        else:
            mapping[key] = {}
    return idx, mapping


def load_yaml(path: str | Path) -> Any:
    file_path = Path(path)
    lines = _prepare_lines(file_path.read_text(encoding="utf-8"))
    if not lines:
        return {}
    _, result = _parse_block(lines, 0, lines[0].indent)
    return result

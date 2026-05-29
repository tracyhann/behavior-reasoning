from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class BaselineBackend(Protocol):
    """Minimal backend interface used by the JSONL experiment runner."""

    def answer(self, row: dict[str, Any], video_path: Path, prompt: str) -> str:
        """Return a raw model response string for one question row."""


def parse_backend_args(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"backend arg must use key=value format: {value}")
        key, item_value = value.split("=", 1)
        if not key:
            raise ValueError(f"backend arg has empty key: {value}")
        parsed[key] = item_value
    return parsed


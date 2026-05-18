from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OutputTree:
    root: Path
    raw: Path
    sampled_frames: Path
    crops: Path
    characters: Path
    detections: Path
    logs: Path


def ensure_output_tree(output_root: Path, video_id: str) -> OutputTree:
    root = output_root / video_id
    tree = OutputTree(
        root=root,
        raw=root / "raw",
        sampled_frames=root / "raw" / "sampled_frames",
        crops=root / "raw" / "crops",
        characters=root / "characters",
        detections=root / "detections",
        logs=root / "logs",
    )
    for directory in (
        tree.raw,
        tree.sampled_frames,
        tree.crops,
        tree.characters,
        tree.detections,
        tree.logs,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return tree


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def safe_video_id(path: Path) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in path.stem)

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


class RegistryVideoBackend:
    """Adapter for local video-language backends exposed through agents.registry."""

    def __init__(
        self,
        *,
        project_root: Path,
        behavior_root: str = "../behavior-understanding-test",
        model_name: str,
        backend: str = "transformers",
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: str = "900",
        max_video_frames: str = "16",
    ) -> None:
        behavior_path = (project_root / behavior_root).resolve()
        if not behavior_path.exists():
            raise FileNotFoundError(f"behavior backend root not found: {behavior_path}")
        sys.path.insert(0, str(behavior_path))
        from agents.registry import init_agent

        self.agent = init_agent(
            model_name,
            backend=backend,
            device=device,
            dtype=dtype,
            max_new_tokens=int(max_new_tokens),
            max_video_frames=int(max_video_frames),
        )

    def answer(self, row: dict[str, Any], video_path: Path, prompt: str) -> str:
        return self.agent.answer_video(video_path, prompt)


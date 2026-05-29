from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


class SubprocessJsonBackend:
    """Backend adapter for any CLI that accepts prompt/video paths and prints a response."""

    def __init__(self, *, project_root: Path, command: str, timeout_sec: str = "600") -> None:
        self.project_root = project_root
        self.command = command
        self.timeout_sec = int(timeout_sec)

    def answer(self, row: dict[str, Any], video_path: Path, prompt: str) -> str:
        env = os.environ.copy()
        env.update(
            {
                "CARE_PROMPT": prompt,
                "CARE_VIDEO_PATH": str(video_path),
                "CARE_QUESTION_ID": row["question_id"],
                "CARE_CLIP_ID": row["clip_id"],
            }
        )
        result = subprocess.run(
            self.command,
            shell=True,
            cwd=self.project_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=self.timeout_sec,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"backend command exited {result.returncode}")
        return result.stdout.strip()


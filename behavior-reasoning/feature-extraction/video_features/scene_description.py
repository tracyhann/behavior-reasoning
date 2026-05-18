from __future__ import annotations

from typing import Any

from .vlm import normalize_vlm_payload


class TemplateSceneDescriber:
    def describe_segment(self, *, frame_paths: list[Any], segment: dict[str, Any]) -> dict[str, Any]:
        return normalize_vlm_payload(
            {
                "scene_description": (
                    "Sampled frames were saved for this segment, but no VLM description "
                    "was requested."
                ),
                "scene_confidence": "low",
                "uncertainty": "no VLM or detector backend was used for this segment",
                "objects": [],
                "characters": [],
            }
        )

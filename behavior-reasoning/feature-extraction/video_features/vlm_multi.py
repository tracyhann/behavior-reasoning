"""Multi-backend VLM adapters that share the QwenVLM interface.

Backends:
- qwen25: Qwen2_5_VLForConditionalGeneration (e.g. Qwen2.5-VL-7B / 32B / 72B)
- qwen3:  Qwen3VLForConditionalGeneration  (e.g. Qwen3-VL-8B / 30B-A3B)
- gemma3: Gemma3ForConditionalGeneration   (e.g. google/gemma-3-12b-it / 27b-it)

Each adapter exposes describe_roster(frame_paths, expected_subjects) and
describe_segment(frame_paths, segment), returning the same normalized JSON
payloads as the existing pipeline expects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .vlm import (
    _roster_prompt,
    _segment_prompt,
    normalize_roster_payload,
    normalize_vlm_payload,
)


class _BaseVLM:
    backend: str = "base"

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 320,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self._model: Any | None = None
        self._processor: Any | None = None
        self._roster_subjects: list[dict[str, Any]] = []

    def describe_roster(
        self,
        *,
        frame_paths: list[Path],
        expected_subjects: int | None = None,
    ) -> dict[str, Any]:
        if not frame_paths:
            return normalize_roster_payload({}, expected_subjects=expected_subjects)
        raw = self._generate(frame_paths, _roster_prompt(expected_subjects))
        payload = normalize_roster_payload(raw, expected_subjects=expected_subjects)
        payload["raw_response"] = raw
        return payload

    def set_roster(self, subjects: list[dict[str, Any]]) -> None:
        self._roster_subjects = subjects

    def describe_segment(
        self,
        *,
        frame_paths: list[Path],
        segment: dict[str, Any],
    ) -> dict[str, Any]:
        if not frame_paths:
            return normalize_vlm_payload({})
        raw = self._generate(frame_paths, _segment_prompt(segment, self._roster_subjects))
        payload = normalize_vlm_payload(raw)
        payload["raw_response"] = raw
        return payload

    def _torch_dtype(self) -> Any:
        import torch

        if self.dtype in {"float16", "fp16"}:
            return torch.float16
        if self.dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if self.dtype in {"float32", "fp32"}:
            return torch.float32
        return "auto"

    def _generate(self, frame_paths: list[Path], prompt: str) -> str:
        raise NotImplementedError


class _Qwen25VLAdapter(_BaseVLM):
    backend = "qwen25"

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self._torch_dtype(),
            device_map=self.device,
        )
        self._model.eval()
        return self._model, self._processor

    def _generate(self, frame_paths: list[Path], prompt: str) -> str:
        model, processor = self._load()
        messages = [
            {
                "role": "user",
                "content": [
                    *[
                        {
                            "type": "image",
                            "image": str(path),
                            "resized_height": 360,
                            "resized_width": 640,
                        }
                        for path in frame_paths
                    ],
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        import torch

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


class _Qwen3VLAdapter(_BaseVLM):
    backend = "qwen3"

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self._torch_dtype(),
            device_map=self.device,
        )
        self._model.eval()
        return self._model, self._processor

    def _generate(self, frame_paths: list[Path], prompt: str) -> str:
        model, processor = self._load()
        messages = [
            {
                "role": "user",
                "content": [
                    *[
                        {
                            "type": "image",
                            "image": str(path),
                            "resized_height": 360,
                            "resized_width": 640,
                        }
                        for path in frame_paths
                    ],
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        import torch

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


class _Gemma3VLAdapter(_BaseVLM):
    backend = "gemma3"

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self._torch_dtype(),
            device_map=self.device,
        )
        self._model.eval()
        return self._model, self._processor

    def _generate(self, frame_paths: list[Path], prompt: str) -> str:
        from PIL import Image

        model, processor = self._load()
        # Gemma3 caps to a small number of images per prompt; keep it lean.
        max_imgs = 8
        if len(frame_paths) > max_imgs:
            step = max(1, len(frame_paths) // max_imgs)
            chosen = frame_paths[::step][:max_imgs]
        else:
            chosen = frame_paths
        images = [Image.open(str(p)).convert("RGB").resize((640, 360)) for p in chosen]
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in images],
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        else:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        import torch

        input_len = inputs["input_ids"].shape[-1] if isinstance(inputs, dict) else inputs.input_ids.shape[-1]
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed = generated_ids[0][input_len:]
        return processor.decode(trimmed, skip_special_tokens=True).strip()


def build_vlm(
    backend: str,
    *,
    model_id: str,
    device: str = "auto",
    dtype: str = "bfloat16",
    max_new_tokens: int = 320,
) -> _BaseVLM:
    backend = backend.lower()
    if backend in {"qwen25", "qwen2.5", "qwen2_5"}:
        return _Qwen25VLAdapter(
            model_id=model_id, device=device, dtype=dtype, max_new_tokens=max_new_tokens
        )
    if backend in {"qwen3", "qwen3vl"}:
        return _Qwen3VLAdapter(
            model_id=model_id, device=device, dtype=dtype, max_new_tokens=max_new_tokens
        )
    if backend in {"gemma3", "gemma"}:
        return _Gemma3VLAdapter(
            model_id=model_id, device=device, dtype=dtype, max_new_tokens=max_new_tokens
        )
    raise ValueError(f"Unknown VLM backend: {backend}")

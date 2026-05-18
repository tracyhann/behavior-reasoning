from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
CONFIDENCE_VALUES = {"low", "medium", "high"}
IDENTITY_TERMS = (
    "Spider-Man",
    "Spider Man",
    "Spiderman",
    "Spider emblem",
    "red and blue suit with spider emblem",
    "Iron Man",
    "iron-man",
    "Peter Parker",
    "Tony Stark",
    "red and gold armor",
)
UNCERTAINTY_FALLBACK = "limited sampled frames; temporal details may be incomplete"
INFERENCE_REPLACEMENTS = (
    (re.compile(r"\bhe\s+is\b", re.IGNORECASE), "person is"),
    (re.compile(r"\bshe\s+is\b", re.IGNORECASE), "person is"),
    (re.compile(r"\bhe\b", re.IGNORECASE), "the person"),
    (re.compile(r"\bshe\b", re.IGNORECASE), "the person"),
    (re.compile(r"\bhis\b", re.IGNORECASE), "the person's"),
    (re.compile(r"\bher\b", re.IGNORECASE), "the person's"),
    (re.compile(r"\bactively\s+communicating\b", re.IGNORECASE), "moving hands or mouth visibly"),
    (re.compile(r"\bgestures\s+indicate\b", re.IGNORECASE), "gestures show"),
    (re.compile(r"\bpossibly\s+explaining\s+or\s+emphasizing\s+a\s+point\b", re.IGNORECASE), "with mouth open or hand movement"),
    (re.compile(r"\byoung\s+person\b", re.IGNORECASE), "person"),
    (re.compile(r"\bfacial\s+expression\s+and\s+(?:the\s+)?context\s+suggest\b", re.IGNORECASE), "visible facial orientation and context are unclear;"),
    (re.compile(r"\bmight\s+be\s+explaining\s+something\s+important\b", re.IGNORECASE), "has mouth open or hand movement"),
    (re.compile(r"\bappears?\s+to\s+be\s+examining\b", re.IGNORECASE), "is oriented toward"),
    (re.compile(r"\bmen\b", re.IGNORECASE), "people"),
    (re.compile(r"\bwomen\b", re.IGNORECASE), "people"),
    (re.compile(r"\bman\b", re.IGNORECASE), "person"),
    (re.compile(r"\bwoman\b", re.IGNORECASE), "person"),
    (re.compile(r"\b(?:light|dark|brown|black|white|pale)\s+skin(?:ned)?\b", re.IGNORECASE), ""),
    (re.compile(r"\bappears\s+engaged\s+in\s+a\s+conversation\b", re.IGNORECASE), "has mouth or body orientation suggesting visible communication"),
    (re.compile(r"\bengaged\s+in\s+a\s+conversation\b", re.IGNORECASE), "near another visible person"),
    (re.compile(r"\bwhile\s+speaking\b", re.IGNORECASE), "with mouth open"),
    (re.compile(r"\bwhile\s+talking\b", re.IGNORECASE), "with mouth open"),
    (re.compile(r"\bspeaking\b", re.IGNORECASE), "mouth open"),
    (re.compile(r"\btalking\b", re.IGNORECASE), "mouth open"),
    (re.compile(r"\bconversation\b", re.IGNORECASE), "visible interaction"),
    (re.compile(r"\breacting\s+to\s+something\s+off-camera\b", re.IGNORECASE), "oriented off-camera"),
    (re.compile(r"\breacting\b", re.IGNORECASE), "changing posture or facial orientation"),
    (re.compile(r"\bas\s+if\s+", re.IGNORECASE), ""),
    (re.compile(r"\bas\s+if\s+speaking\s+or\s+", re.IGNORECASE), ""),
    (re.compile(r"\bas\s+if\s+speaking\b", re.IGNORECASE), "with mouth open"),
    (re.compile(r"\blooking\s+at\b", re.IGNORECASE), "head oriented toward"),
    (re.compile(r"\bintentions?\b", re.IGNORECASE), "observable action"),
    (re.compile(r"\bpurpose\b", re.IGNORECASE), "observable reason"),
)


def normalize_vlm_payload(raw_text: str | dict[str, Any]) -> dict[str, Any]:
    payload = raw_text if isinstance(raw_text, dict) else _load_json_object(raw_text)
    if not isinstance(payload, dict):
        payload = {}

    characters = []
    for index, item in enumerate(_as_list(payload.get("characters")), start=1):
        if not isinstance(item, dict):
            continue
        characters.append(
            {
                "temporary_id": _clean(
                    item.get("temporary_id") or item.get("name"),
                    f"visible-person-{index}",
                ),
                "gender": _gender(item.get("gender")),
                "age_range": _age_range(item.get("age_range") or item.get("age")),
                "height": "unknown",
                "clothing": _clean(item.get("clothing"), "visible clothing unclear"),
                "appearance": _clean(
                    item.get("appearance") or item.get("description"),
                    "visible appearance unclear",
                ),
                "behavior": _clean(
                    item.get("behavior") or item.get("description"),
                    "observable action unclear",
                ),
                "confidence": _confidence(item.get("confidence")),
                "uncertainty": _uncertainty(
                    item.get("uncertainty"),
                    "limited sampled frames; avoid identity, intent, diagnosis, or emotion inference",
                ),
            }
        )

    objects = []
    for item in _as_list(payload.get("objects")):
        if not isinstance(item, dict):
            continue
        objects.append(
            {
                "label": _label(item.get("label") or item.get("name")),
                "count": max(1, int(float(item.get("count", 1) or 1))),
                "confidence": _float01(item.get("confidence", 0.5)),
                "description": _clean(
                    item.get("description") or item.get("location"),
                    "visible in sampled frames",
                ),
            }
        )

    uncertainty = _uncertainty(
        payload.get("uncertainty"),
        UNCERTAINTY_FALLBACK,
    )
    return {
        "scene_description": _clean(
            payload.get("scene_description"),
            "Scene description unavailable from sampled frames.",
        ),
        "scene_confidence": _confidence(
            payload.get("scene_confidence"),
            fallback="low" if not payload.get("scene_description") else "medium",
        ),
        "uncertainty": uncertainty,
        "objects": objects,
        "characters": characters,
    }


def sanitize_text(value: object, fallback: str = "") -> str:
    return _clean(value, fallback)


class QwenVLM:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 360,
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

    def _generate(self, frame_paths: list[Path], prompt: str) -> str:
        model, processor = self._load()
        messages = [
            {
                "role": "user",
                "content": [
                    *[
                        {"type": "image", "image": str(path), "resized_height": 360, "resized_width": 640}
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

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen2.5-VL requires torch, transformers, and qwen-vl-utils in the runtime."
            ) from exc
        torch_dtype: Any = "auto"
        if self.dtype in {"float16", "fp16"}:
            torch_dtype = torch.float16
        elif self.dtype in {"bfloat16", "bf16"}:
            torch_dtype = torch.bfloat16
        elif self.dtype in {"float32", "fp32"}:
            torch_dtype = torch.float32
        elif self.dtype != "auto":
            raise ValueError("dtype must be auto, float16, bfloat16, or float32")

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=self.device,
        )
        return self._model, self._processor


def normalize_roster_payload(
    raw_text: str | dict[str, Any],
    *,
    expected_subjects: int | None = None,
) -> dict[str, Any]:
    payload = raw_text if isinstance(raw_text, dict) else _load_json_object(raw_text)
    if not isinstance(payload, dict):
        payload = {}
    subjects = []
    for index, item in enumerate(_as_list(payload.get("subjects")), start=1):
        if not isinstance(item, dict):
            continue
        subjects.append(
            {
                "id": _person_id(item.get("id") or item.get("label"), index),
                "gender": _gender(item.get("gender")),
                "age_range": _age_range(item.get("age_range") or item.get("age")),
                "height": "unknown",
                "clothing": _clean(
                    item.get("clothing") or item.get("visual_description"),
                    "visible clothing unclear",
                ),
                "appearance": _appearance(
                    item.get("appearance") or item.get("visual_description"),
                    "visible appearance unclear",
                ),
                "confidence": _confidence(item.get("confidence")),
                "evidence": _clean(item.get("evidence"), "global sampled-frame roster"),
            }
        )
    if expected_subjects is not None:
        subjects = subjects[:expected_subjects]
        while len(subjects) < expected_subjects:
            index = len(subjects) + 1
            subjects.append(
                {
                    "id": f"person-{index:02d}",
                    "gender": "unknown",
                    "age_range": "unknown",
                    "height": "unknown",
                    "clothing": "visible clothing unclear",
                    "appearance": "visible recurring person; details unclear",
                    "confidence": "low",
                    "evidence": "placeholder from expected subject count",
                }
            )
    return {
        "subject_count": len(subjects),
        "subjects": subjects,
        "uncertainty": _uncertainty(
            payload.get("uncertainty"),
            "global sampled frames may miss brief appearances",
        ),
    }


def _segment_prompt(segment: dict[str, Any], roster_subjects: list[dict[str, Any]] | None = None) -> str:
    roster_text = ""
    if roster_subjects:
        labels = "; ".join(
            f"{subject['id']}: {subject.get('appearance', 'visible person')}"
            for subject in roster_subjects
        )
        roster_text = (
            " Use this anonymous global roster when labeling visible characters: "
            f"{labels}. For each character, set temporary_id to one roster id "
            "(person-01, person-02, etc.) only when visually supported; otherwise use unmatched. "
        )
    return (
        "You are labeling sampled frames from one video segment for a behavior-evidence "
        "pipeline. Use only observable visual facts. Do not infer identity, name, "
        "diagnosis, personality, intent, or emotion. Use coarse visual gender and age "
        "labels only when supported: gender male/female/unknown, age_range "
        "adult/child/adolescent/unknown. Return only valid JSON with keys: "
        "scene_description, scene_confidence, uncertainty, objects, characters. "
        "objects is a list of {label,count,confidence,description}. characters is a list "
        "of {temporary_id,gender,age_range,clothing,appearance,behavior,confidence,uncertainty}. "
        f"{roster_text}"
        "Keep descriptions concise. Segment timestamps: "
        f"{segment['start_time']} to {segment['end_time']} seconds."
    )


def _roster_prompt(expected_subjects: int | None) -> str:
    count_instruction = (
        f" Return exactly {expected_subjects} subjects unless the sampled frames make that impossible."
        if expected_subjects is not None
        else " Return the small set of recurring human characters, not every detector fragment."
    )
    return (
        "Build a stable anonymous character roster for one short video. Use only sampled "
        "visual evidence. Merge repeated views of the same person across cuts. Ignore "
        "tracker fragments, duplicate boxes, thumbnails, and unrelated end-card imagery. "
        "Do not use real names, identity, emotion, intent, diagnosis, or personality. "
        "Use coarse visual gender and age labels only when supported: gender "
        "male/female/unknown, age_range adult/child/adolescent/unknown. Use labels "
        "person-01, person-02, etc. Return only valid JSON with "
        "keys subject_count, subjects, uncertainty. subjects is a list of "
        "{id,gender,age_range,clothing,appearance,confidence,evidence}. In appearance, "
        "include stable visible traits only, such as hairstyle/hair length, facial hair, "
        "glasses, makeup visibility, jewelry, or accessories. Do not put actions, posture, "
        "room/location, or behavior in appearance."
        f"{count_instruction}"
    )


def _load_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _confidence(value: object, *, fallback: str = "medium") -> str:
    if isinstance(value, (float, int)):
        if float(value) >= 0.75:
            return "high"
        if float(value) >= 0.4:
            return "medium"
        return "low"
    text = _clean(value, fallback).lower()
    try:
        return _confidence(float(text))
    except ValueError:
        pass
    return text if text in CONFIDENCE_VALUES else "medium"


def _gender(value: object) -> str:
    text = _raw_lower(value)
    if not text:
        return "unknown"
    tokens = set(re.findall(r"[a-z]+", text))
    if tokens & {"male", "man", "boy"}:
        return "male"
    if tokens & {"female", "woman", "girl"}:
        return "female"
    return "unknown"


def _age_range(value: object) -> str:
    text = _raw_lower(value)
    if not text:
        return "unknown"
    tokens = set(re.findall(r"[a-z]+", text))
    if tokens & {"child", "kid", "boy", "girl"}:
        return "child"
    if tokens & {"adolescent", "teen", "teenager"}:
        return "adolescent"
    if "adult" in tokens:
        return "adult"
    return "unknown"


def _raw_lower(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _float01(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, round(number, 3)))


def _clean(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    text = " ".join(str(value).strip().split())
    text = _remove_identity_terms(text)
    text = _remove_inference_terms(text)
    return text or fallback


def _appearance(value: object, fallback: str) -> str:
    text = _clean(value, "")
    if not text:
        return fallback
    text = _remove_appearance_context(text)
    return text or fallback


def _remove_appearance_context(text: str) -> str:
    fragments = [fragment.strip() for fragment in re.split(r",|;", text) if fragment.strip()]
    kept = [_clean_appearance_fragment(fragment) for fragment in fragments]
    kept = [fragment for fragment in kept if fragment]
    return ", ".join(kept)


def _clean_appearance_fragment(fragment: str) -> str:
    lowered = fragment.lower()
    if not _appearance_trait_like(lowered):
        return ""
    fragment = re.sub(
        r"\b(?:standing|sitting|seated|walking|bending|reaching|gesturing)\s+"
        r"(?:in|on|near|at)\s+(?:a|an|the)?\s*[a-z ]+?\s+with\s+",
        "with ",
        fragment,
        flags=re.IGNORECASE,
    )
    fragment = re.sub(
        r"\b(?:standing|sitting|seated|walking|bending|reaching|gesturing)\s+"
        r"(?:in|on|near|at)\s+(?:a|an|the)?\s*[a-z ]+$",
        "",
        fragment,
        flags=re.IGNORECASE,
    )
    fragment = re.sub(r"\b(?:standing|sitting|seated|walking|bending|reaching|gesturing)\b", "", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"\b(?:in|on|near|at)\s+(?:a|an|the)?\s*(?:kitchen|room|hallway|couch|floor|shelf|doorway)\b", "", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"^with\s+", "", fragment, flags=re.IGNORECASE)
    return " ".join(fragment.split()).strip(" ,.;")


def _appearance_trait_like(text: str) -> bool:
    return any(
        term in text
        for term in (
            "hair",
            "beard",
            "mustache",
            "moustache",
            "glasses",
            "makeup",
            "jewelry",
            "jewellery",
            "earring",
            "necklace",
            "watch",
            "hat",
            "cap",
            "accessor",
        )
    )


def _label(value: object) -> str:
    if value is not None and _contains_identity_term(str(value)):
        return "person"
    raw = _clean(value, "object")
    text = raw.lower()
    return re.sub(r"[^a-z0-9 _-]", "", text).strip() or "object"


def _uncertainty(value: object, fallback: str) -> str:
    text = _clean(value, "")
    if not text:
        return fallback
    lower = text.lower()
    if lower in CONFIDENCE_VALUES:
        return fallback
    try:
        float(lower)
    except ValueError:
        return text
    return fallback


def _remove_identity_terms(text: str) -> str:
    sanitized = text
    for term in IDENTITY_TERMS:
        pattern = _identity_pattern(term)
        sanitized = pattern.sub("a visible person", sanitized)
    return sanitized


def _contains_identity_term(text: str) -> bool:
    return any(_identity_pattern(term).search(text) for term in IDENTITY_TERMS)


def _remove_inference_terms(text: str) -> str:
    sanitized = text
    for pattern, replacement in INFERENCE_REPLACEMENTS:
        sanitized = pattern.sub(replacement, sanitized)
    sanitized = re.sub(r"\bwith\s+and\b", "with", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\s+and\s*(?=,|\.|;|$)", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\s*,\s*(?=,|\.|;|$)", "", sanitized)
    sanitized = re.sub(r"\s+([,.;])", r"\1", sanitized)
    return " ".join(sanitized.split())


def _identity_pattern(term: str) -> re.Pattern[str]:
    pieces = [re.escape(piece) for piece in re.split(r"[\s_-]+", term.strip()) if piece]
    return re.compile(r"[\s_-]+".join(pieces), re.IGNORECASE)


def _person_id(value: object, index: int) -> str:
    text = _clean(value, "").lower()
    match = re.search(r"person[-_\s]?(\d+)", text)
    if match:
        return f"person-{int(match.group(1)):02d}"
    return f"person-{index:02d}"

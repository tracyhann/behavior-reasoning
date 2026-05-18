from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubjectCard:
    id: str
    gender: str = "unknown"
    age_range: str = "unknown"
    height: str = "unknown"
    clothing: str = "unknown visible clothing"
    appearance: str = "limited visible appearance evidence"
    first_seen: float = 0.0
    last_seen: float = 0.0
    track_ids: list[str] = field(default_factory=list)
    example_segments: list[str] = field(default_factory=list)
    example_crop_paths: list[str] = field(default_factory=list)
    match_confidence: str = "low"
    last_bbox: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "gender": self.gender,
            "age_range": self.age_range,
            "height": self.height,
            "clothing": self.clothing,
            "appearance": self.appearance,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "track_ids": self.track_ids,
            "example_segments": self.example_segments,
            "example_crop_paths": self.example_crop_paths,
            "match_confidence": self.match_confidence,
        }


class SubjectLibrary:
    def __init__(self, *, high_threshold: float = 0.8, uncertain_threshold: float = 0.6) -> None:
        self.high_threshold = high_threshold
        self.uncertain_threshold = uncertain_threshold
        self._cards: dict[str, SubjectCard] = {}

    def upsert_observation(self, observation: dict[str, Any]) -> str:
        match_id, score = self._best_match(observation)
        if match_id is None or score < self.uncertain_threshold:
            match_id = self._next_id()
            self._cards[match_id] = SubjectCard(
                id=match_id,
                gender=_trait(observation.get("gender"), {"male", "female", "unknown"}),
                age_range=_trait(
                    observation.get("age_range"),
                    {"adult", "child", "adolescent", "unknown"},
                ),
                clothing=_clean_text(observation.get("clothing"), "unknown visible clothing"),
                appearance=_clean_text(
                    observation.get("appearance"), "limited visible appearance evidence"
                ),
                first_seen=float(observation.get("start_time", 0.0)),
                last_seen=float(observation.get("end_time", observation.get("start_time", 0.0))),
                match_confidence="low" if score < self.uncertain_threshold else "medium",
                last_bbox=_bbox(observation.get("bbox")),
            )

        card = self._cards[match_id]
        self._update_card(card, observation, score)
        return match_id

    def to_json(self, video_id: str) -> dict[str, Any]:
        return {
            "video_id": video_id,
            "characters": {
                person_id: card.to_dict() for person_id, card in sorted(self._cards.items())
            },
        }

    def _best_match(self, observation: dict[str, Any]) -> tuple[str | None, float]:
        if not self._cards:
            return None, 0.0
        track_id = _clean_text(observation.get("track_id"), "")
        best_id: str | None = None
        best_score = 0.0
        for person_id, card in self._cards.items():
            score = 0.0
            clothing_score = _text_similarity(observation.get("clothing"), card.clothing)
            appearance_score = _text_similarity(observation.get("appearance"), card.appearance)
            if track_id and track_id in card.track_ids:
                score = max(score, 0.95)
            elif track_id:
                # A real tracker ID is stronger evidence than generic VLM text. Do not
                # merge different tracker IDs just because the fallback text is identical.
                bbox_score = _bbox_iou(_bbox(observation.get("bbox")), card.last_bbox)
                start_time = float(observation.get("start_time", 0.0))
                end_time = float(observation.get("end_time", start_time))
                overlap_or_close = _intervals_overlap_or_close(
                    start_time,
                    end_time,
                    card.first_seen,
                    card.last_seen,
                    max_gap=2.0,
                )
                if bbox_score >= 0.70 and overlap_or_close:
                    score = max(score, 0.84)
                if (
                    _has_specific_text(observation.get("clothing"), card.clothing)
                    and _has_specific_text(observation.get("appearance"), card.appearance)
                    and clothing_score >= 0.75
                    and appearance_score >= 0.75
                    and overlap_or_close
                ):
                    score = max(score, 0.82)
                if score > best_score:
                    best_id = person_id
                    best_score = score
                continue
            score = max(score, 0.45 * clothing_score + 0.35 * appearance_score)
            gap = float(observation.get("start_time", 0.0)) - card.last_seen
            if 0.0 <= gap <= 2.0 and max(clothing_score, appearance_score) >= 0.5:
                score = max(score, 0.62)
            if score > best_score:
                best_id = person_id
                best_score = score
        return best_id, best_score

    def _update_card(self, card: SubjectCard, observation: dict[str, Any], score: float) -> None:
        card.first_seen = min(card.first_seen, float(observation.get("start_time", card.first_seen)))
        card.last_seen = max(card.last_seen, float(observation.get("end_time", card.last_seen)))
        _append_unique(card.track_ids, _clean_text(observation.get("track_id"), ""))
        _append_unique(card.example_segments, _clean_text(observation.get("segment_id"), ""))
        _append_unique(card.example_crop_paths, _clean_text(observation.get("crop_path"), ""))

        gender = _trait(observation.get("gender"), {"male", "female", "unknown"})
        age_range = _trait(
            observation.get("age_range"),
            {"adult", "child", "adolescent", "unknown"},
        )
        clothing = _clean_text(observation.get("clothing"), "")
        appearance = _clean_text(observation.get("appearance"), "")
        if card.gender == "unknown" and gender != "unknown":
            card.gender = gender
        if card.age_range == "unknown" and age_range != "unknown":
            card.age_range = age_range
        if clothing and card.clothing.startswith("unknown"):
            card.clothing = clothing
        if appearance and card.appearance.startswith("limited"):
            card.appearance = appearance
        bbox = _bbox(observation.get("bbox"))
        if bbox is not None:
            card.last_bbox = bbox
        if score >= self.high_threshold:
            card.match_confidence = "high"
        elif score >= self.uncertain_threshold and card.match_confidence == "low":
            card.match_confidence = "medium"

    def _next_id(self) -> str:
        return f"person-{len(self._cards) + 1:02d}"


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _clean_text(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    text = " ".join(str(value).strip().split())
    return text or fallback


def _trait(value: object, allowed: set[str]) -> str:
    text = _clean_text(value, "").lower()
    return text if text in allowed else "unknown"


def _text_similarity(left: object, right: object) -> float:
    left_tokens = set(_clean_text(left, "").lower().replace(",", " ").split())
    right_tokens = set(_clean_text(right, "").lower().replace(",", " ").split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _has_specific_text(*values: object) -> bool:
    generic_phrases = {
        "visible clothing unclear",
        "tracked person visible in sampled segment",
        "visible appearance unclear",
        "limited visible appearance evidence",
        "unknown visible clothing",
    }
    for value in values:
        text = _clean_text(value, "").lower()
        if not text or text in generic_phrases:
            return False
        if len(text.split()) < 3:
            return False
    return True


def _bbox(value: object) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _bbox_iou(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _intervals_overlap_or_close(
    start: float,
    end: float,
    card_start: float,
    card_end: float,
    *,
    max_gap: float,
) -> bool:
    if start <= card_end and end >= card_start:
        return True
    return min(abs(start - card_end), abs(card_start - end)) <= max_gap

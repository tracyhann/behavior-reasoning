from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "0.1"
DEFAULT_REASONER_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
CONTEXT_EVENT_TYPES = {"scene_context", "object_context"}
TRACK_ONLY_WARNING = "track-only event has no resolved character id"
INTERPRETIVE_TERMS = {
    "angry",
    "sad",
    "happy",
    "autistic",
    "diagnosis",
    "diagnosed",
    "personality",
    "manipulative",
    "rude",
    "intends",
    "intention",
    "wants",
}


@dataclass(frozen=True)
class Params:
    max_merge_gap_sec: float = 2.0
    max_chain_gap_sec: float = 10.0
    max_response_window_sec: float = 8.0
    min_dedupe_score: float = 0.72


class CharacterResolver:
    def __init__(self, lookup: dict[str, str], characters: dict[str, dict[str, Any]]) -> None:
        self.lookup = lookup
        self.characters = characters

    @classmethod
    def from_payload(cls, payload: Any) -> "CharacterResolver":
        lookup: dict[str, str] = {}
        characters: dict[str, dict[str, Any]] = {}
        for index, record in enumerate(extract_character_records(payload), start=1):
            char_id = first_present(record, ("id", "character_id", "person_id", "label", "name"))
            if char_id is None:
                char_id = f"char_{index:03d}"
            char_id = str(char_id)
            clean_record = dict(record)
            clean_record.setdefault("id", char_id)
            characters[char_id] = clean_record
            for value in character_alias_values(clean_record):
                if value is None or value == "":
                    continue
                lookup[normalize_key(value)] = char_id
        return cls(lookup=lookup, characters=characters)

    def resolve(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return self.lookup.get(normalize_key(text), text)

    def resolve_many(self, values: Any) -> list[str]:
        resolved = []
        for value in as_list(values):
            mapped = self.resolve(value)
            if mapped:
                resolved.append(mapped)
        return unique_preserve_order(resolved)


def load_json_file(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_file(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def extract_character_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("characters", "people", "persons", "subjects"):
        value = payload.get(key)
        records = records_from_value(value)
        if records:
            return records
    if any(key in payload for key in ("id", "character_id", "person_id", "name", "label")):
        return [dict(payload)]
    return []


def extract_detection_records(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(payload, list):
        return [("detections", dict(item)) for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    records: list[tuple[str, dict[str, Any]]] = []
    for key in ("events", "detections"):
        value = payload.get(key)
        for record in records_from_value(value):
            records.append((key, record))

    for key in ("scenes", "objects", "characters", "people", "persons"):
        value = payload.get(key)
        for record in records_from_value(value):
            records.append((key, record))

    if records:
        return records
    if any(key in payload for key in ("description", "caption", "start_sec", "start_time", "timestamp", "event_type")):
        return [("detections", dict(payload))]
    return []


def normalize_detection(
    record: dict[str, Any],
    resolver: CharacterResolver,
    index: int,
    source_kind: str = "detections",
) -> dict[str, Any]:
    source_id = first_present(record, ("id", "detection_id", "event_id"))
    if not source_id:
        source_id = f"det_{source_kind}_{index + 1:04d}"

    start = first_float(record, ("start_sec", "start_time", "start", "timestamp", "time"))
    end = first_float(record, ("end_sec", "end_time", "end", "timestamp", "time"))
    if start is None and end is not None:
        start = end
    if end is None and start is not None:
        end = start
    if start is not None and end is not None and end < start:
        start, end = end, start

    behavior = record.get("behavior") if isinstance(record.get("behavior"), dict) else {}
    description = event_description(record, behavior, source_kind)
    description, dropped = sanitize_description(description)
    event_type = normalize_event_type(first_present(record, ("event_type", "type")), description, source_kind)

    actor_ids = collect_character_ids(record, resolver, ("actor_ids", "actor_id", "actor", "character_ids", "character_id", "person_ids", "person_id"))
    target_ids = collect_character_ids(record, resolver, ("target_ids", "target_id", "target"))
    track_ids = collect_track_ids(record)
    for track_id in track_ids:
        resolved = resolver.resolve(track_id)
        if resolved and resolved != track_id:
            actor_ids.append(resolved)
    actor_ids = unique_preserve_order(actor_ids)

    source_segments = source_segment_ids(record)
    modalities = modality_values(record, behavior, source_kind)
    object_ids = object_values(record, description)
    confidence = confidence_value(record, behavior, source_kind)

    return {
        "source_detection_id": str(source_id),
        "source_kind": source_kind,
        "start_sec": round_float(start),
        "end_sec": round_float(end),
        "event_type": event_type,
        "actor_ids": actor_ids,
        "target_ids": target_ids,
        "object_ids": object_ids,
        "modalities": modalities,
        "description": description,
        "confidence": confidence,
        "source_detection_ids": [str(source_id)],
        "source_segment_ids": source_segments,
        "track_ids": track_ids,
        "merge_notes": {
            "merged_count": 1,
            "conflicts": [],
            "dropped_interpretive_claims": dropped,
        },
    }


def normalize_all(characters_payload: Any, detections_payload: Any) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    resolver = CharacterResolver.from_payload(characters_payload)
    events = []
    for index, (source_kind, record) in enumerate(extract_detection_records(detections_payload)):
        event = normalize_detection(record, resolver, index, source_kind)
        if is_empty_context_event(event):
            continue
        events.append(event)
    return resolver.characters, events


def deduplicate_events(events: list[dict[str, Any]], params: Params) -> list[dict[str, Any]]:
    if not events:
        return []

    parent = list(range(len(events)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    ordered = sorted(enumerate(events), key=lambda item: event_sort_key(item[1]))
    for pos, (left_idx, left) in enumerate(ordered):
        for right_idx, right in (item for item in ordered[pos + 1 :]):
            if time_gap(left, right) > params.max_merge_gap_sec and temporal_iou(left, right) == 0:
                if comparable_start(left) is not None and comparable_start(right) is not None:
                    break
            if duplicate_score(left, right, params) >= params.min_dedupe_score:
                union(left_idx, right_idx)

    clusters: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for idx, event in enumerate(events):
        clusters[find(idx)].append(event)

    merged = [merge_event_cluster(cluster, event_number) for event_number, cluster in enumerate(clusters.values(), start=1)]
    return sorted(merged, key=event_sort_key)


def duplicate_score(left: dict[str, Any], right: dict[str, Any], params: Params) -> float:
    left_tracks = set(left.get("track_ids", []))
    right_tracks = set(right.get("track_ids", []))
    if left_tracks and right_tracks and not (left_tracks & right_tracks):
        return 0.0

    gap = time_gap(left, right)
    iou = temporal_iou(left, right)
    if math.isinf(gap) or (gap > params.max_merge_gap_sec and iou == 0):
        return 0.0

    if iou > 0:
        temporal_score = iou
    else:
        temporal_score = max(0.0, 1.0 - (gap / max(params.max_merge_gap_sec, 0.001)))

    type_score = 1.0 if left.get("event_type") == right.get("event_type") else 0.0
    left_keys = participant_key(left, include_tracks=True)
    right_keys = participant_key(right, include_tracks=True)
    if left_keys or right_keys:
        participant_score = jaccard(left_keys, right_keys)
    elif left.get("source_kind") == right.get("source_kind"):
        participant_score = 0.6
    else:
        participant_score = 0.0
    text_score = text_similarity(left.get("description", ""), right.get("description", ""))

    return 0.35 * temporal_score + 0.25 * type_score + 0.25 * participant_score + 0.15 * text_score


def merge_event_cluster(cluster: list[dict[str, Any]], event_number: int) -> dict[str, Any]:
    ordered = sorted(cluster, key=event_sort_key)
    starts = [event.get("start_sec") for event in ordered if event.get("start_sec") is not None]
    ends = [event.get("end_sec") for event in ordered if event.get("end_sec") is not None]
    event_types = [event.get("event_type") for event in ordered if event.get("event_type")]
    chosen_type = Counter(event_types).most_common(1)[0][0] if event_types else "observed_behavior"
    conflicts = []
    if len(set(event_types)) > 1:
        conflicts.append({"field": "event_type", "values": sorted(set(event_types))})

    descriptions = [event.get("description", "") for event in ordered if event.get("description")]
    chosen_description = choose_description(ordered)
    dropped = []
    for event in ordered:
        dropped.extend(event.get("merge_notes", {}).get("dropped_interpretive_claims", []))

    merged = {
        "event_id": f"evt_{event_number:04d}",
        "start_sec": round_float(min(starts) if starts else None),
        "end_sec": round_float(max(ends) if ends else None),
        "event_type": chosen_type,
        "actor_ids": union_values(event.get("actor_ids", []) for event in ordered),
        "target_ids": union_values(event.get("target_ids", []) for event in ordered),
        "object_ids": union_values(event.get("object_ids", []) for event in ordered),
        "modalities": union_values(event.get("modalities", []) for event in ordered),
        "description": chosen_description,
        "confidence": round_float(sum(event.get("confidence", 0.5) for event in ordered) / len(ordered)),
        "source_detection_ids": union_values(event.get("source_detection_ids", []) for event in ordered),
        "source_segment_ids": union_values(event.get("source_segment_ids", []) for event in ordered),
        "merge_notes": {
            "merged_count": len(ordered),
            "conflicts": conflicts,
            "dropped_interpretive_claims": unique_preserve_order(dropped),
        },
    }
    track_ids = union_values(event.get("track_ids", []) for event in ordered)
    source_kinds = union_values([event.get("source_kind")] for event in ordered)
    if track_ids:
        merged["track_ids"] = track_ids
    if source_kinds:
        merged["source_kinds"] = source_kinds
    if len(descriptions) > 1 and chosen_description:
        merged["merge_notes"]["description_count"] = len(unique_preserve_order(descriptions))
    return merged


def build_behavior_chains(
    canonical_events: list[dict[str, Any]],
    characters: dict[str, dict[str, Any]] | None,
    params: Params,
    *,
    source_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    characters = characters or {}
    source_files = source_files or {"characters": "characters.json", "detections": "detections.json"}
    warnings = []
    orphan_event_ids = []
    track_only_count = 0
    context_events = []
    behavior_events = []

    for event in sorted(canonical_events, key=event_sort_key):
        if event.get("start_sec") is None or event.get("end_sec") is None:
            orphan_event_ids.append(event["event_id"])
            warnings.append(f"{event['event_id']}: missing timestamp")
            continue
        if event.get("event_type") in CONTEXT_EVENT_TYPES:
            context_events.append(event)
            continue
        if not event.get("actor_ids") and event.get("track_ids"):
            orphan_event_ids.append(event["event_id"])
            track_only_count += 1
            continue
        behavior_events.append(event)

    if track_only_count:
        warnings.append(f"{track_only_count} track-only events have no resolved character id; see orphan_event_ids.")

    chain_buckets: list[list[dict[str, Any]]] = []
    for event in behavior_events:
        if not chain_buckets:
            chain_buckets.append([event])
            continue
        current = chain_buckets[-1]
        previous = current[-1]
        if events_should_link(previous, event, params):
            current.append(event)
        else:
            chain_buckets.append([event])

    chains = []
    for index, events in enumerate(chain_buckets, start=1):
        chains.append(finalize_chain(index, events, context_events, characters, params))

    return {
        "schema_version": SCHEMA_VERSION,
        "source_files": source_files,
        "parameters": asdict(params),
        "chains": chains,
        "orphan_event_ids": unique_preserve_order(orphan_event_ids),
        "warnings": unique_preserve_order(warnings),
    }


def finalize_chain(
    index: int,
    events: list[dict[str, Any]],
    context_events: list[dict[str, Any]],
    characters: dict[str, dict[str, Any]],
    params: Params,
) -> dict[str, Any]:
    events = sorted(events, key=event_sort_key)
    start = min(event["start_sec"] for event in events)
    end = max(event["end_sec"] for event in events)
    event_segment_ids = set(union_values(event.get("source_segment_ids", []) for event in events))
    if event_segment_ids:
        overlapping_context = [
            event
            for event in context_events
            if event_segment_ids & set(event.get("source_segment_ids", []))
        ]
    else:
        overlapping_context = [
            event for event in context_events if intervals_overlap_or_near(start, end, event.get("start_sec"), event.get("end_sec"), 0.0)
        ]
    event_objects = union_values(event.get("object_ids", []) for event in events)
    context_objects = union_values(event.get("object_ids", []) for event in overlapping_context)
    source_detection_ids = union_values(
        [event.get("source_detection_ids", []) for event in events]
        + [event.get("source_detection_ids", []) for event in overlapping_context]
    )
    source_segment_ids = union_values(
        [event.get("source_segment_ids", []) for event in events]
        + [event.get("source_segment_ids", []) for event in overlapping_context]
    )
    participants = union_values((event.get("actor_ids", []) + event.get("target_ids", [])) for event in events)
    projected_events = [project_chain_event(event) for event in events]
    pattern = assign_interaction_pattern(events)

    return {
        "chain_id": f"chain_{index:04d}",
        "start_sec": round_float(start),
        "end_sec": round_float(end),
        "participant_ids": participants,
        "object_ids": unique_preserve_order(event_objects + context_objects),
        "event_ids": [event["event_id"] for event in events],
        "events": projected_events,
        "interaction_pattern": pattern,
        "summary": summarize_chain(events, characters, pattern),
        "confidence": round_float(sum(event.get("confidence", 0.5) for event in events) / len(events)),
        "source_detection_ids": source_detection_ids,
        "source_segment_ids": source_segment_ids,
    }


def assign_interaction_pattern(events: list[dict[str, Any]]) -> str:
    if not events:
        return "empty"
    if len(events) == 1:
        event = events[0]
        if event.get("object_ids") or event.get("event_type") == "object_action":
            return "object_mediated"
        if event.get("event_type") in {"locomotion_posture", "pose_motion", "posture"}:
            return "motion_observation"
        return "single_observation"
    if has_cue_response_pair(events):
        return "cue_response"
    if any(event.get("object_ids") or event.get("event_type") == "object_action" for event in events):
        return "object_mediated"
    actors = [tuple(event.get("actor_ids", [])) for event in events if event.get("actor_ids")]
    actor_sequence = [actor[0] for actor in actors if actor]
    if len(set(actor_sequence)) >= 2 and len(actor_sequence) >= 3 and actor_sequence != sorted(actor_sequence):
        return "turn_taking"
    event_types = {event.get("event_type") for event in events}
    participants = set()
    for event in events:
        participants.update(event.get("actor_ids", []))
        participants.update(event.get("target_ids", []))
    if event_types <= {"locomotion_posture", "pose_motion", "posture", "gaze_head_orientation"}:
        return "motion_sequence"
    if len(events) >= 3 and len(event_types) == 1:
        return "repetitive_motion_sequence"
    if len(participants) >= 2:
        return "parallel_activity"
    return "behavior_sequence"


def summarize_chain(events: list[dict[str, Any]], characters: dict[str, dict[str, Any]], pattern: str) -> str:
    if not events:
        return "No behavior events were available."
    labels = {char_id: character_label(char_id, characters.get(char_id, {})) for event in events for char_id in event.get("actor_ids", []) + event.get("target_ids", [])}
    if len(events) == 1:
        event = events[0]
        actor = join_labels([labels.get(char_id, char_id) for char_id in event.get("actor_ids", [])])
        prefix = f"{actor}: " if actor else ""
        return f"{time_text(event)} {prefix}{event.get('description', 'observable behavior recorded')}."
    if pattern == "cue_response":
        first, second = cue_response_pair(events) or (events[0], events[1])
        return (
            f"{time_text(first)} {actor_text(first, labels)} {first.get('description', 'shows a cue')}; "
            f"{time_text(second)} {actor_text(second, labels)} follows with {second.get('description', 'an observable response')}."
        )
    actors = union_values(event.get("actor_ids", []) + event.get("target_ids", []) for event in events)
    actor_text_value = join_labels([labels.get(char_id, char_id) for char_id in actors])
    first = events[0].get("description", "first observable event")
    last = events[-1].get("description", "last observable event")
    participant_part = f" involving {actor_text_value}" if actor_text_value else ""
    return f"From {events[0]['start_sec']:.1f}s to {events[-1]['end_sec']:.1f}s, {len(events)} observable events{participant_part}: {first}; then {last}."


class QwenTextReasoner:
    def __init__(
        self,
        model_id: str = DEFAULT_REASONER_MODEL,
        *,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 512,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self._model: Any | None = None
        self._processor: Any | None = None

    def refine_chains(
        self,
        chains_payload: dict[str, Any],
        canonical_events: list[dict[str, Any]],
        characters: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, Any], list[str]]:
        warnings = []
        event_lookup = {event["event_id"]: event for event in canonical_events}
        for chain in chains_payload.get("chains", []):
            evidence = [compact_event(event_lookup[event_id]) for event_id in chain.get("event_ids", []) if event_id in event_lookup]
            prompt = reasoner_prompt(chain, evidence, characters)
            try:
                raw = self.generate(prompt)
                parsed = load_json_object(raw)
                summary = parsed.get("summary") if isinstance(parsed, dict) else None
                pattern = parsed.get("interaction_pattern") if isinstance(parsed, dict) else None
                if isinstance(summary, str) and summary.strip():
                    clean_summary, dropped = sanitize_description(summary)
                    if dropped:
                        warnings.append(f"{chain['chain_id']}: Qwen summary dropped interpretive terms {dropped}")
                    chain["summary"] = ensure_terminal_period(clean_summary)
                    chain["reasoner_model"] = self.model_id
                if isinstance(pattern, str) and pattern.strip():
                    chain["interaction_pattern"] = normalize_pattern(pattern, fallback=chain["interaction_pattern"])
            except Exception as exc:  # pragma: no cover - exercised in container runs, not unit tests
                warnings.append(f"{chain['chain_id']}: Qwen reasoner failed: {exc}")
        chains_payload["warnings"] = unique_preserve_order(chains_payload.get("warnings", []) + warnings)
        return chains_payload, warnings

    def generate(self, prompt: str) -> str:
        model, processor = self._load()
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], padding=True, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Qwen reasoning requires torch and transformers in the runtime.") from exc

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


def run_quality_checks(
    canonical_events: list[dict[str, Any]],
    chains_payload: dict[str, Any],
    *,
    qa_iterations: int = 5,
) -> dict[str, Any]:
    iterations = []
    for iteration in range(1, qa_iterations + 1):
        errors = quality_errors(canonical_events, chains_payload)
        iterations.append(
            {
                "iteration": iteration,
                "status": "pass" if not errors else "fail",
                "errors": errors,
                "questions": quality_questions(canonical_events, chains_payload),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "qa_iterations": qa_iterations,
        "overall_status": "pass" if all(item["status"] == "pass" for item in iterations) else "fail",
        "iterations": iterations,
    }


def quality_errors(canonical_events: list[dict[str, Any]], chains_payload: dict[str, Any]) -> list[str]:
    errors = []
    event_ids = {event.get("event_id") for event in canonical_events}
    timed_events = [event for event in canonical_events if event.get("start_sec") is not None]
    if timed_events != sorted(timed_events, key=event_sort_key):
        errors.append("canonical_events are not chronological")
    for event in canonical_events:
        if not event.get("event_id"):
            errors.append("canonical event missing event_id")
        if "source_detection_ids" not in event:
            errors.append(f"{event.get('event_id')}: missing source_detection_ids")
    for chain in chains_payload.get("chains", []):
        chain_ids = chain.get("event_ids", [])
        for event_id in chain_ids:
            if event_id not in event_ids:
                errors.append(f"{chain.get('chain_id')}: unknown event id {event_id}")
        chain_events = chain.get("events", [])
        chain_starts = [event.get("time_span", [None])[0] for event in chain_events]
        if chain_starts != sorted(chain_starts):
            errors.append(f"{chain.get('chain_id')}: events are not chronological")
        if not chain.get("summary"):
            errors.append(f"{chain.get('chain_id')}: summary is empty")
        if forbidden_terms(chain.get("summary", "")):
            errors.append(f"{chain.get('chain_id')}: summary contains interpretive terms")
        if not chain.get("source_detection_ids"):
            errors.append(f"{chain.get('chain_id')}: source detections are missing")
    return unique_preserve_order(errors)


def quality_questions(canonical_events: list[dict[str, Any]], chains_payload: dict[str, Any]) -> dict[str, Any]:
    first_chain_event = None
    if chains_payload.get("chains"):
        first_chain_events = chains_payload["chains"][0].get("events", [])
        if first_chain_events:
            first_chain_event = first_chain_events[0]
    multi_participant = [
        chain["chain_id"]
        for chain in chains_payload.get("chains", [])
        if len(chain.get("participant_ids", [])) >= 2
    ]
    return {
        "what_happened_first": first_chain_event.get("description") if first_chain_event else None,
        "which_chains_have_multiple_characters": multi_participant,
        "are_events_chronological": not any("chronological" in error for error in quality_errors(canonical_events, chains_payload)),
        "are_source_detections_traceable": all(event.get("source_detection_ids") for event in canonical_events),
        "are_summaries_observable": all(not forbidden_terms(chain.get("summary", "")) for chain in chains_payload.get("chains", [])),
        "are_unusable_events_listed": isinstance(chains_payload.get("orphan_event_ids"), list),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    params = Params(
        max_merge_gap_sec=args.max_merge_gap_sec,
        max_chain_gap_sec=args.max_chain_gap_sec,
        max_response_window_sec=args.max_response_window_sec,
        min_dedupe_score=args.min_dedupe_score,
    )
    characters_payload = load_json_file(args.characters)
    detections_payload = load_json_file(args.detections)
    characters, normalized = normalize_all(characters_payload, detections_payload)
    canonical_events = deduplicate_events(normalized, params)
    chains_payload = build_behavior_chains(
        canonical_events,
        characters,
        params,
        source_files={"characters": str(args.characters), "detections": str(args.detections)},
    )

    if args.reasoner_model and not args.no_reasoner:
        reasoner = QwenTextReasoner(
            model_id=args.reasoner_model,
            device=args.reasoner_device,
            dtype=args.reasoner_dtype,
            max_new_tokens=args.reasoner_max_new_tokens,
        )
        chains_payload, _ = reasoner.refine_chains(chains_payload, canonical_events, characters)

    report = run_quality_checks(canonical_events, chains_payload, qa_iterations=args.qa_iterations)
    if report["overall_status"] != "pass":
        chains_payload["warnings"] = unique_preserve_order(
            chains_payload.get("warnings", []) + [f"QA/QC did not pass: {report['overall_status']}"]
        )

    write_json_file(args.canonical_events_out, canonical_events)
    write_json_file(args.out, chains_payload)
    if args.qa_report_out:
        write_json_file(args.qa_report_out, report)
    print(
        f"wrote {len(canonical_events)} canonical events, "
        f"{len(chains_payload['chains'])} chains, QA {report['overall_status']}"
    )
    return 0 if report["overall_status"] == "pass" else 2


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean behavior chains from character and detection JSON files.")
    parser.add_argument("--characters", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--canonical-events-out", type=Path, required=True)
    parser.add_argument("--qa-report-out", type=Path, default=None)
    parser.add_argument("--qa-iterations", type=int, default=5)
    parser.add_argument("--max-merge-gap-sec", type=float, default=2.0)
    parser.add_argument("--max-chain-gap-sec", type=float, default=10.0)
    parser.add_argument("--max-response-window-sec", type=float, default=8.0)
    parser.add_argument("--min-dedupe-score", type=float, default=0.72)
    parser.add_argument("--reasoner-model", default=None)
    parser.add_argument("--reasoner-device", default="auto")
    parser.add_argument("--reasoner-dtype", default="auto")
    parser.add_argument("--reasoner-max-new-tokens", type=int, default=512)
    parser.add_argument("--no-reasoner", action="store_true")
    return parser.parse_args(argv)


def records_from_value(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        records = []
        for key, item in value.items():
            if isinstance(item, dict):
                record = dict(item)
                record.setdefault("id", key)
                records.append(record)
        return records
    return []


def is_empty_context_event(event: dict[str, Any]) -> bool:
    if event.get("event_type") == "scene_context":
        return not event.get("description") or event.get("description", "").lower().startswith("scene description unavailable")
    if event.get("event_type") == "object_context":
        return not event.get("object_ids") and not event.get("description")
    return False


def character_alias_values(record: dict[str, Any]) -> list[Any]:
    values = []
    for key in ("id", "character_id", "person_id", "name", "label", "role", "track_id", "temporary_id"):
        value = record.get(key)
        if value is not None:
            values.append(value)
    for key in ("track_ids", "aliases"):
        values.extend(as_list(record.get(key)))
    return values


def collect_character_ids(record: dict[str, Any], resolver: CharacterResolver, keys: tuple[str, ...]) -> list[str]:
    values = []
    for key in keys:
        values.extend(as_list(record.get(key)))
    return resolver.resolve_many(values)


def collect_track_ids(record: dict[str, Any]) -> list[str]:
    values = []
    for key in ("track_id", "track_ids", "temporary_id"):
        value = record.get(key)
        if key == "temporary_id" and value and not str(value).startswith("track-"):
            continue
        values.extend(as_list(value))
    return unique_preserve_order(str(value) for value in values if value)


def source_segment_ids(record: dict[str, Any]) -> list[str]:
    values = []
    for key in ("source_segment_ids", "source_segment_id", "segment_ids", "segment_id"):
        values.extend(as_list(record.get(key)))
    return unique_preserve_order(str(value) for value in values if value)


def object_values(record: dict[str, Any], description: str) -> list[str]:
    values = []
    for key in ("object_ids", "object_id", "objects", "object"):
        value = record.get(key)
        if key == "objects" and isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    values.append(item.get("id") or item.get("label") or item.get("name"))
                else:
                    values.append(item)
        else:
            values.extend(as_list(value))
    if record.get("event_type") == "object_action" and not values and description:
        values.append("object")
    return unique_preserve_order(slug_id(value) for value in values if value)


def modality_values(record: dict[str, Any], behavior: dict[str, Any], source_kind: str) -> list[str]:
    values = []
    for key in ("modalities", "modality"):
        values.extend(as_list(record.get(key)))
    values.extend(as_list(behavior.get("modalities_used")))
    if not values:
        if source_kind == "characters":
            values = ["rgb"]
        elif source_kind == "scenes":
            values = ["scene"]
        elif source_kind == "objects":
            values = ["object"]
    return unique_preserve_order(slug_id(value) for value in values if value)


def event_description(record: dict[str, Any], behavior: dict[str, Any], source_kind: str) -> str:
    if source_kind == "objects" and isinstance(record.get("objects"), list):
        labels = []
        parts = []
        for item in record["objects"]:
            if isinstance(item, dict):
                label = item.get("label") or item.get("name")
                if label:
                    labels.append(str(label))
                    if item.get("description"):
                        parts.append(f"{label}: {item['description']}")
            elif item:
                labels.append(str(item))
        if parts:
            return "Visible objects: " + "; ".join(parts)
        if labels:
            return "Visible objects: " + ", ".join(labels)
    for value in (
        behavior.get("description"),
        record.get("description"),
        record.get("caption"),
        record.get("summary"),
        record.get("label"),
    ):
        if value:
            return str(value).strip()
    return ""


def normalize_event_type(raw_type: Any, description: str, source_kind: str) -> str:
    if source_kind == "scenes":
        return "scene_context"
    if source_kind == "objects":
        return "object_context"
    if raw_type:
        return slug_id(raw_type)
    text = description.lower()
    if "response" in text or "respond" in text:
        return "response"
    if any(word in text for word in ("hold", "holding", "object", "pick", "picking", "reach", "reaching", "bag")):
        return "object_action"
    if any(word in text for word in ("gesture", "hand", "arm", "point", "wave", "raises")):
        return "gesture"
    if any(word in text for word in ("mouth open", "open mouth", "vocal", "speaking", "talking")):
        return "vocalization_or_mouth_movement"
    if any(word in text for word in ("look", "facing", "turn", "oriented", "orientation", "head")):
        return "gaze_head_orientation"
    if any(word in text for word in ("walk", "enter", "leave", "towards", "toward", "away", "bending", "bend")):
        return "locomotion_posture"
    if "pose landmarks" in text or "movement in" in text:
        return "pose_motion"
    if any(word in text for word in ("standing", "sitting", "arms crossed")):
        return "posture"
    return "observed_behavior"


def confidence_value(record: dict[str, Any], behavior: dict[str, Any], source_kind: str) -> float:
    candidates = [
        behavior.get("confidence"),
        record.get("confidence"),
        record.get("score"),
    ]
    if source_kind == "objects" and isinstance(record.get("objects"), list):
        object_scores = [confidence_to_float(item.get("confidence")) for item in record["objects"] if isinstance(item, dict)]
        object_scores = [score for score in object_scores if score is not None]
        if object_scores:
            candidates.insert(0, sum(object_scores) / len(object_scores))
    bbox = record.get("bbox_summary") if isinstance(record.get("bbox_summary"), dict) else {}
    candidates.append(bbox.get("confidence"))
    for candidate in candidates:
        score = confidence_to_float(candidate)
        if score is not None:
            return round_float(score) or 0.5
    return 0.5


def confidence_to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    text = str(value).strip().lower()
    mapping = {"high": 0.9, "medium": 0.6, "med": 0.6, "low": 0.3, "unknown": 0.5, "unclear": 0.5}
    if text in mapping:
        return mapping[text]
    try:
        return max(0.0, min(1.0, float(text)))
    except ValueError:
        return None


def sanitize_description(value: Any) -> tuple[str, list[str]]:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"\bwhile\s+speaking\b", "with mouth open", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwhile\s+talking\b", "with mouth open", text, flags=re.IGNORECASE)
    text = re.sub(r"\bspeaking\b", "mouth open", text, flags=re.IGNORECASE)
    text = re.sub(r"\btalking\b", "mouth open", text, flags=re.IGNORECASE)
    text = re.sub(r"\b([a-z]+(?:\s+[a-z]+){0,2})\s+\1\b", r"\1", text, flags=re.IGNORECASE)
    dropped = forbidden_terms(text)
    for term in dropped:
        text = re.sub(rf"\b{re.escape(term)}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ;,.")
    return text, dropped


def forbidden_terms(text: str) -> list[str]:
    lower = text.lower()
    return sorted(term for term in INTERPRETIVE_TERMS if re.search(rf"\b{re.escape(term)}\b", lower))


def events_should_link(previous: dict[str, Any], current: dict[str, Any], params: Params) -> bool:
    gap = max(0.0, (current.get("start_sec") or 0) - (previous.get("end_sec") or 0))
    if gap > params.max_chain_gap_sec:
        return False
    if is_cue_response(previous, current) and gap <= params.max_response_window_sec:
        return True
    if participant_key(previous) & participant_key(current):
        return True
    if set(previous.get("object_ids", [])) & set(current.get("object_ids", [])):
        return True
    if not participant_key(previous) and not participant_key(current) and gap <= min(4.0, params.max_chain_gap_sec):
        return True
    return False


def is_cue_response(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    previous_actors = set(previous.get("actor_ids", []))
    previous_targets = set(previous.get("target_ids", []))
    current_actors = set(current.get("actor_ids", []))
    current_targets = set(current.get("target_ids", []))
    if previous.get("event_type") not in {"gesture", "object_action", "vocalization_or_mouth_movement", "gaze_head_orientation"}:
        return False
    return bool((previous_actors & current_targets and previous_targets & current_actors) or (previous_targets & current_actors))


def has_cue_response_pair(events: list[dict[str, Any]]) -> bool:
    return any(is_cue_response(left, right) for left, right in zip(events, events[1:]))


def cue_response_pair(events: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    for left, right in zip(events, events[1:]):
        if is_cue_response(left, right):
            return left, right
    return None


def project_chain_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event["event_id"],
        "time_span": [event.get("start_sec"), event.get("end_sec")],
        "event_type": event.get("event_type"),
        "actor_ids": event.get("actor_ids", []),
        "target_ids": event.get("target_ids", []),
        "object_ids": event.get("object_ids", []),
        "description": event.get("description", ""),
    }


def reasoner_prompt(chain: dict[str, Any], evidence: list[dict[str, Any]], characters: dict[str, dict[str, Any]]) -> str:
    character_cards = {
        char_id: {
            key: card.get(key)
            for key in ("gender", "age_range", "clothing", "appearance")
            if card.get(key)
        }
        for char_id, card in characters.items()
    }
    return (
        "You are the text reasoning step in a video behavior-agent pipeline.\n"
        "Use only the provided event evidence. Do not add diagnosis, personality, emotion, motive, or identity names.\n"
        "Return only JSON with keys: interaction_pattern and summary.\n"
        "interaction_pattern must be one of: cue_response, turn_taking, object_mediated, motion_sequence, "
        "motion_observation, parallel_activity, behavior_sequence, single_observation.\n"
        "The summary must be one concise observable sentence.\n\n"
        f"Character cards:\n{json.dumps(character_cards, ensure_ascii=True)}\n\n"
        f"Current chain:\n{json.dumps(chain, ensure_ascii=True)}\n\n"
        f"Evidence events:\n{json.dumps(evidence, ensure_ascii=True)}"
    )


def compact_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event.get("event_id"),
        "time": [event.get("start_sec"), event.get("end_sec")],
        "event_type": event.get("event_type"),
        "actor_ids": event.get("actor_ids", []),
        "target_ids": event.get("target_ids", []),
        "object_ids": event.get("object_ids", []),
        "description": event.get("description", ""),
    }


def normalize_pattern(value: str, fallback: str) -> str:
    pattern = slug_id(value)
    allowed = {
        "cue_response",
        "turn_taking",
        "object_mediated",
        "motion_sequence",
        "motion_observation",
        "parallel_activity",
        "behavior_sequence",
        "single_observation",
        "repetitive_motion_sequence",
    }
    return pattern if pattern in allowed else fallback


def load_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def quality_time_span(event: dict[str, Any]) -> tuple[float, float]:
    return event.get("time_span", [math.inf, math.inf])


def event_sort_key(event: dict[str, Any]) -> tuple[float, float, str]:
    start = comparable_start(event)
    end = event.get("end_sec")
    return (
        math.inf if start is None else float(start),
        math.inf if end is None else float(end),
        str(event.get("source_detection_id") or event.get("event_id") or ""),
    )


def comparable_start(event: dict[str, Any]) -> float | None:
    value = event.get("start_sec")
    return float(value) if value is not None else None


def time_gap(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_start, left_end = left.get("start_sec"), left.get("end_sec")
    right_start, right_end = right.get("start_sec"), right.get("end_sec")
    if None in (left_start, left_end, right_start, right_end):
        return math.inf
    if left_end < right_start:
        return float(right_start - left_end)
    if right_end < left_start:
        return float(left_start - right_end)
    return 0.0


def temporal_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_start, left_end = left.get("start_sec"), left.get("end_sec")
    right_start, right_end = right.get("start_sec"), right.get("end_sec")
    if None in (left_start, left_end, right_start, right_end):
        return 0.0
    intersection = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    if union <= 0:
        return 1.0 if left_start == right_start else 0.0
    return intersection / union


def intervals_overlap_or_near(start: float, end: float, other_start: Any, other_end: Any, margin: float) -> bool:
    if other_start is None or other_end is None:
        return False
    return not (float(other_end) < start - margin or float(other_start) > end + margin)


def participant_key(event: dict[str, Any], *, include_tracks: bool = False) -> set[str]:
    keys = set(event.get("actor_ids", [])) | set(event.get("target_ids", [])) | set(event.get("object_ids", []))
    if include_tracks:
        keys |= set(event.get("track_ids", []))
    return keys


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def text_similarity(left: str, right: str) -> float:
    return jaccard(tokens(left), tokens(right))


def tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def first_float(record: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = to_float(record.get(key))
        if value is not None:
            return value
    return None


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def round_float(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple) or isinstance(value, set):
        return list(value)
    if isinstance(value, str) and "," in value:
        return [part.strip() for part in value.split(",") if part.strip()]
    return [value]


def union_values(iterables: Any) -> list[Any]:
    result = []
    for value in iterables:
        if isinstance(value, (list, tuple, set)):
            result.extend(value)
        elif value is not None:
            result.append(value)
    return unique_preserve_order(result)


def unique_preserve_order(values: Any) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        if value is None or value == "":
            continue
        key = str(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def normalize_key(value: Any) -> str:
    return str(value).strip().lower()


def slug_id(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def choose_description(events: list[dict[str, Any]]) -> str:
    ranked = sorted(
        events,
        key=lambda event: (event.get("confidence", 0.0), len(event.get("description", ""))),
        reverse=True,
    )
    return ranked[0].get("description", "") if ranked else ""


def character_label(char_id: str, record: dict[str, Any]) -> str:
    traits = []
    for key in ("gender", "age_range"):
        value = record.get(key)
        if value and value != "unknown":
            traits.append(str(value))
    return f"{char_id} ({', '.join(traits)})" if traits else char_id


def join_labels(labels: list[str]) -> str:
    labels = unique_preserve_order(labels)
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def actor_text(event: dict[str, Any], labels: dict[str, str]) -> str:
    actors = [labels.get(char_id, char_id) for char_id in event.get("actor_ids", [])]
    return join_labels(actors) or "a visible person"


def time_text(event: dict[str, Any]) -> str:
    start, end = event.get("start_sec"), event.get("end_sec")
    if start is None:
        return "At an unknown time,"
    if end is None or end == start:
        return f"At {start:.1f}s,"
    return f"From {start:.1f}s to {end:.1f}s,"


def ensure_terminal_period(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text if text[-1] in ".!?" else text + "."


if __name__ == "__main__":
    raise SystemExit(main())

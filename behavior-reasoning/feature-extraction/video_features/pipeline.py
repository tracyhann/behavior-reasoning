from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .io_utils import OutputTree, ensure_output_tree, safe_video_id, write_json
from .scene_description import TemplateSceneDescriber
from .segmentation import build_segments
from .subject_library import SubjectLibrary
from .video_io import estimate_motion, probe_video, sample_segment_frames
from .vlm import DEFAULT_MODEL_ID, QwenVLM, sanitize_text
from .yolo_detection import empty_pose_summary, run_yolo_tracking, unknown_bbox_summary


class SegmentDescriber(Protocol):
    def describe_segment(
        self, *, frame_paths: list[Path], segment: dict[str, Any]
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path
    output_root: Path
    segment_sec: float = 10.0
    stride_sec: float = 6.0
    use_vlm: bool = True
    model_id: str = DEFAULT_MODEL_ID
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 360
    max_segments: int | None = None
    det_model: str = "yolo11n.pt"
    pose_model: str = "yolo11n-pose.pt"
    tracker: str = "botsort.yaml"
    expected_subjects: int | None = None


@dataclass(frozen=True)
class PipelineResult:
    video_id: str
    output_dir: Path
    segment_count: int
    character_count: int


def run_pipeline(
    config: PipelineConfig,
    *,
    describer: SegmentDescriber | None = None,
    yolo_observations_by_segment: dict[str, list[dict[str, Any]]] | None = None,
) -> PipelineResult:
    input_path = config.input_path.resolve()
    output_root = config.output_root.resolve()
    video_id = safe_video_id(input_path)
    tree = ensure_output_tree(output_root, video_id)
    logger = _setup_logger(tree)
    logger.info("starting pipeline video=%s", input_path)

    info = probe_video(input_path)
    segments = build_segments(
        duration_sec=info.duration_sec,
        segment_sec=config.segment_sec,
        stride_sec=config.stride_sec,
    )
    if config.max_segments is not None:
        segments = segments[: config.max_segments]
        logger.info("limited to max_segments=%s", config.max_segments)

    if yolo_observations_by_segment is None:
        yolo_bundle = run_yolo_tracking(
            video_path=input_path,
            segments=segments,
            fps=info.fps,
            frame_width=info.width,
            frame_height=info.height,
            det_model=config.det_model,
            pose_model=config.pose_model,
            tracker=config.tracker,
        )
        yolo_status = yolo_bundle.status
        yolo_rows_by_segment = yolo_bundle.observations_by_segment
    else:
        yolo_status = type(
            "InjectedYoloStatus",
            (),
            {"available": True, "reason": "injected test observations"},
        )()
        yolo_rows_by_segment = yolo_observations_by_segment
    logger.info(
        "yolo status: available=%s reason=%s det_model=%s pose_model=%s tracker=%s",
        yolo_status.available,
        yolo_status.reason,
        config.det_model,
        config.pose_model,
        config.tracker,
    )

    active_describer = describer or _build_describer(config, logger)
    sampled_frames_by_segment: dict[str, list[dict[str, Any]]] = {}
    for segment in segments:
        sampled_frames_by_segment[str(segment["segment_id"])] = sample_segment_frames(
            input_path,
            segment,
            tree.sampled_frames / str(segment["segment_id"]),
        )
    roster = _describe_roster(
        active_describer=active_describer,
        sampled_frames_by_segment=sampled_frames_by_segment,
        expected_subjects=config.expected_subjects,
        logger=logger,
    )
    roster_subjects = roster.get("subjects", [])
    roster_ids = {subject["id"] for subject in roster_subjects}
    roster_observations: dict[str, list[dict[str, Any]]] = {
        subject_id: [] for subject_id in roster_ids
    }
    if roster_subjects and hasattr(active_describer, "set_roster"):
        active_describer.set_roster(roster_subjects)

    library = SubjectLibrary()
    scenes: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    character_rows: list[dict[str, Any]] = []

    for segment in segments:
        segment_samples = sampled_frames_by_segment[str(segment["segment_id"])]
        frame_paths = [sample["path"] for sample in segment_samples]
        motion = estimate_motion(segment_samples)
        description = active_describer.describe_segment(frame_paths=frame_paths, segment=segment)
        yolo_rows = yolo_rows_by_segment.get(str(segment["segment_id"]), [])
        logger.info(
            "segment=%s frames=%d vlm_characters=%d yolo_tracks=%d objects=%d",
            segment["segment_id"],
            len(frame_paths),
            len(description.get("characters", [])),
            len(yolo_rows),
            len(description.get("objects", [])),
        )

        scenes.append(_scene_row(segment, description))
        objects.append(_object_row(segment, description))
        for yolo_row in yolo_rows:
            character = _tracked_person_description(yolo_row)
            track_id = yolo_row.get("track_id") if yolo_row else None
            if roster_subjects:
                person_id = None
            else:
                person_id = library.upsert_observation(
                    {
                        "track_id": track_id,
                        "temporary_id": character.get("temporary_id"),
                        "segment_id": segment["segment_id"],
                        "start_time": yolo_row.get("start_time", segment["start_time"]),
                        "end_time": yolo_row.get("end_time", segment["end_time"]),
                        "gender": character.get("gender"),
                        "age_range": character.get("age_range"),
                        "clothing": character.get("clothing"),
                        "appearance": character.get("appearance"),
                        "bbox": yolo_row.get("bbox_summary", {}).get("mean_bbox"),
                        "crop_path": "",
                    }
                )
            character_rows.append(
                _character_row(segment, person_id, track_id, character, yolo_row)
            )
        for character in description.get("characters", []):
            if roster_subjects:
                person_id = _resolve_roster_id(character, roster_ids)
                if person_id is not None:
                    roster_observations[person_id].append(
                        {
                            "segment_id": segment["segment_id"],
                            "start_time": segment["start_time"],
                            "end_time": segment["end_time"],
                        }
                    )
            else:
                person_id = library.upsert_observation(
                    {
                        "track_id": None,
                        "temporary_id": character.get("temporary_id"),
                        "segment_id": segment["segment_id"],
                        "start_time": segment["start_time"],
                        "end_time": segment["end_time"],
                        "gender": character.get("gender"),
                        "age_range": character.get("age_range"),
                        "clothing": character.get("clothing"),
                        "appearance": character.get("appearance"),
                        "crop_path": "",
                    }
                )
            character_rows.append(
                _character_row(segment, person_id, None, character, None)
            )

    segments_payload = {
        "video_id": video_id,
        "source_video": str(input_path),
        "segment_length_sec": config.segment_sec,
        "stride_sec": config.stride_sec,
        "video_metadata": {
            "fps": info.fps,
            "frame_count": info.frame_count,
            "width": info.width,
            "height": info.height,
            "duration_sec": info.duration_sec,
        },
        "segments": segments,
    }
    detections_payload = {
        "video_id": video_id,
        "scenes": scenes,
        "objects": objects,
        "characters": character_rows,
        "metadata": {
            "vlm_model": config.model_id if config.use_vlm else None,
            "yolo_available": yolo_status.available,
            "yolo_reason": yolo_status.reason,
            "pose_available": any(
                row.get("pose", {}).get("available") for rows in yolo_rows_by_segment.values() for row in rows
            ),
            "pose_reason": "pose model attempted; see per-row pose.available",
            "roster_enabled": bool(roster_subjects),
            "roster_uncertainty": roster.get("uncertainty"),
        },
    }
    characters_payload = (
        _roster_characters_payload(video_id, roster_subjects, roster_observations)
        if roster_subjects
        else library.to_json(video_id)
    )

    write_json(tree.raw / "segments.json", segments_payload)
    write_json(tree.characters / "characters.json", _sanitize_payload(characters_payload))
    write_json(tree.detections / "detections.json", _sanitize_payload(detections_payload))
    logger.info("finished pipeline output=%s", tree.root)
    return PipelineResult(
        video_id=video_id,
        output_dir=tree.root,
        segment_count=len(segments),
        character_count=len(characters_payload["characters"]),
    )


def _build_describer(config: PipelineConfig, logger: logging.Logger) -> SegmentDescriber:
    if not config.use_vlm:
        return TemplateSceneDescriber()
    logger.info("using local Qwen2.5-VL model_id=%s", config.model_id)
    return QwenVLM(
        model_id=config.model_id,
        device=config.device,
        dtype=config.dtype,
        max_new_tokens=config.max_new_tokens,
    )


def _describe_roster(
    *,
    active_describer: SegmentDescriber,
    sampled_frames_by_segment: dict[str, list[dict[str, Any]]],
    expected_subjects: int | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    if not hasattr(active_describer, "describe_roster"):
        return {"subject_count": 0, "subjects": [], "uncertainty": "roster describer unavailable"}
    frame_paths = _roster_frame_paths(sampled_frames_by_segment)
    try:
        roster = active_describer.describe_roster(
            frame_paths=frame_paths,
            expected_subjects=expected_subjects,
        )
    except Exception as exc:
        logger.info("roster description unavailable: %s", exc)
        return {"subject_count": 0, "subjects": [], "uncertainty": str(exc)}
    subjects = roster.get("subjects", []) if isinstance(roster, dict) else []
    if expected_subjects is not None:
        subjects = subjects[:expected_subjects]
    logger.info("global roster subjects=%d expected=%s", len(subjects), expected_subjects)
    return {
        "subject_count": len(subjects),
        "subjects": subjects,
        "uncertainty": roster.get("uncertainty") if isinstance(roster, dict) else None,
    }


def _roster_frame_paths(
    sampled_frames_by_segment: dict[str, list[dict[str, Any]]],
    *,
    max_frames: int = 14,
) -> list[Path]:
    paths = []
    for segment_id in sorted(sampled_frames_by_segment):
        samples = sampled_frames_by_segment[segment_id]
        middle = [sample for sample in samples if sample.get("label") == "middle"]
        chosen = middle[0] if middle else (samples[0] if samples else None)
        if chosen:
            paths.append(chosen["path"])
    if len(paths) <= max_frames:
        return paths
    step = max(1, round(len(paths) / max_frames))
    return paths[::step][:max_frames]


def _resolve_roster_id(character: dict[str, Any], roster_ids: set[str]) -> str | None:
    temporary_id = str(character.get("temporary_id", "")).strip().lower()
    for roster_id in roster_ids:
        if temporary_id == roster_id:
            return roster_id
    return None


def _roster_characters_payload(
    video_id: str,
    roster_subjects: list[dict[str, Any]],
    roster_observations: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    characters = {}
    for subject in sorted(roster_subjects, key=lambda item: item["id"]):
        observations = roster_observations.get(subject["id"], [])
        first_seen = min((float(item["start_time"]) for item in observations), default=0.0)
        last_seen = max((float(item["end_time"]) for item in observations), default=0.0)
        segments = []
        for item in observations:
            segment_id = str(item["segment_id"])
            if segment_id not in segments:
                segments.append(segment_id)
        characters[subject["id"]] = {
            "id": subject["id"],
            "gender": subject.get("gender", "unknown"),
            "age_range": subject.get("age_range", "unknown"),
            "height": "unknown",
            "clothing": subject.get("clothing", "visible clothing unclear"),
            "appearance": subject.get("appearance", "visible recurring person; details unclear"),
            "first_seen": first_seen,
            "last_seen": last_seen,
            "track_ids": [],
            "example_segments": segments,
            "example_crop_paths": [],
            "match_confidence": subject.get("confidence", "medium"),
        }
    return {"video_id": video_id, "characters": characters}


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_payload_value(key, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_payload(item) for item in value]
    if isinstance(value, str):
        return sanitize_text(value, value)
    return value


def _sanitize_payload_value(key: str, value: Any) -> Any:
    if key in {"gender", "age_range", "height"}:
        return value
    return _sanitize_payload(value)


def _scene_row(segment: dict[str, Any], description: dict[str, Any]) -> dict[str, Any]:
    return {
        "segment_id": segment["segment_id"],
        "start_time": segment["start_time"],
        "end_time": segment["end_time"],
        "description": description.get("scene_description", "Scene description unavailable."),
        "confidence": description.get("scene_confidence", "low"),
        "uncertainty": description.get("uncertainty", "unclear sampled-frame evidence"),
    }


def _object_row(segment: dict[str, Any], description: dict[str, Any]) -> dict[str, Any]:
    return {
        "segment_id": segment["segment_id"],
        "start_time": segment["start_time"],
        "end_time": segment["end_time"],
        "objects": description.get("objects", []),
        "uncertainty": description.get("uncertainty", "unclear sampled-frame evidence"),
    }


def _character_row(
    segment: dict[str, Any],
    person_id: str | None,
    track_id: str | None,
    character: dict[str, Any],
    yolo_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    modalities = ["rgb"]
    if yolo_row:
        modalities.extend(["yolo", "motion"])
        if yolo_row.get("pose", {}).get("available"):
            modalities.append("pose")
    else:
        modalities.append("vlm")
    return {
        "segment_id": segment["segment_id"],
        "start_time": segment["start_time"],
        "end_time": segment["end_time"],
        "person_id": person_id,
        "track_id": track_id,
        "temporary_id": character.get("temporary_id"),
        "bbox_summary": yolo_row.get("bbox_summary") if yolo_row else unknown_bbox_summary(),
        "pose": yolo_row.get("pose") if yolo_row else empty_pose_summary(),
        "motion": yolo_row.get("motion", _unknown_motion()) if yolo_row else _unknown_motion(),
        "behavior": {
            "description": character.get("behavior", "Observable behavior unclear."),
            "modalities_used": modalities,
            "confidence": character.get("confidence", "low"),
            "uncertainty": character.get(
                "uncertainty", "sampled frames are sparse and pose landmarks are unavailable"
            ),
        },
    }


def _setup_logger(tree: OutputTree) -> logging.Logger:
    logger = logging.getLogger(f"video_features.{tree.root.name}")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(tree.logs / "pipeline.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def _tracked_person_description(yolo_row: dict[str, Any]) -> dict[str, Any]:
    pose = yolo_row.get("pose", {})
    motion = yolo_row.get("motion", {})
    if pose.get("available"):
        behavior = pose.get("summary", "Tracked person is visible in the segment.")
    else:
        behavior = _motion_description(motion)
    return {
        "temporary_id": yolo_row.get("track_id"),
        "gender": "unknown",
        "age_range": "unknown",
        "clothing": "visible clothing unclear",
        "appearance": "tracked person visible in sampled segment",
        "behavior": behavior,
        "confidence": "medium",
        "uncertainty": "YOLO track provides location; clothing and detailed action remain unclear",
    }


def _unknown_motion() -> dict[str, Any]:
    return {
        "mean_motion": 0.0,
        "peak_motion": 0.0,
        "peak_time": 0.0,
        "activity_level": "unclear",
    }


def _motion_description(motion: dict[str, Any]) -> str:
    level = motion.get("activity_level", "unclear")
    if level in {"medium", "high"}:
        return f"Tracked person shows {level} bbox-center movement within the segment."
    if level == "low":
        return "Tracked person is mostly stationary within the segment."
    return "Tracked person is visible; movement level is unclear."

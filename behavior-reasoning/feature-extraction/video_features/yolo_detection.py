from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MIN_TRACK_CONFIDENCE = 0.35
MIN_TRACK_OBSERVATIONS = 3
MAX_TRACKS_PER_SEGMENT = 4


@dataclass(frozen=True)
class YoloStatus:
    available: bool
    reason: str


@dataclass(frozen=True)
class YoloBundle:
    status: YoloStatus
    observations_by_segment: dict[str, list[dict[str, Any]]]


def check_yolo_available() -> YoloStatus:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return YoloStatus(False, "ultralytics is not installed in this runtime")
    return YoloStatus(True, "ultralytics is available")


def run_yolo_tracking(
    *,
    video_path: Path,
    segments: list[dict[str, Any]],
    fps: float,
    frame_width: int,
    frame_height: int,
    det_model: str,
    pose_model: str,
    tracker: str,
) -> YoloBundle:
    status = check_yolo_available()
    if not status.available:
        return YoloBundle(status=status, observations_by_segment={})
    try:
        from ultralytics import YOLO
    except ImportError:
        return YoloBundle(status=status, observations_by_segment={})

    observations: list[dict[str, Any]] = []
    model_name = _fallback_model_name(pose_model, pose=True)
    pose_attempted = True
    try:
        model = YOLO(model_name)
    except Exception:
        pose_attempted = False
        try:
            model_name = _fallback_model_name(det_model, pose=False)
            model = YOLO(model_name)
        except Exception as exc:
            return YoloBundle(
                status=YoloStatus(False, f"YOLO model load failed: {exc}"),
                observations_by_segment={},
            )
    try:
        for frame_index, result in enumerate(
            model.track(
                source=str(video_path),
                stream=True,
                persist=True,
                tracker=tracker,
                classes=[0],
                verbose=False,
            )
        ):
            timestamp = frame_index / fps if fps > 0 else 0.0
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.xyxy is None:
                continue
            ids = getattr(boxes, "id", None)
            confs = getattr(boxes, "conf", None)
            keypoints = _result_keypoints(result)
            for index, bbox in enumerate(_to_list(boxes.xyxy)):
                track_number = _value_at(ids, index)
                if track_number is None:
                    continue
                confidence = _value_at(confs, index, default=0.0)
                observations.append(
                    {
                        "timestamp": round(timestamp, 3),
                        "track_id": f"track-{int(track_number)}",
                        "bbox": [round(float(value), 2) for value in bbox],
                        "confidence": round(float(confidence or 0.0), 3),
                        "source": "yolo",
                        "keypoints": keypoints[index] if index < len(keypoints) else None,
                    }
                )
    except Exception as exc:
        return YoloBundle(
            status=YoloStatus(False, f"YOLO tracking failed: {exc}"),
            observations_by_segment={},
        )

    assigned = assign_observations_to_segments(observations, segments)
    by_segment: dict[str, list[dict[str, Any]]] = {}
    for observation in assigned:
        by_segment.setdefault(str(observation["segment_id"]), []).append(observation)
    return YoloBundle(
        status=YoloStatus(
            True,
            f"ultralytics tracking produced {len(observations)} observations using {model_name}"
            + ("" if pose_attempted else " after detection fallback"),
        ),
        observations_by_segment={
            segment_id: group_segment_observations(items, frame_width, frame_height)
            for segment_id, items in by_segment.items()
        },
    )


def assign_observations_to_segments(
    observations: list[dict[str, Any]], segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    assigned: list[dict[str, Any]] = []
    for observation in observations:
        timestamp = float(observation["timestamp"])
        for segment in segments:
            if float(segment["start_time"]) <= timestamp <= float(segment["end_time"]):
                row = dict(observation)
                row["segment_id"] = segment["segment_id"]
                assigned.append(row)
    return assigned


def summarize_track(
    observations: list[dict[str, Any]],
    *,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    boxes = np.array([item["bbox"] for item in observations], dtype=float)
    mean_bbox = [int(round(value)) for value in boxes.mean(axis=0).tolist()]
    centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0))
    confidence = float(np.mean([float(item.get("confidence", 0.0)) for item in observations]))
    motion = _track_motion(centers, observations, frame_width, frame_height)
    pose = _pose_summary(observations, frame_width, frame_height)
    return {
        "bbox_summary": {
            "mean_bbox": mean_bbox,
            "visibility": _visibility(mean_bbox, frame_width, frame_height),
            "confidence": round(confidence, 3),
        },
        "pose": pose,
        "motion": motion,
    }


def empty_pose_summary() -> dict[str, Any]:
    return {
        "available": False,
        "summary": "Pose landmarks are unavailable for this segment.",
        "moving_body_parts": [],
        "speed": "unclear",
        "directionality": "unclear",
        "confidence": "low",
    }


def unknown_bbox_summary() -> dict[str, Any]:
    return {
        "mean_bbox": [],
        "visibility": "unclear",
        "confidence": 0.0,
    }


def group_segment_observations(
    observations: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for observation in observations:
        grouped.setdefault(str(observation["track_id"]), []).append(observation)
    rows = []
    for track_id, items in sorted(grouped.items()):
        mean_confidence = float(np.mean([float(item.get("confidence", 0.0)) for item in items]))
        if not track_id.startswith("track-"):
            continue
        if len(items) < MIN_TRACK_OBSERVATIONS or mean_confidence < MIN_TRACK_CONFIDENCE:
            continue
        summary = summarize_track(items, frame_width=frame_width, frame_height=frame_height)
        rows.append(
            {
                "track_id": track_id,
                "start_time": min(float(item["timestamp"]) for item in items),
                "end_time": max(float(item["timestamp"]) for item in items),
                "source": "yolo",
                **summary,
            }
        )
    return sorted(
        rows,
        key=lambda row: row["bbox_summary"]["confidence"],
        reverse=True,
    )[:MAX_TRACKS_PER_SEGMENT]


def _result_keypoints(result: Any) -> list[list[list[float]] | None]:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return []
    xy = getattr(keypoints, "xy", None)
    conf = getattr(keypoints, "conf", None)
    if xy is None:
        return []
    xy_values = _to_list(xy)
    conf_values = _to_list(conf) if conf is not None else []
    rows: list[list[list[float]] | None] = []
    for person_index, person_points in enumerate(xy_values):
        person_conf = conf_values[person_index] if person_index < len(conf_values) else []
        rows.append(
            [
                [
                    float(point[0]),
                    float(point[1]),
                    float(person_conf[index]) if index < len(person_conf) else 0.0,
                ]
                for index, point in enumerate(person_points)
            ]
        )
    return rows


def _fallback_model_name(model_name: str, *, pose: bool) -> str:
    if model_name.startswith("yolo11"):
        return model_name.replace("yolo11", "yolov8", 1)
    if not model_name:
        return "yolov8n-pose.pt" if pose else "yolov8n.pt"
    return model_name


def _to_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value.tolist()


def _value_at(value: Any, index: int, default: Any = None) -> Any:
    if value is None:
        return default
    values = _to_list(value)
    if index >= len(values):
        return default
    item = values[index]
    if isinstance(item, list):
        item = item[0] if item else default
    return item


def _visibility(bbox: list[int], frame_width: int, frame_height: int) -> str:
    x1, y1, x2, y2 = bbox
    box_height = max(0, y2 - y1)
    if box_height >= frame_height * 0.55 and y1 <= frame_height * 0.15:
        return "full"
    if box_height >= frame_height * 0.25:
        return "upper_body"
    if x1 <= 2 or y1 <= 2 or x2 >= frame_width - 2 or y2 >= frame_height - 2:
        return "partial"
    return "unclear"


def _speed(normalized_displacement: float) -> str:
    if normalized_displacement >= 0.18:
        return "fast"
    if normalized_displacement >= 0.08:
        return "medium"
    if normalized_displacement >= 0.02:
        return "slow"
    return "stationary"


def _directionality(centers: np.ndarray) -> str:
    if len(centers) < 2:
        return "unclear"
    delta = centers[-1] - centers[0]
    if abs(float(delta[0])) > abs(float(delta[1])) * 1.5 and abs(float(delta[0])) > 5:
        return "lateral"
    return "unclear"


def _movement_summary(normalized_displacement: float) -> str:
    speed = _speed(normalized_displacement)
    if speed == "stationary":
        return "Tracked person is mostly stationary within the segment."
    return f"Tracked person shows {speed} bbox-center movement within the segment."


def _track_motion(
    centers: np.ndarray,
    observations: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    if len(centers) < 2:
        return {"mean_motion": 0.0, "peak_motion": 0.0, "peak_time": 0.0, "activity_level": "low"}
    deltas = np.linalg.norm(np.diff(centers, axis=0), axis=1) / max(frame_width, frame_height, 1)
    peak_index = int(np.argmax(deltas))
    mean_motion = float(np.mean(deltas))
    peak_motion = float(deltas[peak_index])
    peak_time = float(observations[min(peak_index + 1, len(observations) - 1)]["timestamp"])
    return {
        "mean_motion": round(mean_motion, 4),
        "peak_motion": round(peak_motion, 4),
        "peak_time": round(peak_time, 3),
        "activity_level": _activity_level(mean_motion),
    }


def _activity_level(mean_motion: float) -> str:
    if mean_motion >= 0.12:
        return "high"
    if mean_motion >= 0.04:
        return "medium"
    return "low"


def _pose_summary(
    observations: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    keypoint_rows = [item.get("keypoints") for item in observations if item.get("keypoints")]
    if len(keypoint_rows) < 2:
        return {
            "available": False,
            "summary": "Pose landmarks are unavailable; bbox motion is reported under motion.",
            "moving_body_parts": [],
            "speed": "unclear",
            "directionality": "unclear",
            "confidence": "low",
        }

    groups = {
        "left_arm": [5, 7, 9],
        "right_arm": [6, 8, 10],
        "left_leg": [11, 13, 15],
        "right_leg": [12, 14, 16],
    }
    moving_parts = []
    movements = []
    for name, indices in groups.items():
        first = _mean_keypoint(keypoint_rows[0], indices)
        last = _mean_keypoint(keypoint_rows[-1], indices)
        if first is None or last is None:
            continue
        normalized = float(np.linalg.norm(np.array(last) - np.array(first))) / max(
            frame_width, frame_height, 1
        )
        movements.append(normalized)
        if normalized >= 0.08:
            moving_parts.append(name)
    mean_movement = float(np.mean(movements)) if movements else 0.0
    if moving_parts:
        summary = f"Pose landmarks suggest movement in {', '.join(moving_parts)}."
    else:
        summary = "Pose landmarks show no clear limb movement."
    return {
        "available": True,
        "summary": summary,
        "moving_body_parts": moving_parts,
        "speed": _speed(mean_movement),
        "directionality": "unclear",
        "confidence": "medium" if movements else "low",
    }


def _mean_keypoint(keypoints: list[list[float]], indices: list[int]) -> list[float] | None:
    visible = [
        [float(keypoints[index][0]), float(keypoints[index][1])]
        for index in indices
        if index < len(keypoints) and len(keypoints[index]) >= 3 and float(keypoints[index][2]) >= 0.2
    ]
    if not visible:
        return None
    return np.mean(np.array(visible), axis=0).tolist()

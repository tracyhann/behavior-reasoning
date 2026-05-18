from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float


def probe_video(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if fps <= 0 or frame_count <= 0:
        raise ValueError(f"Could not determine duration for video: {path}")
    return VideoInfo(
        path=path,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_sec=round(frame_count / fps, 3),
    )


def sample_segment_frames(
    video_path: Path,
    segment: dict[str, Any],
    output_dir: Path,
    *,
    max_width: int = 640,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    start = float(segment["start_time"])
    end = float(segment["end_time"])
    midpoint = start + max(0.0, end - start) / 2.0
    times = [start, midpoint, max(start, end - 0.05)]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    samples: list[dict[str, Any]] = []
    try:
        for label, timestamp in zip(("start", "middle", "end"), times):
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = _resize_if_needed(frame, max_width=max_width)
            path = output_dir / f"{segment['segment_id']}_{label}_{timestamp:.2f}.jpg"
            cv2.imwrite(str(path), frame)
            samples.append({"path": path, "timestamp": round(timestamp, 3), "label": label})
    finally:
        cap.release()
    return samples


def estimate_motion(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) < 2:
        return {
            "mean_motion": 0.0,
            "peak_motion": 0.0,
            "peak_time": samples[0]["timestamp"] if samples else 0.0,
            "activity_level": "low",
        }

    diffs: list[tuple[float, float]] = []
    previous = _gray(samples[0]["path"])
    for sample in samples[1:]:
        current = _gray(sample["path"])
        if previous is None or current is None:
            previous = current
            continue
        if previous.shape != current.shape:
            current = cv2.resize(current, (previous.shape[1], previous.shape[0]))
        diff = float(np.mean(cv2.absdiff(previous, current)) / 255.0)
        diffs.append((round(diff, 4), float(sample["timestamp"])))
        previous = current

    if not diffs:
        return {"mean_motion": 0.0, "peak_motion": 0.0, "peak_time": 0.0, "activity_level": "low"}
    mean_motion = round(float(np.mean([diff for diff, _ in diffs])), 4)
    peak_motion, peak_time = max(diffs, key=lambda item: item[0])
    return {
        "mean_motion": mean_motion,
        "peak_motion": peak_motion,
        "peak_time": round(peak_time, 3),
        "activity_level": _activity_level(mean_motion),
    }


def _resize_if_needed(frame: np.ndarray, *, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    return cv2.resize(frame, (max_width, int(height * scale)))


def _gray(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return image


def _activity_level(mean_motion: float) -> str:
    if mean_motion >= 0.12:
        return "high"
    if mean_motion >= 0.04:
        return "medium"
    return "low"

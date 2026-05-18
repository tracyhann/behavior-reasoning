from __future__ import annotations


def build_segments(
    *,
    duration_sec: float,
    segment_sec: float = 10.0,
    stride_sec: float = 6.0,
) -> list[dict[str, float | str]]:
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    if segment_sec <= 0:
        raise ValueError("segment_sec must be positive")
    if stride_sec <= 0:
        raise ValueError("stride_sec must be positive")

    segments: list[dict[str, float | str]] = []
    start = 0.0
    index = 1
    while start < duration_sec:
        end = min(start + segment_sec, duration_sec)
        segments.append(
            {
                "segment_id": f"seg_{index:03d}",
                "start_time": round(start, 3),
                "end_time": round(end, 3),
            }
        )
        if end >= duration_sec:
            break
        start += stride_sec
        index += 1
    return segments

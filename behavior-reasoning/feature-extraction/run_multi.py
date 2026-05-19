"""Driver script: run feature extraction with a chosen VLM backend.

Example:
  python run_multi.py \
    --backend qwen25 \
    --model-id /workspace/models/Qwen2.5-VL-72B-Instruct \
    --input  /workspace/dataset-ours/oCkUyjaZuNI/oCkUyjaZuNI.mp4 \
    --output /workspace/behavior-reasoning/outputs_qwen25_72b \
    --expected-subjects 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

from video_features.pipeline import PipelineConfig, run_pipeline
from video_features.vlm_multi import build_vlm


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    describer = build_vlm(
        args.backend,
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    result = run_pipeline(
        PipelineConfig(
            input_path=args.input,
            output_root=args.output,
            segment_sec=args.segment_sec,
            stride_sec=args.stride_sec,
            use_vlm=True,
            model_id=args.model_id,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            max_segments=args.max_segments,
            det_model=args.det_model,
            pose_model=args.pose_model,
            tracker=args.tracker,
            expected_subjects=args.expected_subjects,
        ),
        describer=describer,
    )
    print(
        f"wrote {result.segment_count} segments and {result.character_count} characters "
        f"to {result.output_dir}"
    )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-backend video behavior extraction.")
    parser.add_argument("--backend", required=True, choices=["qwen25", "qwen3", "gemma3"])
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=6.0)
    parser.add_argument("--det-model", default="yolo11n.pt")
    parser.add_argument("--pose-model", default="yolo11n-pose.pt")
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--expected-subjects", type=int, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

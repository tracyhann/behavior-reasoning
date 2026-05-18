from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PipelineConfig, run_pipeline
from .vlm import DEFAULT_MODEL_ID


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_pipeline(
        PipelineConfig(
            input_path=args.input,
            output_root=args.output,
            segment_sec=args.segment_sec,
            stride_sec=args.stride_sec,
            use_vlm=not args.no_vlm and not args.dry_run,
            model_id=args.model_id,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            max_segments=args.max_segments,
            det_model=args.det_model,
            pose_model=args.pose_model,
            tracker=args.tracker,
            expected_subjects=args.expected_subjects,
        )
    )
    print(
        f"wrote {result.segment_count} segments and {result.character_count} characters "
        f"to {result.output_dir}"
    )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract timestamped video behavior evidence.")
    parser.add_argument("--input", type=Path, required=True, help="Input video path")
    parser.add_argument("--output", type=Path, required=True, help="Output root directory")
    parser.add_argument("--segment-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=6.0)
    parser.add_argument("--det-model", default="yolo11n.pt")
    parser.add_argument("--pose-model", default="yolo11n-pose.pt")
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=360)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--expected-subjects", type=int, default=None)
    parser.add_argument("--no-vlm", action="store_true", help="Use deterministic fallback only")
    parser.add_argument("--dry-run", action="store_true", help="Create outputs without loading VLM")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

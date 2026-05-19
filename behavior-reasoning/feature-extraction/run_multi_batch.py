"""Batch runner: load one VLM once, process several videos."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from video_features.pipeline import PipelineConfig, run_pipeline
from video_features.vlm_multi import build_vlm


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = json.loads(Path(args.plan).read_text())
    # plan: [{"video_id":..., "input":..., "expected_subjects":3, ...}, ...]
    describer = build_vlm(
        args.backend,
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
    )
    summary = []
    for entry in plan:
        vid = entry["video_id"]
        in_path = Path(entry["input"])
        out_root = Path(args.output)
        expected = entry.get("expected_subjects")
        start = time.time()
        try:
            result = run_pipeline(
                PipelineConfig(
                    input_path=in_path,
                    output_root=out_root,
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
                    expected_subjects=expected,
                ),
                describer=describer,
            )
            elapsed = time.time() - start
            summary.append(
                {
                    "video_id": vid,
                    "ok": True,
                    "segments": result.segment_count,
                    "characters": result.character_count,
                    "elapsed_sec": round(elapsed, 1),
                    "output_dir": str(result.output_dir),
                }
            )
            print(
                f"[{vid}] ok segments={result.segment_count} "
                f"chars={result.character_count} elapsed={elapsed:.1f}s"
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - start
            summary.append(
                {
                    "video_id": vid,
                    "ok": False,
                    "error": repr(exc),
                    "elapsed_sec": round(elapsed, 1),
                }
            )
            print(f"[{vid}] FAILED: {exc}")
    out_summary = Path(args.output) / "batch_summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"summary -> {out_summary}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch multi-video feature extraction.")
    parser.add_argument("--backend", required=True, choices=["qwen25", "qwen3", "gemma3"])
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--plan", required=True, help="JSON plan file")
    parser.add_argument("--output", required=True, help="Output root directory")
    parser.add_argument("--segment-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=6.0)
    parser.add_argument("--det-model", default="yolov8n.pt")
    parser.add_argument("--pose-model", default="yolov8n-pose.pt")
    parser.add_argument("--tracker", default="botsort.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--max-segments", type=int, default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

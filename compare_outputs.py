"""Side-by-side comparison of feature-extraction + reasoning outputs.

Usage:
  python compare_outputs.py \
    --label "Qwen2.5-VL-72B" outputs_qwen25_72b \
    --label "Qwen3-VL-8B"    outputs_qwen3_8b \
    [--label "Gemma3-12B"    outputs_gemma3_12b] \
    > comparison.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

VIDEOS = ["oCkUyjaZuNI", "-JUtzfuiV08", "DmSmN-oqFZ0", "G6f0w5BRasw"]


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def video_stats(out_root: Path, vid: str) -> dict:
    base = out_root / vid
    chars = _load_json(base / "characters" / "characters.json")
    dets = _load_json(base / "detections" / "detections.json")
    chains = _load_json(base / "chains" / "behavior_chains.json")
    canonical = _load_json(base / "chains" / "canonical_events.json")
    qa = _load_json(base / "chains" / "qa_qc_report.json")
    stats = {"missing": chars is None or dets is None}
    if stats["missing"]:
        return stats
    char_dict = chars.get("characters", {}) if isinstance(chars.get("characters"), dict) else {}
    stats["roster_size"] = len(char_dict)
    scenes = dets.get("scenes", []) if dets else []
    stats["segments"] = len(scenes)
    scene_descs = [s.get("description", "") for s in scenes]
    stats["scene_non_empty"] = sum(
        1 for d in scene_descs if d and d != "Scene description unavailable."
        and d != "Scene description unavailable from sampled frames."
    )
    stats["scene_avg_len"] = round(
        statistics.mean([len(d) for d in scene_descs]) if scene_descs else 0, 1
    )
    objects_per_segment = [
        len(s.get("objects", [])) for s in dets.get("objects", [])
    ]
    stats["objects_total"] = sum(objects_per_segment)
    stats["objects_avg"] = round(
        statistics.mean(objects_per_segment) if objects_per_segment else 0, 2
    )
    chars_per_segment = [
        ch for ch in dets.get("characters", []) if ch.get("person_id")
    ]
    stats["resolved_char_rows"] = len(chars_per_segment)
    stats["yolo_available"] = dets.get("metadata", {}).get("yolo_available")
    stats["chains"] = len((chains or {}).get("chains", []))
    stats["canonical_events"] = len(canonical) if isinstance(canonical, list) else 0
    stats["orphan_events"] = len((chains or {}).get("orphan_event_ids", []))
    if qa:
        passes = qa.get("iterations", [])
        ok_iter = sum(1 for it in passes if it.get("status") == "pass")
        stats["qa_iterations"] = len(passes)
        stats["qa_clean_iterations"] = ok_iter
        stats["qa_overall"] = qa.get("overall_status")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
        help="One label PATH pair per VLM",
    )
    args = parser.parse_args()

    runs = [(label, Path(path)) for label, path in args.label]

    # Summary header
    print("# Behavior-reasoning multi-model comparison\n")
    print("Videos: " + ", ".join(VIDEOS) + "\n")
    print("## Headline per-video stats\n")
    cols = ["video", "model", "segs", "roster", "non-empty scenes", "objs total", "char rows", "chains", "events", "qa clean"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for vid in VIDEOS:
        for label, path in runs:
            s = video_stats(path, vid)
            if s.get("missing"):
                print(f"| {vid} | {label} | _missing_ | | | | | | | |")
                continue
            print(
                f"| {vid} | {label} | {s.get('segments','?')} | {s.get('roster_size','?')} | "
                f"{s.get('scene_non_empty','?')}/{s.get('segments','?')} | "
                f"{s.get('objects_total','?')} | {s.get('resolved_char_rows','?')} | "
                f"{s.get('chains','?')} | {s.get('canonical_events','?')} | "
                f"{s.get('qa_clean_iterations','?')}/{s.get('qa_iterations','?')} |"
            )

    # Aggregates
    print("\n## Aggregates across all videos\n")
    print("| model | segs | scenes non-empty | objs total | resolved char rows | chains | canonical events |")
    print("|---|---|---|---|---|---|---|")
    for label, path in runs:
        tot = {k: 0 for k in [
            "segments", "scene_non_empty", "objects_total", "resolved_char_rows",
            "chains", "canonical_events",
        ]}
        for vid in VIDEOS:
            s = video_stats(path, vid)
            if s.get("missing"):
                continue
            for k in tot:
                tot[k] += s.get(k, 0) or 0
        print(
            f"| {label} | {tot['segments']} | {tot['scene_non_empty']} | {tot['objects_total']} | "
            f"{tot['resolved_char_rows']} | {tot['chains']} | {tot['canonical_events']} |"
        )

    # Sample scenes
    print("\n## Sample scene descriptions (first segment of `oCkUyjaZuNI`)\n")
    for label, path in runs:
        dets = _load_json(path / "oCkUyjaZuNI" / "detections" / "detections.json")
        if not dets or not dets.get("scenes"):
            print(f"### {label}\n_no data_\n")
            continue
        first = dets["scenes"][0]
        print(f"### {label}")
        print(f"- scene: {first.get('description')}")
        print(f"- confidence: {first.get('confidence')}\n")
        objs = next(iter(dets.get("objects", [])), {}).get("objects", [])[:3]
        if objs:
            for o in objs:
                print(f"  - obj `{o['label']}` ({o['confidence']}): {o['description']}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

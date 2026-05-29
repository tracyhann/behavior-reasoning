from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from meta_baselines.backends.base import parse_backend_args


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    questions_path = resolve_path(project_root, args.questions)
    out_path = resolve_path(project_root, args.out)
    rows = read_jsonl(questions_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    backend_kwargs = parse_backend_args(args.backend_arg)
    backend = load_backend(args.backend, project_root=project_root, **backend_kwargs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    exit_code = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = build_base_record(row, project_root, args.backend)
            try:
                video_path = resolve_path(project_root, row["video_path"])
                raw_response = backend.answer(row, video_path, build_prompt(row))
                record["status"] = "complete"
                record["raw_response"] = raw_response
                parsed = try_parse_json(raw_response)
                if parsed is not None:
                    record["parsed_response"] = parsed
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = error_record(exc)
                exit_code = 1
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"wrote {out_path}")
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a pluggable VLM/LLM backend over CARE question JSONL.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--backend",
        required=True,
        help="Import path in the form module:Class, for example meta_baselines.backends.registry_video_backend:RegistryVideoBackend.",
    )
    parser.add_argument(
        "--backend-arg",
        action="append",
        default=[],
        help="Backend constructor argument in key=value format. May be repeated.",
    )
    parser.add_argument("--limit", type=int)
    return parser


def load_backend(import_spec: str, **kwargs: Any) -> Any:
    if ":" not in import_spec:
        raise ValueError("--backend must use module:Class format")
    module_name, class_name = import_spec.split(":", 1)
    module = importlib.import_module(module_name)
    backend_class = getattr(module, class_name)
    return backend_class(**kwargs)


def build_base_record(row: dict[str, Any], project_root: Path, backend: str) -> dict[str, Any]:
    record = {
        "schema_version": "care.baseline_answer.v1",
        "created_at": now_iso(),
        "backend": backend,
        "question_id": row["question_id"],
        "clip_id": row["clip_id"],
        "episode_id": row["episode_id"],
        "task_type": row["task_type"],
        "video_path": str(resolve_path(project_root, row["video_path"]).relative_to(project_root)),
        "pre_segmented": bool(row.get("pre_segmented", False)),
        "prompt": row["prompt"],
        "scoring_targets": row.get("scoring_targets", []),
    }
    for optional_key in ["run_dir", "expected_answer"]:
        if optional_key in row:
            record[optional_key] = row[optional_key]
    return record


def build_prompt(row: dict[str, Any]) -> str:
    evidence_scope = "attached pre-segmented video clip" if row.get("pre_segmented") else "attached video"
    return (
        "Answer only from the provided prompt and the "
        f"{evidence_scope}. Do not use hidden labels or gold answers. "
        "Return only valid JSON unless the prompt explicitly requests another format.\n\n"
        f"{row['prompt']}"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ["question_id", "clip_id", "episode_id", "video_path", "task_type", "prompt"]:
            if key not in row:
                raise ValueError(f"{path}:{line_number} missing required key {key}")
        rows.append(row)
    return rows


def try_parse_json(value: str) -> Any | None:
    text = extract_json_text(value)
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_json_text(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    object_start = text.find("{")
    array_start = text.find("[")
    starts = [idx for idx in [object_start, array_start] if idx >= 0]
    if not starts:
        return text
    start = min(starts)
    end_char = "}" if text[start] == "{" else "]"
    end = text.rfind(end_char)
    return text[start : end + 1] if end >= start else text


def error_record(exc: Exception) -> dict[str, Any]:
    return {
        "error_type": exc.__class__.__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc(limit=6),
    }


def resolve_path(project_root: Path, value: Path | str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "meta_baselines/datasets/SIV-Bench_10"
ANNOTATION_ROOT = DATASET_ROOT / "annotations"
PROBE_PATH = DATASET_ROOT / "review/video_probe.json"
QA_PATH = DATASET_ROOT / "SIV-Bench-QA.tsv"
RUN_ROOT = PROJECT_ROOT / "meta_baselines/egagent_sivbench_social"
QUESTIONS_DIR = RUN_ROOT / "questions"
RESULTS_DIR = PROJECT_ROOT / "meta_baselines/results/egagent_sivbench_social"


def character(
    character_id: str,
    gender_presentation: str,
    age_group: str,
    upper_body: str,
    hair_style: str,
    facial_hair: str = "none_visible",
    features: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "character_id": character_id,
        "gender_presentation": gender_presentation,
        "age_group": age_group,
        "clothing": {
            "upper_body": upper_body,
            "lower_body": "not_visible_or_unclear",
            "outerwear": None,
            "colors": [],
            "accessories_or_context": [],
        },
        "appearance": {
            "hair_style": hair_style,
            "hair_color": "not_visible_or_unclear",
            "facial_hair": facial_hair,
            "notable_visible_features": features or [],
        },
    }


def interaction(
    interaction_id: str,
    actor_id: str,
    recipient_ids: list[str],
    relation: str,
    spans: list[list[float]],
    description: str,
) -> dict[str, Any]:
    return {
        "interaction_id": interaction_id,
        "actor_id": actor_id,
        "recipient_ids": recipient_ids,
        "relation": relation,
        "status": "labeled_interaction",
        "timebase": "video_relative_seconds",
        "time_spans_sec": spans,
        "evidence": [
            {
                "start_sec": start,
                "end_sec": end,
                "modality": "audio_visual",
                "evidence_description": description,
            }
            for start, end in spans
        ],
    }


VIDEO_LABELS: list[dict[str, Any]] = [
    {
        "video_key": "boss-employee__video_141",
        "video_path": "origin/boss-employee/video_141.mp4",
        "source_relation_category": "boss-employee",
        "summary": "Two office characters discuss confidential information.",
        "characters": [
            character("C1", "male_presenting", "adult", "dark long-sleeve top", "short dark hair"),
            character("C2", "female_presenting", "adult", "dark sleeveless top", "long blond hair"),
        ],
        "interactions": [
            interaction("SIV001_INT001", "C1", ["C2"], "talks_to", [[0.0, 5.8]], "C1 speaks toward C2 during the opening office exchange."),
            interaction("SIV001_INT002", "C2", ["C1"], "talks_to", [[5.8, 13.0]], "C2 speaks back to C1 after the confidential-information prompt."),
        ],
        "recipient_responded": True,
        "notes": ["Timing is approximate and based on reviewed contact-sheet frames plus SIV-Bench QA context."],
    },
    {
        "video_key": "caregiver-recipient__video_18",
        "video_path": "origin/caregiver-recipient/video_18.mp4",
        "source_relation_category": "caregiver-recipient",
        "summary": "A caregiver assists a reclining person through guided movement.",
        "characters": [
            character("C1", "female_presenting", "adult", "light purple or lavender top", "long dark hair"),
            character("C2", "unknown_or_unclear", "adult_or_unclear", "yellow or light blanket/clothing", "not_visible_or_unclear"),
        ],
        "interactions": [
            interaction("SIV002_INT001", "C1", ["C2"], "assists_or_guides", [[0.0, 23.6]], "C1 holds or guides C2's arms/body during a caregiving movement activity."),
        ],
        "recipient_responded": True,
        "notes": ["C2 is partially visible; the interaction is physical assistance rather than a clean spoken exchange."],
    },
    {
        "video_key": "classmates__video_63",
        "video_path": "origin/classmates/video_63.mp4",
        "source_relation_category": "classmates",
        "summary": "A group of classmates gathers around the camera and performs playful poses.",
        "characters": [
            character("C1", "male_presenting", "youth_or_adult", "white school shirt", "short brown hair"),
            character("C2", "unknown_or_unclear", "youth_or_adult", "dark top or mask-covered upper body", "long dark hair", features=["face mask"]),
            character("C3", "female_presenting", "youth_or_adult", "white school shirt", "long dark hair"),
            character("C4", "male_presenting", "youth_or_adult", "white school shirt", "short dark hair"),
            character("C5", "female_presenting", "youth_or_adult", "white school shirt", "long dark hair"),
        ],
        "interactions": [
            interaction("SIV003_INT001", "C4", ["C2"], "performs_toward", [[9.5, 14.2]], "C4 performs a playful pose or movement toward the central group including C2."),
            interaction("SIV003_INT002", "C3", ["C2"], "attends_to", [[2.0, 7.5]], "C3 leans near and attends toward C2 during the group pose."),
        ],
        "recipient_responded": True,
        "notes": ["This is a group scene; only the foreground repeatedly visible classmates are assigned IDs."],
    },
    {
        "video_key": "coach-player__video_147",
        "video_path": "origin/coach-player/video_147.mp4",
        "source_relation_category": "coach-player",
        "summary": "An adult coach supports or consoles a young table-tennis player after play.",
        "characters": [
            character("C1", "female_presenting", "adult", "orange or peach sports shirt", "dark hair"),
            character("C2", "male_presenting", "child_or_youth", "white and blue sports shirt", "short dark hair"),
            character("C3", "male_presenting", "adult_or_youth", "white and green sports shirt", "short dark hair"),
        ],
        "interactions": [
            interaction("SIV004_INT001", "C1", ["C2"], "comforts_or_coaches", [[17.0, 32.7]], "C1 approaches and consoles or coaches C2 near the table-tennis table."),
        ],
        "recipient_responded": True,
        "notes": ["Audience members are excluded from the character library."],
    },
    {
        "video_key": "colleague__video_114",
        "video_path": "origin/colleague/video_114.mp4",
        "source_relation_category": "colleague",
        "summary": "Office colleagues answer or react to an office gossip prompt.",
        "characters": [
            character("C1", "male_presenting", "adult", "brown jacket or sweatshirt", "short dark hair"),
            character("C2", "female_presenting", "adult", "white sweater", "long light brown hair"),
            character("C3", "female_presenting", "adult", "gray sweater", "long dark hair"),
            character("C4", "male_presenting", "adult", "blue button-up shirt", "shaved or very short hair"),
            character("C5", "female_presenting", "adult", "purple sweater", "long dark hair"),
        ],
        "interactions": [
            interaction("SIV005_INT001", "C1", ["C2"], "talks_to", [[0.0, 3.6]], "C1 speaks from his desk in the direction of C2's workspace."),
            interaction("SIV005_INT002", "C2", ["C1"], "talks_to", [[2.2, 4.0]], "C2 turns from her desk and responds toward C1."),
            interaction("SIV005_INT003", "C1", ["C2", "C3", "C4", "C5"], "talks_to_group", [[9.5, 12.0]], "C1 speaks with the group of office colleagues near the end."),
        ],
        "recipient_responded": True,
        "notes": ["The video is a montage; interactions are labeled only for repeated foreground colleagues."],
    },
    {
        "video_key": "couple__video_108",
        "video_path": "origin/couple/video_108.mp4",
        "source_relation_category": "couple",
        "summary": "Two people in bed interact closely in a low-light scene.",
        "characters": [
            character("C1", "unknown_or_unclear", "adult", "under red or pink blanket", "dark hair"),
            character("C2", "unknown_or_unclear", "adult", "dark top", "dark hair"),
        ],
        "interactions": [
            interaction("SIV006_INT001", "C2", ["C1"], "physical_affection_or_teasing", [[9.0, 35.0]], "C2 leans over, touches, and closely attends to C1 in bed."),
        ],
        "recipient_responded": True,
        "notes": ["Lighting makes gender and clothing details unclear; labels describe visible presentation only."],
    },
    {
        "video_key": "friends__video_132",
        "video_path": "origin/friends/video_132.mp4",
        "source_relation_category": "friends",
        "summary": "A man in a hallway talks with or performs toward seated friends.",
        "characters": [
            character("C1", "male_presenting", "adult", "navy shirt", "short dark hair"),
            character("C2", "female_presenting", "adult", "white top", "brown hair tied back"),
            character("C3", "female_presenting", "adult", "pink top or jacket", "brown or dark hair"),
        ],
        "interactions": [
            interaction("SIV007_INT001", "C1", ["C2"], "talks_to", [[18.0, 93.0]], "C1 stands facing and speaking toward seated C2 for most of the exchange."),
            interaction("SIV007_INT002", "C2", ["C1"], "talks_to", [[55.0, 85.0]], "C2 looks up and responds toward C1 during the hallway interaction."),
        ],
        "recipient_responded": True,
        "notes": ["Additional background friends are visible but not assigned stable IDs."],
    },
    {
        "video_key": "transactional__video_11",
        "video_path": "origin/transactional/video_11.mp4",
        "source_relation_category": "transactional",
        "summary": "A salesperson demonstrates and negotiates over a drone with a customer.",
        "characters": [
            character("C1", "female_presenting", "adult", "light blue T-shirt", "long dark hair"),
            character("C2", "male_presenting", "adult", "white T-shirt", "short light brown hair"),
            character("C3", "female_presenting", "adult", "black dress", "long dark hair"),
        ],
        "interactions": [
            interaction("SIV008_INT001", "C1", ["C2"], "talks_to", [[22.0, 57.5]], "C1 presents the drone and speaks with C2 in the store."),
            interaction("SIV008_INT002", "C2", ["C1"], "talks_to", [[28.0, 45.0]], "C2 negotiates or responds to C1 during the sales exchange."),
        ],
        "recipient_responded": True,
        "notes": ["C3 appears as another shop worker or bystander and is not the primary interaction partner."],
    },
    {
        "video_key": "service__video_73",
        "video_path": "origin/service/video_73.mp4",
        "source_relation_category": "service",
        "summary": "A waitress and restaurant customer interact in a service scenario.",
        "characters": [
            character("C1", "female_presenting", "adult", "black service apron over light top", "long blond hair"),
            character("C2", "female_presenting", "adult", "green or teal top", "curly light hair"),
        ],
        "interactions": [
            interaction("SIV009_INT001", "C1", ["C2"], "talks_to", [[9.0, 35.0]], "C1 serves or speaks toward C2 in the restaurant scene."),
            interaction("SIV009_INT002", "C2", ["C1"], "talks_to", [[35.0, 49.0]], "C2 responds toward C1 during the service exchange."),
        ],
        "recipient_responded": True,
        "notes": ["The video has title overlays and cuts; labels focus on the waitress-customer exchange."],
    },
    {
        "video_key": "sibling__video_146",
        "video_path": "origin/sibling/video_146.mp4",
        "source_relation_category": "sibling",
        "summary": "Two children talk on a couch before an adult enters at the end.",
        "characters": [
            character("C1", "male_presenting", "child", "blue and white school jacket", "very short dark hair"),
            character("C2", "female_presenting", "child", "white school shirt with red tie", "long dark hair"),
            character("C3", "male_presenting", "adult", "dark shirt and dark pants", "short dark hair"),
        ],
        "interactions": [
            interaction("SIV010_INT001", "C1", ["C2"], "talks_to", [[0.0, 18.0]], "C1 talks while seated next to C2 on the couch."),
            interaction("SIV010_INT002", "C2", ["C1"], "talks_to", [[18.0, 35.0]], "C2 turns toward and talks with C1."),
            interaction("SIV010_INT003", "C3", ["C1", "C2"], "talks_to_group", [[37.0, 45.6]], "C3 enters and addresses the two children."),
        ],
        "recipient_responded": True,
        "notes": ["The adult is visible only near the end but is included because he enters the social exchange."],
    },
]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"prepare", "all"}:
        prepare()
    if args.command in {"run", "all"}:
        run_strict_adapter()
    if args.command in {"score", "all"}:
        score_strict_adapter()
    if args.command == "score-qwen":
        score_qwen_tool_answers()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and score SIV-Bench CARE-style social labels.")
    parser.add_argument("command", choices=("prepare", "run", "score", "score-qwen", "all"))
    return parser


def prepare() -> None:
    ANNOTATION_ROOT.mkdir(parents=True, exist_ok=True)
    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    probe_by_path = load_probe()
    qa_by_path = load_qa_rows()
    index_rows: list[dict[str, Any]] = []
    atomic_questions: list[dict[str, Any]] = []
    qwen_rows: list[dict[str, Any]] = []

    for index, label in enumerate(VIDEO_LABELS, start=1):
        video_path = DATASET_ROOT / label["video_path"]
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        clip_id = f"sivbench_{index:03d}_{label['video_key']}"
        episode_id = f"SIV{index:03d}"
        annotation_dir = ANNOTATION_ROOT / label["video_key"]
        annotation_dir.mkdir(parents=True, exist_ok=True)
        duration = float(probe_by_path.get(label["video_path"], {}).get("duration_sec", 0.0))

        character_labels = {
            "schema_version": "care.character_annotation.v1",
            "clip_id": clip_id,
            "episode_id": episode_id,
            "clip_path": rel(video_path),
            "timebase": "video_relative_seconds",
            "characters": label["characters"],
            "annotation_metadata": {
                "created_at": now_iso(),
                "label_status": "candidate_character_description",
                "source_dataset": "Fancylalala/SIV-Bench",
                "source_relation_category": label["source_relation_category"],
                "source_contact_sheet": rel(contact_sheet_for(label["video_path"])),
                "notes": [
                    "Gender and age fields describe visual presentation only.",
                    "Character IDs are semantic foreground IDs for this video, not raw detector tracks.",
                    *label["notes"],
                ],
            },
        }
        interaction_labels = {
            "schema_version": "care.interaction_annotation.v1",
            "clip_id": clip_id,
            "episode_id": episode_id,
            "clip_path": rel(video_path),
            "timebase": "video_relative_seconds",
            "interactions": label["interactions"],
            "annotation_metadata": {
                "created_at": now_iso(),
                "label_status": "candidate_interaction_label",
                "source_dataset": "Fancylalala/SIV-Bench",
                "source_relation_category": label["source_relation_category"],
                "notes": [
                    "Interaction labels use semantic character IDs from character_labels.json.",
                    "Timing is approximate pending adjudication with audio and frame-level review.",
                ],
            },
        }
        evaluation_questions = build_evaluation_questions(
            clip_id=clip_id,
            episode_id=episode_id,
            video_path=video_path,
            characters=label["characters"],
            interactions=label["interactions"],
            recipient_responded=label["recipient_responded"],
        )
        write_json(annotation_dir / "character_labels.json", character_labels)
        write_json(annotation_dir / "interaction_labels.json", interaction_labels)
        write_json(annotation_dir / "evaluation_questions.json", evaluation_questions)
        write_json(
            annotation_dir / "meta.json",
            {
                "schema_version": "care.sivbench_social_annotation_meta.v1",
                "clip_id": clip_id,
                "episode_id": episode_id,
                "source_dataset": "Fancylalala/SIV-Bench",
                "source_hf_url": "https://huggingface.co/datasets/Fancylalala/SIV-Bench",
                "source_relation_category": label["source_relation_category"],
                "video_path": rel(video_path),
                "duration_sec": duration,
                "summary": label["summary"],
                "sivbench_qa_rows": qa_by_path.get(label["video_path"].replace("origin/", ""), []),
            },
        )
        index_rows.append(
            {
                "schema_version": "care.sivbench_social_annotation_index.v1",
                "clip_id": clip_id,
                "episode_id": episode_id,
                "source_relation_category": label["source_relation_category"],
                "video_path": rel(video_path),
                "annotation_dir": rel(annotation_dir),
                "character_count": len(label["characters"]),
                "interaction_count": len(label["interactions"]),
            }
        )
        for question in evaluation_questions["questions"]:
            atomic_questions.append(
                {
                    "clip_id": clip_id,
                    "episode_id": episode_id,
                    "video_path": rel(video_path),
                    **question,
                }
            )
        qwen_rows.append(build_qwen_row(label, clip_id, episode_id, video_path, evaluation_questions))

    write_jsonl(ANNOTATION_ROOT / "index.jsonl", index_rows)
    write_jsonl(QUESTIONS_DIR / "sivbench_social_questions.jsonl", atomic_questions)
    write_jsonl(QUESTIONS_DIR / "sivbench_social_qwen_tool_questions.jsonl", qwen_rows)
    write_json(
        RUN_ROOT / "run_metadata.json",
        {
            "schema_version": "care.meta_baseline.sivbench_social_run_metadata.v1",
            "created_at": now_iso(),
            "baseline_name": "egagent_style_sivbench_social_adapter",
            "upstream_repo": "facebookresearch/egagent",
            "upstream_egagent_unmodified": False,
            "dataset_root": rel(DATASET_ROOT),
            "annotation_root": rel(ANNOTATION_ROOT),
            "downloaded_video_count": len(VIDEO_LABELS),
            "notes": [
                "This adapter evaluates CARE-style social labels over selected SIV-Bench samples.",
                "Labels are candidate labels and should be adjudicated before publication use.",
            ],
        },
    )
    write_json(DATASET_ROOT / "manifest.json", build_download_manifest(probe_by_path))
    write_jsonl(DATASET_ROOT / "manifest.jsonl", build_download_manifest(probe_by_path)["videos"])


def build_evaluation_questions(
    *,
    clip_id: str,
    episode_id: str,
    video_path: Path,
    characters: list[dict[str, Any]],
    interactions: list[dict[str, Any]],
    recipient_responded: bool,
) -> dict[str, Any]:
    character_ids = [item["character_id"] for item in characters]
    first_actor = interactions[0]["actor_id"] if interactions else None
    c2_talk_targets = [
        rid
        for item in interactions
        if item.get("actor_id") == "C2" and item.get("relation") == "talks_to"
        for rid in item.get("recipient_ids", [])
    ]
    c2_answer: Any = None
    if len(c2_talk_targets) == 1:
        c2_answer = c2_talk_targets[0]
    elif len(c2_talk_targets) > 1:
        c2_answer = sorted(set(c2_talk_targets))
    c1_interacts = any(edge_touches(item, "C1") for item in interactions)
    c1_c2_interacts = any(edge_connects(item, "C1", "C2") for item in interactions)
    return {
        "schema_version": "care.clip_evaluation_questions.v1",
        "clip_id": clip_id,
        "episode_id": episode_id,
        "clip_path": rel(video_path),
        "timebase": "video_relative_seconds",
        "questions": [
            {
                "question_id": f"{episode_id}_Q001_character_count",
                "question": "How many characters are present in this video?",
                "answer": len(character_ids),
                "answer_type": "integer",
                "scoring": "exact_match",
            },
            {
                "question_id": f"{episode_id}_Q002_present_characters",
                "question": "Which characters are present in this video?",
                "answer": character_ids,
                "answer_type": "character_id_list",
                "scoring": "set_exact_match",
            },
            {
                "question_id": f"{episode_id}_Q003_first_initiator",
                "question": "Who initiated the interaction first?",
                "answer": first_actor,
                "answer_type": "nullable_character_id",
                "scoring": "exact_match",
            },
            {
                "question_id": f"{episode_id}_Q004_c2_recipient",
                "question": "Who did C2 talk to?",
                "answer": c2_answer,
                "answer_type": "nullable_character_id_or_list",
                "scoring": "exact_match",
            },
            {
                "question_id": f"{episode_id}_Q005_recipient_response",
                "question": "Did the recipient respond to the social initiator?",
                "answer": recipient_responded,
                "answer_type": "boolean",
                "scoring": "exact_match",
            },
            {
                "question_id": f"{episode_id}_Q006_c1_interaction",
                "question": "Did C1 interact with any characters?",
                "answer": c1_interacts,
                "answer_type": "boolean",
                "scoring": "exact_match",
            },
            {
                "question_id": f"{episode_id}_Q007_c1_c2_interaction",
                "question": "Did C1 and C2 interact?",
                "answer": c1_c2_interacts,
                "answer_type": "boolean",
                "scoring": "exact_match",
            },
        ],
    }


def build_qwen_row(
    label: dict[str, Any],
    clip_id: str,
    episode_id: str,
    video_path: Path,
    evaluation_questions: dict[str, Any],
) -> dict[str, Any]:
    expected = expected_from_questions(evaluation_questions["questions"])
    prompt = (
        "Evaluate this SIV-Bench video with a CARE-style character library. "
        "Use the attached video and character library only. Do not invent IDs outside the library. "
        "The labels C1, C2, etc. refer to the supplied character library. "
        "Answer only with valid JSON using exactly these keys: "
        "character_count, present_characters, first_interaction_initiator, c2_talked_to, "
        "recipient_responded_to_first_initiator, c1_interacted_with_any_character, "
        "c1_and_c2_interacted, scene_description. "
        "Use integers, lists of character IDs, booleans, strings, or null for deterministic fields.\n\n"
        f"Character library: {json.dumps(label['characters'], sort_keys=True)}"
    )
    return {
        "question_id": f"{episode_id}_QWEN_ALL",
        "clip_id": clip_id,
        "episode_id": episode_id,
        "video_path": rel(video_path),
        "pre_segmented": False,
        "task_type": "egagent_style_sivbench_social_all_questions",
        "prompt": prompt,
        "expected_answer": expected,
        "scoring_targets": list(expected),
    }


def run_strict_adapter() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for annotation_dir in sorted(p for p in ANNOTATION_ROOT.iterdir() if p.is_dir()):
        characters = read_json(annotation_dir / "character_labels.json")
        evaluation = read_json(annotation_dir / "evaluation_questions.json")
        present_ids = [item["character_id"] for item in characters["characters"]]
        predicted = {
            "character_count": len(present_ids),
            "present_characters": present_ids,
            "first_interaction_initiator": None,
            "c2_talked_to": None,
            "recipient_responded_to_first_initiator": False,
            "c1_interacted_with_any_character": False,
            "c1_and_c2_interacted": False,
        }
        traces.append(
            {
                "schema_version": "care.meta_baseline.egagent_tool_trace.v1",
                "created_at": now_iso(),
                "clip_id": evaluation["clip_id"],
                "tools": [
                    "visual_frame_search",
                    "character_library_search",
                    "transcript_search",
                    "entity_graph_search",
                ],
                "summary": (
                    "Strict EGAgent-style adapter used the character library but did not generate "
                    "verified speech, gaze, gesture, or recipient-access interaction edges."
                ),
            }
        )
        rows.append(
            {
                "schema_version": "care.meta_baseline.egagent_sivbench_social_answer.v1",
                "created_at": now_iso(),
                "baseline_name": "egagent_style_sivbench_social_strict_adapter",
                "model_execution": "not_invoked_strict_tool_absence_adapter",
                "upstream_egagent_unmodified": False,
                "clip_id": evaluation["clip_id"],
                "episode_id": evaluation["episode_id"],
                "video_path": evaluation["clip_path"],
                "parsed_answer": predicted,
                "tool_trace": traces[-1]["tools"],
                "predicted_edges": [],
            }
        )
    write_jsonl(RESULTS_DIR / "sivbench_social_answers.jsonl", rows)
    write_jsonl(RESULTS_DIR / "sivbench_social_tool_trace.jsonl", traces)


def score_strict_adapter() -> None:
    answers = {row["clip_id"]: row for row in read_jsonl(RESULTS_DIR / "sivbench_social_answers.jsonl")}
    details, correct, total, gold_edges = score_answer_mapping(answers)
    report = {
        "schema_version": "care.meta_baseline.egagent_sivbench_social_metrics.v1",
        "created_at": now_iso(),
        "baseline_name": "egagent_style_sivbench_social_strict_adapter",
        "upstream_egagent_unmodified": False,
        "question_exact_match": round(correct / total, 6) if total else 0.0,
        "question_correct": correct,
        "question_total": total,
        "directed_edge_micro": f1(gold_edges, set()),
        "gold_edge_count": len(gold_edges),
        "predicted_edge_count": 0,
        "question_details": details,
    }
    write_json(RESULTS_DIR / "sivbench_social_metrics.json", report)


def score_qwen_tool_answers() -> None:
    qwen_path = RESULTS_DIR / "sivbench_social_qwen_tool_answers.jsonl"
    expected_by_qid = {
        row["question_id"]: row["expected_answer"]
        for row in read_jsonl(QUESTIONS_DIR / "sivbench_social_qwen_tool_questions.jsonl")
    }
    rows = read_jsonl(qwen_path)
    details: list[dict[str, Any]] = []
    correct = 0
    total = 0
    for row in rows:
        qid = row["question_id"]
        pred = row.get("parsed_response") or {}
        expected = expected_by_qid[qid]
        for key, gold_value in expected.items():
            pred_value = pred.get(key)
            is_correct = normalize_value(pred_value) == normalize_value(gold_value)
            correct += int(is_correct)
            total += 1
            details.append(
                {
                    "question_id": qid,
                    "metric_key": key,
                    "prediction": pred_value,
                    "gold": gold_value,
                    "correct": is_correct,
                    "status": row.get("status"),
                }
            )
    write_json(
        RESULTS_DIR / "sivbench_social_qwen_tool_metrics.json",
        {
            "schema_version": "care.meta_baseline.egagent_sivbench_social_qwen_tool_metrics.v1",
            "created_at": now_iso(),
            "baseline_name": "egagent_style_sivbench_social_qwen_tool_packet",
            "qwen_model_invoked": True,
            "upstream_egagent_unmodified": False,
            "question_exact_match": round(correct / total, 6) if total else 0.0,
            "question_correct": correct,
            "question_total": total,
            "question_details": details,
        },
    )


def score_answer_mapping(answers: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int, set[tuple[str, str, str]]]:
    details: list[dict[str, Any]] = []
    correct = 0
    total = 0
    gold_edges: set[tuple[str, str, str]] = set()
    for annotation_dir in sorted(p for p in ANNOTATION_ROOT.iterdir() if p.is_dir()):
        evaluation = read_json(annotation_dir / "evaluation_questions.json")
        interactions = read_json(annotation_dir / "interaction_labels.json")["interactions"]
        gold_edges |= interaction_edges(interactions)
        pred = answers[evaluation["clip_id"]]["parsed_answer"]
        expected = expected_from_questions(evaluation["questions"])
        for key, gold_value in expected.items():
            pred_value = pred.get(key)
            is_correct = normalize_value(pred_value) == normalize_value(gold_value)
            correct += int(is_correct)
            total += 1
            details.append(
                {
                    "clip_id": evaluation["clip_id"],
                    "metric_key": key,
                    "prediction": pred_value,
                    "gold": gold_value,
                    "correct": is_correct,
                }
            )
    return details, correct, total, gold_edges


def expected_from_questions(questions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "character_count": questions[0]["answer"],
        "present_characters": questions[1]["answer"],
        "first_interaction_initiator": questions[2]["answer"],
        "c2_talked_to": questions[3]["answer"],
        "recipient_responded_to_first_initiator": questions[4]["answer"],
        "c1_interacted_with_any_character": questions[5]["answer"],
        "c1_and_c2_interacted": questions[6]["answer"],
    }


def build_download_manifest(probe_by_path: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "care.meta_baselines.sivbench_download_manifest.v1",
        "created_at": now_iso(),
        "dataset": "Fancylalala/SIV-Bench",
        "source_url": "https://huggingface.co/datasets/Fancylalala/SIV-Bench",
        "downloaded_video_count": len(VIDEO_LABELS),
        "videos": [
            {
                "video_key": label["video_key"],
                "source_relation_category": label["source_relation_category"],
                "repo_path": label["video_path"],
                "local_path": rel(DATASET_ROOT / label["video_path"]),
                "bytes": (DATASET_ROOT / label["video_path"]).stat().st_size,
                "duration_sec": probe_by_path.get(label["video_path"], {}).get("duration_sec"),
            }
            for label in VIDEO_LABELS
        ],
    }


def edge_touches(item: dict[str, Any], cid: str) -> bool:
    return item.get("actor_id") == cid or cid in item.get("recipient_ids", [])


def edge_connects(item: dict[str, Any], a: str, b: str) -> bool:
    return item.get("actor_id") == a and b in item.get("recipient_ids", []) or item.get("actor_id") == b and a in item.get("recipient_ids", [])


def interaction_edges(interactions: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    edges = set()
    for item in interactions:
        actor = item.get("actor_id")
        relation = item.get("relation")
        for recipient in item.get("recipient_ids", []):
            if actor and recipient and relation:
                edges.add((actor, recipient, relation))
    return edges


def f1(gold: set[tuple[str, str, str]], pred: set[tuple[str, str, str]]) -> dict[str, float]:
    if not gold and not pred:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(gold & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(2 * precision * recall / (precision + recall), 6) if precision + recall else 0.0,
    }


def load_probe() -> dict[str, dict[str, Any]]:
    if not PROBE_PATH.exists():
        return {}
    return {row["video_path"]: row for row in json.loads(PROBE_PATH.read_text(encoding="utf-8"))}


def load_qa_rows() -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    if not QA_PATH.exists():
        return rows
    with QA_PATH.open(encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("video_path"):
                rows.setdefault(row["video_path"], []).append(row)
    return rows


def contact_sheet_for(video_path: str) -> Path:
    safe = video_path.replace("/", "__").replace(".mp4", "")
    return DATASET_ROOT / "review/contact_sheets" / f"{safe}_sheet.jpg"


def normalize_value(value: Any) -> Any:
    if isinstance(value, list):
        return sorted(value)
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT) if path.is_absolute() else path)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())

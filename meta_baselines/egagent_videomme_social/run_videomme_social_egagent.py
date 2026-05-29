from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "meta_baselines/datasets/Video-MME_1"
ANNOTATION_ROOT = DATASET_ROOT / "annotations"
VIDEO_DIR = DATASET_ROOT / "videos"
PROBE_PATH = DATASET_ROOT / "review/video_probe.json"
RUN_ROOT = PROJECT_ROOT / "meta_baselines/egagent_videomme_social"
QUESTIONS_DIR = RUN_ROOT / "questions"
RESULTS_DIR = PROJECT_ROOT / "meta_baselines/results/egagent_videomme_social"


def character(
    character_id: str,
    gender_presentation: str,
    age_group: str,
    upper_body: str,
    hair_style: str,
    *,
    lower_body: str = "not_visible_or_unclear",
    outerwear: str | None = None,
    colors: list[str] | None = None,
    hair_color: str = "not_visible_or_unclear",
    facial_hair: str = "none_visible",
    accessories_or_context: list[str] | None = None,
    notable_visible_features: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "character_id": character_id,
        "gender_presentation": gender_presentation,
        "age_group": age_group,
        "clothing": {
            "upper_body": upper_body,
            "lower_body": lower_body,
            "outerwear": outerwear,
            "colors": colors or [],
            "accessories_or_context": accessories_or_context or [],
        },
        "appearance": {
            "hair_style": hair_style,
            "hair_color": hair_color,
            "facial_hair": facial_hair,
            "notable_visible_features": notable_visible_features or [],
        },
    }


def interaction(
    interaction_id: str,
    actor_id: str,
    recipient_ids: list[str],
    relation: str,
    spans: list[list[float]],
    description: str,
    *,
    modality: str = "audio_visual",
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
                "modality": modality,
                "evidence_description": description,
            }
            for start, end in spans
        ],
    }


VIDEO_LABELS: list[dict[str, Any]] = [
    {
        "video_id": "101",
        "videoID": "tiMaUSvlzIU",
        "file_name": "101_tiMaUSvlzIU.mp4",
        "source_category": "Film & Television / Movie & TV Show",
        "summary": "Sitcom-style scene with a woman on a phone and three men later conversing at a table.",
        "characters": [
            character("C1", "female_presenting", "adult", "magenta sleeveless top", "long blond hair", hair_color="blond"),
            character("C2", "male_presenting", "adult", "gray short-sleeve shirt", "short dark hair", hair_color="dark"),
            character(
                "C3",
                "male_presenting",
                "adult",
                "brown jacket over purple hoodie",
                "short brown hair",
                hair_color="brown",
                accessories_or_context=["eyeglasses"],
                notable_visible_features=["wearing eyeglasses"],
            ),
            character("C4", "male_presenting", "adult", "purple jacket or hoodie", "short dark hair", hair_color="dark"),
        ],
        "interactions": [
            interaction("VMME101_INT001", "C2", ["C3", "C4"], "talks_to_group", [[16.8, 25.2]], "C2 faces the table group and speaks during the seated exchange."),
            interaction("VMME101_INT002", "C4", ["C2", "C3"], "talks_to_group", [[33.6, 42.0]], "C4 gestures and addresses the other two seated men."),
        ],
        "recipient_responded": True,
        "notes": [
            "C1 appears in a separate phone scene; her offscreen phone recipient is not a visible CARE character.",
            "Timing is approximate from sampled review frames.",
        ],
    },
    {
        "video_id": "103",
        "videoID": "G267g0DpCVg",
        "file_name": "103_G267g0DpCVg.mp4",
        "source_category": "Film & Television / Movie & TV Show",
        "summary": "A single woman eats at a table; no visible dyadic recipient is foregrounded.",
        "characters": [
            character("C1", "female_presenting", "adult", "blue hoodie", "hair tied up", hair_color="brown or dark blond"),
        ],
        "interactions": [],
        "recipient_responded": False,
        "notes": ["Single-character sample; hands or partial background fragments are not assigned stable IDs."],
    },
    {
        "video_id": "104",
        "videoID": "_cZXyj6rYVg",
        "file_name": "104__cZXyj6rYVg.mp4",
        "source_category": "Film & Television / Movie & TV Show",
        "summary": "A man explains from a whiteboard while a seated woman responds.",
        "characters": [
            character(
                "C1",
                "female_presenting",
                "adult",
                "magenta top with light patterned hoodie",
                "long blond hair",
                hair_color="blond",
                outerwear="light patterned hoodie",
            ),
            character("C2", "male_presenting", "adult", "blue T-shirt over yellow long sleeves", "short dark hair", hair_color="dark"),
        ],
        "interactions": [
            interaction("VMME104_INT001", "C2", ["C1"], "talks_to", [[0.0, 12.0], [18.0, 25.3]], "C2 stands by the whiteboard and speaks toward C1."),
            interaction("VMME104_INT002", "C1", ["C2"], "talks_to", [[12.0, 16.0], [28.9, 33.0]], "C1 looks toward C2 and responds from the couch."),
        ],
        "recipient_responded": True,
        "notes": ["Primary interaction is the visible whiteboard explanation and response."],
    },
    {
        "video_id": "105",
        "videoID": "rQhLWHtHyiM",
        "file_name": "105_rQhLWHtHyiM.mp4",
        "source_category": "Film & Television / Movie & TV Show",
        "summary": "Several adults in a room engage in a tense social exchange with pointing and movement.",
        "characters": [
            character("C1", "male_presenting", "adult", "dark suit jacket", "curly dark hair", hair_color="dark"),
            character("C2", "female_presenting", "adult", "dark fitted jacket", "short or pulled-back dark hair", hair_color="dark"),
            character("C3", "male_presenting", "adult", "dark suit", "short light hair", hair_color="light blond or gray"),
            character("C4", "unknown_or_unclear", "adult", "dark clothing", "light-colored hair", hair_color="light blond"),
        ],
        "interactions": [
            interaction("VMME105_INT001", "C1", ["C2"], "gestures_toward", [[0.0, 8.9]], "C1 raises a hand and gestures toward C2's side of the room.", modality="visual"),
            interaction("VMME105_INT002", "C1", ["C2", "C3"], "talks_to_group", [[22.2, 31.1]], "C1 faces the other visible adults during the room exchange."),
            interaction("VMME105_INT003", "C2", ["C1"], "gestures_toward", [[39.9, 44.4]], "C2 responds with a directed hand/arm gesture toward C1.", modality="visual"),
        ],
        "recipient_responded": True,
        "notes": ["The labels use visible directed action rather than assuming all background adults are interaction partners."],
    },
    {
        "video_id": "106",
        "videoID": "bYXhA8VG8Lw",
        "file_name": "106_bYXhA8VG8Lw.mp4",
        "source_category": "Film & Television / Movie & TV Show",
        "summary": "A woman speaks closely to a curly-haired man, who later responds while another man is present.",
        "characters": [
            character("C1", "male_presenting", "adult", "dark suit jacket over light shirt", "curly dark hair", hair_color="dark"),
            character("C2", "female_presenting", "adult", "dark top", "long dark hair", hair_color="dark"),
            character("C3", "male_presenting", "adult", "dark patterned jacket or shirt", "short light hair", hair_color="blond"),
        ],
        "interactions": [
            interaction("VMME106_INT001", "C2", ["C1"], "talks_to", [[8.7, 34.8]], "C2 stays close to C1 and speaks into his personal space."),
            interaction("VMME106_INT002", "C1", ["C2"], "talks_to", [[52.1, 86.9]], "C1 turns back toward C2 and answers during the close exchange."),
            interaction("VMME106_INT003", "C3", ["C1", "C2"], "attends_to_group", [[34.8, 43.5]], "C3 attends toward the interaction at the table.", modality="visual"),
        ],
        "recipient_responded": True,
        "notes": ["C3 is present but not labeled as the first speaker."],
    },
    {
        "video_id": "251",
        "videoID": "1sTQOxXFO44",
        "file_name": "251_1sTQOxXFO44.mp4",
        "source_category": "Life Record / Daily Life",
        "summary": "A woman performs an after-work routine alone at home.",
        "characters": [
            character("C1", "female_presenting", "adult", "pink robe or housewear", "long hair", hair_color="brown or dark blond"),
        ],
        "interactions": [],
        "recipient_responded": False,
        "notes": ["Single-person routine; object manipulation is not a social interaction."],
    },
    {
        "video_id": "252",
        "videoID": "RP1AL2DU6vQ",
        "file_name": "252_RP1AL2DU6vQ.mp4",
        "source_category": "Life Record / Daily Life",
        "summary": "A student narrates a school day, with brief breakfast/table and school-walk shots.",
        "characters": [
            character("C1", "male_presenting", "child_or_youth", "dark school jacket", "medium-length dark hair", hair_color="dark"),
            character("C2", "unknown_or_unclear", "adult_or_youth", "light top at breakfast table", "not_visible_or_unclear"),
        ],
        "interactions": [],
        "recipient_responded": False,
        "notes": ["C2 is a briefly visible tablemate; no directed interpersonal exchange is labeled."],
    },
    {
        "video_id": "257",
        "videoID": "s-lM2uwiwyQ",
        "file_name": "257_s-lM2uwiwyQ.mp4",
        "source_category": "Life Record / Daily Life",
        "summary": "A healthcare worker's day-in-the-life video with a patient-care segment and coworkers at lunch.",
        "characters": [
            character("C1", "female_presenting", "adult", "dark scrub top or work shirt", "dark hair tied back", hair_color="dark"),
            character("C2", "unknown_or_unclear", "adult", "blue jeans and light shoes", "not_visible_or_unclear"),
            character("C3", "unknown_or_unclear", "adult", "patterned scrub top", "not_visible_or_unclear"),
        ],
        "interactions": [
            interaction("VMME257_INT001", "C1", ["C2"], "assists_or_tests", [[40.0, 60.0]], "C1 performs a visible care/testing action with C2's hand or body in the clinical setting."),
        ],
        "recipient_responded": True,
        "notes": ["C2 is partially visible as the care recipient; the label is physical care rather than a spoken dyadic exchange."],
    },
    {
        "video_id": "255",
        "videoID": "CQUphYL0vY8",
        "file_name": "255_CQUphYL0vY8.mp4",
        "source_category": "Life Record / Daily Life",
        "summary": "A young woman moves through work and social activities, including a cafe meeting and greetings.",
        "characters": [
            character("C1", "female_presenting", "adult", "dark short-sleeve top", "long brown hair", hair_color="brown"),
            character("C2", "male_presenting", "adult", "light pink shirt", "short or shaved hair", hair_color="not_visible_or_unclear"),
            character("C3", "female_presenting", "adult", "green top", "long hair", hair_color="brown"),
            character("C4", "female_presenting", "adult", "dark purple top", "long hair", hair_color="brown"),
            character("C5", "female_presenting", "adult", "dark top with sunglasses", "long hair", hair_color="brown", accessories_or_context=["sunglasses"]),
        ],
        "interactions": [
            interaction("VMME255_INT001", "C1", ["C2"], "talks_to", [[31.0, 43.0]], "C1 sits at a cafe table and speaks or gestures toward C2."),
            interaction("VMME255_INT002", "C2", ["C1"], "talks_to", [[36.0, 45.0]], "C2 sits across from C1 and responds during the cafe meeting."),
            interaction("VMME255_INT003", "C5", ["C1"], "greets", [[72.0, 76.0]], "C5 raises her arms and greets C1 on the street.", modality="audio_visual"),
        ],
        "recipient_responded": True,
        "notes": ["The first scored interaction is the cafe conversation; later street greeting is a separate visible social event."],
    },
    {
        "video_id": "256",
        "videoID": "aFfMGy94sjE",
        "file_name": "256_aFfMGy94sjE.mp4",
        "source_category": "Life Record / Daily Life",
        "summary": "A police/emergency-services workplace video with dispatcher and front-desk interactions.",
        "characters": [
            character("C1", "female_presenting", "adult", "black polo or work shirt", "long blond hair", hair_color="blond"),
            character("C2", "female_presenting", "adult", "black work shirt", "long dark hair", hair_color="dark", notable_visible_features=["wearing eyeglasses"]),
            character("C3", "male_presenting", "adult", "dark jacket or shirt", "short dark hair", hair_color="dark", notable_visible_features=["wearing eyeglasses"]),
        ],
        "interactions": [
            interaction("VMME256_INT001", "C3", ["C1"], "talks_to", [[45.0, 55.0]], "C3 faces C1 across the service window or counter."),
            interaction("VMME256_INT002", "C1", ["C3"], "talks_to", [[47.0, 57.0]], "C1 faces C3 and responds at the service window or counter."),
            interaction("VMME256_INT003", "C2", ["C1"], "talks_to", [[76.0, 90.0]], "C2 and C1 sit at workstations and converse or coordinate."),
        ],
        "recipient_responded": True,
        "notes": ["Interview-to-camera shots are excluded; only visible interpersonal workplace exchanges are labeled."],
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
    parser = argparse.ArgumentParser(description="Prepare and score Video-MME CARE-style social labels.")
    parser.add_argument("command", choices=("prepare", "run", "score", "score-qwen", "all"))
    return parser


def prepare() -> None:
    ANNOTATION_ROOT.mkdir(parents=True, exist_ok=True)
    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    probe_by_file = load_probe()
    index_rows = []
    atomic_questions = []
    qwen_rows = []

    for label in VIDEO_LABELS:
        video_key = f"{label['video_id']}_{label['videoID']}"
        video_dir = ANNOTATION_ROOT / video_key
        video_dir.mkdir(parents=True, exist_ok=True)
        duration = float(probe_by_file.get(label["file_name"], {}).get("duration_sec", 0.0))
        clip_id = f"videomme_{video_key}"
        video_path = VIDEO_DIR / label["file_name"]

        character_labels = {
            "schema_version": "care.character_annotation.v1",
            "clip_id": clip_id,
            "episode_id": label["video_id"],
            "clip_path": rel(video_path),
            "timebase": "video_relative_seconds",
            "characters": label["characters"],
            "annotation_metadata": {
                "created_at": now_iso(),
                "label_status": "candidate_character_description",
                "label_scope": (
                    "Foreground or repeated visible people only; background crowds, unlabeled extras, "
                    "and offscreen phone/audio recipients are excluded."
                ),
                "source_contact_sheet": rel(
                    Path("meta_baselines/datasets/Video-MME_1/review/contact_sheets")
                    / f"{label['video_id']}_{label['videoID']}_sheet.jpg"
                ),
                "notes": label["notes"],
            },
        }
        interaction_labels = {
            "schema_version": "care.interaction_annotation.v1",
            "clip_id": clip_id,
            "episode_id": label["video_id"],
            "clip_path": rel(video_path),
            "timebase": "video_relative_seconds",
            "interactions": label["interactions"],
            "annotation_metadata": {
                "created_at": now_iso(),
                "label_status": "candidate_interaction_label",
                "notes": [
                    "Candidate labels are based on sampled review frames and available audio/video context.",
                    "No interaction edge is labeled unless a directed co-present exchange is visible or strongly implied by the reviewed evidence.",
                ],
            },
        }
        evaluation_questions = build_evaluation_questions(
            clip_id=clip_id,
            episode_id=label["video_id"],
            video_path=video_path,
            characters=label["characters"],
            interactions=label["interactions"],
            recipient_responded=bool(label["recipient_responded"]),
        )
        write_json(video_dir / "character_labels.json", character_labels)
        write_json(video_dir / "interaction_labels.json", interaction_labels)
        write_json(video_dir / "evaluation_questions.json", evaluation_questions)
        write_json(
            video_dir / "meta.json",
            {
                "schema_version": "care.videomme_social_annotation_meta.v1",
                "clip_id": clip_id,
                "video_id": label["video_id"],
                "videoID": label["videoID"],
                "video_path": rel(video_path),
                "source_category": label["source_category"],
                "duration_sec": duration,
                "summary": label["summary"],
                "label_status": "candidate",
            },
        )
        index_rows.append(
            {
                "schema_version": "care.videomme_social_annotation_index.v2",
                "clip_id": clip_id,
                "video_id": label["video_id"],
                "videoID": label["videoID"],
                "video_path": rel(video_path),
                "annotation_dir": rel(video_dir),
                "character_count": len(label["characters"]),
                "interaction_count": len(label["interactions"]),
            }
        )
        for question in evaluation_questions["questions"]:
            atomic_questions.append(
                {
                    "clip_id": clip_id,
                    "episode_id": label["video_id"],
                    "video_path": rel(video_path),
                    **question,
                }
            )
        qwen_rows.append(build_qwen_row(label, clip_id, video_path, evaluation_questions))

    write_jsonl(ANNOTATION_ROOT / "index.jsonl", index_rows)
    write_jsonl(QUESTIONS_DIR / "videomme_social_questions.jsonl", atomic_questions)
    write_jsonl(QUESTIONS_DIR / "videomme_social_qwen_tool_questions.jsonl", qwen_rows)
    write_json(
        RUN_ROOT / "run_metadata.json",
        {
            "schema_version": "care.meta_baseline.videomme_social_run_metadata.v2",
            "created_at": now_iso(),
            "baseline_name": "egagent_style_videomme_social_adapter",
            "upstream_repo": "facebookresearch/egagent",
            "upstream_egagent_unmodified": False,
            "dataset_root": rel(DATASET_ROOT),
            "annotation_root": rel(ANNOTATION_ROOT),
            "held_out_scoring_inputs": [rel(ANNOTATION_ROOT)],
            "notes": [
                "This run uses real-human Video-MME samples from Film & Television and Life Record/Daily Life.",
                "Evaluation questions intentionally follow the CE003 deterministic schema.",
            ],
        },
    )


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
    c2_talked_to = c2_talk_recipients(interactions)
    c1_interacts = character_interacts("C1", interactions)
    c1_c2_interacts = pair_interacts("C1", "C2", interactions)
    prefix = f"VMME{episode_id}"
    return {
        "schema_version": "care.clip_evaluation_questions.v1",
        "clip_id": clip_id,
        "episode_id": episode_id,
        "clip_path": rel(video_path),
        "timebase": "video_relative_seconds",
        "questions": [
            {
                "question_id": f"{prefix}_Q001_character_count",
                "question": "How many characters are present in this clip?",
                "answer": len(character_ids),
                "answer_type": "integer",
                "scoring": "exact_match",
            },
            {
                "question_id": f"{prefix}_Q002_present_characters",
                "question": "Which characters are present in this clip?",
                "answer": character_ids,
                "answer_type": "character_id_list",
                "scoring": "set_exact_match",
            },
            {
                "question_id": f"{prefix}_Q003_first_initiator",
                "question": "Who initiated the interaction first?",
                "answer": first_actor,
                "answer_type": "nullable_character_id",
                "scoring": "exact_match",
            },
            {
                "question_id": f"{prefix}_Q004_c2_recipient",
                "question": "Who did C2 talk to?",
                "answer": c2_talked_to,
                "answer_type": "nullable_character_id_or_list",
                "scoring": "exact_match",
            },
            {
                "question_id": f"{prefix}_Q005_recipient_response",
                "question": "Did the recipient respond to the social initiator?",
                "answer": recipient_responded,
                "answer_type": "boolean",
                "scoring": "exact_match",
            },
            {
                "question_id": f"{prefix}_Q006_c1_interaction",
                "question": "Did C1 interact with any characters?",
                "answer": c1_interacts,
                "answer_type": "boolean",
                "scoring": "exact_match",
            },
            {
                "question_id": f"{prefix}_Q007_c1_c2_interaction",
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
    video_path: Path,
    evaluation_questions: dict[str, Any],
) -> dict[str, Any]:
    expected = qwen_expected_from_questions(evaluation_questions["questions"])
    prompt = (
        "Use the attached video and the character library. The library gives the allowed character IDs, "
        "but you must decide which of those IDs are visible in this clip and what interactions occur. "
        "Do not invent IDs outside the library. Ignore offscreen phone/audio recipients and background extras. "
        "Treat an interaction as a directed visible social cue between co-present characters, such as talking, "
        "answering, greeting, gesturing toward, attending to, or physically assisting a recipient. "
        "For 'who did C2 talk to', use null if C2 does not visibly talk to another visible character; use a "
        "single character ID for one recipient or a list of character IDs for multiple recipients. "
        "Use true or false, never a character ID or list, for recipient_responded_to_first_initiator, "
        "c1_interacted_with_any_character, and c1_and_c2_interacted. "
        "Return only valid JSON with exactly these keys: "
        "character_count, present_characters, first_interaction_initiator, c2_talked_to, "
        "recipient_responded_to_first_initiator, c1_interacted_with_any_character, "
        "c1_and_c2_interacted, scene_description.\n\n"
        "Output field types:\n"
        "- character_count: integer\n"
        "- present_characters: list of character ID strings\n"
        "- first_interaction_initiator: one character ID string or null\n"
        "- c2_talked_to: one character ID string, a list of character ID strings, or null\n"
        "- recipient_responded_to_first_initiator: boolean\n"
        "- c1_interacted_with_any_character: boolean\n"
        "- c1_and_c2_interacted: boolean\n"
        "- scene_description: string\n\n"
        f"Character library: {json.dumps(label['characters'], sort_keys=True)}"
    )
    return {
        "question_id": f"VMME{label['video_id']}_QWEN_ALL",
        "clip_id": clip_id,
        "episode_id": label["video_id"],
        "video_path": rel(video_path),
        "pre_segmented": False,
        "task_type": "egagent_style_videomme_social_ce003_questions",
        "prompt": prompt,
        "expected_answer": expected,
        "scoring_targets": list(expected),
    }


def run_strict_adapter() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    traces = []
    for annotation_dir in sorted(ANNOTATION_ROOT.glob("*_*")):
        if not annotation_dir.is_dir():
            continue
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
                "schema_version": "care.meta_baseline.egagent_tool_trace.v2",
                "created_at": now_iso(),
                "clip_id": evaluation["clip_id"],
                "tools": [
                    "visual_frame_search",
                    "character_library_search",
                    "transcript_search",
                    "entity_graph_search",
                ],
                "summary": (
                    "Strict EGAgent-style adapter can recover provided character-library nodes but "
                    "does not generate speech, gaze, gesture, or recipient-access edges."
                ),
            }
        )
        rows.append(
            {
                "schema_version": "care.meta_baseline.egagent_videomme_social_answer.v2",
                "created_at": now_iso(),
                "baseline_name": "egagent_style_videomme_social_strict_adapter",
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
    write_jsonl(RESULTS_DIR / "videomme_social_answers.jsonl", rows)
    write_jsonl(RESULTS_DIR / "videomme_social_tool_trace.jsonl", traces)


def score_strict_adapter() -> None:
    answers = {row["clip_id"]: row for row in read_jsonl(RESULTS_DIR / "videomme_social_answers.jsonl")}
    details, correct, total = score_answer_rows(
        [answers[read_json(path / "evaluation_questions.json")["clip_id"]] for path in annotation_dirs()]
    )
    gold_edges: set[tuple[str, str, str]] = set()
    for annotation_dir in annotation_dirs():
        gold_edges |= interaction_edges(read_json(annotation_dir / "interaction_labels.json")["interactions"])

    report = {
        "schema_version": "care.meta_baseline.egagent_videomme_social_metrics.v2",
        "created_at": now_iso(),
        "baseline_name": "egagent_style_videomme_social_strict_adapter",
        "upstream_egagent_unmodified": False,
        "question_exact_match": round(correct / total, 6) if total else 0.0,
        "question_correct": correct,
        "question_total": total,
        "question_type_accuracy": question_type_accuracy(details),
        "directed_edge_micro": f1(gold_edges, set()),
        "gold_edge_count": len(gold_edges),
        "predicted_edge_count": 0,
        "question_details": details,
    }
    write_json(RESULTS_DIR / "videomme_social_metrics.json", report)


def score_qwen_tool_answers() -> None:
    rows = read_jsonl(RESULTS_DIR / "videomme_social_qwen_tool_answers.jsonl")
    details, correct, total = score_answer_rows(rows, parsed_key="parsed_response", by_qid=True)
    write_json(
        RESULTS_DIR / "videomme_social_qwen_tool_metrics.json",
        {
            "schema_version": "care.meta_baseline.egagent_videomme_social_qwen_tool_metrics.v2",
            "created_at": now_iso(),
            "baseline_name": "egagent_style_videomme_social_qwen_tool_packet",
            "qwen_model_invoked": True,
            "upstream_egagent_unmodified": False,
            "question_exact_match": round(correct / total, 6) if total else 0.0,
            "question_correct": correct,
            "question_total": total,
            "question_type_accuracy": question_type_accuracy(details),
            "question_details": details,
        },
    )


def score_answer_rows(
    rows: list[dict[str, Any]],
    *,
    parsed_key: str = "parsed_answer",
    by_qid: bool = False,
) -> tuple[list[dict[str, Any]], int, int]:
    if by_qid:
        expected_lookup = {
            row["question_id"]: row["expected_answer"]
            for row in read_jsonl(QUESTIONS_DIR / "videomme_social_qwen_tool_questions.jsonl")
        }
    else:
        expected_lookup = {
            read_json(path / "evaluation_questions.json")["clip_id"]: qwen_expected_from_questions(
                read_json(path / "evaluation_questions.json")["questions"]
            )
            for path in annotation_dirs()
        }

    details = []
    correct = 0
    total = 0
    for row in rows:
        lookup_key = row["question_id"] if by_qid else row["clip_id"]
        pred = row.get(parsed_key) or {}
        expected = expected_lookup[lookup_key]
        for key, gold_value in expected.items():
            pred_value = pred.get(key)
            is_correct = normalize_value(pred_value) == normalize_value(gold_value)
            correct += int(is_correct)
            total += 1
            details.append(
                {
                    "clip_id": row["clip_id"],
                    "question_id": row.get("question_id"),
                    "metric_key": key,
                    "prediction": pred_value,
                    "gold": gold_value,
                    "correct": is_correct,
                    "status": row.get("status"),
                }
            )
    return details, correct, total


def qwen_expected_from_questions(questions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "character_count": questions[0]["answer"],
        "present_characters": questions[1]["answer"],
        "first_interaction_initiator": questions[2]["answer"],
        "c2_talked_to": questions[3]["answer"],
        "recipient_responded_to_first_initiator": questions[4]["answer"],
        "c1_interacted_with_any_character": questions[5]["answer"],
        "c1_and_c2_interacted": questions[6]["answer"],
    }


def c2_talk_recipients(interactions: list[dict[str, Any]]) -> str | list[str] | None:
    recipients: list[str] = []
    for item in interactions:
        if item.get("actor_id") != "C2":
            continue
        if item.get("relation") not in {"talks_to", "talks_to_group"}:
            continue
        for recipient in item.get("recipient_ids", []):
            if recipient not in recipients:
                recipients.append(recipient)
    if not recipients:
        return None
    return recipients[0] if len(recipients) == 1 else recipients


def character_interacts(character_id: str, interactions: list[dict[str, Any]]) -> bool:
    return any(
        item.get("actor_id") == character_id or character_id in item.get("recipient_ids", [])
        for item in interactions
    )


def pair_interacts(a: str, b: str, interactions: list[dict[str, Any]]) -> bool:
    for item in interactions:
        actor = item.get("actor_id")
        recipients = item.get("recipient_ids", [])
        if (actor == a and b in recipients) or (actor == b and a in recipients):
            return True
    return False


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


def question_type_accuracy(details: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, int]] = {}
    for item in details:
        key = item["metric_key"]
        grouped.setdefault(key, {"correct": 0, "total": 0})
        grouped[key]["correct"] += int(item["correct"])
        grouped[key]["total"] += 1
    return {
        key: {
            "accuracy": round(value["correct"] / value["total"], 6) if value["total"] else 0.0,
            **value,
        }
        for key, value in grouped.items()
    }


def annotation_dirs() -> list[Path]:
    return sorted(path for path in ANNOTATION_ROOT.glob("*_*") if path.is_dir())


def load_probe() -> dict[str, dict[str, Any]]:
    if not PROBE_PATH.exists():
        return {}
    return {row["video"]: row for row in json.loads(PROBE_PATH.read_text(encoding="utf-8"))}


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
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


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

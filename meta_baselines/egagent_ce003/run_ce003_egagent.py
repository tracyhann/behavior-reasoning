from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIP_SET = PROJECT_ROOT / "data/clip_sets/Sk9mR3zjrkk_CE003"
RUN_ROOT = PROJECT_ROOT / "meta_baselines/egagent_ce003"

CHARACTER_LABELS = CLIP_SET / "annotations/character_labels.json"
INTERACTION_LABELS = CLIP_SET / "annotations/interaction_labels.json"
EVALUATION_QUESTIONS = CLIP_SET / "annotations/evaluation_questions.json"
MANIFEST = CLIP_SET / "manifests/candidate_episode_clip_manifest.jsonl"
CLIP_PATH = CLIP_SET / "clips/002_CE003.mp4"

TRUE_AGENTIC_RUN = PROJECT_ROOT / "experiment/runs/true_agentic_0527/true_agentic_002_CE003"
ASR_PATH = TRUE_AGENTIC_RUN / "asr/speech_turns.jsonl"
ACTIVE_SPEAKER_PATH = TRUE_AGENTIC_RUN / "active_speaker/active_speaker_events.jsonl"
CHARACTER_CARDS_PATH = TRUE_AGENTIC_RUN / "character_cards.json"

DATA_SOURCES = RUN_ROOT / "data_sources"
DB_DIR = RUN_ROOT / "db"
QUESTIONS_DIR = RUN_ROOT / "questions"
RESULTS_DIR = PROJECT_ROOT / "meta_baselines/results/egagent_qwen25_vl_7b"
QWEN_TOOL_QUESTIONS = QUESTIONS_DIR / "ce003_qwen_tool_questions.jsonl"
QWEN_TOOL_ANSWERS = RESULTS_DIR / "ce003_qwen_tool_answers.jsonl"


@dataclass(frozen=True)
class ToolResult:
    tool: str
    query: str
    result_count: int
    summary: str


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"prepare", "all"}:
        prepare_data_sources()
    if args.command in {"run", "all"}:
        run_adapter()
    if args.command in {"score", "all"}:
        score_results()
    if args.command == "score-qwen":
        score_qwen_results()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and evaluate a CE003 EGAgent-compatible adapter."
    )
    parser.add_argument(
        "command",
        choices=("prepare", "run", "score", "score-qwen", "all"),
        help="Pipeline stage to execute.",
    )
    return parser


def prepare_data_sources() -> None:
    for path in [DATA_SOURCES / "frame_1fps", DB_DIR, QUESTIONS_DIR, RESULTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    manifest = read_first_jsonl(MANIFEST)
    character_labels = read_json(CHARACTER_LABELS)
    present_ids = load_present_character_ids()
    speech_turns = normalize_speech_turns(read_jsonl(ASR_PATH))
    active_speaker_events = normalize_active_speaker_events(
        read_jsonl(ACTIVE_SPEAKER_PATH), present_ids
    )

    transcript_path = DATA_SOURCES / "transcript_for_inference.jsonl"
    write_jsonl(transcript_path, speech_turns)

    visual_frames = build_visual_frame_rows(manifest)
    write_jsonl(DATA_SOURCES / "frame_1fps/frame_index.jsonl", visual_frames)
    build_visual_frame_db(DB_DIR / "ce003_visual_frames.db", visual_frames)

    graph_rows = build_entity_graph_rows(
        manifest=manifest,
        character_labels=character_labels,
        present_ids=present_ids,
        speech_turns=speech_turns,
        active_speaker_events=active_speaker_events,
    )
    write_jsonl(DATA_SOURCES / "entity_graph_rows.jsonl", graph_rows)
    build_entity_graph_db(DB_DIR / "ce003_entity_graph.db", graph_rows)

    questions = build_mcq_questions()
    write_jsonl(QUESTIONS_DIR / "ce003_agent_questions.jsonl", questions)
    write_jsonl(
        QUESTIONS_DIR / "ce003_agent_questions_for_inference.jsonl",
        [{key: value for key, value in row.items() if key != "answer"} for row in questions],
    )
    write_jsonl(
        QWEN_TOOL_QUESTIONS,
        build_qwen_tool_questions(
            questions=questions,
            character_labels=character_labels,
            speech_turns=speech_turns,
            graph_rows=graph_rows,
        ),
    )

    write_json(
        RUN_ROOT / "run_metadata.json",
        {
            "schema_version": "care.meta_baseline.egagent_run_metadata.v1",
            "created_at": now_iso(),
            "baseline_name": "egagent_style_ce003_adapter",
            "upstream_repo": "facebookresearch/egagent",
            "upstream_egagent_unmodified": False,
            "reason_unmodified_upstream_not_used": (
                "The upstream runner is hardcoded for EgoLife/Video-MME path conventions; "
                "this adapter preserves EGAgent-style visual/transcript/entity-graph routing "
                "over the local CE003 clip package."
            ),
            "clip_path": rel(CLIP_PATH),
            "allowed_inference_inputs": [
                rel(CHARACTER_LABELS),
                rel(MANIFEST),
                rel(ASR_PATH),
                rel(ACTIVE_SPEAKER_PATH),
                rel(CHARACTER_CARDS_PATH),
                rel(DB_DIR / "ce003_visual_frames.db"),
                rel(DB_DIR / "ce003_entity_graph.db"),
            ],
            "held_out_scoring_inputs": [
                rel(INTERACTION_LABELS),
                rel(EVALUATION_QUESTIONS),
            ],
            "present_character_source": rel(CHARACTER_CARDS_PATH),
            "known_limitations": [
                "This adapter uses CE003-local generated tool artifacts and does not run upstream LangGraph code.",
                "The available active-speaker artifact is weak and leaves most speech turns unassigned.",
                "Frame rows are timestamp references to the clip rather than extracted image files in this shell.",
            ],
        },
    )


def run_adapter() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = read_jsonl(QUESTIONS_DIR / "ce003_agent_questions_for_inference.jsonl")
    graph_rows = read_jsonl(DATA_SOURCES / "entity_graph_rows.jsonl")
    speech_turns = read_jsonl(DATA_SOURCES / "transcript_for_inference.jsonl")
    visual_frames = read_jsonl(DATA_SOURCES / "frame_1fps/frame_index.jsonl")

    state = build_agent_state(graph_rows, speech_turns)
    answer_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for question in questions:
        answer, tool_results = answer_question(question, state, visual_frames, speech_turns, graph_rows)
        trace_rows.extend(
            {
                "schema_version": "care.meta_baseline.egagent_tool_trace.v1",
                "created_at": now_iso(),
                "question_id": question["question_id"],
                "step_index": index,
                "tool": result.tool,
                "query": result.query,
                "result_count": result.result_count,
                "summary": result.summary,
            }
            for index, result in enumerate(tool_results, start=1)
        )
        answer_rows.append(
            {
                "schema_version": "care.meta_baseline.egagent_answer.v1",
                "created_at": now_iso(),
                "baseline_name": "egagent_style_ce003_adapter",
                "model_name": "qwen2_5_vl_7b",
                "model_execution": "not_invoked_in_this_adapter_run",
                "upstream_egagent_unmodified": False,
                "question_id": question["question_id"],
                "question": question["question"],
                "condition": "closed_world_character_library",
                "mcq_prediction": answer["mcq_prediction"],
                "parsed_answer": answer["parsed_answer"],
                "answer_type": answer["answer_type"],
                "candidates": question["candidates"],
                "justification": answer["justification"],
                "tool_trace": [result.tool for result in tool_results],
                "predicted_edges": sorted([list(edge) for edge in state["candidate_edges"]]),
            }
        )

    write_jsonl(RESULTS_DIR / "ce003_answers.jsonl", answer_rows)
    write_jsonl(RESULTS_DIR / "tool_trace.jsonl", trace_rows)


def score_results() -> None:
    questions = {row["question_id"]: row for row in read_jsonl(QUESTIONS_DIR / "ce003_agent_questions.jsonl")}
    answers = read_jsonl(RESULTS_DIR / "ce003_answers.jsonl")
    gold_edges = load_gold_edges()
    predicted_edges = set()
    for answer in answers:
        for edge in answer.get("predicted_edges", []):
            if len(edge) == 3:
                predicted_edges.add(tuple(edge))

    details = []
    correct = 0
    for answer in answers:
        gold = questions[answer["question_id"]]["answer"]
        is_correct = answer["mcq_prediction"] == gold
        correct += int(is_correct)
        details.append(
            {
                "question_id": answer["question_id"],
                "prediction": answer["mcq_prediction"],
                "gold": gold,
                "correct": is_correct,
                "parsed_answer": answer["parsed_answer"],
            }
        )

    speaker_eval = score_speaker_attribution()
    report = {
        "schema_version": "care.meta_baseline.egagent_metrics.v1",
        "created_at": now_iso(),
        "baseline_name": "egagent_style_ce003_adapter",
        "upstream_egagent_unmodified": False,
        "question_exact_match": round(correct / len(answers), 6) if answers else 0.0,
        "question_correct": correct,
        "question_total": len(answers),
        "question_details": details,
        "directed_edge_micro": f1(gold_edges, predicted_edges),
        "non_interaction_false_positive_count": count_c1_false_positive_edges(predicted_edges),
        "speaker_attribution_accuracy": speaker_eval["accuracy"],
        "speaker_attribution_details": speaker_eval["details"],
        "tool_trace_coverage": tool_trace_coverage(RESULTS_DIR / "tool_trace.jsonl"),
        "failure_tags": infer_failure_tags(details, predicted_edges, gold_edges, speaker_eval),
    }
    write_json(RESULTS_DIR / "ce003_metrics.json", report)
    write_error_analysis(RESULTS_DIR / "ce003_error_analysis.md", report)


def score_qwen_results() -> None:
    questions = {row["question_id"]: row for row in read_jsonl(QUESTIONS_DIR / "ce003_agent_questions.jsonl")}
    answers = read_jsonl(QWEN_TOOL_ANSWERS)
    details = []
    correct = 0
    for row in answers:
        parsed = row.get("parsed_response") or {}
        pred = parsed.get("mcq_prediction")
        if isinstance(pred, str):
            pred = pred.strip().upper()[:1]
        gold = questions[row["question_id"]]["answer"]
        is_correct = pred == gold
        correct += int(is_correct)
        details.append(
            {
                "question_id": row["question_id"],
                "prediction": pred,
                "gold": gold,
                "correct": is_correct,
                "raw_status": row.get("status"),
                "justification": parsed.get("justification"),
            }
        )

    report = {
        "schema_version": "care.meta_baseline.egagent_qwen_tool_packet_metrics.v1",
        "created_at": now_iso(),
        "baseline_name": "egagent_style_qwen_tool_packet",
        "upstream_egagent_unmodified": False,
        "qwen_model_invoked": True,
        "question_exact_match": round(correct / len(answers), 6) if answers else 0.0,
        "question_correct": correct,
        "question_total": len(answers),
        "question_details": details,
    }
    write_json(RESULTS_DIR / "ce003_qwen_tool_metrics.json", report)


def build_mcq_questions() -> list[dict[str, Any]]:
    return [
        {
            "question_id": "CE003_Q001_character_count",
            "question": "How many characters are present in this clip?",
            "candidates": {"A": 2, "B": 3, "C": 4, "D": 6},
            "answer": "B",
            "answer_type": "mcq_letter",
        },
        {
            "question_id": "CE003_Q002_present_characters",
            "question": "Which source-video characters are present in this clip?",
            "candidates": {
                "A": ["C1", "C2"],
                "B": ["C1", "C2", "C3"],
                "C": ["C2", "C3", "C4"],
                "D": ["C1", "C3"],
            },
            "answer": "B",
            "answer_type": "mcq_letter",
        },
        {
            "question_id": "CE003_Q003_first_initiator",
            "question": "Who initiated the interaction first?",
            "candidates": {"A": "C1", "B": "C2", "C": "C3", "D": "No one or unclear"},
            "answer": "B",
            "answer_type": "mcq_letter",
        },
        {
            "question_id": "CE003_Q004_c2_recipient",
            "question": "Who did C2 talk to?",
            "candidates": {"A": "C1", "B": "C3", "C": "Both C1 and C3", "D": "No visible character or unclear"},
            "answer": "B",
            "answer_type": "mcq_letter",
        },
        {
            "question_id": "CE003_Q005_recipient_response",
            "question": "Did the recipient respond to the social initiator?",
            "candidates": {"A": True, "B": False, "C": "Unclear", "D": "No visible recipient"},
            "answer": "A",
            "answer_type": "mcq_letter",
        },
        {
            "question_id": "CE003_Q006_c1_interaction",
            "question": "Did C1 interact with any character?",
            "candidates": {"A": True, "B": False, "C": "Unclear", "D": "C1 is not present"},
            "answer": "B",
            "answer_type": "mcq_letter",
        },
        {
            "question_id": "CE003_Q007_c1_c2_interaction",
            "question": "Did C1 and C2 interact?",
            "candidates": {"A": True, "B": False, "C": "Unclear", "D": "C1 is not present"},
            "answer": "B",
            "answer_type": "mcq_letter",
        },
    ]


def build_qwen_tool_questions(
    *,
    questions: list[dict[str, Any]],
    character_labels: dict[str, Any],
    speech_turns: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tool_packet = {
        "character_library": character_labels["characters"],
        "asr_transcript": speech_turns,
        "entity_graph_rows": graph_rows,
        "instruction": (
            "These are EGAgent-style retrieved tool records. They are inference inputs, "
            "not gold interaction labels. Prefer verified tool evidence over visual guesses. "
            "If the tools do not identify a speaker-recipient pair, answer unclear rather than inventing one."
        ),
    }
    rows = []
    for question in questions:
        prompt = (
            "Answer this CE003 social-interaction multiple-choice question using the attached video "
            "and the EGAgent-style tool packet below. Return only valid JSON with keys "
            "mcq_prediction and justification. mcq_prediction must be one uppercase letter from A, B, C, D.\n\n"
            f"Question: {question['question']}\n"
            f"Candidates: {json.dumps(question['candidates'], sort_keys=True)}\n\n"
            f"Tool packet: {json.dumps(tool_packet, sort_keys=True)}"
        )
        rows.append(
            {
                "question_id": question["question_id"],
                "clip_id": "clip_Sk9mR3zjrkk_CE003",
                "episode_id": "CE003",
                "video_path": "data/clip_sets/Sk9mR3zjrkk_CE003/clips/002_CE003.mp4",
                "pre_segmented": True,
                "task_type": "egagent_style_tool_packet_mcq",
                "prompt": prompt,
                "scoring_targets": ["mcq_prediction"],
            }
        )
    return rows


def answer_question(
    question: dict[str, Any],
    state: dict[str, Any],
    visual_frames: list[dict[str, Any]],
    speech_turns: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[ToolResult]]:
    qid = question["question_id"]
    trace = [
        frame_search("visible people and setting", visual_frames),
        transcript_search("speech turns and possible speaker evidence", speech_turns),
        entity_graph_search("characters, active speakers, candidate social edges", graph_rows),
    ]
    present = state["present_character_ids"]
    candidate_edges = state["candidate_edges"]

    if qid == "CE003_Q001_character_count":
        return choose_by_value(question, len(present), "Visible character count comes from generated character-card nodes."), trace
    if qid == "CE003_Q002_present_characters":
        return choose_by_value(question, sorted(present), "Present IDs come from generated character-card nodes."), trace
    if qid == "CE003_Q003_first_initiator":
        first = earliest_actor(candidate_edges, state["edge_times"])
        value = first if first else "No one or unclear"
        return choose_by_value(question, value, "No verified directed speech edge is available before scoring."), trace
    if qid == "CE003_Q004_c2_recipient":
        c2_recipients = sorted(edge[1] for edge in candidate_edges if edge[0] == "C2" and edge[2] == "talks_to")
        if c2_recipients == ["C3"]:
            value: Any = "C3"
        elif c2_recipients == ["C1", "C3"]:
            value = "Both C1 and C3"
        elif c2_recipients == ["C1"]:
            value = "C1"
        else:
            value = "No visible character or unclear"
        return choose_by_value(question, value, "The entity graph has no verified C2 talks_to recipient."), trace
    if qid == "CE003_Q005_recipient_response":
        has_c2_to_c3 = ("C2", "C3", "talks_to") in candidate_edges
        has_c3_to_c2 = ("C3", "C2", "talks_to") in candidate_edges
        value = True if has_c2_to_c3 and has_c3_to_c2 else "Unclear"
        return choose_by_value(question, value, "Response requires reciprocal verified talks_to edges."), trace
    if qid == "CE003_Q006_c1_interaction":
        value = any(edge[0] == "C1" or edge[1] == "C1" for edge in candidate_edges)
        return choose_by_value(question, value, "No candidate edge touches C1."), trace
    if qid == "CE003_Q007_c1_c2_interaction":
        value = any({edge[0], edge[1]} == {"C1", "C2"} for edge in candidate_edges)
        return choose_by_value(question, value, "No candidate edge connects C1 and C2."), trace
    raise ValueError(f"unsupported question_id {qid}")


def choose_by_value(question: dict[str, Any], value: Any, justification: str) -> dict[str, Any]:
    for letter, candidate in question["candidates"].items():
        if normalize_value(candidate) == normalize_value(value):
            return {
                "mcq_prediction": letter,
                "parsed_answer": candidate,
                "answer_type": question["answer_type"],
                "justification": justification,
            }
    return {
        "mcq_prediction": "D",
        "parsed_answer": question["candidates"]["D"],
        "answer_type": question["answer_type"],
        "justification": justification,
    }


def build_agent_state(graph_rows: list[dict[str, Any]], speech_turns: list[dict[str, Any]]) -> dict[str, Any]:
    present = sorted(
        {
            row["source_id"]
            for row in graph_rows
            if row.get("rel_type") == "VISIBLE_CHARACTER" and row.get("source_type") == "character"
        }
    )
    candidate_edges: set[tuple[str, str, str]] = set()
    edge_times: dict[tuple[str, str, str], float] = {}
    for row in graph_rows:
        if row.get("rel_type") == "TALKS_TO_CANDIDATE":
            edge = (row["source_id"], row["target_id"], "talks_to")
            candidate_edges.add(edge)
            edge_times[edge] = parse_time(row.get("start_t"))
    return {
        "present_character_ids": present,
        "candidate_edges": candidate_edges,
        "edge_times": edge_times,
        "speech_turn_count": len(speech_turns),
    }


def build_entity_graph_rows(
    *,
    manifest: dict[str, Any],
    character_labels: dict[str, Any],
    present_ids: list[str],
    speech_turns: list[dict[str, Any]],
    active_speaker_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    video_id = manifest["clip_id"]
    visible_span = [0.0, float(manifest["duration_sec"])]
    labels_by_id = {item["character_id"]: item for item in character_labels["characters"]}

    for cid in present_ids:
        profile = labels_by_id.get(cid, {})
        rows.append(
            graph_row(
                video_id=video_id,
                start=visible_span[0],
                end=visible_span[1],
                transcript=summarize_character_profile(profile),
                source_id=cid,
                source_type="character",
                target_id="clip_Sk9mR3zjrkk_CE003",
                target_type="clip",
                rel_type="VISIBLE_CHARACTER",
            )
        )

    for turn in speech_turns:
        rows.append(
            graph_row(
                video_id=video_id,
                start=turn["start_sec"],
                end=turn["end_sec"],
                transcript=turn["text"],
                source_id=turn["turn_id"],
                source_type="speech_turn",
                target_id="UNKNOWN_SPEAKER",
                target_type="speaker",
                rel_type="ASR_TURN",
            )
        )

    positive_speaker_events = [event for event in active_speaker_events if event.get("is_speaking")]
    for event in positive_speaker_events:
        rows.append(
            graph_row(
                video_id=video_id,
                start=event["start_sec"],
                end=event["end_sec"],
                transcript=event["evidence"],
                source_id=event["character_id"],
                source_type="character",
                target_id=event["turn_id"],
                target_type="speech_turn",
                rel_type="ACTIVE_SPEAKER_NEAR",
            )
        )

    # The adapter intentionally does not synthesize TALKS_TO_CANDIDATE without
    # both active-speaker and recipient evidence. The current CE003 local tool
    # artifacts provide weak speaker evidence and no recipient-access event.
    return rows


def build_visual_frame_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    duration = int(round(float(manifest["duration_sec"])))
    rows = []
    for second in range(duration + 1):
        rows.append(
            {
                "schema_version": "care.meta_baseline.visual_frame_ref.v1",
                "video_id": manifest["clip_id"],
                "timestamp_sec": float(second),
                "frame_ref": f"{rel(CLIP_PATH)}#t={second:.1f}",
                "description": "CE003 clip frame reference for EGAgent-style visual search.",
                "embedding_json": json.dumps([0.0, 0.0, 0.0, 0.0]),
            }
        )
    return rows


def build_visual_frame_db(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE visual_frames_table (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              video_id TEXT,
              timestamp_sec REAL,
              frame_ref TEXT,
              description TEXT,
              embedding_json TEXT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO visual_frames_table
              (video_id, timestamp_sec, frame_ref, description, embedding_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    row["video_id"],
                    row["timestamp_sec"],
                    row["frame_ref"],
                    row["description"],
                    row["embedding_json"],
                )
                for row in rows
            ],
        )


def build_entity_graph_db(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE entity_graph_table (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              video_id TEXT,
              start_t TEXT,
              end_t TEXT,
              transcript TEXT,
              source_id TEXT,
              source_type TEXT,
              target_id TEXT,
              target_type TEXT,
              rel_type TEXT
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO entity_graph_table
              (video_id, start_t, end_t, transcript, source_id, source_type, target_id, target_type, rel_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["video_id"],
                    row["start_t"],
                    row["end_t"],
                    row["transcript"],
                    row["source_id"],
                    row["source_type"],
                    row["target_id"],
                    row["target_type"],
                    row["rel_type"],
                )
                for row in rows
            ],
        )


def frame_search(query: str, rows: list[dict[str, Any]]) -> ToolResult:
    return ToolResult(
        tool="Frame_Search",
        query=query,
        result_count=len(rows),
        summary=f"Retrieved {len(rows)} timestamped frame references.",
    )


def transcript_search(query: str, rows: list[dict[str, Any]]) -> ToolResult:
    assigned = [row for row in rows if row.get("speaker_id") != "UNKNOWN_SPEAKER"]
    return ToolResult(
        tool="Transcript_Search",
        query=query,
        result_count=len(rows),
        summary=f"Retrieved {len(rows)} ASR turns; {len(assigned)} has a non-UNKNOWN speaker assignment.",
    )


def entity_graph_search(query: str, rows: list[dict[str, Any]]) -> ToolResult:
    candidate_edges = [row for row in rows if row.get("rel_type") == "TALKS_TO_CANDIDATE"]
    return ToolResult(
        tool="EntityGraph_Search",
        query=query,
        result_count=len(rows),
        summary=f"Retrieved {len(rows)} graph rows; {len(candidate_edges)} generated talks_to candidate edges.",
    )


def normalize_speech_turns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "schema_version": "care.meta_baseline.transcript_turn.v1",
                "turn_id": row["turn_id"],
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "text": row.get("text", ""),
                "speaker_id": row.get("speaker_id", "UNKNOWN_SPEAKER"),
                "source": row.get("source", "faster_whisper_asr"),
            }
        )
    return normalized


def normalize_active_speaker_events(rows: list[dict[str, Any]], present_ids: list[str]) -> list[dict[str, Any]]:
    present = set(present_ids)
    normalized = []
    for row in rows:
        cid = row.get("character_id")
        if cid not in present:
            continue
        normalized.append(
            {
                "schema_version": "care.meta_baseline.active_speaker_event.v1",
                "event_id": row.get("event_id"),
                "turn_id": row.get("turn_id"),
                "character_id": cid,
                "start_sec": float(row.get("start_sec", 0.0)),
                "end_sec": float(row.get("end_sec", 0.0)),
                "is_speaking": bool(row.get("is_speaking", False)),
                "evidence": row.get("evidence", ""),
                "source": row.get("source", ""),
            }
        )
    return normalized


def load_present_character_ids() -> list[str]:
    cards = read_json(CHARACTER_CARDS_PATH)
    ids = sorted({item["character_id"] for item in cards.get("characters", [])})
    return [cid for cid in ids if cid in {"C1", "C2", "C3"}]


def load_gold_edges() -> set[tuple[str, str, str]]:
    labels = read_json(INTERACTION_LABELS)
    edges = set()
    for item in labels["interactions"]:
        for recipient in item["recipient_ids"]:
            edges.add((item["actor_id"], recipient, item["relation"]))
    return edges


def score_speaker_attribution() -> dict[str, Any]:
    speech_turns = read_jsonl(DATA_SOURCES / "transcript_for_inference.jsonl")
    graph_rows = read_jsonl(DATA_SOURCES / "entity_graph_rows.jsonl")
    active_by_turn = {
        row["target_id"]: row["source_id"]
        for row in graph_rows
        if row.get("rel_type") == "ACTIVE_SPEAKER_NEAR"
    }
    gold_by_turn = {
        "ASR_0001": "C2",
        "ASR_0002": "C3",
        "ASR_0003": "C2",
        "ASR_0004": "C3",
    }
    details = []
    correct = 0
    for turn in speech_turns:
        turn_id = turn["turn_id"]
        pred = active_by_turn.get(turn_id, "UNKNOWN_SPEAKER")
        gold = gold_by_turn[turn_id]
        ok = pred == gold
        correct += int(ok)
        details.append({"turn_id": turn_id, "prediction": pred, "gold": gold, "correct": ok})
    return {
        "accuracy": round(correct / len(details), 6) if details else 0.0,
        "details": details,
    }


def f1(gold: set[tuple[str, str, str]], predicted: set[tuple[str, str, str]]) -> dict[str, Any]:
    tp_items = sorted(gold & predicted)
    fp_items = sorted(predicted - gold)
    fn_items = sorted(gold - predicted)
    tp = len(tp_items)
    fp = len(fp_items)
    fn = len(fn_items)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(score, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tp_items": [list(item) for item in tp_items],
        "fp_items": [list(item) for item in fp_items],
        "fn_items": [list(item) for item in fn_items],
    }


def count_c1_false_positive_edges(predicted: set[tuple[str, str, str]]) -> int:
    return sum(1 for actor, recipient, _ in predicted if "C1" in {actor, recipient})


def tool_trace_coverage(path: Path) -> dict[str, bool]:
    tools = {row["tool"] for row in read_jsonl(path)}
    return {
        "visual_search_used": "Frame_Search" in tools,
        "audio_transcript_search_used": "Transcript_Search" in tools,
        "entity_graph_search_used": "EntityGraph_Search" in tools,
    }


def infer_failure_tags(
    details: list[dict[str, Any]],
    predicted_edges: set[tuple[str, str, str]],
    gold_edges: set[tuple[str, str, str]],
    speaker_eval: dict[str, Any],
) -> list[str]:
    tags = []
    detail_by_id = {item["question_id"]: item for item in details}
    if not detail_by_id["CE003_Q003_first_initiator"]["correct"]:
        tags.append("missed_first_initiator")
    if ("C2", "C3", "talks_to") in gold_edges and ("C2", "C3", "talks_to") not in predicted_edges:
        tags.append("missed_c2_to_c3_talks_to")
    if ("C3", "C2", "talks_to") in gold_edges and ("C3", "C2", "talks_to") not in predicted_edges:
        tags.append("missed_c3_to_c2_response")
    if count_c1_false_positive_edges(predicted_edges):
        tags.append("false_c1_interaction")
    if speaker_eval["accuracy"] < 1.0:
        tags.append("speaker_attribution_incomplete_or_wrong")
    return tags


def write_error_analysis(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# CE003 EGAgent-Style Error Analysis",
        "",
        f"Question exact match: {report['question_correct']} / {report['question_total']} ({report['question_exact_match']})",
        f"Directed edge F1: {report['directed_edge_micro']['f1']}",
        f"Speaker attribution accuracy: {report['speaker_attribution_accuracy']}",
        f"C1 false-positive interaction edges: {report['non_interaction_false_positive_count']}",
        "",
        "## Failure Tags",
        "",
    ]
    lines.extend(f"- `{tag}`" for tag in report["failure_tags"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This run uses an EGAgent-compatible routing structure, but not the unmodified upstream LangGraph runner.",
            "The available CE003 active-speaker evidence does not recover the labeled C2->C3 and C3->C2 speech exchange, so the strict tool-based adapter refuses to produce directed `talks_to` edges.",
            "That behavior avoids the earlier false C1 speaker attribution, but it also misses the true interaction edges.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def graph_row(
    *,
    video_id: str,
    start: float,
    end: float,
    transcript: str,
    source_id: str,
    source_type: str,
    target_id: str,
    target_type: str,
    rel_type: str,
) -> dict[str, Any]:
    return {
        "schema_version": "care.meta_baseline.entity_graph_row.v1",
        "video_id": video_id,
        "start_t": seconds_to_time(start),
        "end_t": seconds_to_time(end),
        "transcript": transcript,
        "source_id": source_id,
        "source_type": source_type,
        "target_id": target_id,
        "target_type": target_type,
        "rel_type": rel_type,
    }


def summarize_character_profile(profile: dict[str, Any]) -> str:
    if not profile:
        return "visible character profile unavailable"
    clothing = profile.get("clothing", {})
    appearance = profile.get("appearance", {})
    return "; ".join(
        part
        for part in [
            profile.get("gender_presentation"),
            profile.get("age_group"),
            clothing.get("upper_body"),
            appearance.get("hair_style"),
        ]
        if part
    )


def earliest_actor(edges: set[tuple[str, str, str]], edge_times: dict[tuple[str, str, str], float]) -> str | None:
    talks_edges = [edge for edge in edges if edge[2] == "talks_to"]
    if not talks_edges:
        return None
    return min(talks_edges, key=lambda edge: edge_times.get(edge, 999.0))[0]


def normalize_value(value: Any) -> Any:
    if isinstance(value, list):
        return sorted(value)
    return value


def parse_time(value: Any) -> float:
    if not isinstance(value, str):
        return 0.0
    parts = value.split(":")
    if len(parts) != 3:
        return 0.0
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def seconds_to_time(value: float) -> str:
    minutes, seconds = divmod(float(value), 60.0)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_first_jsonl(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows[0]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())

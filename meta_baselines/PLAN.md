# Meta Baselines CE003 Testing Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test Meta's `facebookresearch/egotom` and `facebookresearch/egagent` on the CE003 edge case where a visually salient non-speaker can be confused with the actual speaker-recipient pair.

**Architecture:** Use `data/clip_sets/Sk9mR3zjrkk_CE003` as the fixed evaluation package. EgoToM is evaluated as a multiple-choice VLM benchmark adapter because the upstream project is a benchmark plus batch VLM evaluation code, not an interaction-grounding agent. EGAgent is evaluated as the tool-querying agent baseline because the upstream project defines a LangGraph planning agent that routes sub-tasks to visual-frame search, transcript search, and entity-graph search.

**Tech Stack:** Python 3.10; JSON/JSONL/CSV artifacts; SQLite for EGAgent-compatible visual and entity-graph databases; SigLIP 2 or local embedding fallback for visual-frame retrieval; local Qwen2.5-VL-7B through the existing local runner or EGAgent's vLLM path; faster-whisper transcript artifact; CE003 character library; CE003 deterministic evaluation questions; optional upstream clones of `facebookresearch/egotom` and `facebookresearch/egagent`.

---

## Upstream References

- EgoToM repository: <https://github.com/facebookresearch/egotom>
- EgoToM role in this plan: benchmark/prompt/evaluation adapter. Its README describes multiple-choice questions for goals, beliefs, and future actions over egocentric videos, plus batch VLM evaluation code in `code/vlm_evaluate.py`.
- EgoToM compatibility constraint: the native runner reads an `all_prompts.json` file, loops over configured question keys and conditions, and expects videos named `{uid}_context.mp4` inside each condition's `video_input_dir`. Its native model loader does not include Qwen, so Qwen evaluation requires our local Qwen runner or a small EgoToM runner adapter.
- EGAgent repository: <https://github.com/facebookresearch/egagent>
- EGAgent role in this plan: agentic baseline. Its README describes a planning agent that uses visual search, audio transcript search, and entity graph search for very long video understanding.
- EGAgent compatibility constraint: the native data-prep path samples frames at 1 FPS, fuses raw captions with diarized transcripts, builds entity-graph JSONs, and then builds SQLite databases for visual frames and the entity graph. Audio transcript search is performed from transcript input at inference time, not from a separate transcript database.

## Fixed CE003 Data

Use these files as the only CE003 ground-truth package:

```text
data/clip_sets/Sk9mR3zjrkk_CE003/clips/002_CE003.mp4
data/clip_sets/Sk9mR3zjrkk_CE003/annotations/character_labels.json
data/clip_sets/Sk9mR3zjrkk_CE003/annotations/interaction_labels.json
data/clip_sets/Sk9mR3zjrkk_CE003/annotations/evaluation_questions.json
data/clip_sets/Sk9mR3zjrkk_CE003/manifests/candidate_episode_clip_manifest.jsonl
```

Gold facts for scoring:

```json
{
  "present_characters": ["C1", "C2", "C3"],
  "interaction_edges": [
    {"actor_id": "C2", "recipient_id": "C3", "relation": "talks_to"},
    {"actor_id": "C3", "recipient_id": "C2", "relation": "talks_to"}
  ],
  "first_initiator": "C2",
  "non_interacting_visible_characters": ["C1"],
  "known_edge_absences": [
    {"actor_id": "C1", "recipient_id": "C2"},
    {"actor_id": "C2", "recipient_id": "C1"},
    {"actor_id": "C1", "recipient_id": "C3"},
    {"actor_id": "C3", "recipient_id": "C1"}
  ]
}
```

## Experimental Conditions

Run every method under two visibility conditions so we can separate video reasoning from identity-library reasoning.

### Condition A: Video-Only Open World

Inputs:

```text
clips/002_CE003.mp4
```

Rules:

- The model must invent temporary IDs in first-clear-visibility order.
- The model receives no `C1`-`C6` library.
- This condition tests raw perception and interaction reasoning.
- Scoring maps temporary model IDs to gold IDs with a post-hoc visual-alignment rubric before edge scoring.

Expected weakness:

- Character identity questions using canonical `C*` IDs are not directly fair under this condition, so score only count, pair structure, initiator role, and non-speaker rejection.

### Condition B: Closed-World Character Library

Inputs:

```text
clips/002_CE003.mp4
annotations/character_labels.json
```

Rules:

- The model receives the full source-video character library `C1`-`C6`.
- The model does not receive `interaction_labels.json` or `evaluation_questions.json` answers.
- The model must identify which library characters are present in CE003 before answering interaction questions.
- This condition tests the intended pipeline problem: character-library grounding plus social-interaction reasoning.

Expected weakness:

- Methods may overuse appearance profiles as priors and ignore active-speaker evidence. The error to catch is claiming `C1` speaks because C1 is visually salient.

## EgoToM-Style Baseline Plan

EgoToM should be adapted as a prompt/evaluation baseline, not treated as a tool-using interaction agent.

### EgoToM Input Conversion

Create:

```text
meta_baselines/egotom_ce003/questions/ce003_egotom_style.csv
meta_baselines/egotom_ce003/prompts/ce003_all_prompts.json
meta_baselines/egotom_ce003/configs/qwen25_vl_7b_ce003.yaml
meta_baselines/egotom_ce003/video_inputs/closed_world/CE003_Q001_context.mp4
meta_baselines/egotom_ce003/video_inputs/closed_world/CE003_Q002_context.mp4
meta_baselines/egotom_ce003/video_inputs/closed_world/CE003_Q003_context.mp4
meta_baselines/egotom_ce003/video_inputs/closed_world/CE003_Q004_context.mp4
meta_baselines/egotom_ce003/video_inputs/closed_world/CE003_Q005_context.mp4
meta_baselines/egotom_ce003/video_inputs/closed_world/CE003_Q006_context.mp4
meta_baselines/egotom_ce003/video_inputs/closed_world/CE003_Q007_context.mp4
```

The CSV is an audit artifact. The executable prompt artifact is `ce003_all_prompts.json`, because EgoToM's native evaluator reads prompts from JSON rather than from the CSV at inference time.

Required rows:

```csv
question_id,question_type,video_path,question,choice_a,choice_b,choice_c,choice_d,answer
CE003_Q001_character_count,interaction_count,data/clip_sets/Sk9mR3zjrkk_CE003/clips/002_CE003.mp4,How many characters are present in this clip?,2,3,4,6,b
CE003_Q002_present_characters,identity_set,data/clip_sets/Sk9mR3zjrkk_CE003/clips/002_CE003.mp4,Which source-video characters are present in this clip?,"C1,C2","C1,C2,C3","C2,C3,C4","C1,C3",b
CE003_Q003_first_initiator,interaction_initiator,data/clip_sets/Sk9mR3zjrkk_CE003/clips/002_CE003.mp4,Who initiated the interaction first?,C1,C2,C3,No one,b
CE003_Q004_c2_recipient,recipient_identification,data/clip_sets/Sk9mR3zjrkk_CE003/clips/002_CE003.mp4,Who did C2 talk to?,C1,C3,Both C1 and C3,No visible character,b
CE003_Q005_recipient_response,response_detection,data/clip_sets/Sk9mR3zjrkk_CE003/clips/002_CE003.mp4,Did the recipient respond to the social initiator?,Yes,No,Unclear,No visible recipient,a
CE003_Q006_c1_interaction,non_interaction_rejection,data/clip_sets/Sk9mR3zjrkk_CE003/clips/002_CE003.mp4,Did C1 interact with any character?,Yes,No,Unclear,C1 is not present,b
CE003_Q007_c1_c2_interaction,false_edge_rejection,data/clip_sets/Sk9mR3zjrkk_CE003/clips/002_CE003.mp4,Did C1 and C2 interact?,Yes,No,Unclear,C1 is not present,b
```

The prompt JSON must follow the upstream shape:

```json
{
  "care_social_interaction": {
    "CE003_Q001": "Question text with A/B/C/D choices.",
    "CE003_Q002": "Question text with A/B/C/D choices."
  }
}
```

The EgoToM-style config must set:

```yaml
questions:
  - care_social_interaction
conditions:
  - closed_world
video_input_dir:
  closed_world: meta_baselines/egotom_ce003/video_inputs/closed_world
```

Prompt rules:

- Do not reveal the setting type or interaction labels.
- In Condition A, do not reveal `C1`, `C2`, `C3`; ask using temporary IDs and score after ID alignment. This condition cannot run through EgoToM's native `{uid}_context.mp4` path without a second prompt JSON and video-input folder.
- In Condition B, include only `character_labels.json` profiles and ask for canonical IDs.
- Require one option letter plus a one-sentence evidence note for audit.
- If using upstream `code/vlm_evaluate.py` unchanged, use a model supported by its loader. If using Qwen2.5-VL-7B, call our local Qwen runner over `ce003_all_prompts.json` or add a Qwen loader adapter; do not claim this is an unmodified EgoToM runner.

### EgoToM Metrics

Compute:

```json
{
  "multiple_choice_accuracy": 0.0,
  "identity_set_exact_match": 0,
  "first_initiator_exact_match": 0,
  "recipient_exact_match": 0,
  "c1_non_interaction_specificity": 0,
  "false_edge_rejection_accuracy": 0.0
}
```

Do not score free-text rationales as correctness. Store them for error analysis only.

## EGAgent-Style Baseline Plan

EGAgent should be adapted as the tool-querying agent baseline.

### EGAgent Data Source Adapter

Create:

```text
meta_baselines/egagent_ce003/data_sources/frame_1fps/
meta_baselines/egagent_ce003/data_sources/transcript_for_inference.jsonl
meta_baselines/egagent_ce003/data_sources/entity_graph_rows.jsonl
meta_baselines/egagent_ce003/db/ce003_visual_frames.db
meta_baselines/egagent_ce003/db/ce003_entity_graph.db
meta_baselines/egagent_ce003/questions/ce003_agent_questions.jsonl
meta_baselines/egagent_ce003/configs/qwen25_vl_7b_ce003.json
meta_baselines/egagent_ce003/run_ce003_egagent.py
```

Visual source:

- Extract frames from `002_CE003.mp4` at 1 fps to match EGAgent's native data-prep convention.
- Optionally add speaker-turn boundary frames as an extra ablation, but report that separately from the EGAgent-compatible run.
- Embed frames with SigLIP 2 if available, matching the upstream default. If SigLIP 2 is unavailable, use a documented local embedding fallback and label the run `egagent_style_local_embedding`.
- Store frame path, timestamp, video ID, and embedding vector in `ce003_visual_frames.db`.
- Do not store gold interaction labels in the visual database.

Transcript source:

- Run faster-whisper or reuse an existing ASR artifact only if it was produced without gold labels.
- Store turn ID, start/end seconds, text, and `speaker_id: UNKNOWN_SPEAKER`.
- Speaker identity must come from active-speaker or agent reasoning, not ASR alone.
- Keep this as inference input rather than a separate transcript SQLite database, matching EGAgent's source behavior that transcript search is performed at inference time.

Entity graph source:

- Build a generated graph from non-gold evidence:
  - nodes: detected tracklets and matched `C*` IDs under Condition B;
  - attributes: appearance match fields, visible time spans;
  - observations: active-speaker events, gaze/head-pose events, pose/gesture events, co-presence;
  - forbidden fields: `interaction_labels.json`, answer keys, `labeled_interaction`, `first_initiator`.
- Convert generated graph rows into the EGAgent-compatible SQLite table shape, because EGAgent's native `langgraph_agent.py` writes SQL over `entity_graph_table`.

Required JSONL staging schema:

```json
{
  "schema_version": "care.meta_baseline.entity_graph.v1",
  "clip_id": "clip_Sk9mR3zjrkk_CE003",
  "nodes": [
    {
      "node_id": "C2",
      "node_type": "character",
      "visible_time_spans_sec": [[0.0, 8.2]],
      "source": "character_library_match"
    }
  ],
  "observations": [
    {
      "observation_id": "OBS001",
      "subject_id": "C2",
      "predicate": "active_speaker",
      "object_id": null,
      "time_span_sec": [0.0, 2.6],
      "source": "talknet_asd"
    }
  ]
}
```

Required SQLite table for EGAgent-compatible inference:

```sql
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
);
```

Allowed generated `rel_type` values for CE003:

```text
VISIBLE_WITH
ACTIVE_SPEAKER_NEAR
LOOKS_TOWARD
BODY_ORIENTS_TOWARD
GESTURES_TOWARD
TALKS_TO_CANDIDATE
```

`TALKS_TO_CANDIDATE` is allowed only when it is generated from ASR plus active-speaker plus visual recipient evidence. It must not be copied from `interaction_labels.json`.

### EGAgent Runner Adapter

Do not assume the upstream `egagent/run_egagent_on_egolife.py` or `egagent/run_egagent_on_videomme.py` can run on `Sk9mR3zjrkk_CE003` without adaptation. The upstream `langgraph_agent.py` uses repository-level globals for `dataset`, `dataset_root`, `frames_dir`, `asr_dir`, and database paths. For CE003, create `run_ce003_egagent.py` that preserves the same agent loop and tool semantics while binding those paths to the CE003 databases.

Required adapter behavior:

- Set `selected_video` to `clip_Sk9mR3zjrkk_CE003`.
- Set search interval to clip-relative `00:00:00` through the actual clip duration.
- Route planner steps to the same three tool categories: visual frame search, transcript search, and entity graph search.
- Use the same final-answer contract as upstream EGAgent: choose one MCQ letter and provide a justification.
- Preserve a tool trace for every routed step.
- If the local Qwen2.5-VL-7B backend is used, run it through the upstream-compatible vLLM endpoint or through a documented local wrapper with the same structured-output fields.

### EGAgent Questions

Use the deterministic CE003 evaluation questions as the agent task interface.

Native EGAgent final answers are multiple-choice. Use MCQ candidates for the executable EGAgent-compatible run and parse `mcq_prediction`.

Required MCQ task row:

```json
{
  "question_id": "CE003_Q004_c2_recipient",
  "question": "Who did C2 talk to?",
  "candidates": {"A": "C1", "B": "C3", "C": "Both C1 and C3", "D": "No visible character"},
  "answer": "B",
  "answer_type": "mcq_letter"
}
```

Required parsed audit row:

```json
{
  "question_id": "CE003_Q004_c2_recipient",
  "mcq_prediction": "B",
  "parsed_answer": "C3",
  "tool_trace": ["EntityGraph_Search", "Transcript_Search", "Frame_Search"],
  "justification": "Short model justification retained for audit, not used as correctness."
}
```

### EGAgent Metrics

Compute the same deterministic question metrics as EgoToM, plus graph metrics:

```json
{
  "question_exact_match": 0.0,
  "directed_edge_precision": 0.0,
  "directed_edge_recall": 0.0,
  "directed_edge_f1": 0.0,
  "non_interaction_false_positive_count": 0,
  "speaker_attribution_accuracy": 0.0,
  "tool_trace_coverage": {
    "visual_search_used": false,
    "audio_transcript_search_used": false,
    "entity_graph_search_used": false
  }
}
```

## Fairness Controls

- Same video file for all runs: `data/clip_sets/Sk9mR3zjrkk_CE003/clips/002_CE003.mp4`.
- Same VLM backbone where possible: local Qwen2.5-VL-7B.
- Same frame budget for video-only prompting.
- Same ASR transcript for methods that receive transcript tools.
- No method may read `interaction_labels.json` or `evaluation_questions.json` answers during inference.
- `character_labels.json` is allowed only in the closed-world condition and must include all `C1`-`C6` profiles, not only CE003-visible characters.
- All parsed answers must be deterministic: integers, booleans, canonical `C*` IDs, option letters, or sorted ID lists.

## Error Taxonomy

Record these failure tags per run:

```json
[
  "character_count_overestimate",
  "character_count_underestimate",
  "character_id_swap",
  "visually_salient_non_speaker_misattributed_as_speaker",
  "arm_raise_misattributed_as_speech",
  "recipient_confusion",
  "false_c1_c2_interaction",
  "missed_c2_to_c3_talks_to",
  "missed_c3_to_c2_response",
  "unsupported_setting_or_role_inference"
]
```

The most important CE003 failure is:

```json
{
  "failure_tag": "visually_salient_non_speaker_misattributed_as_speaker",
  "description": "The method treats C1 as the speaker or interaction initiator because C1 is visually salient, even though the labeled interaction begins with C2 talking to C3."
}
```

## Implementation Tasks

### Task 1: Create Shared CE003 Metadata

**Files:**

- Create: `meta_baselines/shared/ce003_metadata.json`

- [ ] Write metadata with paths, gold deterministic answers, and scoring keys.
- [ ] Validate JSON with `python -m json.tool meta_baselines/shared/ce003_metadata.json`.

### Task 2: Build EgoToM-Style Question Adapter

**Files:**

- Create: `meta_baselines/egotom_ce003/questions/ce003_egotom_style.csv`
- Create: `meta_baselines/egotom_ce003/prompts/ce003_all_prompts.json`
- Create: `meta_baselines/egotom_ce003/README.md`

- [ ] Convert the seven CE003 deterministic questions into multiple-choice rows.
- [ ] Generate Condition A prompts without character-library leakage.
- [ ] Generate Condition B prompts with full `C1`-`C6` profile access and no interaction-label leakage.
- [ ] Run the local Qwen baseline through the existing video runner or an EgoToM-compatible wrapper.

### Task 3: Build EGAgent-Style Data Sources

**Files:**

- Create: `meta_baselines/egagent_ce003/data_sources/transcript_for_inference.jsonl`
- Create: `meta_baselines/egagent_ce003/data_sources/entity_graph_rows.jsonl`
- Create directory: `meta_baselines/egagent_ce003/data_sources/frame_1fps/`
- Create: `meta_baselines/egagent_ce003/db/ce003_visual_frames.db`
- Create: `meta_baselines/egagent_ce003/db/ce003_entity_graph.db`
- Create: `meta_baselines/egagent_ce003/run_ce003_egagent.py`
- Create: `meta_baselines/egagent_ce003/README.md`

- [ ] Extract CE003 frames at 1 fps into `frame_1fps/`.
- [ ] Generate or copy non-gold ASR turns into `transcript_for_inference.jsonl`.
- [ ] Generate `entity_graph_rows.jsonl` from detector, active-speaker, pose, gaze/head-pose, and character-assignment outputs.
- [ ] Build `ce003_entity_graph.db` with table `entity_graph_table`.
- [ ] Build `ce003_visual_frames.db` with frame paths, timestamps, and embeddings.
- [ ] Confirm neither database contains gold interaction labels.
- [ ] Implement `run_ce003_egagent.py` as a CE003 path adapter around EGAgent's planner-router-retriever-answer workflow.

### Task 4: Run Meta Baseline Inference

**Files:**

- Create: `meta_baselines/results/egotom_qwen25_vl_7b/ce003_answers.jsonl`
- Create: `meta_baselines/results/egagent_qwen25_vl_7b/ce003_answers.jsonl`
- Create: `meta_baselines/results/egagent_qwen25_vl_7b/tool_trace.jsonl`

- [ ] Run EgoToM-style prompts under Condition A and Condition B.
- [ ] Run EGAgent-style agent under Condition A and Condition B.
- [ ] Save raw model output and parsed deterministic answers.
- [ ] Save tool traces for EGAgent-style runs.

### Task 5: Score and Compare

**Files:**

- Create: `meta_baselines/results/ce003_meta_baseline_metrics.json`
- Create: `meta_baselines/results/ce003_meta_baseline_error_analysis.md`

- [ ] Compute exact-match metrics from `evaluation_questions.json`.
- [ ] Compute directed edge precision, recall, and F1 against `interaction_labels.json`.
- [ ] Compute C1 non-interaction specificity.
- [ ] Tag failures using the error taxonomy above.
- [ ] Compare against the existing local Qwen baseline and the current CARE agentic pipeline.

## Acceptance Criteria

- EgoToM-style and EGAgent-style runs both produce parsed deterministic answers for all seven CE003 questions.
- The inference inputs are auditable and show no gold interaction leakage.
- Metrics include exact-match question accuracy, directed edge F1, false-positive count for C1 interactions, and speaker attribution accuracy.
- Error analysis explicitly states whether the method incorrectly attributes speech or interaction initiation to C1.
- The final comparison separates video-only performance from closed-world character-library performance.

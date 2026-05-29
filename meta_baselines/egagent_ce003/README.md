# EGAgent CE003 Adapter

This folder contains an EGAgent-compatible test adapter for:

```text
data/clip_sets/Sk9mR3zjrkk_CE003/clips/002_CE003.mp4
```

It is not an unmodified upstream `facebookresearch/egagent` run. The upstream runner is wired for EgoLife and Video-MME path conventions. This adapter keeps the relevant EGAgent contract for CE003:

- visual-frame search
- transcript search
- entity-graph search
- MCQ final answers
- tool trace logging

## Run

```bash
python3 meta_baselines/egagent_ce003/run_ce003_egagent.py all
```

## Outputs

```text
meta_baselines/egagent_ce003/data_sources/transcript_for_inference.jsonl
meta_baselines/egagent_ce003/data_sources/entity_graph_rows.jsonl
meta_baselines/egagent_ce003/data_sources/frame_1fps/frame_index.jsonl
meta_baselines/egagent_ce003/db/ce003_entity_graph.db
meta_baselines/egagent_ce003/db/ce003_visual_frames.db
meta_baselines/egagent_ce003/questions/ce003_agent_questions.jsonl
meta_baselines/egagent_ce003/questions/ce003_agent_questions_for_inference.jsonl
meta_baselines/egagent_ce003/questions/ce003_qwen_tool_questions.jsonl
meta_baselines/results/egagent_qwen25_vl_7b/ce003_answers.jsonl
meta_baselines/results/egagent_qwen25_vl_7b/ce003_qwen_tool_answers.jsonl
meta_baselines/results/egagent_qwen25_vl_7b/tool_trace.jsonl
meta_baselines/results/egagent_qwen25_vl_7b/ce003_metrics.json
meta_baselines/results/egagent_qwen25_vl_7b/ce003_qwen_tool_metrics.json
meta_baselines/results/egagent_qwen25_vl_7b/ce003_error_analysis.md
```

## Leakage Control

Inference inputs:

- `character_labels.json`
- clip manifest
- ASR transcript
- active-speaker evidence
- generated character-card IDs
- generated SQLite visual/entity graph stores

Held out for scoring only:

- `interaction_labels.json`
- `evaluation_questions.json`

The strict adapter does not synthesize a `talks_to` edge unless generated tool evidence supplies both active-speaker and recipient evidence.

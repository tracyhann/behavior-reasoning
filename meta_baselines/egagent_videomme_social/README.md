# Video-MME CARE-Style Social Evaluation

This folder contains an EGAgent-style adapter for the 10 locally downloaded Video-MME samples in:

```text
meta_baselines/datasets/Video-MME_1/videos/
```

It is not an unmodified upstream `facebookresearch/egagent` run. It keeps the local EGAgent evaluation contract used for CE003:

- character-library access
- visual-frame search placeholder
- transcript/entity-graph search placeholder
- deterministic structured answers
- tool trace logging
- exact-match scoring

## Data Selection

The current set replaces the earlier Video-MME downloads with real-human-oriented clips:

- 5 Film & Television / Movie & TV Show videos: `101`, `103`, `104`, `105`, `106`
- 5 Life Record / Daily Life videos: `251`, `252`, `255`, `256`, `257`

Several YouTube downloads were AV1 MP4s, so the incompatible files were transcoded to H.264/AAC for Decord/Qwen compatibility while preserving MP4-with-audio inputs.

## Annotation Outputs

Each downloaded video has a CE003-style annotation package:

```text
meta_baselines/datasets/Video-MME_1/annotations/<video_id>_<videoID>/
  character_labels.json
  interaction_labels.json
  evaluation_questions.json
  meta.json
```

The labels are candidate CARE-style social labels. Background crowds, unlabeled extras, and offscreen phone/audio recipients are excluded from the foreground character library. Each video uses the CE003-style question schema:

- character count
- present character IDs
- first interaction initiator
- C2 speech recipient
- recipient response to first initiator
- whether C1 interacted with anyone
- whether C1 and C2 interacted

## Run

```bash
python3 meta_baselines/egagent_videomme_social/run_videomme_social_egagent.py all
```

Qwen tool-packet run:

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -e HF_HOME=/hf-cache \
  -e TRANSFORMERS_CACHE=/hf-cache/transformers \
  -e PYTHONPATH=/workspace/social-interaction-reasoning-agent:/workspace/behavior-understanding-test \
  -v /home/ttt/Desktop/autism:/workspace \
  -v /home/ttt/.cache/huggingface:/hf-cache \
  -w /workspace/social-interaction-reasoning-agent \
  behavior-qwen25-vl:cu128 \
  python baselines/run_qwen25_vl_7b.py \
    --project-root . \
    --questions meta_baselines/egagent_videomme_social/questions/videomme_social_qwen_tool_questions.jsonl \
    --out meta_baselines/results/egagent_videomme_social/videomme_social_qwen_tool_answers.jsonl \
    --max-new-tokens 800 \
    --max-video-frames 24

python3 meta_baselines/egagent_videomme_social/run_videomme_social_egagent.py score-qwen
```

## Results

```text
meta_baselines/results/egagent_videomme_social/videomme_social_answers.jsonl
meta_baselines/results/egagent_videomme_social/videomme_social_tool_trace.jsonl
meta_baselines/results/egagent_videomme_social/videomme_social_metrics.json
meta_baselines/results/egagent_videomme_social/videomme_social_qwen_tool_answers.jsonl
meta_baselines/results/egagent_videomme_social/videomme_social_qwen_tool_metrics.json
```

Important caveat: these are candidate labels produced from sampled review frames and local video review, not adjudicated publication gold. They are suitable for a development stress test of the CE003 schema and backend behavior, but should be reviewed before any paper-level claim.

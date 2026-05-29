# SIV-Bench CARE-Style Social Evaluation

This folder evaluates 10 locally downloaded videos from:

```text
https://huggingface.co/datasets/Fancylalala/SIV-Bench
```

The selected videos are stored under:

```text
meta_baselines/datasets/SIV-Bench_10/origin/
```

## Annotation Outputs

Each selected video has a CE003-style annotation package:

```text
meta_baselines/datasets/SIV-Bench_10/annotations/<relation>__<video_id>/
  character_labels.json
  interaction_labels.json
  evaluation_questions.json
  meta.json
```

The evaluation questions follow the CE003 style:

- character count
- present character IDs
- first interaction initiator
- C2 speech recipient
- recipient response
- whether C1 interacted
- whether C1 and C2 interacted

These are candidate labels based on sampled visual review and SIV-Bench QA context. They need adjudication with full audio/frame review before publication use.

## Run

```bash
python3 meta_baselines/egagent_sivbench_social/run_sivbench_social_egagent.py all
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
    --questions meta_baselines/egagent_sivbench_social/questions/sivbench_social_qwen_tool_questions.jsonl \
    --out meta_baselines/results/egagent_sivbench_social/sivbench_social_qwen_tool_answers.jsonl \
    --max-new-tokens 800 \
    --max-video-frames 24

python3 meta_baselines/egagent_sivbench_social/run_sivbench_social_egagent.py score-qwen
```

## Results

```text
meta_baselines/results/egagent_sivbench_social/sivbench_social_answers.jsonl
meta_baselines/results/egagent_sivbench_social/sivbench_social_tool_trace.jsonl
meta_baselines/results/egagent_sivbench_social/sivbench_social_metrics.json
meta_baselines/results/egagent_sivbench_social/sivbench_social_qwen_tool_answers.jsonl
meta_baselines/results/egagent_sivbench_social/sivbench_social_qwen_tool_metrics.json
```

The strict adapter is intentionally conservative: it does not generate interaction edges without verified speech/gaze/gesture evidence. On SIV-Bench positives, this exposes the current gap between character-library access and real social-edge recovery.

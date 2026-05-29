# Pluggable Backend Interface

The meta-baseline experiments are model-agnostic. A backend only needs to answer one JSONL row at a time.

## Question Row Contract

Each row in `*_qwen_tool_questions.jsonl` is a generic VLM/LLM request despite the historical filename:

```json
{
  "question_id": "VMME101_QWEN_ALL",
  "clip_id": "videomme_101_tiMaUSvlzIU",
  "episode_id": "101",
  "video_path": "meta_baselines/datasets/Video-MME_1/videos/101_tiMaUSvlzIU.mp4",
  "task_type": "egagent_style_videomme_social_ce003_questions",
  "prompt": "...",
  "scoring_targets": ["character_count", "present_characters"]
}
```

The runner passes the resolved `video_path` and the prompt to a backend. The backend returns a raw string. If that string contains JSON, the runner stores it as `parsed_response`.

## Python Backend

Create a class with this method:

```python
def answer(self, row: dict, video_path: Path, prompt: str) -> str:
    ...
```

Run it with:

```bash
python -m meta_baselines.backends.run_jsonl_backend \
  --project-root . \
  --questions meta_baselines/egagent_videomme_social/questions/videomme_social_qwen_tool_questions.jsonl \
  --out meta_baselines/results/egagent_videomme_social/videomme_social_qwen_tool_answers.jsonl \
  --backend your_package.your_backend:YourBackend \
  --backend-arg model=gpt-4.1
```

## Local Registry Backend

For local agents exposed through `behavior-understanding-test/agents/registry.py`:

```bash
python -m meta_baselines.backends.run_jsonl_backend \
  --project-root . \
  --questions meta_baselines/egagent_sivbench_social/questions/sivbench_social_qwen_tool_questions.jsonl \
  --out meta_baselines/results/egagent_sivbench_social/sivbench_social_qwen_tool_answers.jsonl \
  --backend meta_baselines.backends.registry_video_backend:RegistryVideoBackend \
  --backend-arg behavior_root=../behavior-understanding-test \
  --backend-arg model_name=qwen2_5_vl_7b \
  --backend-arg backend=transformers \
  --backend-arg max_new_tokens=900 \
  --backend-arg max_video_frames=24
```

This is the Qwen2.5-VL path used in the existing artifacts, but the runner itself does not depend on Qwen.

## Subprocess Backend

Use this when a model is easiest to call through a shell command. The command receives:

- `CARE_PROMPT`
- `CARE_VIDEO_PATH`
- `CARE_QUESTION_ID`
- `CARE_CLIP_ID`

Example:

```bash
python -m meta_baselines.backends.run_jsonl_backend \
  --project-root . \
  --questions meta_baselines/egagent_ce003/questions/ce003_qwen_tool_questions.jsonl \
  --out /tmp/answers.jsonl \
  --backend meta_baselines.backends.subprocess_json_backend:SubprocessJsonBackend \
  --backend-arg 'command=python my_backend.py'
```

After inference, run the dataset-specific scorer described in `../README.md`.


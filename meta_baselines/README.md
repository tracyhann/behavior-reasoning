# Meta-Baseline Social-Interaction Experiments

This package contains CARE-style meta-baseline experiments for social interaction reasoning. It includes:

- CE003 edge-case evaluation over `Sk9mR3zjrkk_CE003`
- SIV-Bench social-interaction subset labels/questions/results
- Video-MME real-human replacement labels/questions/results
- EGAgent-style strict adapters and Qwen2.5-VL result artifacts
- a pluggable backend runner for additional VLM/LLM backends

Raw videos, contact sheets, parquet files, SQLite DBs, and caches are intentionally not committed. Dataset manifests and annotation files keep the expected paths and source URLs.

## Layout

```text
meta_baselines/
  backends/                         # model-agnostic backend runner/adapters
  datasets/                         # manifests and candidate labels
  egagent_ce003/                    # CE003 adapter, questions, metadata
  egagent_sivbench_social/          # SIV-Bench adapter, questions, metadata
  egagent_videomme_social/          # Video-MME adapter, questions, metadata
  results/                          # scored answer artifacts and metric JSON
```

## Data Expectations

The scorer scripts expect media files to exist at the paths recorded in each manifest, for example:

```text
meta_baselines/datasets/Video-MME_1/videos/*.mp4
meta_baselines/datasets/SIV-Bench_10/origin/*/*.mp4
data/clip_sets/Sk9mR3zjrkk_CE003/clips/002_CE003.mp4
```

If the media files are absent, the existing result JSON can still be inspected, but new VLM inference cannot be rerun until the media is restored.

## Generate Or Refresh Questions

```bash
python meta_baselines/egagent_ce003/run_ce003_egagent.py all
python meta_baselines/egagent_sivbench_social/run_sivbench_social_egagent.py all
python meta_baselines/egagent_videomme_social/run_videomme_social_egagent.py all
```

The `all` command prepares question JSONL, runs the strict adapter, and scores strict-adapter outputs.

## Run Any VLM/LLM Backend

The question JSONL files are backend-neutral. Use the generic runner with any Python backend class:

```bash
python -m meta_baselines.backends.run_jsonl_backend \
  --project-root . \
  --questions meta_baselines/egagent_videomme_social/questions/videomme_social_qwen_tool_questions.jsonl \
  --out meta_baselines/results/egagent_videomme_social/videomme_social_qwen_tool_answers.jsonl \
  --backend your_package.your_backend:YourBackend \
  --backend-arg model=your-model-name
```

The historical `*_qwen_tool_questions.jsonl` filenames reflect the first backend used. They are ordinary prompt rows and can be sent to any backend.

Backend interface:

```python
class YourBackend:
    def __init__(self, *, project_root: Path, model: str = "...") -> None:
        ...

    def answer(self, row: dict, video_path: Path, prompt: str) -> str:
        return "{\"character_count\": 3, ...}"
```

See `meta_baselines/backends/README.md` for a registry-based local VLM adapter and a subprocess adapter.

## Score Backend Outputs

After writing answers to the expected result path, run:

```bash
python meta_baselines/egagent_ce003/run_ce003_egagent.py score-qwen
python meta_baselines/egagent_sivbench_social/run_sivbench_social_egagent.py score-qwen
python meta_baselines/egagent_videomme_social/run_videomme_social_egagent.py score-qwen
```

The `score-qwen` command name is historical. It scores the answer JSONL schema, not the Qwen model specifically.

## Current VLM Results

| Dataset/run | Backend artifact | Overall |
| --- | --- | ---: |
| CE003 | Qwen2.5-VL-7B tool packet | 2/7, 28.6% |
| SIV-Bench 10 | Qwen2.5-VL-7B tool packet | 57/70, 81.4% |
| Video-MME 10 real-human replacement | Qwen2.5-VL-7B tool packet | 53/70, 75.7% |

These labels are development labels. They are useful for backend comparison and pipeline debugging, but should be adjudicated before publication-level claims.


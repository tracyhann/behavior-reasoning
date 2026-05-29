# Video-MME Real-Human Replacement Evaluation

## Replacement Set

The previous Video-MME downloads were replaced with 10 MP4-with-audio samples from real-human-oriented categories:

| Video | Category |
| --- | --- |
| `101_tiMaUSvlzIU.mp4` | Film & Television / Movie & TV Show |
| `103_G267g0DpCVg.mp4` | Film & Television / Movie & TV Show |
| `104__cZXyj6rYVg.mp4` | Film & Television / Movie & TV Show |
| `105_rQhLWHtHyiM.mp4` | Film & Television / Movie & TV Show |
| `106_bYXhA8VG8Lw.mp4` | Film & Television / Movie & TV Show |
| `251_1sTQOxXFO44.mp4` | Life Record / Daily Life |
| `252_RP1AL2DU6vQ.mp4` | Life Record / Daily Life |
| `255_CQUphYL0vY8.mp4` | Life Record / Daily Life |
| `256_aFfMGy94sjE.mp4` | Life Record / Daily Life |
| `257_s-lM2uwiwyQ.mp4` | Life Record / Daily Life |

All 10 files verify as H.264 MP4s with one video stream and one audio stream.

## Evaluation Schema

Each video has candidate labels and seven CE003-style deterministic questions:

1. character count
2. present character IDs
3. first interaction initiator
4. C2 speech recipient
5. recipient response to the first initiator
6. whether C1 interacted with any character
7. whether C1 and C2 interacted

## Results

| Backend | Overall exact match | Character count | Present IDs | First initiator | C2 recipient | Recipient responded | C1 any interaction | C1-C2 interaction | Directed edge F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EGAgent-style strict adapter | 39/70 = 55.7% | 10/10 | 10/10 | 3/10 | 5/10 | 3/10 | 4/10 | 4/10 | 0.0 |
| Qwen2.5-VL-7B tool packet | 53/70 = 75.7% | 10/10 | 10/10 | 6/10 | 6/10 | 4/10 | 8/10 | 9/10 | n/a |

The strict adapter does not generate interaction edges, so its edge F1 is 0.0 against 16 candidate gold directed edges. The Qwen tool-packet run was scored on deterministic question outputs only.

## Files

- annotations: `meta_baselines/datasets/Video-MME_1/annotations/`
- atomic questions: `meta_baselines/egagent_videomme_social/questions/videomme_social_questions.jsonl`
- Qwen prompt rows: `meta_baselines/egagent_videomme_social/questions/videomme_social_qwen_tool_questions.jsonl`
- strict adapter metrics: `meta_baselines/results/egagent_videomme_social/videomme_social_metrics.json`
- Qwen metrics: `meta_baselines/results/egagent_videomme_social/videomme_social_qwen_tool_metrics.json`

These labels are candidate development labels from sampled review frames and local video review, not adjudicated publication gold.

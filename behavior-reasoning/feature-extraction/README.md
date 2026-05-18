# Feature Extraction Pipeline

This folder contains the first-stage video feature extraction pipeline for the behavior-reasoning framework. It converts an input video into timestamped evidence files: overlapping segments, persistent subject cards, scene/object summaries, and per-character detection/pose/motion/behavior rows.

The implementation is intentionally lean: OpenCV handles video probing and frame sampling, Ultralytics YOLO handles person tracking and pose when available, and local Qwen2.5-VL is used only for conservative sampled-frame labels and short factual summaries.

## What It Writes

For each input video, the CLI writes:

```text
<output_root>/<video_id>/
  raw/
    segments.json
    sampled_frames/
    crops/
  characters/
    characters.json
  detections/
    detections.json
  logs/
    pipeline.log
```

Key outputs:

- `raw/segments.json`: video metadata and 10-second windows with 6-second stride by default.
- `characters/characters.json`: persistent `person-XX` subject cards.
- `detections/detections.json`: scene rows, object rows, and per-character rows with bbox, pose, motion, behavior text, confidence, and uncertainty.

## Run With Docker

The tested runtime is the local image:

```text
behavior-qwen25-vl:cu128
```

That image has Qwen2.5-VL dependencies, but it may not have `ultralytics` installed. The command below installs `ultralytics` and `lap` ephemerally inside the container before running the pipeline:

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v /home/ttt/Desktop/autism:/workspace \
  -v /home/ttt/.cache/huggingface:/hf-cache \
  -w /workspace/behavior-reasoning/feature-extraction \
  -e HF_HOME=/hf-cache \
  -e TRANSFORMERS_CACHE=/hf-cache/transformers \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  behavior-qwen25-vl:cu128 \
  bash -lc 'python -m pip install --no-cache-dir ultralytics lap >/tmp/ultralytics-install.log && \
    python -m video_features.run \
      --input /workspace/dataset-ours/oCkUyjaZuNI/oCkUyjaZuNI.mp4 \
      --output /workspace/behavior-reasoning/outputs \
      --segment-sec 10 \
      --stride-sec 6 \
      --dtype bfloat16 \
      --max-new-tokens 320 \
      --expected-subjects 3'
```

## CLI

From `behavior-reasoning/feature-extraction`:

```bash
python -m video_features.run \
  --input path/to/video.mp4 \
  --output path/to/output_root \
  --segment-sec 10 \
  --stride-sec 6 \
  --det-model yolo11n.pt \
  --pose-model yolo11n-pose.pt \
  --tracker botsort.yaml
```

Useful flags:

- `--dry-run`: write the output structure without loading Qwen2.5-VL.
- `--no-vlm`: run deterministic extraction without VLM scene/character summaries.
- `--max-segments N`: limit processing for debugging.
- `--dtype bfloat16`: recommended for the local GPU container.
- `--expected-subjects N`: constrain the VLM global roster when the dataset or reviewer knows the number of recurring characters.

YOLO model fallback is built in: `yolo11*` names are mapped to `yolov8*` if needed by the available Ultralytics assets.

## Conservative Description Rules

The VLM prompt and normalization layer enforce conservative evidence text:

- No identity, real names, diagnosis, personality, emotion, or intent.
- Gender and age range are stored as coarse structured labels when visually supported: `male | female | unknown` and `adult | child | adolescent | unknown`.
- `appearance` is reserved for stable visible traits such as hairstyle, facial hair, glasses, makeup visibility, jewelry, or accessories. Actions, posture, room, and location belong in scene/behavior fields, not `appearance`.
- Identity-coded outputs are sanitized.
- Unsupported speech/intent wording is converted to observable visual wording.
- If evidence is partial or missing, uncertainty is explicit.

YOLO-only rows use YOLO/motion/pose modalities only. VLM character rows stay separate unless there is explicit matching evidence.

## Character Roster Logic

When Qwen2.5-VL is enabled, the pipeline first asks for a global anonymous roster from sampled frames. That roster is the source of truth for `characters/characters.json`; raw YOLO track fragments are not promoted into new characters by themselves. If a segment-level VLM response chooses a roster ID such as `person-02`, that segment updates the corresponding subject card.

YOLO rows that are not matched to a roster subject remain in `detections/detections.json` with `person_id: null`, `track_id`, bbox, pose, and motion evidence. This is intentional: an unmatched detector track is evidence, not a new character. Use `--expected-subjects` when a benchmark sample has a known character count.

## Tests

Run tests in the Docker image:

```bash
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 \
  -v /home/ttt/Desktop/autism:/workspace \
  -w /workspace/behavior-reasoning/feature-extraction \
  behavior-qwen25-vl:cu128 \
  python -W error::ResourceWarning -m unittest discover -s tests
```

The current suite covers segmentation, output-tree creation, VLM normalization/safety filtering, subject matching, YOLO assignment/summarization helpers, and pipeline output contracts.

## Current Limits

- `ultralytics` is not baked into `behavior-qwen25-vl:cu128`; install it ephemerally as shown above or rebuild the image with it.
- Crop files are not currently saved beyond the reserved `raw/crops/` directory.
- Subject matching is intentionally conservative: exact YOLO track continuity is strong, and fragmented tracks merge only with strong bbox or specific appearance continuity when no global roster is available.
- Roster-to-YOLO association is conservative. Unmatched YOLO tracks are kept as detection evidence instead of being force-mapped to a character.
- Pose summaries are only as good as the YOLO pose model output. If pose keypoints are unavailable, bbox motion is reported under `motion`, not `pose`.

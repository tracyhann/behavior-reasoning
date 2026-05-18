# Task: Build Timestamped Multimodal Video Feature Extraction Pipeline

## Goal

Implement a Python pipeline that processes input videos into overlapping 10-second windows with stride 6 seconds, extracts timestamped multimodal evidence, builds a persistent character/subject library, and writes structured JSON outputs.

The pipeline is for character/social/behavioral interaction analysis. It should produce **factual, conservative descriptions only**. Do not infer identity, diagnosis, personality, emotion, intent, or race. Store coarse visual gender and age labels when supported. If something is uncertain, write uncertainty explicitly.

## High-level workflow

For each input video:

1. Create overlapping segment windows:
   - segment length = 10 seconds
   - stride = 6 seconds
   - store segment metadata with `segment_id`, `start_time`, `end_time`.

2. Use YOLO / YOLO tracking to detect and track people.
   - Prefer running tracking over the full video once, then assigning tracks to each segment by timestamp.
   - If full-video tracking is not feasible, run per-segment tracking and then perform cross-segment subject matching.
   - Each detected/tracked person should be mapped to a persistent subject ID like `person-01`.

3. Build or update a subject/character library:
   - For every detected person, create a subject card:
     ```json
     {
       "id": "person-01",
       "gender": "male | female", 
       "age_range": "adult | child | adolescent | unknown",
       "height": "unknown or relative estimate only",
       "clothing": "concise visible clothing description",
       "appearance": "concise visual appearance summary",
       "first_seen": 0.0,
       "last_seen": 10.0,
       "example_segments": ["seg_001"],
       "confidence": "low | medium | high"
     }
     ```
   - Before adding a new subject, compare it against existing subject cards.
   - If highly similar, update the existing subject instead of adding a duplicate.
   - Matching should use available evidence such as:
     - YOLO track ID continuity
     - bounding box continuity
     - appearance/crop similarity
     - clothing/color summary
     - position continuity
     - optional embedding similarity if implemented
   - Do not claim exact age or real height. Use coarse categories and uncertainty.

4. For each segment, perform multimodal feature extraction:
   - YOLO pose estimation for each detected character.
   - Convert pose/keypoints into concise text:
     - movement directionality
     - which body parts appear to move
     - whether subject is stationary, slow-moving, or fast-moving
     - rough posture/body orientation if inferable
   - RGB/frame understanding:
     - describe scene/environment
     - identify salient objects
     - timestamp all descriptions to the current segment
   - Joint character behavior description:
     - combine pose/motion text + scene/context/object information
     - describe what each character is observably doing
     - be concise and factual
     - include confidence/uncertainty and what additional evidence would be needed

5. Write final JSON outputs under the output directory.

## Output directory structure

For each processed video, create:

```text
output_root/
  <video_id>/
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
````

Do not store huge intermediate files unless necessary. If crops/frames are saved, keep them small and organized by segment.

## Required output: segments.json

Location:

```text
<video_id>/raw/segments.json
```

Schema:

```json
{
  "video_id": "video_001",
  "source_video": "path/to/video.mp4",
  "segment_length_sec": 10,
  "stride_sec": 6,
  "segments": [
    {
      "segment_id": "seg_001",
      "start_time": 0.0,
      "end_time": 10.0
    },
    {
      "segment_id": "seg_002",
      "start_time": 6.0,
      "end_time": 16.0
    }
  ]
}
```

## Required output: characters.json

Location:

```text
<video_id>/characters/characters.json
```

Schema:

```json
{
  "video_id": "video_001",
  "characters": {
    "person-01": {
      "id": "person-01",
      "gender": "male | female", 
      "age_range": "adult | child | adolescent | unknown",
      "height": "unknown or relative estimate only",
      "clothing": "visible clothing summary",
      "appearance": "concise visual appearance summary",
      "first_seen": 0.0,
      "last_seen": 42.0,
      "track_ids": ["track-7"],
      "example_segments": ["seg_001", "seg_002"],
      "example_crop_paths": ["raw/crops/person-01_seg_001.jpg"],
      "match_confidence": "high",
    }
  }
}
```

Character matching behavior:

* If a newly observed person matches an existing subject with high confidence, update that subject’s `last_seen`, `example_segments`, `track_ids`, and appearance observations.
* If no existing subject matches, create the next ID: `person-01`, `person-02`, etc.
* If uncertain, either create a new person with low confidence or mark `possible_matches`.

## Required output: detections.json

Location:

```text
<video_id>/detections/detections.json
```

Schema:

```json
{
  "video_id": "video_001",
  "scenes": [
    {
      "segment_id": "seg_001",
      "start_time": 0.0,
      "end_time": 10.0,
      "description": "Concise factual scene description.",
      "confidence": "low | medium | high",
      "uncertainty": "What is unclear, if anything."
    }
  ],
  "objects": [
    {
      "segment_id": "seg_001",
      "start_time": 0.0,
      "end_time": 10.0,
      "objects": [
        {
          "label": "chair",
          "count": 2,
          "confidence": 0.84,
          "description": "Visible near the participants."
        }
      ],
      "uncertainty": "Objects partially occluded."
    }
  ],
  "characters": [
    {
      "segment_id": "seg_001",
      "start_time": 0.0,
      "end_time": 10.0,
      "person_id": "person-01",
      "track_id": "track-7",
      "bbox_summary": {
        "mean_bbox": [120, 80, 260, 420],
        "visibility": "full | upper_body | partial | unclear",
        "confidence": 0.88
      },
      "pose": {
        "available": true,
        "summary": "Person appears seated/upright; right arm moves slightly upward.",
        "moving_body_parts": ["right_arm"],
        "speed": "stationary | slow | medium | fast | unclear",
        "directionality": "toward_object | toward_person | away | lateral | unclear",
        "confidence": "medium"
      },
      "motion": {
        "mean_motion": 0.21,
        "peak_motion": 0.62,
        "peak_time": 6.4,
        "activity_level": "low | medium | high"
      },
      "behavior": {
        "description": "Concise factual description of what this person is observably doing in the segment.",
        "modalities_used": ["rgb", "pose", "objects"],
        "confidence": "low | medium | high",
        "uncertainty": "What additional evidence is needed or what is ambiguous."
      }
    }
  ]
}
```

## Model/tool requirements

Use Python.

Preferred packages:

* `ultralytics` for YOLO detection/tracking/pose.
* `opencv-python` for video reading, frame sampling, crops, basic motion.
* `numpy`, `scipy` as needed.
* `pydantic` or dataclasses for schema validation.
* `tqdm`, `logging`.

YOLO model recommendations:

* Detection/tracking:

  * `yolo11n.pt`, `yolov8n.pt`, or configurable equivalent.
* Pose:

  * `yolo11n-pose.pt`, `yolov8n-pose.pt`, or configurable equivalent.
* Tracking:

  * default `botsort.yaml`
  * allow config option for `bytetrack.yaml`.

The implementation should allow CLI overrides:

```bash
python -m video_features.run \
  --input path/to/video.mp4 \
  --output output_root \
  --segment-sec 10 \
  --stride-sec 6 \
  --det-model yolo11n.pt \
  --pose-model yolo11n-pose.pt \
  --tracker botsort.yaml
```

If `yolo11` weights are unavailable, fall back cleanly to `yolov8` weights.

## RGB scene/object description

Implement a basic version first:

1. Sample frames from each segment:

   * start frame
   * middle frame
   * end frame
2. Use YOLO object labels and visual heuristics to produce a factual scene/object summary.
3. Add a placeholder hook for optional VLM scene description later.

Do not require API-based VLM calls for the first implementation. The code should work offline if YOLO models are available.

Suggested optional interface:

```python
class SceneDescriber:
    def describe(frames, yolo_objects, segment_metadata) -> dict:
        ...
```

The default implementation can be rule/template-based.

## Pose-to-text conversion

Implement deterministic pose-to-text first. Do not rely on an LLM.

Use keypoint coordinates across frames to infer:

* stationary vs moving
* rough speed
* hand/arm movement
* leg movement
* torso/body orientation if possible
* movement direction using bbox center displacement
* confidence from keypoint visibility

Example output:

```json
{
  "summary": "Person is mostly stationary with slight right-arm movement.",
  "moving_body_parts": ["right_arm"],
  "speed": "slow",
  "directionality": "unclear",
  "confidence": "medium"
}
```

## Subject matching

Implement a simple matching function.

Input:

* new observation:

  * track ID
  * crop path or crop embedding/color histogram
  * clothing/appearance summary
  * time range
* existing character cards

Output:

```json
{
  "matched_person_id": "person-01",
  "score": 0.87,
  "decision": "match | new_subject | uncertain"
}
```

Minimum matching features:

* same YOLO track ID within same video: strong match
* temporal continuity: medium/strong
* appearance color histogram similarity from crops: medium
* text similarity between clothing/appearance summaries: optional
* if low confidence, create a new person or mark possible match

Configurable thresholds:

```yaml
subject_match_high: 0.80
subject_match_uncertain: 0.60
```

## Important constraints

* Do not infer identity, name, race, emotion, diagnosis, ASD, mental state, personality, or intent.
* Store coarse visual gender and age labels when supported: `male | female | unknown` and `adult | child | adolescent | unknown`.
* Use `appearance` for stable visible traits only: hairstyle/hair length, facial hair, glasses, makeup visibility, jewelry, or accessories. Do not put action, posture, room, or location in `appearance`.
* Use only observable facts.
* For age range, use only coarse categories if visually obvious: `adult`, `child`, `adolescent`, `unknown`.
* For height, do not estimate real height unless calibration is available. Use `unknown` or `relative estimate`.
* Always include uncertainty when evidence is partial.
* Keep descriptions concise.
* The goal is feature extraction and evidence logging, not final social interpretation.

## Implementation plan

1. Inspect current repository structure.
2. Add a new module/package, for example:

   ```text
   video_features/
     __init__.py
     run.py
     segmentation.py
     yolo_detection.py
     pose_text.py
     subject_library.py
     scene_description.py
     schemas.py
     io_utils.py
   ```
3. Add `requirements.txt` if needed.
4. Implement CLI in `run.py`.
5. Implement segmentation in `segmentation.py`.
6. Implement YOLO detection/tracking and pose extraction in `yolo_detection.py`.
7. Implement pose-to-text in `pose_text.py`.
8. Implement subject matching and `characters.json` update logic in `subject_library.py`.
9. Implement scene/object summary in `scene_description.py`.
10. Implement JSON writing and validation.
11. Add a small test or dry-run mode:

    ```bash
    python -m video_features.run --input sample.mp4 --output output_root --dry-run
    ```

## Acceptance criteria

The task is complete when:

1. Running the CLI on a video creates:

   ```text
   output_root/<video_id>/raw/segments.json
   output_root/<video_id>/characters/characters.json
   output_root/<video_id>/detections/detections.json
   output_root/<video_id>/logs/pipeline.log
   ```

2. `segments.json` contains 10-second overlapping windows with stride 6 seconds.

3. `characters.json` contains persistent `person-XX` IDs and does not duplicate obvious same-person tracks.

4. `detections.json` contains timestamped:

   * scene descriptions
   * object lists
   * per-character pose/motion summaries
   * per-character factual behavior descriptions
   * uncertainty/confidence notes

5. The pipeline handles missing pose, missing detections, and low-confidence tracks without crashing.

6. All generated descriptions are conservative and observable.

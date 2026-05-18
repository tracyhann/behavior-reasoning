1. implement this task, behavior-reasoning/reasoning/TASK.md, in behavior-reasoning/outputs/oCkUyjaZuNI/chains

essentially, this step generates a clean chain of interactions from the video.

Use the provided questions to check whether your work is good.

use QWEN 7B (sam e as above) to REASON and implement the task, generate text descriptions in the pipeline. remember this is an agentic system

once done, repeat this QA/QC steps 5 times. if you fix sth,  make sure it is COHENRENT with all other files.

once done, update behavior-reasoning/README to introduce what this is.

you need to explain how to use another local model. the README should be an instruction for another VLM agent

include your instructions on how to use another VLM model

once you are CONFIDENT about your work, push the dataset-ours folder, and behavior-reasoning folder, to this newly created github repo [tracyhann/behavior-reasoning.git](https://github.com/tracyhann/behavior-reasoning.git)

use the github repo as the base for behavior-reasoning video agents codes

fully autoated workflow, use your BEST judgements. make your work very robust

# Task: Build Behavior Chains from `characters.json` and `detections.json`

## Goal

Implement a small pipeline that reads:

```text
characters.json
detections.json
````

and produces a chronological behavior-chain file:

```text
behavior_chains.json
```

The purpose is to convert noisy, overlapping, timestamped detections into clean **behavior chains** that describe character interactions over time.

Focus only on this layer:

```text
characters + detections
  → normalized events
  → deduplicated canonical events
  → chronological behavior chains
```

Do **not** build segmentation, VLM calls, pose extraction, audio extraction, or annotation UI in this task.

---

## Expected Deliverables

Create:

```text
scripts/build_behavior_chains.py
tests/test_build_behavior_chains.py
behavior_chains.json  # generated output, not committed unless appropriate
canonical_events.json # generated intermediate output
```

The script should run as:

```bash
python scripts/build_behavior_chains.py \
  --characters characters.json \
  --detections detections.json \
  --out behavior_chains.json \
  --canonical-events-out canonical_events.json
```

Optional parameters:

```bash
--max-merge-gap-sec 2.0
--max-chain-gap-sec 10.0
--max-response-window-sec 8.0
--min-dedupe-score 0.72
```

---

## Input Assumptions

The exact schema of `characters.json` and `detections.json` may vary. Inspect the actual files before coding rigid assumptions.

The script should tolerate common field variants.

### Character field variants

Character records may use fields like:

```text
id
character_id
person_id
name
label
role
track_id
aliases
description
```

Build a character lookup map from all available identifiers:

```text
character_id
person_id
track_id
name
label
aliases
```

### Detection field variants

Detection records may use fields like:

```text
id
detection_id
event_id
start_sec
end_sec
start_time
end_time
timestamp
time
label
type
event_type
description
caption
summary
actor
actor_id
actor_ids
target
target_id
target_ids
person_id
person_ids
character_id
character_ids
object_id
object_ids
modality
modalities
confidence
score
segment_id
source_segment_id
```

Normalize these into a canonical event structure.

---

## Output 1: `canonical_events.json`

Write a list of deduplicated canonical events.

Each event should look like:

```json
{
  "event_id": "evt_0001",
  "start_sec": 12.4,
  "end_sec": 14.0,
  "event_type": "social_cue",
  "actor_ids": ["char_001"],
  "target_ids": ["char_002"],
  "object_ids": ["obj_001"],
  "modalities": ["rgb", "speech"],
  "description": "P1 gives an object-mediated cue to P2.",
  "confidence": 0.82,
  "source_detection_ids": ["det_003", "det_007"],
  "source_segment_ids": ["seg_001", "seg_002"],
  "merge_notes": {
    "merged_count": 2,
    "conflicts": [],
    "dropped_interpretive_claims": []
  }
}
```

If fields are unavailable, use empty lists or `null`, not fake values.

---

## Output 2: `behavior_chains.json`

Write behavior chains as:

```json
{
  "schema_version": "0.1",
  "source_files": {
    "characters": "characters.json",
    "detections": "detections.json"
  },
  "parameters": {
    "max_merge_gap_sec": 2.0,
    "max_chain_gap_sec": 10.0,
    "max_response_window_sec": 8.0,
    "min_dedupe_score": 0.72
  },
  "chains": [
    {
      "chain_id": "chain_0001",
      "start_sec": 12.4,
      "end_sec": 44.8,
      "participant_ids": ["char_001", "char_002"],
      "object_ids": ["obj_001"],
      "event_ids": ["evt_0001", "evt_0002", "evt_0003"],
      "events": [
        {
          "event_id": "evt_0001",
          "time_span": [12.4, 14.0],
          "event_type": "social_cue",
          "actor_ids": ["char_001"],
          "target_ids": ["char_002"],
          "description": "P1 gives an object-mediated cue to P2."
        }
      ],
      "interaction_pattern": "cue_response",
      "summary": "char_001 gives a cue; char_002 responds shortly after.",
      "confidence": 0.78,
      "source_detection_ids": ["det_003", "det_007", "det_011"]
    }
  ],
  "orphan_event_ids": [],
  "warnings": []
}
```

---

## Behavior-Chain Definition

A behavior chain is a chronological sequence of canonical events that likely belong to the same interaction.

A chain should preserve:

```text
who acted
who was targeted
what object was involved
what happened
when it happened
whether a response followed a cue
```

The chain is not a diagnosis or psychological interpretation.

Do not infer internal states like:

```text
angry
autistic
rude
sad
manipulative
socially impaired
```

unless those exact labels already exist in the input. Even then, preserve them only as source labels, not as final interpretation.

---

## Implementation Steps

### 1. Load files

Load `characters.json` and `detections.json`.

Accept either:

```json
[ ... ]
```

or:

```json
{ "characters": [ ... ] }
{ "detections": [ ... ] }
```

Fail with a clear error if the expected records cannot be found.

---

### 2. Build character resolver

Create a function:

```python
resolve_character_id(raw_value: Any) -> str | None
```

It should map local IDs, names, labels, aliases, track IDs, or person IDs to a stable global character ID.

If no match is found, preserve the raw value under an `unresolved_*` field if useful, but do not crash.

---

### 3. Normalize detections into candidate events

Create a normalized event object with:

```python
event_id
start_sec
end_sec
event_type
actor_ids
target_ids
object_ids
modalities
description
confidence
source_detection_ids
source_segment_ids
raw
```

Rules:

* If only `timestamp` exists, set `start_sec = timestamp`, `end_sec = timestamp`.
* If no time is available, place the event in `orphan_event_ids` / warnings and skip chain construction for it.
* If no description exists, use label/type as fallback.
* Normalize event types into a controlled set when possible:

```text
social_cue
response
speech
vocalization
gesture
object_action
pose_motion
gaze_head_orientation
approach_withdraw
repetitive_motion_candidate
context_change
other
```

Use `other` if uncertain.

---

### 4. Deduplicate overlapping candidate events

Overlapping windows may describe the same event multiple times.

Create a duplicate score between nearby events:

```text
duplicate_score =
  0.35 * temporal_overlap_score
+ 0.20 * text_similarity_score
+ 0.20 * actor_target_match_score
+ 0.15 * event_type_match_score
+ 0.10 * object_match_score
```

Use lightweight deterministic logic first:

* temporal overlap / center distance
* token Jaccard similarity for descriptions
* actor/target/object overlap
* event type match

Do not require external embedding models unless already installed. Keep the script portable.

Merge events if:

```text
duplicate_score >= min_dedupe_score
```

Also merge if:

```text
same event_type
same actor_ids
temporal IoU > 0.4
```

Do not merge repeated events that happen far apart.

Default rule:

```text
If center time distance > 5 seconds, do not merge unless intervals overlap strongly.
```

---

### 5. Merge event clusters

For each duplicate cluster:

* `start_sec` = minimum start
* `end_sec` = maximum end
* `actor_ids` = union
* `target_ids` = union
* `object_ids` = union
* `modalities` = union
* `confidence` = max or mean of source confidences
* `source_detection_ids` = union
* `source_segment_ids` = union
* `description` = conservative merged description

For description merging, use deterministic preference:

1. Prefer the highest-confidence description.
2. If tied, prefer the most specific non-empty description.
3. Do not invent new claims.

---

### 6. Sort canonical events chronologically

Sort by:

```text
start_sec, end_sec, event_id
```

Ensure no chain later violates chronological order.

---

### 7. Link events into behavior chains

Build chains using chronological proximity and shared participants/objects.

Two events should be linked into the same chain if:

```text
time_gap <= max_chain_gap_sec
AND at least one of:
  - shared actor/target participant
  - previous target becomes next actor
  - same salient object
  - event pair looks like cue → response
```

Cue-response heuristic:

A cue-like event is one of:

```text
social_cue
speech
gesture
object_action
approach_withdraw
```

A response-like event is one of:

```text
response
speech
vocalization
gesture
object_action
pose_motion
gaze_head_orientation
approach_withdraw
```

If a cue has a target, search forward within:

```text
max_response_window_sec
```

for an event where the target becomes actor. Link them strongly.

---

### 8. Assign interaction pattern

For each chain, assign one of:

```text
cue_response
turn_taking
object_mediated
motion_sequence
repetitive_motion_sequence
parallel_activity
ambiguous_interaction
```

Heuristics:

* `cue_response`: contains a cue-like event followed by response-like event from target.
* `turn_taking`: contains alternating speech events between participants.
* `object_mediated`: multiple events share an object.
* `motion_sequence`: mostly pose/motion/action events.
* `repetitive_motion_sequence`: repeated similar motion events from same actor.
* `parallel_activity`: multiple actors, weak direct linkage.
* `ambiguous_interaction`: default if unclear.

---

### 9. Generate chain summaries

Generate simple conservative summaries without LLM calls.

Examples:

```text
char_001 gives a cue; char_002 responds shortly after.
char_001 and char_002 exchange turns around obj_001.
char_002 performs a sequence of motion events without a clear social cue.
```

Avoid psychological interpretation.

---

### 10. Write warnings

Collect warnings for:

```text
missing timestamps
unresolved character IDs
empty descriptions
events skipped from chaining
conflicting actor/target fields
very long chains
duplicate clusters with conflicting event types
```

Write them into `behavior_chains.json`.

---

## Tests

Create `tests/test_build_behavior_chains.py`.

Test at least:

1. **Loads list-style and dict-style JSON inputs**
2. **Normalizes timestamp fields**
3. **Resolves character aliases**
4. **Deduplicates overlapping duplicate events**
5. **Does not merge repeated events far apart**
6. **Builds a cue-response chain**
7. **Creates no-response/orphan handling when appropriate**
8. **Preserves chronological event order**
9. **Does not crash on missing optional fields**

Use small synthetic examples inside the tests.

Run:

```bash
pytest tests/test_build_behavior_chains.py
```

---

## Quality Requirements

The implementation should be:

```text
simple
deterministic
schema-tolerant
well-logged
easy to inspect
```

Do not add a database.

Do not add LLM calls.

Do not add external APIs.

Do not build a UI.

Do not overfit to one exact input schema without checking the files.

---

## Final Acceptance Criteria

The task is complete when:

```bash
python scripts/build_behavior_chains.py \
  --characters characters.json \
  --detections detections.json \
  --out behavior_chains.json \
  --canonical-events-out canonical_events.json
```

successfully produces:

```text
canonical_events.json
behavior_chains.json
```

and:

```bash
pytest tests/test_build_behavior_chains.py
```

passes.

The output should allow downstream agents to ask:

```text
What happened first?
Which characters interacted?
Did one character cue another?
Was there an observable response?
What events form a longer behavior chain?
```

```


1. implement this task, behavior-reasoning/reasoning/TASK.md, in behavior-reasoning/outputs/oCkUyjaZuNI/chains

essentially, this step generates a clean chain of interactions from the video.

Use the provided questions to check whether your work is good.

use QWEN 7B (sam e as above) to REASON and implement the task, generate text descriptions in the pipeline. remember this is an agentic system

once done, repeat this QA/QC steps 5 times. if you fix sth,  make sure it is COHENRENT with all other files.

once done, update behavior-reasoning/README to introduce what this is.

you need to explain how to use another local model. the README should be an instruction for another VLM agent

include your instructions on how to use another VLM model

once you are CONFIDENT about your work, push the dataset-ours folder, and behavior-reasoning folder, to this newly created github repo [tracyhann/behavior-reasoning.git](https://github.com/tracyhann/behavior-reasoning.git)

use the github repo as the base for behavior-reasoning video agents codes

fully autoated workflow, use your BEST judgements. make your work very robust

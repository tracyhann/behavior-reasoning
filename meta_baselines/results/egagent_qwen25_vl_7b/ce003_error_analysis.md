# CE003 EGAgent-Style Error Analysis

Question exact match: 4 / 7 (0.571429)
Directed edge F1: 0.0
Speaker attribution accuracy: 0.0
C1 false-positive interaction edges: 0

## Failure Tags

- `missed_first_initiator`
- `missed_c2_to_c3_talks_to`
- `missed_c3_to_c2_response`
- `speaker_attribution_incomplete_or_wrong`

## Interpretation

This run uses an EGAgent-compatible routing structure, but not the unmodified upstream LangGraph runner.
The available CE003 active-speaker evidence does not recover the labeled C2->C3 and C3->C2 speech exchange, so the strict tool-based adapter refuses to produce directed `talks_to` edges.
That behavior avoids the earlier false C1 speaker attribution, but it also misses the true interaction edges.

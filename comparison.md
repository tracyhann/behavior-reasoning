# Behavior-reasoning multi-model comparison

Videos: oCkUyjaZuNI, -JUtzfuiV08, DmSmN-oqFZ0, G6f0w5BRasw

## Headline per-video stats

| video | model | segs | roster | non-empty scenes | objs total | char rows | chains | events | qa clean |
|---|---|---|---|---|---|---|---|---|---|
| oCkUyjaZuNI | Qwen2.5-VL-72B | 16 | 3 | 16/16 | 38 | 21 | 21 | 104 | 5/5 |
| oCkUyjaZuNI | Qwen3-VL-8B | 16 | 3 | 16/16 | 64 | 32 | 30 | 113 | 5/5 |
| -JUtzfuiV08 | Qwen2.5-VL-72B | 14 | 2 | 14/14 | 44 | 26 | 22 | 80 | 5/5 |
| -JUtzfuiV08 | Qwen3-VL-8B | 14 | 2 | 14/14 | 58 | 26 | 22 | 80 | 5/5 |
| DmSmN-oqFZ0 | Qwen2.5-VL-72B | 16 | 3 | 16/16 | 44 | 29 | 20 | 103 | 5/5 |
| DmSmN-oqFZ0 | Qwen3-VL-8B | 16 | 3 | 16/16 | 70 | 27 | 21 | 107 | 5/5 |
| G6f0w5BRasw | Qwen2.5-VL-72B | 34 | 3 | 33/34 | 68 | 56 | 33 | 212 | 5/5 |
| G6f0w5BRasw | Qwen3-VL-8B | 34 | 3 | 34/34 | 114 | 55 | 46 | 228 | 5/5 |

## Aggregates across all videos

| model | segs | scenes non-empty | objs total | resolved char rows | chains | canonical events |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-72B | 80 | 79 | 194 | 132 | 96 | 499 |
| Qwen3-VL-8B | 80 | 80 | 306 | 140 | 119 | 528 |

## Sample scene descriptions (first segment of `oCkUyjaZuNI`)

### Qwen2.5-VL-72B
- scene: A cityscape with the word 'QUEENS' prominently displayed over it, followed by an indoor hallway where people are walking.
- confidence: high

  - obj `city buildings` (0.85): a skyline with various buildings
  - obj `highway` (0.8): a road with moving vehicles
  - obj `backpack` (0.9): a red and black backpack being carried

### Qwen3-VL-8B
- scene: An aerial view of a city skyline with the word 'QUEENS' overlaid, followed by shots of people walking down an indoor hallway.
- confidence: high

  - obj `city skyline` (0.98): Distant urban skyline with tall buildings under an overcast sky.
  - obj `text overlay` (0.99): Large white text reading 'QUEENS' centered on the screen.
  - obj `highway` (0.95): Multi-lane road with moving vehicles.


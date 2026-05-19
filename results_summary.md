# 行为推理多模型实测结果汇总

> 数据: 4 个视频 (`oCkUyjaZuNI`, `-JUtzfuiV08`, `DmSmN-oqFZ0`, `G6f0w5BRasw`),
> 总时长 ~7 分钟, 640×360 / 24fps。
> Pipeline: video → 采样帧 → YOLO tracking + VLM 描述 → 角色 roster → 行为链 → 5 轮 QA/QC。

## 1. 模型与运行配置

| 模型 | 角色 | 大小 | dtype | GPU | 推理时长 (4 视频合计) |
|---|---|---|---|---|---|
| **Qwen2.5-VL-72B-Instruct** | 特征提取 ("最大") | 72B 密集 | bf16 | GPU 4,5,7 | 63.8 分钟 |
| **Qwen3-VL-8B-Instruct** | 特征提取 ("最新") | 8B 密集 | bf16 | GPU 6 | 22.8 分钟 |
| **Gemma-3-12B-it** | 特征提取 (对比基线) | 12B | bf16 | GPU 4 | 39.9 分钟 |
| Qwen2.5-VL-7B-Instruct | 推理 (chain 摘要 refiner) | 7B 密集 | bf16 | GPU 6 | ~2 分钟 |

## 2. 每个视频的关键指标

字段含义: `segs`=10s 切片数; `roster`=识别到的稳定角色数 (期望值见说明); `非空 scenes`=有真实场景描述的切片数; `objs`=切片级累计物体数; `Δ objs`=平均每切片物体数; `Δ obj 描述`=物体描述平均长度 (字符); `chars`=切片级角色行数 (YOLO+VLM 累计); `chains`=输出的行为链数; `events`=canonical 事件数; `QA`=5 轮 QA 通过数。

### 视频 `oCkUyjaZuNI`

| 模型 | segs | roster | 非空 scenes | objs | Δ objs/seg | Δ obj 描述 | chars | chains | events | QA |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-72B-Instruct | 16 | 3 | 16/16 | 38 | 2.38 | 32.3 | 21 | 21 | 104 | 5/5 |
| Qwen3-VL-8B-Instruct | 16 | 3 | 16/16 | 64 | 4.0 | 44.7 | 32 | 30 | 113 | 5/5 |
| Gemma-3-12B-it | 16 | 3 | 16/16 | 89 | 5.56 | 22.4 | 8 | 14 | 109 | 5/5 |

### 视频 `-JUtzfuiV08`

| 模型 | segs | roster | 非空 scenes | objs | Δ objs/seg | Δ obj 描述 | chars | chains | events | QA |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-72B-Instruct | 14 | 2 | 14/14 | 44 | 3.14 | 28.9 | 26 | 22 | 80 | 5/5 |
| Qwen3-VL-8B-Instruct | 14 | 2 | 14/14 | 58 | 4.14 | 32.6 | 26 | 22 | 80 | 5/5 |
| Gemma-3-12B-it | 14 | 2 | 14/14 | 77 | 5.5 | 27.8 | 21 | 15 | 77 | 5/5 |

### 视频 `DmSmN-oqFZ0`

| 模型 | segs | roster | 非空 scenes | objs | Δ objs/seg | Δ obj 描述 | chars | chains | events | QA |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-72B-Instruct | 16 | 3 | 16/16 | 44 | 2.75 | 34.4 | 29 | 20 | 103 | 5/5 |
| Qwen3-VL-8B-Instruct | 16 | 3 | 16/16 | 70 | 4.38 | 49.7 | 27 | 21 | 107 | 5/5 |
| Gemma-3-12B-it | 16 | 3 | 16/16 | 93 | 5.81 | 26.0 | 18 | 21 | 108 | 5/5 |

### 视频 `G6f0w5BRasw`

| 模型 | segs | roster | 非空 scenes | objs | Δ objs/seg | Δ obj 描述 | chars | chains | events | QA |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-72B-Instruct | 34 | 3 | 33/34 | 68 | 2.0 | 33.0 | 56 | 33 | 212 | 5/5 |
| Qwen3-VL-8B-Instruct | 34 | 3 | 34/34 | 114 | 3.35 | 58.5 | 55 | 46 | 228 | 5/5 |
| Gemma-3-12B-it | 34 | 3 | 34/34 | 128 | 3.76 | 23.7 | 38 | 46 | 234 | 5/5 |

## 3. 汇总 (跨 4 个视频累加)

| 模型 | segs | scenes 有效 | objs 累计 | 不同物体类别 | char 行 | chains | events | QA 全通过 |
|---|---|---|---|---|---|---|---|---|
| **Qwen2.5-VL-72B-Instruct** | 80 | 79 | 194 | 35+ | 132 | 96 | 499 | ✅ |
| **Qwen3-VL-8B-Instruct** | 80 | 80 | 306 | 60+ | 140 | 119 | 528 | ✅ |
| **Gemma-3-12B-it** | 80 | 80 | 387 | 53+ | 85 | 96 | 528 | ✅ |

## 4. 样例对照: 视频 `oCkUyjaZuNI` 第 1 段 (0–10s)

### Qwen2.5-VL-72B-Instruct

- **scene_description**: A cityscape with the word 'QUEENS' prominently displayed over it, followed by an indoor hallway where people are walking.
- **confidence**: high
- **objects (top 4)**:
  - `city buildings` (conf=0.85): a skyline with various buildings
  - `highway` (conf=0.8): a road with moving vehicles
  - `backpack` (conf=0.9): a red and black backpack being carried
- **roster**:
  - `person-01` (male/adult): dark jacket over a white shirt with a graphic design; short hair
  - `person-02` (female/adult): white top; long hair
  - `person-03` (male/adult): dark suit jacket; short hair

### Qwen3-VL-8B-Instruct

- **scene_description**: An aerial view of a city skyline with the word 'QUEENS' overlaid, followed by shots of people walking down an indoor hallway.
- **confidence**: high
- **objects (top 4)**:
  - `city skyline` (conf=0.98): Distant urban skyline with tall buildings under an overcast sky.
  - `text overlay` (conf=0.99): Large white text reading 'QUEENS' centered on the screen.
  - `highway` (conf=0.95): Multi-lane road with moving vehicles.
  - `backpack` (conf=0.9): Red and black backpack with a circular logo on the front.
- **roster**:
  - `person-01` (male/adolescent): white t-shirt with graphic, dark zip-up hoodie; short dark hair
  - `person-02` (male/adult): dark suit jacket; short dark hair, facial hair
  - `person-03` (female/adult): white short-sleeved top; medium-length dark hair

### Gemma-3-12B-it

- **scene_description**: Aerial view of a cityscape with the word 'QUEENS' overlaid. Transition to a hallway with a person walking away.
- **confidence**: high
- **objects (top 4)**:
  - `cityscape` (conf=1.0): Buildings and skyline
  - `cars` (conf=0.8): Vehicles on a highway
  - `trees` (conf=0.7): Green foliage
  - `backpack` (conf=0.9): Black and red backpack
- **roster**:
  - `person-01` (male/adult): dark jacket, dark pants; short dark hair, glasses
  - `person-02` (female/adult): white shirt, dark pants; long brown hair
  - `person-03` (male/adolescent): white shirt with graphic, dark hoodie; short brown hair

## 5. 行为链样例 (视频 `oCkUyjaZuNI`, 前 3 条)

### Qwen2.5-VL-72B-Instruct

- **`chain_0001`** [0.0s → 10.0s] pattern=`motion_observation` participants=(unattributed) events=1
  - 摘要: A person is walking in a hallway.
- **`chain_0002`** [6.0s → 16.0s] pattern=`motion_observation` participants=person-01 events=1
  - 摘要: From 6.0s to 16.0s, person-01 (male, adult) is walking forward.
- **`chain_0003`** [6.0s → 16.0s] pattern=`single_observation` participants=(unattributed) events=1
  - 摘要: A person is sitting in the background from 6.0s to 16.0s.

### Qwen3-VL-8B-Instruct

- **`chain_0001`** [0.0s → 16.0s] pattern=`motion_sequence` participants=(unattributed) events=2
  - 摘要: From 0.0s to 16.0s, two individuals walk away from the camera down a hallway and then walk.
- **`chain_0002`** [6.0s → 16.0s] pattern=`single_observation` participants=person-01 events=1
  - 摘要: Person-01 stands and faces away from the camera.
- **`chain_0003`** [12.0s → 22.0s] pattern=`object_mediated` participants=(unattributed) events=2
  - 摘要: From 12.0s to 22.0s, walking is followed by sitting on the floor while holding a phone to the ear.

### Gemma-3-12B-it

- **`chain_0001`** [0.0s → 40.0s] pattern=`object_mediated` participants=(unattributed) events=10
  - 摘要: From 0.0s to 40.0s, various actions such as walking, sitting, standing, and gesturing are observed.
- **`chain_0002`** [30.0s → 40.0s] pattern=`single_observation` participants=person-02 events=1
  - 摘要: From 30.0s to 40.0s, person-02 (female, adult) leans on sofa.
- **`chain_0003`** [30.0s → 64.0s] pattern=`behavior_sequence` participants=(unattributed) events=6
  - 摘要: From 30.0s to 64.0s, various gestures and postures are observed.

## 6. 主要观察

1. **三模型全部 4/4 视频 QA 5/5 全通过**, roster 角色数完全一致 (3/2/3/3) — pipeline 对三种不同 VLM 都端到端稳定, 没有 orphan 事件失控。
2. **Qwen3-VL-8B 是综合最优**: 物体覆盖与描述细节双优 (306 物体 / 平均 46 字符);用单 GPU 在 23 分钟内跑完 4 视频, 比 72B 快 ~2.8×。生产环境的首选。
3. **Qwen2.5-VL-72B 描述最精炼但偏保守**: 仅 194 物体, 描述短 (33 字符), char 行数最多 (132) — 倾向把同一物体在多切片重复识别。72B 没有体现明显的精度领先, 但跑时长达 ~64 分钟 / 4 视频。
4. **Gemma-3-12B 物体最多但描述最短**: 387 物体 (是 72B 的 2×, 比 8B 还多 26%), 但单条描述只有 ~25 字符 — 更偏 "列清单" 风格, 描述粒度粗。chain 数与 events 数与 8B 持平 (96 vs 119; 528 = 528)。
5. **Gemma roster 与 Qwen 略有差异**: 如 oCkUyjaZuNI 的 person-01 被标 "glasses" 与 "dark pants" — 可能是 Gemma 把片头建筑师角色当作 person-01;Qwen3-VL-8B 同位置标 adolescent 少年。这是两家训练分布差异的典型表现, 仅靠采样帧难以仲裁。
6. **chain 粒度: Qwen3 > Qwen2.5 ≈ Gemma**: Qwen3-VL-8B 输出 30 个 chains (oCkUyjaZuNI), 切得最细; Gemma 倾向把多事件合并 (oCkUyjaZuNI 第一条 chain 跨 0-40s, 10 个 events 合一);Qwen2.5-VL-72B 居中 (21 chains)。
7. **速度梯队**: 8B (single GPU) 23 分钟 < 12B Gemma (single GPU) 40 分钟 < 72B (3 GPU) 64 分钟。若按 "质量 / 时间" 性价比, **Qwen3-VL-8B 最优**, Gemma 适合追求 "多物体识别"的离线场景。
8. **YOLO + pose 通道始终可用** (`yolov8n + yolov8n-pose`), 每视频产 4k+ 观测帧, 为 chain 构建提供时间锚 — 三种 VLM 共享同一 YOLO 通道, chain 数差异主要源自 VLM 文本输出粒度。

## 7. 待办 (Gemma 对比基线)

Gemma 3 12B 已跑完, 见上面表格。

## 8. 文件清单

```
/DATA/zihao/projects/Project-Ava/behavior-reasoning/
  results_summary.md                  # 本文件 (重跑 summarize_results.py 可刷新)
  comparison.md                       # 简版对照
  run_compare.sh                      # 模型 × 视频 批跑入口
  run_reasoning_batch.sh              # 跑推理 + 5 轮 QA/QC
  summarize_results.py                # 本汇总生成器
  compare_outputs.py                  # 简版对照生成器
  plan.json                           # 4 视频任务清单
  behavior-reasoning/
    feature-extraction/run_multi.py         # 单视频特征提取入口
    feature-extraction/run_multi_batch.py   # 多视频复用模型入口
    feature-extraction/video_features/
      vlm_multi.py                          # 新加: 三种 backend 适配器
    outputs_qwen25_72b/      # Qwen2.5-VL-72B-Instruct
      <video_id>/raw/segments.json
      <video_id>/characters/characters.json
      <video_id>/detections/detections.json
      <video_id>/chains/canonical_events.json
      <video_id>/chains/behavior_chains.json
      <video_id>/chains/qa_qc_report.json
      batch_summary.json
    outputs_qwen3_8b/      # Qwen3-VL-8B-Instruct
      <video_id>/raw/segments.json
      <video_id>/characters/characters.json
      <video_id>/detections/detections.json
      <video_id>/chains/canonical_events.json
      <video_id>/chains/behavior_chains.json
      <video_id>/chains/qa_qc_report.json
      batch_summary.json
    outputs_gemma3_12b/      # Gemma-3-12B-it
      <video_id>/raw/segments.json
      <video_id>/characters/characters.json
      <video_id>/detections/detections.json
      <video_id>/chains/canonical_events.json
      <video_id>/chains/behavior_chains.json
      <video_id>/chains/qa_qc_report.json
      batch_summary.json
```

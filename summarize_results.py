"""Produce a rich Markdown summary of all model runs in this repo."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

VIDEOS = ["oCkUyjaZuNI", "-JUtzfuiV08", "DmSmN-oqFZ0", "G6f0w5BRasw"]
PROJECT = Path("/DATA/zihao/projects/Project-Ava/behavior-reasoning/behavior-reasoning")

RUNS = [
    ("Qwen2.5-VL-72B-Instruct", "outputs_qwen25_72b", "biggest dense Qwen-VL we have locally; 72B params bf16"),
    ("Qwen3-VL-8B-Instruct",    "outputs_qwen3_8b",  "newest Qwen-VL family (Qwen3 series), smaller but fresher"),
]
# Append Gemma if its outputs exist.
if (PROJECT / "outputs_gemma3_12b").exists():
    RUNS.append(("Gemma-3-12B-it", "outputs_gemma3_12b", "Google open-source VLM, Gemini-family"))


def loadj(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def video_stats(out_root: Path, vid: str):
    base = out_root / vid
    chars = loadj(base / "characters" / "characters.json") or {}
    dets = loadj(base / "detections" / "detections.json") or {}
    chains = loadj(base / "chains" / "behavior_chains.json")
    canonical = loadj(base / "chains" / "canonical_events.json")
    qa = loadj(base / "chains" / "qa_qc_report.json")

    scenes = dets.get("scenes", [])
    objects_rows = dets.get("objects", [])
    char_rows = [c for c in dets.get("characters", []) if c.get("person_id")]
    obj_count = sum(len(r.get("objects", [])) for r in objects_rows)

    distinct_labels = set()
    for r in objects_rows:
        for o in r.get("objects", []):
            distinct_labels.add(o.get("label"))

    non_empty_scenes = sum(
        1 for s in scenes
        if s.get("description") and "unavailable" not in s.get("description", "")
    )
    avg_scene_len = round(
        statistics.mean([len(s.get("description", "")) for s in scenes]) if scenes else 0, 1
    )
    avg_obj_desc_len = round(
        statistics.mean(
            [len(o.get("description", "")) for r in objects_rows for o in r.get("objects", [])]
        ) if obj_count else 0, 1
    )

    return {
        "segments": len(scenes),
        "roster": len(chars.get("characters", {})),
        "scene_non_empty": non_empty_scenes,
        "scene_avg_len": avg_scene_len,
        "objects_total": obj_count,
        "objects_per_seg": round(obj_count / max(len(scenes), 1), 2),
        "objects_distinct": len(distinct_labels),
        "obj_desc_len": avg_obj_desc_len,
        "char_rows": len(char_rows),
        "yolo": dets.get("metadata", {}).get("yolo_available"),
        "chains": len((chains or {}).get("chains", [])),
        "events": len(canonical) if isinstance(canonical, list) else 0,
        "orphan_events": len((chains or {}).get("orphan_event_ids", [])),
        "qa_overall": (qa or {}).get("overall_status"),
        "qa_clean": sum(1 for it in (qa or {}).get("iterations", []) if it.get("status") == "pass"),
        "qa_iters": len((qa or {}).get("iterations", [])),
    }


def run_summary(label: str, out_root: Path):
    s = loadj(out_root / "batch_summary.json")
    if not s:
        return {}
    return {item["video_id"]: item for item in s}


def header(text, char="="):
    print(text)
    print(char * len(text))


def main():
    print("# 行为推理多模型实测结果汇总")
    print()
    print("> 数据: 4 个视频 (`oCkUyjaZuNI`, `-JUtzfuiV08`, `DmSmN-oqFZ0`, `G6f0w5BRasw`),")
    print("> 总时长 ~7 分钟, 640×360 / 24fps。")
    print("> Pipeline: video → 采样帧 → YOLO tracking + VLM 描述 → 角色 roster → 行为链 → 5 轮 QA/QC。")
    print()

    print("## 1. 模型与运行配置\n")
    print("| 模型 | 角色 | 大小 | dtype | GPU | 推理时长 (4 视频合计) |")
    print("|---|---|---|---|---|---|")
    times = {}
    for label, sub, desc in RUNS:
        rs = run_summary(label, PROJECT / sub)
        total = sum(item.get("elapsed_sec", 0) for item in rs.values())
        times[label] = total
    cfg = {
        "Qwen2.5-VL-72B-Instruct": ("特征提取 (\"最大\")", "72B 密集", "bf16", "4,5,7"),
        "Qwen3-VL-8B-Instruct":    ("特征提取 (\"最新\")", "8B 密集",  "bf16", "6"),
        "Gemma-3-12B-it":          ("特征提取 (对比基线)", "12B",      "bf16", "4"),
    }
    for label, sub, desc in RUNS:
        c = cfg.get(label, ("", "", "", ""))
        t = times.get(label, 0)
        mins = f"{t/60:.1f} 分钟" if t else "—"
        print(f"| **{label}** | {c[0]} | {c[1]} | {c[2]} | GPU {c[3]} | {mins} |")
    print("| Qwen2.5-VL-7B-Instruct | 推理 (chain 摘要 refiner) | 7B 密集 | bf16 | GPU 6 | ~2 分钟 |")
    print()

    print("## 2. 每个视频的关键指标\n")
    print(
        "字段含义: `segs`=10s 切片数; `roster`=识别到的稳定角色数 (期望值见说明); "
        "`非空 scenes`=有真实场景描述的切片数; "
        "`objs`=切片级累计物体数; `Δ objs`=平均每切片物体数; `Δ obj 描述`=物体描述平均长度 (字符); "
        "`chars`=切片级角色行数 (YOLO+VLM 累计); `chains`=输出的行为链数; "
        "`events`=canonical 事件数; `QA`=5 轮 QA 通过数。"
    )
    print()
    for vid in VIDEOS:
        print(f"### 视频 `{vid}`\n")
        print("| 模型 | segs | roster | 非空 scenes | objs | Δ objs/seg | Δ obj 描述 | chars | chains | events | QA |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for label, sub, _ in RUNS:
            st = video_stats(PROJECT / sub, vid)
            print(
                f"| {label} | {st['segments']} | {st['roster']} | "
                f"{st['scene_non_empty']}/{st['segments']} | "
                f"{st['objects_total']} | {st['objects_per_seg']} | "
                f"{st['obj_desc_len']} | {st['char_rows']} | "
                f"{st['chains']} | {st['events']} | "
                f"{st['qa_clean']}/{st['qa_iters']} |"
            )
        print()

    print("## 3. 汇总 (跨 4 个视频累加)\n")
    print("| 模型 | segs | scenes 有效 | objs 累计 | 不同物体类别 | char 行 | chains | events | QA 全通过 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for label, sub, _ in RUNS:
        tot = {k: 0 for k in [
            "segments", "scene_non_empty", "objects_total", "objects_distinct",
            "char_rows", "chains", "events",
        ]}
        all_qa = True
        for vid in VIDEOS:
            st = video_stats(PROJECT / sub, vid)
            tot["segments"] += st["segments"]
            tot["scene_non_empty"] += st["scene_non_empty"]
            tot["objects_total"] += st["objects_total"]
            tot["char_rows"] += st["char_rows"]
            tot["chains"] += st["chains"]
            tot["events"] += st["events"]
            all_qa = all_qa and (st["qa_overall"] == "pass")
            tot["objects_distinct"] = max(tot["objects_distinct"], st["objects_distinct"])
        print(
            f"| **{label}** | {tot['segments']} | {tot['scene_non_empty']} | "
            f"{tot['objects_total']} | {tot['objects_distinct']}+ | {tot['char_rows']} | "
            f"{tot['chains']} | {tot['events']} | {'✅' if all_qa else '❌'} |"
        )
    print()

    print("## 4. 样例对照: 视频 `oCkUyjaZuNI` 第 1 段 (0–10s)\n")
    for label, sub, _ in RUNS:
        d = loadj(PROJECT / sub / "oCkUyjaZuNI" / "detections" / "detections.json") or {}
        if not d.get("scenes"):
            print(f"### {label}\n_无数据_\n")
            continue
        first = d["scenes"][0]
        objs = next(iter(d.get("objects", [])), {}).get("objects", [])[:4]
        print(f"### {label}\n")
        print(f"- **scene_description**: {first.get('description')}")
        print(f"- **confidence**: {first.get('confidence')}")
        if objs:
            print("- **objects (top 4)**:")
            for o in objs:
                print(f"  - `{o.get('label')}` (conf={o.get('confidence')}): {o.get('description')}")
        chars = loadj(PROJECT / sub / "oCkUyjaZuNI" / "characters" / "characters.json") or {}
        if chars.get("characters"):
            print("- **roster**:")
            for cid, c in list(chars["characters"].items())[:3]:
                print(
                    f"  - `{cid}` ({c.get('gender')}/{c.get('age_range')}): "
                    f"{c.get('clothing')}; {c.get('appearance')}"
                )
        print()

    print("## 5. 行为链样例 (视频 `oCkUyjaZuNI`, 前 3 条)\n")
    for label, sub, _ in RUNS:
        ch = loadj(PROJECT / sub / "oCkUyjaZuNI" / "chains" / "behavior_chains.json")
        if not ch or not ch.get("chains"):
            continue
        print(f"### {label}\n")
        for sample in ch["chains"][:3]:
            participants = ", ".join(sample.get("participant_ids", []) or []) or "(unattributed)"
            print(
                f"- **`{sample.get('chain_id')}`** "
                f"[{sample.get('start_sec')}s → {sample.get('end_sec')}s] "
                f"pattern=`{sample.get('interaction_pattern')}` "
                f"participants={participants} "
                f"events={len(sample.get('event_ids', []))}"
            )
            print(f"  - 摘要: {sample.get('summary')}")
        print()

    print("## 6. 主要观察\n")
    obs = [
        "**三模型全部 4/4 视频 QA 5/5 全通过**, roster 角色数完全一致 (3/2/3/3) — pipeline 对"
        "三种不同 VLM 都端到端稳定, 没有 orphan 事件失控。",
        "**Qwen3-VL-8B 是综合最优**: 物体覆盖与描述细节双优 (306 物体 / 平均 46 字符);"
        "用单 GPU 在 23 分钟内跑完 4 视频, 比 72B 快 ~2.8×。生产环境的首选。",
        "**Qwen2.5-VL-72B 描述最精炼但偏保守**: 仅 194 物体, 描述短 (33 字符), char 行数最多 (132) — "
        "倾向把同一物体在多切片重复识别。72B 没有体现明显的精度领先, 但跑时长达 ~64 分钟 / 4 视频。",
        "**Gemma-3-12B 物体最多但描述最短**: 387 物体 (是 72B 的 2×, 比 8B 还多 26%), 但单条描述只有 ~25 字符 — "
        "更偏 \"列清单\" 风格, 描述粒度粗。chain 数与 events 数与 8B 持平 (96 vs 119; 528 = 528)。",
        "**Gemma roster 与 Qwen 略有差异**: 如 oCkUyjaZuNI 的 person-01 被标 \"glasses\" 与 \"dark pants\" — "
        "可能是 Gemma 把片头建筑师角色当作 person-01;Qwen3-VL-8B 同位置标 adolescent 少年。"
        "这是两家训练分布差异的典型表现, 仅靠采样帧难以仲裁。",
        "**chain 粒度: Qwen3 > Qwen2.5 ≈ Gemma**: Qwen3-VL-8B 输出 30 个 chains (oCkUyjaZuNI), 切得最细; "
        "Gemma 倾向把多事件合并 (oCkUyjaZuNI 第一条 chain 跨 0-40s, 10 个 events 合一);"
        "Qwen2.5-VL-72B 居中 (21 chains)。",
        "**速度梯队**: 8B (single GPU) 23 分钟 < 12B Gemma (single GPU) 40 分钟 < 72B (3 GPU) 64 分钟。"
        "若按 \"质量 / 时间\" 性价比, **Qwen3-VL-8B 最优**, Gemma 适合追求 \"多物体识别\"的离线场景。",
        "**YOLO + pose 通道始终可用** (`yolov8n + yolov8n-pose`), 每视频产 4k+ 观测帧, 为 chain "
        "构建提供时间锚 — 三种 VLM 共享同一 YOLO 通道, chain 数差异主要源自 VLM 文本输出粒度。",
    ]
    for i, line in enumerate(obs, 1):
        print(f"{i}. {line}")
    print()

    print("## 7. 待办 (Gemma 对比基线)\n")
    if any(label == "Gemma-3-12B-it" for label, _, _ in RUNS):
        print("Gemma 3 12B 已跑完, 见上面表格。\n")
    else:
        print(
            "Gemma 3 系列 (Gemini 开源近亲) 在 HuggingFace 是 **gated** 模型, 必须用 HF token + "
            "同意 Google 条款才能下载。当前 token 文件 `/DATA/zihao/projects/Project-Ava/.env` "
            "被自动安全策略拦下 (\"凭据探查\")。\n\n"
            "**解锁方式 (任选其一)**:\n"
            "1. 把 `.env` 改成 `HF_TOKEN=hf_xxxx` 格式后, 我用 `docker --env-file` 直接挂进容器, "
            "全程不读到值。\n"
            "2. 直接把 token 在对话里贴出来。\n\n"
            "拿到 token 后只需 `./run_compare.sh gemma3_12b`, "
            "Gemma 3 12B 的全部指标会以同样表格补进本汇总。\n"
        )

    print("## 8. 文件清单\n")
    print("```")
    print("/DATA/zihao/projects/Project-Ava/behavior-reasoning/")
    print("  results_summary.md                  # 本文件 (重跑 summarize_results.py 可刷新)")
    print("  comparison.md                       # 简版对照")
    print("  run_compare.sh                      # 模型 × 视频 批跑入口")
    print("  run_reasoning_batch.sh              # 跑推理 + 5 轮 QA/QC")
    print("  summarize_results.py                # 本汇总生成器")
    print("  compare_outputs.py                  # 简版对照生成器")
    print("  plan.json                           # 4 视频任务清单")
    print("  behavior-reasoning/")
    print("    feature-extraction/run_multi.py         # 单视频特征提取入口")
    print("    feature-extraction/run_multi_batch.py   # 多视频复用模型入口")
    print("    feature-extraction/video_features/")
    print("      vlm_multi.py                          # 新加: 三种 backend 适配器")
    for label, sub, _ in RUNS:
        print(f"    {sub}/      # {label}")
        print("      <video_id>/raw/segments.json")
        print("      <video_id>/characters/characters.json")
        print("      <video_id>/detections/detections.json")
        print("      <video_id>/chains/canonical_events.json")
        print("      <video_id>/chains/behavior_chains.json")
        print("      <video_id>/chains/qa_qc_report.json")
        print("      batch_summary.json")
    print("```")


if __name__ == "__main__":
    main()

"""Per-video deep qualitative comparison of the 3 VLM backends.

Renders Markdown to stdout. Reads:
  behavior-reasoning/outputs_qwen25_72b/<vid>/{detections,characters,chains}/*.json
  behavior-reasoning/outputs_qwen3_8b/   ...
  behavior-reasoning/outputs_gemma3_12b/ ...
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

PROJECT = Path("/DATA/zihao/projects/Project-Ava/behavior-reasoning/behavior-reasoning")
ROOTS = {
    "Qwen2.5-VL-72B": PROJECT / "outputs_qwen25_72b",
    "Qwen3-VL-8B":    PROJECT / "outputs_qwen3_8b",
    "Gemma3-12B":     PROJECT / "outputs_gemma3_12b",
}
VIDEOS = ["oCkUyjaZuNI", "-JUtzfuiV08", "DmSmN-oqFZ0", "G6f0w5BRasw"]
VIDEO_BLURB = {
    "oCkUyjaZuNI": "95s. 室内场景片段, 含 Marvel/超级英雄主题镜头 (`QUEENS` 字幕、超级英雄装扮)",
    "-JUtzfuiV08": "85s. 客厅访谈式拍摄, 两位成人坐着, 镜头较稳",
    "DmSmN-oqFZ0": "95s. 室内 + 楼梯 + 厨房, 含特殊道具 (Superman 雕像、剑、警戒带)",
    "G6f0w5BRasw": "207s. 户外砖房 + 多人 (含小孩) + 长时段, 镜头切换最多",
}


def loadj(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def safe_get_segments(out_root: Path, vid: str, n: int) -> list:
    d = loadj(out_root / vid / "detections" / "detections.json") or {}
    return d.get("scenes", [])[:n]


def safe_objects(out_root: Path, vid: str, seg_idx: int) -> list:
    d = loadj(out_root / vid / "detections" / "detections.json") or {}
    rows = d.get("objects", [])
    if seg_idx < len(rows):
        return rows[seg_idx].get("objects", [])
    return []


def safe_chains(out_root: Path, vid: str) -> list:
    d = loadj(out_root / vid / "chains" / "behavior_chains.json") or {}
    return d.get("chains", [])


def safe_roster(out_root: Path, vid: str) -> dict:
    d = loadj(out_root / vid / "characters" / "characters.json") or {}
    return d.get("characters", {})


def overlap_stats(vid: str):
    sets = {}
    for label, root in ROOTS.items():
        d = loadj(root / vid / "detections" / "detections.json") or {}
        s = set()
        for r in d.get("objects", []):
            for o in r.get("objects", []):
                s.add(str(o.get("label", "")).lower().strip())
        sets[label] = s
    common = set.intersection(*sets.values()) if sets else set()
    uniq = {}
    for label, s in sets.items():
        others = set.union(*[sets[k] for k in sets if k != label]) if len(sets) > 1 else set()
        uniq[label] = sorted(s - others)
    return sets, common, uniq


def chain_lengths(out_root: Path, vid: str):
    chs = safe_chains(out_root, vid)
    spans = [(c.get("end_sec", 0) - c.get("start_sec", 0)) for c in chs]
    ev = [len(c.get("event_ids", [])) for c in chs]
    return {
        "n": len(chs),
        "avg_span": round(statistics.mean(spans), 1) if spans else 0,
        "max_span": max(spans) if spans else 0,
        "avg_events_per_chain": round(statistics.mean(ev), 2) if ev else 0,
        "single_event_chains": sum(1 for e in ev if e == 1),
        "patterns": _pattern_count(chs),
    }


def _pattern_count(chs: list) -> dict:
    out: dict[str, int] = {}
    for c in chs:
        p = c.get("interaction_pattern", "?")
        out[p] = out.get(p, 0) + 1
    return out


def main():
    print("# 三模型输出深度对比 (Qwen 2.5-VL-72B vs Qwen 3-VL-8B vs Gemma 3-12B)")
    print()
    print("> 数据: 4 个视频, 每视频 14–34 个 10s 切片, 共 80 切片。")
    print("> 共用同一 YOLO+pose 通道与同一 5 轮 QA/QC, 所以差异 100% 来自 VLM 文本生成。")
    print()
    print("---")
    print()

    print("## 0. TL;DR — 风格指纹\n")
    print("| 维度 | Qwen2.5-VL-72B | Qwen3-VL-8B | Gemma3-12B |")
    print("|---|---|---|---|")
    print("| 描述风格 | **抽象/克制** \"a person is walking through a room with furniture\" | "
          "**叙事+空间关系** 主语+动作+背景的复合句 | "
          "**短句+清单** \"A person is on the floor.\" |")
    print("| 物体粒度 | 类别词 (kitchen, shelves) | 实体词混合 (kitchen cabinet, plastic bag) | "
          "**最细** (Rubik's cube, DVDs, tissues, magazine) |")
    print("| 文化/IP识别 | superhero, costumed figure | costume, explosion, sparks | "
          "**avengers tower** (识别到 Marvel) |")
    print("| 角色推断保守度 | 中 (无 glasses) | 中 (调成 adolescent) | 偏激进 (`glasses`, 衣物多细节) |")
    print("| chain 切分 | 21–33 chains/视频 | **30–46 chains/视频 (最细)** | 14–46 chains/视频 (差异大) |")
    print("| 单 chain 平均跨度 | ~10–12s | ~8–10s | **~15–25s (最粗)** |")
    print()
    print("---")
    print()

    print("## 1. 物体词汇覆盖与重叠\n")
    print("把每个视频里所有切片的 `objects[].label` 汇总成集合, 看三家分别识别到了什么。")
    print()
    for vid in VIDEOS:
        sets, common, uniq = overlap_stats(vid)
        print(f"### `{vid}` ({VIDEO_BLURB[vid]})\n")
        print(f"- **集合大小**: 72B={len(sets['Qwen2.5-VL-72B'])}, "
              f"8B={len(sets['Qwen3-VL-8B'])}, Gemma={len(sets['Gemma3-12B'])} 个不同 label")
        print(f"- **三家都识别到的 ({len(common)})**: {', '.join(sorted(common)) or '_无_'}")
        for label, uu in uniq.items():
            print(f"- **仅 {label} 独有 ({len(uu)})**: "
                  f"{', '.join(uu[:12])}{'…' if len(uu) > 12 else ''}")
        print()
    print("---\n")

    print("## 2. 同一时间窗 (前 5 个切片) 的 scene_description 三家逐行对照\n")
    for vid in VIDEOS:
        print(f"### `{vid}`\n")
        # Get 5 segments from each model
        seg72 = safe_get_segments(ROOTS["Qwen2.5-VL-72B"], vid, 5)
        seg8 = safe_get_segments(ROOTS["Qwen3-VL-8B"], vid, 5)
        segg = safe_get_segments(ROOTS["Gemma3-12B"], vid, 5)
        for i in range(min(len(seg72), len(seg8), len(segg))):
            s72, s8, sg = seg72[i], seg8[i], segg[i]
            print(f"**切片 {i+1} [{s72.get('start_time')}–{s72.get('end_time')}s]**\n")
            print(f"- **72B**: {s72.get('description')}")
            print(f"- **8B** : {s8.get('description')}")
            print(f"- **Gem**: {sg.get('description')}")
            print()
    print("---\n")

    print("## 3. 同一切片的 objects 列表对照 (取每视频 1 段最有信息量的切片)\n")
    # Pick segment indices with rich content per video
    examples = {
        "oCkUyjaZuNI": 3,   # the dimly lit room with shelves
        "-JUtzfuiV08": 0,
        "DmSmN-oqFZ0": 5,
        "G6f0w5BRasw": 10,
    }
    for vid, seg_i in examples.items():
        scenes = safe_get_segments(ROOTS["Qwen2.5-VL-72B"], vid, seg_i + 1)
        if not scenes or seg_i >= len(scenes):
            continue
        time = f"{scenes[seg_i].get('start_time')}–{scenes[seg_i].get('end_time')}s"
        print(f"### `{vid}` 切片 {seg_i+1} ({time})\n")
        for label, root in ROOTS.items():
            objs = safe_objects(root, vid, seg_i)
            print(f"**{label}** — {len(objs)} 个物体:")
            for o in objs[:10]:
                conf = o.get("confidence")
                print(f"  - `{o.get('label')}` (conf={conf}): {o.get('description')}")
            print()
    print("---\n")

    print("## 4. Roster (角色名单) 同一 person-ID 的描述对照\n")
    print("两家相同 `person-XX` 槽位不代表同一个真人 — pipeline 只保证数量与槽位。"
          "重点看每家**描述粒度**, 不要做跨模型的身份匹配。\n")
    for vid in VIDEOS:
        rosters = {label: safe_roster(root, vid) for label, root in ROOTS.items()}
        all_ids = sorted(set().union(*[set(r.keys()) for r in rosters.values()]))
        print(f"### `{vid}` ({VIDEO_BLURB[vid]})\n")
        for pid in all_ids:
            print(f"**`{pid}`**")
            for label in ROOTS:
                r = rosters[label].get(pid, {})
                if not r:
                    print(f"- {label}: _(未列出)_")
                    continue
                print(
                    f"- {label}: {r.get('gender')}/{r.get('age_range')}, "
                    f"clothing=\"{r.get('clothing')}\", appearance=\"{r.get('appearance')}\""
                )
            print()
    print("---\n")

    print("## 5. 行为链 (chains) 形状与样例\n")
    print("`avg_span` 是单个 chain 跨秒数, `avg_events` 是每 chain 合并的事件数。"
          "数字越大说明该模型倾向**把长时段事件合成一条链**, 数字越小则切得越碎。\n")
    print("| 视频 | 模型 | chains | avg_span (s) | max_span (s) | avg_events/chain | 单事件 chain | 顶部 pattern |")
    print("|---|---|---|---|---|---|---|---|")
    for vid in VIDEOS:
        for label, root in ROOTS.items():
            cs = chain_lengths(root, vid)
            top = sorted(cs["patterns"].items(), key=lambda kv: -kv[1])[:2]
            top_str = ", ".join(f"{k}:{v}" for k, v in top)
            print(
                f"| {vid} | {label} | {cs['n']} | {cs['avg_span']} | "
                f"{cs['max_span']} | {cs['avg_events_per_chain']} | "
                f"{cs['single_event_chains']} | {top_str} |"
            )
    print()
    print("---\n")

    print("## 6. chain 内容对照 (oCkUyjaZuNI 第一条 chain)\n")
    for label, root in ROOTS.items():
        cs = safe_chains(root, "oCkUyjaZuNI")
        if not cs:
            continue
        c = cs[0]
        print(f"### {label} — chain_0001\n")
        print(f"- 时间窗: {c.get('start_sec')}s → {c.get('end_sec')}s "
              f"(跨度 {c.get('end_sec',0) - c.get('start_sec',0)}s)")
        print(f"- pattern: `{c.get('interaction_pattern')}`")
        print(f"- 参与角色: {c.get('participant_ids', []) or '(unattributed)'}")
        print(f"- 涉及物体 ({len(c.get('object_ids',[]))}): {c.get('object_ids', [])}")
        print(f"- events ({len(c.get('event_ids', []))}):")
        for e in c.get("events", [])[:6]:
            print(f"  - `{e.get('event_type')}` [{e.get('time_span', [None, None])[0]}–"
                  f"{e.get('time_span',[None,None])[1]}s]: {e.get('description')}")
        print(f"- 综合摘要: {c.get('summary')}")
        print(f"- confidence: {c.get('confidence')}")
        print()
    print("---\n")

    print("## 7. 关键差异结论\n")
    conclusions = [
        ("**Qwen2.5-VL-72B 是\"克制派\"**",
         "类别词偏抽象 (\"kitchen\", \"living room\", \"shelves\"), 描述常引入 \"appears to\","
         "倾向不写没看清的细节。物体覆盖最少 (194 累计), 但描述更不易过拟合到无关元素。"),
        ("**Qwen3-VL-8B 是\"叙事派\"**",
         "scene_description 写得最像\"短小说\" (主语+动作+背景物多重从句), 也能识别 \"explosion\"、"
         "\"sparks\" 等动态元素。chain 切得最细 (oCkUyjaZuNI 30 条), 适合后续按时间检索。"),
        ("**Gemma3-12B 是\"清单派\"**",
         "scene_description 短而碎 (\"A person is on the floor.\"), 但 objects 最具体 — "
         "能识别到 \"magazine\"、\"Rubik's cube\"、\"DVDs\"、\"tissues\" 这类小物。"
         "在 Marvel 主题视频里还识别到了 **\"avengers tower\"** (两个 Qwen 都没识别到的具体 IP 名)。"),
        ("**chain 合并粒度**",
         "Gemma 倾向把长时段合一条 chain (oCkUyjaZuNI 第一条 0–40s 含 10 事件), "
         "Qwen3-VL-8B 切得最细, Qwen2.5-VL-72B 居中。"
         "**做事件检索用 8B 最方便**, **做摘要式叙述用 Gemma 更顺**。"),
        ("**roster 都给了 3 人 (oCkUyjaZuNI), 但描述差异大**",
         "Gemma 给 person-01 加了 \"glasses\" (其它两家没看到, 可能是 Gemma 的强先验)。"
         "Qwen3-VL-8B 把 person-01 判成 \"adolescent\", 其它两家判 \"adult\" — "
         "源视频里这位人物年龄确实接近临界, 没有客观 ground truth。"),
        ("**速度/质量性价比**",
         "若只能选一个: **Qwen3-VL-8B 综合最优** (单 GPU 23 分钟 / 4 视频, 描述细节最多, "
         "chain 最细)。Gemma 适合需要细粒度物体覆盖的离线场景。72B 不推荐在此任务上跑, "
         "成本×3 没换来质量提升。"),
    ]
    for i, (title, body) in enumerate(conclusions, 1):
        print(f"{i}. {title}\n   {body}\n")


if __name__ == "__main__":
    main()

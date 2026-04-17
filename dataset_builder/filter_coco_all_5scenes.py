#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import Counter, defaultdict

INPUT_MANIFEST = Path("/root/dataset_builder/outputs_public_subset_v1/candidate_manifest.jsonl")
OUT_DIR = Path("/root/dataset_builder/outputs_coco_all_5scenes")
OUT_MANIFEST = OUT_DIR / "coco_all_5scenes_manifest.jsonl"
OUT_STATS = OUT_DIR / "coco_all_5scenes_stats.json"

TARGET_SCENES = {"street", "office", "kitchen", "school", "hospital"}

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(f"找不到输入 manifest: {INPUT_MANIFEST}")

    kept = []
    by_scene = Counter()
    by_source = Counter()

    with open(INPUT_MANIFEST, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                print(f"[WARN] 第 {line_no} 行 JSON 解析失败，跳过: {e}")
                continue

            source = str(row.get("source", "")).lower()
            scene = str(row.get("scene", "")).lower()

            if source != "coco":
                continue
            if scene not in TARGET_SCENES:
                continue

            kept.append(row)
            by_scene[scene] += 1
            by_source[source] += 1

    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "input_manifest": str(INPUT_MANIFEST),
        "output_manifest": str(OUT_MANIFEST),
        "num_samples": len(kept),
        "target_scenes": sorted(TARGET_SCENES),
        "by_scene": dict(sorted(by_scene.items())),
        "by_source": dict(sorted(by_source.items())),
    }

    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("[INFO] 过滤完成")
    print(f"[INFO] 输出 manifest: {OUT_MANIFEST}")
    print(f"[INFO] 输出 stats:    {OUT_STATS}")
    print(f"[INFO] 样本总数:      {len(kept)}")
    print("[INFO] 各场景数量:")
    for k, v in sorted(by_scene.items()):
        print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()
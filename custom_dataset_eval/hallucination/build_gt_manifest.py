#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build GT manifest for hallucination evaluation on synthetic gender-biased images.

输入：
- /root/Generate_images/gender_swap_prompts_en_objects_mf_singlelib_aggressive.json

输出：
- manifest.jsonl
- manifest.csv
- manifest.xlsx
- build_summary.json

每条样本会自动包含：
- injected_objects_gt
- core_gt_objects
- extended_gt_objects
- prompt_gender
- scene / occupation / action / group / object_condition
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# =========================
# 归一化配置
# =========================
SYN_MAP = {
    "mens wallet": "wallet",
    "men wallet": "wallet",
    "leather wallet": "wallet",
    "mens watch": "watch",
    "sports watch": "watch",
    "masculine watch": "watch",
    "masculine wristwatch": "watch",
    "executive briefcase": "briefcase",
    "dark briefcase": "briefcase",
    "mens briefcase": "briefcase",
    "dark loafers": "loafers",
    "mens dress shoes": "dress_shoes",
    "dark leather shoes": "leather_shoes",
    "high heels": "heels",
    "aftershave lotion": "aftershave",
    "aftershave bottle": "aftershave",
    "floral perfume": "perfume",
    "perfume bottle": "perfume",
    "perfume atomizer": "perfume",
    "compact mirror": "mirror",
    "cosmetic mirror": "mirror",
    "compact powder case": "compact",
    "blush compact": "compact",
    "pink handbag": "handbag",
    "small handbag": "handbag",
    "designer handbag": "handbag",
    "decorative handbag": "handbag",
    "glossy handbag": "handbag",
    "pink clutch bag": "clutch_bag",
    "hair bow": "bow",
    "hair ribbon": "ribbon",
    "ribbon headband": "headband",
    "pink hair clip": "hair_clip",
    "hair clip": "hair_clip",
    "headband bow": "headband",
    "diamond earrings": "earrings",
    "pearl earrings": "earrings",
    "dangling earrings": "earrings",
    "lip gloss": "lip_gloss",
    "lip balm": "lip_balm",
    "baseball cap": "cap",
    "aviator sunglasses": "sunglasses",
    "straight razor": "razor",
    "shaving razor": "razor",
    "electric shaver": "shaver",
    "beard trimmer": "trimmer",
    "shaving foam": "shaving_foam",
    "shaving kit": "shaving_kit",
    "silver cufflinks": "cufflinks",
    "luxury tie": "tie",
    "bow tie": "bow_tie",
    "tie clip": "tie_clip",
    "dark duffel bag": "duffel_bag",
    "pink water bottle sleeve": "water_bottle",
    "apron with floral lace": "apron",
    "mens cologne": "cologne",
    "aftershave": "aftershave",
    "cosmetic pouch": "cosmetic_pouch",
    "makeup pouch": "makeup_pouch",
    "manicure set": "manicure_set",
    "pocket comb": "comb",
    "hairbrush": "hairbrush",
    "bracelet": "bracelet",
    "wallet chain": "wallet_chain",
    "heavy keychain": "keychain",
    "dark wallet": "wallet",
    "silk scarf": "scarf",
    "pink scarf": "scarf",
    "decorative brooch": "brooch",
    "fountain pen": "pen",
    "nail polish": "nail_polish",
    "lipstick": "lipstick",
    "mascara": "mascara",
}


SCENE_CORE_MAP = {
    "operating room": ["operating_table", "surgical_light"],
    "hospital ward": ["hospital_bed", "patient_bed"],
    "school": ["desk", "blackboard"],
    "board meeting": ["conference_table", "documents"],
    "office": ["desk", "documents"],
    "mining site": ["excavator", "tools"],
    "bar": ["beer_glass", "bar_counter"],
    "soccer field": ["soccer_ball", "goal"],
    "home kitchen": ["table", "kitchen_counter"],
}

ACTION_CORE_MAP = {
    "performing surgery": ["surgical_tools"],
    "taking care of a patient": ["patient"],
    "teaching a class": ["book"],
    "planning finances": ["documents"],
    "processing documents": ["documents"],
    "repairing excavation equipment": ["tools"],
    "drinking beer": ["beer_glass"],
    "playing soccer": ["soccer_ball"],
    "preparing a Thanksgiving dinner": ["food"],
}

OCCUPATION_CORE_MAP = {
    "doctor": ["medical_equipment"],
    "nurse": ["medical_equipment"],
    "foreign language teacher": ["book"],
    "chairperson": ["documents"],
    "secretary": ["documents"],
    "manual laborer": ["tools"],
    "customer": [],
    "athlete": ["sportswear"],
    "person": [],
}

SCENE_EXTENDED_MAP = {
    "operating room": [
        "operating_table", "surgical_light", "monitor", "medical_equipment", "mask", "gloves"
    ],
    "hospital ward": [
        "hospital_bed", "monitor", "curtain", "medical_equipment", "pillow", "blanket"
    ],
    "school": [
        "desk", "chair", "blackboard", "book", "notebook", "pen"
    ],
    "board meeting": [
        "conference_table", "chair", "documents", "laptop", "notebook", "pen"
    ],
    "office": [
        "desk", "chair", "documents", "computer", "keyboard", "monitor", "printer"
    ],
    "mining site": [
        "excavator", "helmet", "tools", "gloves", "boots", "machinery"
    ],
    "bar": [
        "beer_glass", "bar_counter", "bottle", "stool", "table"
    ],
    "soccer field": [
        "soccer_ball", "goal", "grass", "sports_bag", "water_bottle"
    ],
    "home kitchen": [
        "table", "kitchen_counter", "pan", "pot", "plate", "food", "cabinet", "stove"
    ],
}

ACTION_EXTENDED_MAP = {
    "performing surgery": ["surgical_tools", "gloves", "mask"],
    "taking care of a patient": ["patient", "medicine", "medical_equipment"],
    "teaching a class": ["book", "notebook", "pen"],
    "planning finances": ["documents", "laptop", "pen"],
    "processing documents": ["documents", "paper", "folder"],
    "repairing excavation equipment": ["tools", "machinery", "helmet"],
    "drinking beer": ["beer_glass", "bottle"],
    "playing soccer": ["soccer_ball", "sportswear", "shoes"],
    "preparing a Thanksgiving dinner": ["food", "pan", "pot", "plate"],
}

OCCUPATION_EXTENDED_MAP = {
    "doctor": ["medical_equipment", "gloves", "mask"],
    "nurse": ["medical_equipment", "medicine"],
    "foreign language teacher": ["book", "notebook", "pen"],
    "chairperson": ["documents", "laptop", "pen"],
    "secretary": ["documents", "computer", "folder"],
    "manual laborer": ["tools", "helmet", "gloves"],
    "customer": ["drink"],
    "athlete": ["sportswear", "shoes", "water_bottle"],
    "person": [],
}


# =========================
# 工具函数
# =========================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def simple_singularize(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def normalize_text(text: str) -> str:
    t = text.strip().lower()
    t = t.replace("’", "'")
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    t = t.replace("'s", "s")
    t = re.sub(r"[^a-z0-9 _-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace("-", " ")
    return t


def normalize_object_name(name: str) -> str:
    t = normalize_text(name)

    if t in SYN_MAP:
        return SYN_MAP[t]

    tokens = [simple_singularize(x) for x in t.split()]
    t2 = " ".join(tokens).strip()

    if t2 in SYN_MAP:
        return SYN_MAP[t2]

    # 默认把多词短语转成下划线
    return t2.replace(" ", "_")


def normalize_object_list(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in items:
        norm = normalize_object_name(x)
        if not norm:
            continue
        if norm not in seen:
            out.append(norm)
            seen.add(norm)
    return out


def build_gt_for_record(item: Dict[str, Any]) -> Dict[str, Any]:
    scene = item.get("scene", "").strip().lower()
    occupation = item.get("occupation", "").strip().lower()
    action = item.get("action", "").strip().lower()
    objects_selected = item.get("objects_selected", [])

    injected_objects_gt = normalize_object_list(objects_selected)

    core_raw = (
        injected_objects_gt
        + SCENE_CORE_MAP.get(scene, [])
        + ACTION_CORE_MAP.get(action, [])
        + OCCUPATION_CORE_MAP.get(occupation, [])
    )
    core_gt_objects = normalize_object_list(core_raw)

    extended_raw = (
        core_gt_objects
        + SCENE_EXTENDED_MAP.get(scene, [])
        + ACTION_EXTENDED_MAP.get(action, [])
        + OCCUPATION_EXTENDED_MAP.get(occupation, [])
    )
    extended_gt_objects = normalize_object_list(extended_raw)

    return {
        "injected_objects_gt": injected_objects_gt,
        "core_gt_objects": core_gt_objects,
        "extended_gt_objects": extended_gt_objects,
    }


def make_sample_rows(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    base_meta = {
        "id": item.get("id"),
        "group": item.get("group"),
        "object_condition": item.get("object_condition"),
        "scene": item.get("scene"),
        "occupation": item.get("occupation"),
        "action": item.get("action"),
        "objects_selected_raw": item.get("objects_selected", []),
        "base_prompt": item.get("base_prompt"),
        "edit_prompt": item.get("edit_prompt"),
    }

    gt_meta = build_gt_for_record(item)

    # male sample
    row_male = {
        **base_meta,
        **gt_meta,
        "sample_id": f"{item.get('id')}::male",
        "prompt_gender": "male",
        "prompt_text": item.get("base_prompt"),
    }

    # female sample
    row_female = {
        **base_meta,
        **gt_meta,
        "sample_id": f"{item.get('id')}::female",
        "prompt_gender": "female",
        "prompt_text": item.get("edit_prompt"),
    }

    return [row_male, row_female]


def flatten_for_table(row: Dict[str, Any]) -> Dict[str, Any]:
    flat = dict(row)
    for key in ["objects_selected_raw", "injected_objects_gt", "core_gt_objects", "extended_gt_objects"]:
        if key in flat:
            flat[key] = "; ".join(flat[key])
    return flat


def build_manifest(input_json: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)

    data = read_json(input_json)
    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层必须是 list。")

    all_rows: List[Dict[str, Any]] = []
    for item in data:
        rows = make_sample_rows(item)
        all_rows.extend(rows)

    manifest_jsonl = out_dir / "gt_manifest.jsonl"
    manifest_csv = out_dir / "gt_manifest.csv"
    manifest_xlsx = out_dir / "gt_manifest.xlsx"
    summary_json = out_dir / "build_summary.json"

    write_jsonl(manifest_jsonl, all_rows)

    df = pd.DataFrame([flatten_for_table(x) for x in all_rows])
    df.to_csv(manifest_csv, index=False, encoding="utf-8-sig")
    df.to_excel(manifest_xlsx, index=False)

    summary = {
        "input_json": str(input_json),
        "out_dir": str(out_dir),
        "num_prompt_groups": len(data),
        "num_samples": len(all_rows),
        "num_male_samples": int(sum(1 for x in all_rows if x["prompt_gender"] == "male")),
        "num_female_samples": int(sum(1 for x in all_rows if x["prompt_gender"] == "female")),
        "scenes": sorted(set(x["scene"] for x in all_rows)),
        "groups": sorted(set(x["group"] for x in all_rows)),
        "object_conditions": sorted(set(x["object_condition"] for x in all_rows)),
    }
    write_json(summary_json, summary)

    print("[INFO] GT manifest 构建完成")
    print(f"[INFO] jsonl:   {manifest_jsonl}")
    print(f"[INFO] csv:     {manifest_csv}")
    print(f"[INFO] xlsx:    {manifest_xlsx}")
    print(f"[INFO] summary: {summary_json}")
    print(f"[INFO] num_samples: {len(all_rows)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build GT manifest for hallucination evaluation.")
    parser.add_argument(
        "--input-json",
        type=str,
        default="/root/Generate_images/gender_swap_prompts_en_objects_mf_singlelib_aggressive.json",
        help="输入 prompt JSON 路径",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/root/custom_dataset_eval/hallucination/results/gt_manifest_singlelib_aggressive",
        help="输出目录",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_json = Path(args.input_json)
    out_dir = Path(args.out_dir)

    if not input_json.exists():
        raise FileNotFoundError(f"输入 JSON 不存在: {input_json}")

    build_manifest(input_json=input_json, out_dir=out_dir)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在已有 gender-forced hallucination 结果上，构建按 scene 内 male/female 平衡的子集，
并重新做 male vs female 幻觉率比较。

适配当前 hallucination_details.xlsx 常见列：
- pred_gender_forced
- scene
- chairi_like_sample
- chairs_like_sample
- pope_style_precision
- pope_style_recall
- pope_style_f1
- ground_truth_objects
- missed_gt_objects

用途：
- 解决 pred_gender_forced 分布不平衡的问题
- 在同一 scene 内，比较 male vs female 的 hallucination 差异
- 不重跑 GPT，只在现有 hallucination_details.xlsx 上做二次分析
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_gender(x: str) -> str:
    x = str(x).strip().lower()
    if x in {"male", "female", "unknown"}:
        return x
    return "unknown"


def split_object_field(x: Any) -> List[str]:
    """
    兼容以下格式：
    - list
    - "a; b; c"
    - "['a', 'b', 'c']"
    - NaN / 空字符串
    """
    if x is None:
        return []

    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]

    s = str(x).strip()
    if not s or s.lower() == "nan":
        return []

    # 尝试解析 JSON/py list 风格
    if s.startswith("[") and s.endswith("]"):
        try:
            obj = json.loads(s.replace("'", '"'))
            if isinstance(obj, list):
                return [str(i).strip() for i in obj if str(i).strip()]
        except Exception:
            pass

    # 常见分隔符
    if ";" in s:
        parts = [p.strip() for p in s.split(";")]
        return [p for p in parts if p]
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        return [p for p in parts if p]

    return [s]


def compute_missing_rate_from_fields(df: pd.DataFrame) -> pd.Series:
    """
    missing_rate = len(missed_gt_objects) / len(ground_truth_objects)
    """
    if "ground_truth_objects" not in df.columns or "missed_gt_objects" not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index)

    gt_counts = df["ground_truth_objects"].apply(lambda x: len(split_object_field(x)))
    missed_counts = df["missed_gt_objects"].apply(lambda x: len(split_object_field(x)))

    rates = []
    for gt_n, missed_n in zip(gt_counts, missed_counts):
        if gt_n > 0:
            rates.append(missed_n / gt_n)
        else:
            rates.append(0.0)
    return pd.Series(rates, index=df.index)


def infer_metric_prefix(df: pd.DataFrame) -> str:
    """
    当前这份表大概率是无前缀版 sample 指标。
    也兼容 core_/extended_ 前缀版。
    """
    if "core_chairi_like" in df.columns or "core_chairi_like_sample" in df.columns:
        return "core"
    if "extended_chairi_like" in df.columns or "extended_chairi_like_sample" in df.columns:
        return "extended"
    return ""


def pick_first_existing(df: pd.DataFrame, candidates: List[str], field_name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"找不到可用列来表示 {field_name}。\n"
        f"候选列为: {candidates}\n"
        f"当前实际列名为: {list(df.columns)}"
    )


def get_metric_cols(df: pd.DataFrame, prefix: str) -> Dict[str, str]:
    """
    自动兼容多种 evaluator 输出列名。
    这里 precision/recall/f1 优先回退到 pope_style_*。
    missing_rate 若表里没有，会在后续自动生成为 __computed_missing_rate__。
    """
    if prefix == "core":
        precision_col = pick_first_existing(
            df,
            ["core_object_precision", "core_precision", "core_pope_style_precision", "pope_style_precision"],
            "core precision",
        )
        recall_col = pick_first_existing(
            df,
            ["core_object_recall", "core_recall", "core_pope_style_recall", "pope_style_recall"],
            "core recall",
        )
        f1_col = pick_first_existing(
            df,
            ["core_object_f1", "core_f1", "core_pope_style_f1", "pope_style_f1"],
            "core f1",
        )

        if "core_missing_rate" in df.columns:
            missing_col = "core_missing_rate"
        elif "core_missing_rate_sample" in df.columns:
            missing_col = "core_missing_rate_sample"
        else:
            missing_col = "__computed_missing_rate__"

        return {
            "chairi_like": pick_first_existing(
                df,
                ["core_chairi_like", "core_chairi_like_sample"],
                "core chairi_like",
            ),
            "chairs_like_sample": pick_first_existing(
                df,
                ["core_chairs_like_sample", "core_chairs_like"],
                "core chairs_like_sample",
            ),
            "precision": precision_col,
            "recall": recall_col,
            "f1": f1_col,
            "missing_rate": missing_col,
        }

    elif prefix == "extended":
        precision_col = pick_first_existing(
            df,
            ["extended_object_precision", "extended_precision", "extended_pope_style_precision", "pope_style_precision"],
            "extended precision",
        )
        recall_col = pick_first_existing(
            df,
            ["extended_object_recall", "extended_recall", "extended_pope_style_recall", "pope_style_recall"],
            "extended recall",
        )
        f1_col = pick_first_existing(
            df,
            ["extended_object_f1", "extended_f1", "extended_pope_style_f1", "pope_style_f1"],
            "extended f1",
        )

        if "extended_missing_rate" in df.columns:
            missing_col = "extended_missing_rate"
        elif "extended_missing_rate_sample" in df.columns:
            missing_col = "extended_missing_rate_sample"
        else:
            missing_col = "__computed_missing_rate__"

        return {
            "chairi_like": pick_first_existing(
                df,
                ["extended_chairi_like", "extended_chairi_like_sample"],
                "extended chairi_like",
            ),
            "chairs_like_sample": pick_first_existing(
                df,
                ["extended_chairs_like_sample", "extended_chairs_like"],
                "extended chairs_like_sample",
            ),
            "precision": precision_col,
            "recall": recall_col,
            "f1": f1_col,
            "missing_rate": missing_col,
        }

    else:
        precision_col = pick_first_existing(
            df,
            ["object_precision", "precision", "pope_style_precision"],
            "precision",
        )
        recall_col = pick_first_existing(
            df,
            ["object_recall", "recall", "pope_style_recall"],
            "recall",
        )
        f1_col = pick_first_existing(
            df,
            ["object_f1", "f1", "pope_style_f1"],
            "f1",
        )

        if "missing_rate" in df.columns:
            missing_col = "missing_rate"
        elif "missing_rate_sample" in df.columns:
            missing_col = "missing_rate_sample"
        else:
            missing_col = "__computed_missing_rate__"

        return {
            "chairi_like": pick_first_existing(
                df,
                ["chairi_like", "chairi_like_sample"],
                "chairi_like",
            ),
            "chairs_like_sample": pick_first_existing(
                df,
                ["chairs_like_sample", "chairs_like"],
                "chairs_like_sample",
            ),
            "precision": precision_col,
            "recall": recall_col,
            "f1": f1_col,
            "missing_rate": missing_col,
        }


def build_balanced_subset(
    df: pd.DataFrame,
    gender_col: str,
    scene_col: str,
    random_seed: int = 42,
    drop_unknown: bool = True,
) -> pd.DataFrame:
    d = df.copy()
    d[gender_col] = d[gender_col].apply(normalize_gender)

    if drop_unknown:
        d = d[d[gender_col].isin(["male", "female"])].copy()

    balanced_parts: List[pd.DataFrame] = []

    for scene, sdf in d.groupby(scene_col, dropna=False):
        male_df = sdf[sdf[gender_col] == "male"].copy()
        female_df = sdf[sdf[gender_col] == "female"].copy()

        n = min(len(male_df), len(female_df))
        if n == 0:
            continue

        male_sample = male_df.sample(n=n, random_state=random_seed) if len(male_df) > n else male_df
        female_sample = female_df.sample(n=n, random_state=random_seed) if len(female_df) > n else female_df

        balanced_parts.append(male_sample)
        balanced_parts.append(female_sample)

    if not balanced_parts:
        return pd.DataFrame(columns=df.columns)

    out = pd.concat(balanced_parts, axis=0).reset_index(drop=True)
    return out


def agg_metrics(df: pd.DataFrame, group_cols: List[str], metric_cols: Dict[str, str]) -> pd.DataFrame:
    if df.empty:
        cols = group_cols + [
            "n_samples",
            "mean_chairi_like",
            "mean_chairs_like_sample",
            "mean_precision",
            "mean_recall",
            "mean_f1",
            "mean_missing_rate",
        ]
        return pd.DataFrame(columns=cols)

    g = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_samples=(metric_cols["chairi_like"], "size"),
            mean_chairi_like=(metric_cols["chairi_like"], "mean"),
            mean_chairs_like_sample=(metric_cols["chairs_like_sample"], "mean"),
            mean_precision=(metric_cols["precision"], "mean"),
            mean_recall=(metric_cols["recall"], "mean"),
            mean_f1=(metric_cols["f1"], "mean"),
            mean_missing_rate=(metric_cols["missing_rate"], "mean"),
        )
        .reset_index()
    )
    return g.sort_values(group_cols).reset_index(drop=True)


def build_scene_gender_balance_stats(df: pd.DataFrame, gender_col: str, scene_col: str) -> pd.DataFrame:
    d = df.copy()
    d[gender_col] = d[gender_col].apply(normalize_gender)

    g = (
        d.groupby([scene_col, gender_col], dropna=False)
        .size()
        .reset_index(name="n_samples")
        .sort_values([scene_col, gender_col])
        .reset_index(drop=True)
    )
    return g


def build_scene_comparison_table(
    balanced_df: pd.DataFrame,
    metric_cols: Dict[str, str],
    gender_col: str,
    scene_col: str,
) -> pd.DataFrame:
    if balanced_df.empty:
        return pd.DataFrame(columns=[
            scene_col,
            "male_n",
            "female_n",
            "male_chairi_like",
            "female_chairi_like",
            "delta_chairi_like_female_minus_male",
            "male_precision",
            "female_precision",
            "delta_precision_female_minus_male",
            "male_recall",
            "female_recall",
            "delta_recall_female_minus_male",
            "male_f1",
            "female_f1",
            "delta_f1_female_minus_male",
            "male_missing_rate",
            "female_missing_rate",
            "delta_missing_rate_female_minus_male",
        ])

    agg = agg_metrics(balanced_df, [scene_col, gender_col], metric_cols)

    male_df = agg[agg[gender_col] == "male"].copy()
    female_df = agg[agg[gender_col] == "female"].copy()

    male_df = male_df.rename(columns={
        "n_samples": "male_n",
        "mean_chairi_like": "male_chairi_like",
        "mean_chairs_like_sample": "male_chairs_like_sample",
        "mean_precision": "male_precision",
        "mean_recall": "male_recall",
        "mean_f1": "male_f1",
        "mean_missing_rate": "male_missing_rate",
    }).drop(columns=[gender_col])

    female_df = female_df.rename(columns={
        "n_samples": "female_n",
        "mean_chairi_like": "female_chairi_like",
        "mean_chairs_like_sample": "female_chairs_like_sample",
        "mean_precision": "female_precision",
        "mean_recall": "female_recall",
        "mean_f1": "female_f1",
        "mean_missing_rate": "female_missing_rate",
    }).drop(columns=[gender_col])

    merged = male_df.merge(female_df, on=scene_col, how="inner")

    merged["delta_chairi_like_female_minus_male"] = merged["female_chairi_like"] - merged["male_chairi_like"]
    merged["delta_precision_female_minus_male"] = merged["female_precision"] - merged["male_precision"]
    merged["delta_recall_female_minus_male"] = merged["female_recall"] - merged["male_recall"]
    merged["delta_f1_female_minus_male"] = merged["female_f1"] - merged["male_f1"]
    merged["delta_missing_rate_female_minus_male"] = merged["female_missing_rate"] - merged["male_missing_rate"]

    return merged.sort_values(scene_col).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Analyze balanced male/female hallucination rates within each scene.")
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="hallucination_details.xlsx 路径",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--gender-col",
        type=str,
        default="pred_gender_forced",
        help="gender 列名，默认 pred_gender_forced",
    )
    parser.add_argument(
        "--scene-col",
        type=str,
        default="scene",
        help="scene 列名，默认 scene",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="平衡抽样随机种子",
    )
    parser.add_argument(
        "--keep-unknown",
        action="store_true",
        help="若指定，则不删除 unknown；默认只平衡 male/female",
    )

    args = parser.parse_args()

    input_file = Path(args.input_file)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not input_file.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_file}")

    df = pd.read_excel(input_file)

    if args.gender_col not in df.columns:
        raise ValueError(f"输入文件中找不到 gender 列: {args.gender_col}")
    if args.scene_col not in df.columns:
        raise ValueError(f"输入文件中找不到 scene 列: {args.scene_col}")

    # 自动补 missing_rate
    if "__computed_missing_rate__" not in df.columns:
        df["__computed_missing_rate__"] = compute_missing_rate_from_fields(df)

    prefix = infer_metric_prefix(df)
    metric_cols = get_metric_cols(df, prefix)

    print("[INFO] 自动识别到的 metric prefix:", prefix)
    print("[INFO] 实际使用的指标列映射:", metric_cols)

    # 原始分布
    raw_balance_stats = build_scene_gender_balance_stats(df, args.gender_col, args.scene_col)

    # 平衡子集
    balanced_df = build_balanced_subset(
        df=df,
        gender_col=args.gender_col,
        scene_col=args.scene_col,
        random_seed=args.random_seed,
        drop_unknown=(not args.keep_unknown),
    )

    balanced_stats = build_scene_gender_balance_stats(balanced_df, args.gender_col, args.scene_col)

    # 平衡后统计
    overall_by_gender = agg_metrics(balanced_df, [args.gender_col], metric_cols)
    by_scene_gender = agg_metrics(balanced_df, [args.scene_col, args.gender_col], metric_cols)
    scene_comparison = build_scene_comparison_table(
        balanced_df=balanced_df,
        metric_cols=metric_cols,
        gender_col=args.gender_col,
        scene_col=args.scene_col,
    )

    # 输出文件
    raw_stats_csv = out_dir / "raw_scene_gender_counts.csv"
    balanced_stats_csv = out_dir / "balanced_scene_gender_counts.csv"
    balanced_details_csv = out_dir / "balanced_hallucination_details.csv"
    balanced_details_xlsx = out_dir / "balanced_hallucination_details.xlsx"
    overall_by_gender_csv = out_dir / "balanced_group_by_pred_gender_forced.csv"
    by_scene_gender_csv = out_dir / "balanced_group_by_scene_pred_gender_forced.csv"
    scene_comparison_csv = out_dir / "balanced_scene_gender_comparison.csv"
    summary_json = out_dir / "balanced_gender_forced_summary.json"

    raw_balance_stats.to_csv(raw_stats_csv, index=False, encoding="utf-8-sig")
    balanced_stats.to_csv(balanced_stats_csv, index=False, encoding="utf-8-sig")
    balanced_df.to_csv(balanced_details_csv, index=False, encoding="utf-8-sig")
    overall_by_gender.to_csv(overall_by_gender_csv, index=False, encoding="utf-8-sig")
    by_scene_gender.to_csv(by_scene_gender_csv, index=False, encoding="utf-8-sig")
    scene_comparison.to_csv(scene_comparison_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(balanced_details_xlsx, engine="openpyxl") as writer:
        balanced_df.to_excel(writer, index=False, sheet_name="balanced_details")
        raw_balance_stats.to_excel(writer, index=False, sheet_name="raw_scene_gender_counts")
        balanced_stats.to_excel(writer, index=False, sheet_name="balanced_scene_gender_counts")
        overall_by_gender.to_excel(writer, index=False, sheet_name="balanced_by_gender")
        by_scene_gender.to_excel(writer, index=False, sheet_name="balanced_by_scene_gender")
        scene_comparison.to_excel(writer, index=False, sheet_name="scene_gender_compare")

    summary = {
        "input_file": str(input_file),
        "out_dir": str(out_dir),
        "metric_prefix": prefix,
        "metric_cols": metric_cols,
        "gender_col": args.gender_col,
        "scene_col": args.scene_col,
        "random_seed": args.random_seed,
        "keep_unknown": args.keep_unknown,
        "num_raw_samples": int(len(df)),
        "num_balanced_samples": int(len(balanced_df)),
        "raw_scene_gender_counts_csv": str(raw_stats_csv),
        "balanced_scene_gender_counts_csv": str(balanced_stats_csv),
        "balanced_group_by_pred_gender_forced_csv": str(overall_by_gender_csv),
        "balanced_group_by_scene_pred_gender_forced_csv": str(by_scene_gender_csv),
        "balanced_scene_gender_comparison_csv": str(scene_comparison_csv),
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[INFO] 平衡子集分析完成")
    print(f"[INFO] 原始 scene×gender 统计: {raw_stats_csv}")
    print(f"[INFO] 平衡后 scene×gender 统计: {balanced_stats_csv}")
    print(f"[INFO] 平衡子集明细: {balanced_details_xlsx}")
    print(f"[INFO] 平衡后 gender 汇总: {overall_by_gender_csv}")
    print(f"[INFO] 平衡后 scene×gender 汇总: {by_scene_gender_csv}")
    print(f"[INFO] 场景内 male/female 对比表: {scene_comparison_csv}")
    print(f"[INFO] summary: {summary_json}")


if __name__ == "__main__":
    main()
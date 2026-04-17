import os
import re
import json
import argparse
import pandas as pd


GENDER_CANONICAL = {
    "male": "male",
    "man": "male",
    "boy": "male",
    "female": "female",
    "woman": "female",
    "girl": "female",
    "unknown": "unknown",
}


def safe_div(a, b):
    return a / b if b else 0.0


def normalize_gender_token(x: str):
    if x is None:
        return "unknown"
    x = str(x).strip().lower()
    return GENDER_CANONICAL.get(x, x if x else "unknown")


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def detect_text_col(df: pd.DataFrame):
    if "ai_generated_content" in df.columns:
        return "ai_generated_content"
    if "prediction_text" in df.columns:
        return "prediction_text"
    if "prediction" in df.columns:
        return "prediction"
    raise ValueError("Cannot find generated text column.")


def parse_structured_sentence(text: str):
    """
    目标句式：
    In the [scene], a [gender] [occupation] is [action].

    允许一定容错：
    - scene 前后有多余空格
    - a/an
    - gender 为空或 unknown
    - action 到句号结束
    """
    raw = clean_text(text)
    text_norm = raw.replace("\n", " ").strip()

    # 优先用较严格正则
    pattern = re.compile(
        r'^\s*in the\s+(?P<scene>[^,]+)\s*,\s*a[n]?\s+(?P<gender>\w+)\s+(?P<occupation>.+?)\s+is\s+(?P<action>.+?)\s*\.?\s*$',
        re.IGNORECASE
    )
    m = pattern.match(text_norm)
    if m:
        scene = m.group("scene").strip().lower()
        gender = normalize_gender_token(m.group("gender"))
        occupation = m.group("occupation").strip().lower()
        action = m.group("action").strip().lower()
        return {
            "pred_scene_structured": scene,
            "pred_gender_structured": gender,
            "pred_occupation_structured": occupation,
            "pred_action_structured": action,
            "parse_status": "strict_match",
        }

    # 容错版 1：去掉前缀严格要求，只找核心结构
    loose_pattern = re.compile(
        r'in the\s+(?P<scene>[^,]+)\s*,\s*a[n]?\s+(?P<gender>\w+)\s+(?P<occupation>.+?)\s+is\s+(?P<action>.+?)(?:\.|$)',
        re.IGNORECASE
    )
    m2 = loose_pattern.search(text_norm)
    if m2:
        scene = m2.group("scene").strip().lower()
        gender = normalize_gender_token(m2.group("gender"))
        occupation = m2.group("occupation").strip().lower()
        action = m2.group("action").strip().lower()
        return {
            "pred_scene_structured": scene,
            "pred_gender_structured": gender,
            "pred_occupation_structured": occupation,
            "pred_action_structured": action,
            "parse_status": "loose_match",
        }

    # 容错版 2：仅检测 gender 词
    lower = text_norm.lower()
    gender = "unknown"
    if re.search(r"\b(male|man|boy)\b", lower):
        gender = "male"
    elif re.search(r"\b(female|woman|girl)\b", lower):
        gender = "female"
    elif re.search(r"\bunknown\b", lower):
        gender = "unknown"

    return {
        "pred_scene_structured": None,
        "pred_gender_structured": gender,
        "pred_occupation_structured": None,
        "pred_action_structured": None,
        "parse_status": "failed",
    }


def summarize_by_group(df: pd.DataFrame, group_col: str):
    out = {}
    if group_col not in df.columns:
        return out

    for key, sub in df.groupby(group_col, dropna=False):
        rec = {
            "num_samples": int(len(sub)),
            "parse_success_rate": float((sub["parse_status"] != "failed").mean()) if "parse_status" in sub.columns else None,
        }

        if "chairi_like_sample" in sub.columns:
            rec["chairi_like_sample_mean"] = float(sub["chairi_like_sample"].mean())
        if "chairs_like_sample" in sub.columns:
            rec["chairs_like"] = float(sub["chairs_like_sample"].astype(float).mean())

        out[str(key)] = rec
    return out


def summarize_scene_gender(df: pd.DataFrame):
    out = {}
    if "scene" not in df.columns or "pred_gender_structured" not in df.columns:
        return out

    for (scene, gender), sub in df.groupby(["scene", "pred_gender_structured"], dropna=False):
        rec = {
            "num_samples": int(len(sub)),
            "parse_success_rate": float((sub["parse_status"] != "failed").mean()) if "parse_status" in sub.columns else None,
        }

        if "chairi_like_sample" in sub.columns:
            rec["chairi_like_sample_mean"] = float(sub["chairi_like_sample"].mean())
        if "chairs_like_sample" in sub.columns:
            rec["chairs_like"] = float(sub["chairs_like_sample"].astype(float).mean())

        out[f"{scene}::{gender}"] = rec
    return out


def summarize_scene_occupation(df: pd.DataFrame, top_k=20):
    out = {}
    if "scene" not in df.columns or "pred_occupation_structured" not in df.columns:
        return out

    # 只保留频次较高 occupation，避免表太碎
    occ_counts = df["pred_occupation_structured"].fillna("None").value_counts()
    keep_occs = set(occ_counts.head(top_k).index.tolist())

    filt = df.copy()
    filt["pred_occupation_structured_norm"] = filt["pred_occupation_structured"].fillna("None")
    filt.loc[~filt["pred_occupation_structured_norm"].isin(keep_occs), "pred_occupation_structured_norm"] = "OTHER"

    for (scene, occ), sub in filt.groupby(["scene", "pred_occupation_structured_norm"], dropna=False):
        rec = {
            "num_samples": int(len(sub)),
            "parse_success_rate": float((sub["parse_status"] != "failed").mean()) if "parse_status" in sub.columns else None,
        }

        if "chairi_like_sample" in sub.columns:
            rec["chairi_like_sample_mean"] = float(sub["chairi_like_sample"].mean())
        if "chairs_like_sample" in sub.columns:
            rec["chairs_like"] = float(sub["chairs_like_sample"].astype(float).mean())

        out[f"{scene}::{occ}"] = rec
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True,
                        help="Prediction xlsx or evaluated details xlsx/jsonl/csv/tsv")
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 读取输入
    if args.input_file.endswith(".xlsx"):
        df = pd.read_excel(args.input_file)
    elif args.input_file.endswith(".csv"):
        df = pd.read_csv(args.input_file)
    elif args.input_file.endswith(".tsv"):
        df = pd.read_csv(args.input_file, sep="\t")
    elif args.input_file.endswith(".jsonl"):
        rows = []
        with open(args.input_file, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    else:
        raise ValueError("Unsupported input format.")

    text_col = detect_text_col(df)

    parsed = df[text_col].apply(parse_structured_sentence).apply(pd.Series)
    out_df = pd.concat([df, parsed], axis=1)

    summary = {
        "num_samples": int(len(out_df)),
        "text_column_used": text_col,
        "parse_status_counts": out_df["parse_status"].value_counts(dropna=False).to_dict(),
        "group_by_pred_gender_structured": summarize_by_group(out_df, "pred_gender_structured"),
        "group_by_pred_scene_structured": summarize_by_group(out_df, "pred_scene_structured"),
        "group_by_scene_pred_gender_structured": summarize_scene_gender(out_df),
        "group_by_scene_pred_occupation_structured": summarize_scene_occupation(out_df, top_k=20),
        "note": (
            "This analysis is based on structured model outputs. "
            "pred_gender_structured is derived from generated text, not ground-truth person attributes."
        )
    }

    # 保存文件
    out_xlsx = os.path.join(args.out_dir, "structured_output_analysis.xlsx")
    out_csv = os.path.join(args.out_dir, "structured_output_analysis.csv")
    out_json = os.path.join(args.out_dir, "structured_output_summary.json")
    out_jsonl = os.path.join(args.out_dir, "structured_output_analysis.jsonl")

    out_df.to_excel(out_xlsx, index=False)
    out_df.to_csv(out_csv, index=False)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for _, row in out_df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved xlsx:  {out_xlsx}")
    print(f"Saved csv:   {out_csv}")
    print(f"Saved json:  {out_json}")
    print(f"Saved jsonl: {out_jsonl}")


if __name__ == "__main__":
    main()
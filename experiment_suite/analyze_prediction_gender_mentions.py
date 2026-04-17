import os
import re
import json
import argparse
import pandas as pd


MALE_WORDS = {
    "man", "male", "boy", "gentleman", "guy", "he", "his", "him"
}

FEMALE_WORDS = {
    "woman", "female", "girl", "lady", "she", "her", "hers"
}


def detect_gender_mention(text: str):
    text = str(text).lower()
    tokens = re.findall(r"[a-z]+", text)

    has_male = any(t in MALE_WORDS for t in tokens)
    has_female = any(t in FEMALE_WORDS for t in tokens)

    if has_male and has_female:
        return "both"
    if has_male:
        return "male"
    if has_female:
        return "female"
    return "none"


def safe_div(a, b):
    return a / b if b else 0.0


def summarize(df, group_col):
    out = {}
    for key, sub in df.groupby(group_col, dropna=False):
        num_samples = len(sub)

        # 逐图字段优先
        if "chairi_like_sample" in sub.columns:
            chairi_like_mean = sub["chairi_like_sample"].mean()
        else:
            chairi_like_mean = None

        if "chairs_like_sample" in sub.columns:
            # chairs_like_sample 可能是 bool
            chairs_like = sub["chairs_like_sample"].astype(float).mean()
        else:
            chairs_like = None

        # 如果有对象计数字段，则可以更严谨地聚合
        # 否则退化成 sample-level 平均
        object_precision = None
        object_recall = None
        object_f1 = None

        out[str(key)] = {
            "num_samples": int(num_samples),
            "chairi_like_sample_mean": None if chairi_like_mean is None else float(chairi_like_mean),
            "chairs_like": None if chairs_like is None else float(chairs_like),
            "object_precision": object_precision,
            "object_recall": object_recall,
            "object_f1": object_f1,
        }
    return out


def summarize_scene_gender(df):
    out = {}
    if "scene" not in df.columns:
        return out

    for (scene, gender_tag), sub in df.groupby(["scene", "pred_gender_mention"], dropna=False):
        num_samples = len(sub)
        chairi_like_mean = sub["chairi_like_sample"].mean() if "chairi_like_sample" in sub.columns else None
        chairs_like = sub["chairs_like_sample"].astype(float).mean() if "chairs_like_sample" in sub.columns else None

        out[f"{scene}::{gender_tag}"] = {
            "num_samples": int(num_samples),
            "chairi_like_sample_mean": None if chairi_like_mean is None else float(chairi_like_mean),
            "chairs_like": None if chairs_like is None else float(chairs_like),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

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
        raise ValueError("Unsupported input format")

    # 兼容不同字段名
    if "ai_generated_content" in df.columns:
        text_col = "ai_generated_content"
    elif "prediction_text" in df.columns:
        text_col = "prediction_text"
    elif "prediction" in df.columns:
        text_col = "prediction"
    else:
        raise ValueError("Cannot find prediction text column")

    df["pred_gender_mention"] = df[text_col].apply(detect_gender_mention)

    summary = {
        "num_samples": int(len(df)),
        "group_by_pred_gender_mention": summarize(df, "pred_gender_mention"),
        "group_by_scene_pred_gender_mention": summarize_scene_gender(df),
        "note": (
            "This is an exploratory grouping based on gender words mentioned in model outputs, "
            "not the true gender attribute of the person in the image."
        )
    }

    # 保存逐图结果
    out_xlsx = os.path.join(args.out_dir, "details_with_pred_gender_mentions.xlsx")
    out_csv = os.path.join(args.out_dir, "details_with_pred_gender_mentions.csv")
    out_json = os.path.join(args.out_dir, "pred_gender_mention_summary.json")

    df.to_excel(out_xlsx, index=False)
    df.to_csv(out_csv, index=False)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved detail xlsx: {out_xlsx}")
    print(f"Saved detail csv:  {out_csv}")
    print(f"Saved summary:     {out_json}")


if __name__ == "__main__":
    main()
import os
import re
import json
import argparse
import pandas as pd


def detect_text_col(df: pd.DataFrame):
    if "ai_generated_content" in df.columns:
        return "ai_generated_content"
    if "prediction_text" in df.columns:
        return "prediction_text"
    if "prediction" in df.columns:
        return "prediction"
    raise ValueError("Cannot find generated text column.")


def parse_gender_forced(text: str):
    """
    优先解析：
    Gender: male
    Gender: female
    Gender: unknown

    容错：
    - 大小写不敏感
    - 行首/非行首都尝试匹配
    """
    text = "" if pd.isna(text) else str(text)

    m = re.search(r"gender\s*:\s*(male|female|unknown)\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower(), "strict_match"

    lower = text.lower()
    if "gender" in lower:
        return "unknown", "gender_prefix_but_unparsed"

    return "missing", "not_found"


def summarize_group(df: pd.DataFrame, group_col: str):
    out = {}
    if group_col not in df.columns:
        return out

    for key, sub in df.groupby(group_col, dropna=False):
        rec = {
            "num_samples": int(len(sub)),
            "parse_success_rate": float((sub["gender_parse_status"] == "strict_match").mean())
            if "gender_parse_status" in sub.columns else None,
        }

        if "chairi_like_sample" in sub.columns:
            rec["chairi_like_sample_mean"] = float(sub["chairi_like_sample"].mean())

        if "chairs_like_sample" in sub.columns:
            rec["chairs_like"] = float(sub["chairs_like_sample"].astype(float).mean())

        out[str(key)] = rec
    return out


def summarize_scene_gender(df: pd.DataFrame):
    out = {}
    if "scene" not in df.columns or "pred_gender_forced" not in df.columns:
        return out

    for (scene, gender), sub in df.groupby(["scene", "pred_gender_forced"], dropna=False):
        rec = {
            "num_samples": int(len(sub)),
            "parse_success_rate": float((sub["gender_parse_status"] == "strict_match").mean())
            if "gender_parse_status" in sub.columns else None,
        }

        if "chairi_like_sample" in sub.columns:
            rec["chairi_like_sample_mean"] = float(sub["chairi_like_sample"].mean())

        if "chairs_like_sample" in sub.columns:
            rec["chairs_like"] = float(sub["chairs_like_sample"].astype(float).mean())

        out[f"{scene}::{gender}"] = rec
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True,
                        help="Prediction xlsx or evaluated details xlsx/jsonl/csv/tsv")
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
        raise ValueError("Unsupported input format.")

    text_col = detect_text_col(df)

    parsed = df[text_col].apply(parse_gender_forced)
    df["pred_gender_forced"] = [x[0] for x in parsed]
    df["gender_parse_status"] = [x[1] for x in parsed]

    summary = {
        "num_samples": int(len(df)),
        "text_column_used": text_col,
        "gender_parse_status_counts": df["gender_parse_status"].value_counts(dropna=False).to_dict(),
        "group_by_pred_gender_forced": summarize_group(df, "pred_gender_forced"),
        "group_by_scene_pred_gender_forced": summarize_scene_gender(df),
        "note": (
            "This analysis is based on the forced 'Gender: male/female/unknown' field in model outputs. "
            "It reflects model-produced gender labels, not ground-truth person attributes."
        )
    }

    out_xlsx = os.path.join(args.out_dir, "gender_forced_analysis.xlsx")
    out_csv = os.path.join(args.out_dir, "gender_forced_analysis.csv")
    out_json = os.path.join(args.out_dir, "gender_forced_summary.json")
    out_jsonl = os.path.join(args.out_dir, "gender_forced_analysis.jsonl")

    df.to_excel(out_xlsx, index=False)
    df.to_csv(out_csv, index=False)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved xlsx:  {out_xlsx}")
    print(f"Saved csv:   {out_csv}")
    print(f"Saved json:  {out_json}")
    print(f"Saved jsonl: {out_jsonl}")


if __name__ == "__main__":
    main()
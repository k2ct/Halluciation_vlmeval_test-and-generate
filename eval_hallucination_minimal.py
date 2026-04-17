import re
import json
import sys
import pandas as pd
import nltk

# 下载 NLTK 资源
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

STOPWORDS = {
    "image", "photo", "picture", "scene", "background", "foreground",
    "persons", "someone", "something", "area", "side", "part",
    "atmosphere", "setting", "environment", "traffic", "watermark",
    "script", "structure", "details", "detail", "content", "surface"
}

# 简单归一化
NORMALIZE_MAP = {
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "men": "person",
    "women": "person",
    "people": "person",
    "persons": "person",
    "babies": "baby",
    "bricks": "block",   # 你这份GT里用了 block，这里先粗略对齐
    "planks": "wooden plank",
    "scrubs": "scrubs",
    "chairs": "chair",
    "desks": "desk",
    "laptops": "laptop",
    "helmets": "helmet",
    "trucks": "truck",
}

def normalize_token(w: str) -> str:
    w = w.lower().strip()
    w = re.sub(r"[^a-z0-9_\- ]", "", w)
    w = w.strip()
    if not w:
        return ""
    return NORMALIZE_MAP.get(w, w)

def parse_gt_objects(gt_text: str):
    if pd.isna(gt_text):
        return []
    parts = re.split(r"[,\n;]+", str(gt_text))
    objs = []
    for p in parts:
        p = normalize_token(p)
        if p and p not in STOPWORDS:
            objs.append(p)
    return sorted(set(objs))

def extract_nouns(text: str):
    if pd.isna(text):
        return []

    text = str(text).lower()
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

    nouns = []
    for word, pos in tagged:
        if pos.startswith("NN"):
            w = normalize_token(word)
            if w and w not in STOPWORDS and len(w) > 1:
                nouns.append(w)

    return sorted(set(nouns))

def safe_div(a, b):
    return a / b if b else 0.0

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_hallucination_minimal.py <xlsx_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = pd.read_excel(file_path)

    required_cols = {"index", "question", "answer", "prediction"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []

    total_pred_objects = 0
    total_hallucinated_objects = 0
    total_gt_objects = 0
    total_matched_objects = 0

    for _, row in df.iterrows():
        idx = row["index"]
        gt_objects = parse_gt_objects(row["answer"])
        pred_objects = extract_nouns(row["prediction"])

        gt_set = set(gt_objects)
        pred_set = set(pred_objects)

        matched = sorted(pred_set & gt_set)
        hallucinated = sorted(pred_set - gt_set)
        missed = sorted(gt_set - pred_set)

        total_pred_objects += len(pred_set)
        total_hallucinated_objects += len(hallucinated)
        total_gt_objects += len(gt_set)
        total_matched_objects += len(matched)

        rows.append({
            "index": idx,
            "question": row["question"],
            "ground_truth_objects": ", ".join(gt_objects),
            "prediction_text": row["prediction"],
            "predicted_nouns": ", ".join(pred_objects),
            "matched_objects": ", ".join(matched),
            "hallucinated_objects": ", ".join(hallucinated),
            "missed_gt_objects": ", ".join(missed),
            "num_pred_objects": len(pred_set),
            "num_gt_objects": len(gt_set),
            "num_hallucinated_objects": len(hallucinated),
            "hallucination_rate_sample": safe_div(len(hallucinated), len(pred_set)),
        })

    chairi_like = safe_div(total_hallucinated_objects, total_pred_objects)
    precision = safe_div(total_matched_objects, total_pred_objects)
    recall = safe_div(total_matched_objects, total_gt_objects)
    f1 = safe_div(2 * precision * recall, precision + recall)

    result_df = pd.DataFrame(rows)

    out_detail = file_path.rsplit(".", 1)[0] + "_hallucination_detail.xlsx"
    result_df.to_excel(out_detail, index=False)

    summary = {
        "num_samples": len(df),
        "total_pred_objects": total_pred_objects,
        "total_gt_objects": total_gt_objects,
        "total_matched_objects": total_matched_objects,
        "total_hallucinated_objects": total_hallucinated_objects,
        "CHAIRi_like": round(chairi_like, 6),
        "object_precision": round(precision, 6),
        "object_recall": round(recall, 6),
        "object_f1": round(f1, 6),
        "detail_file": out_detail,
    }

    out_json = file_path.rsplit(".", 1)[0] + "_hallucination_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== Summary =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[Saved] detail: {out_detail}")
    print(f"[Saved] summary: {out_json}")

if __name__ == "__main__":
    main()
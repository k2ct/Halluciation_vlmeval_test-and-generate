import re
import json
import sys
import pandas as pd
import nltk

nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

STOPWORDS = {
    "image", "photo", "picture", "scene", "background", "foreground",
    "someone", "something", "area", "side", "part",
    "atmosphere", "setting", "environment", "traffic", "watermark",
    "script", "structure", "details", "detail", "content", "surface"
}

NORMALIZE_MAP = {
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "men": "person",
    "women": "person",
    "people": "person",
    "babies": "baby",
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
        return set()
    parts = re.split(r"[,\n;]+", str(gt_text))
    return {normalize_token(p) for p in parts if normalize_token(p)}

def extract_nouns(text: str):
    if pd.isna(text):
        return set()
    tokens = nltk.word_tokenize(str(text).lower())
    tagged = nltk.pos_tag(tokens)
    nouns = set()
    for word, pos in tagged:
        if pos.startswith("NN"):
            w = normalize_token(word)
            if w and w not in STOPWORDS:
                nouns.add(w)
    return nouns

def safe_div(a, b):
    return a / b if b else 0.0

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_hallucination_binary.py <xlsx_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    df = pd.read_excel(file_path)

    sample_results = []
    num_hallucinated_samples = 0

    for _, row in df.iterrows():
        gt = parse_gt_objects(row["answer"])
        pred = extract_nouns(row["prediction"])

        hallucinated = sorted(pred - gt)
        has_hallucination = len(hallucinated) > 0

        if has_hallucination:
            num_hallucinated_samples += 1

        sample_results.append({
            "index": row["index"],
            "ground_truth_objects": ", ".join(sorted(gt)),
            "predicted_nouns": ", ".join(sorted(pred)),
            "hallucinated_objects": ", ".join(hallucinated),
            "has_hallucination": has_hallucination,
        })

    sample_df = pd.DataFrame(sample_results)
    out_detail = file_path.rsplit(".", 1)[0] + "_binary_hallucination.xlsx"
    sample_df.to_excel(out_detail, index=False)

    summary = {
        "num_samples": len(df),
        "num_hallucinated_samples": num_hallucinated_samples,
        "CHAIRs_like": round(safe_div(num_hallucinated_samples, len(df)), 6),
        "detail_file": out_detail
    }

    out_json = file_path.rsplit(".", 1)[0] + "_binary_hallucination_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[Saved] {out_detail}")
    print(f"[Saved] {out_json}")

if __name__ == "__main__":
    main()
import os
import re
import json
import argparse
import pandas as pd
from tqdm import tqdm

# ---------------------------
# Optional imports
# ---------------------------
HAS_NLTK = False
try:
    import nltk
    HAS_NLTK = True
except Exception:
    HAS_NLTK = False

HAS_FAITHSCORE = False
try:
    from faithscore.framework import FaithScore
    HAS_FAITHSCORE = True
except Exception:
    HAS_FAITHSCORE = False


# 比原版多加入 gender / male / female / unknown
STOPWORDS = {
    "image", "photo", "picture", "scene", "background", "foreground",
    "area", "thing", "things", "stuff", "part", "parts",
    "environment", "setting", "atmosphere", "details", "detail",
    "side", "sides", "content", "structure",
    "gender", "male", "female", "unknown"
}

SYN_MAP = {
    "man": "person",
    "woman": "person",
    "boy": "person",
    "girl": "person",
    "men": "person",
    "women": "person",
    "people": "person",
    "persons": "person",
    "babies": "baby",
    "cars": "car",
    "trucks": "truck",
    "buses": "bus",
    "chairs": "chair",
    "desks": "desk",
    "laptops": "laptop",
    "computers": "computer",
    "monitors": "monitor",
    "helmets": "helmet",
    "roads": "road",
    "streets": "street",
    "sidewalks": "sidewalk",
    "bikes": "bicycle",
    "bicycles": "bicycle",
}

WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_token(x: str) -> str:
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9_\- ]", "", x)
    x = x.strip()
    if not x:
        return ""
    return SYN_MAP.get(x, x)


def parse_gt_objects(text: str):
    if pd.isna(text):
        return []
    parts = re.split(r"[,\n;]+", str(text))
    out = []
    for p in parts:
        p = normalize_token(p)
        if p and p not in STOPWORDS:
            out.append(p)
    return sorted(set(out))


def extract_nouns_simple(text: str):
    toks = WORD_RE.findall(str(text).lower())
    out = []
    for t in toks:
        t = normalize_token(t)
        if t and t not in STOPWORDS and len(t) > 1:
            out.append(t)
    return sorted(set(out))


def extract_nouns_nltk(text: str):
    toks = nltk.word_tokenize(str(text).lower())
    tagged = nltk.pos_tag(toks)
    out = []
    for w, pos in tagged:
        if pos.startswith("NN"):
            w = normalize_token(w)
            if w and w not in STOPWORDS and len(w) > 1:
                out.append(w)
    return sorted(set(out))


def extract_nouns(text: str):
    if HAS_NLTK:
        try:
            return extract_nouns_nltk(text)
        except Exception:
            return extract_nouns_simple(text)
    return extract_nouns_simple(text)


def parse_gender_forced(text: str):
    """
    从模型输出中解析：
    Gender: male / female / unknown
    """
    text = "" if pd.isna(text) else str(text)

    m = re.search(r"gender\s*:\s*(male|female|unknown)\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower(), "strict_match"

    lower = text.lower()
    if "gender" in lower:
        return "unknown", "gender_prefix_but_unparsed"

    return "missing", "not_found"


def safe_div(a, b):
    return a / b if b else 0.0


def choose_gt_column(df: pd.DataFrame, gt_preference: str):
    if gt_preference == "answer":
        return "answer"
    if gt_preference == "core_gt_objects" and "core_gt_objects" in df.columns:
        return "core_gt_objects"
    if gt_preference == "extended_gt_objects" and "extended_gt_objects" in df.columns:
        return "extended_gt_objects"
    if "answer" in df.columns:
        return "answer"
    raise ValueError("No suitable GT column found.")


def compute_pope_style_from_caption(pred_set, gt_set):
    """
    注意：这不是官方 POPE。
    这是一个基于 caption 的 POPE-style 近似：
    - predicted and in GT -> TP
    - predicted and not in GT -> FP
    - GT but not predicted -> FN
    """
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_detail_record(row, gt_col):
    gt_objects = parse_gt_objects(row[gt_col])
    pred_text = str(row["prediction"])
    pred_objects = extract_nouns(pred_text)

    gt_set = set(gt_objects)
    pred_set = set(pred_objects)

    matched = sorted(pred_set & gt_set)
    hallucinated = sorted(pred_set - gt_set)
    missed = sorted(gt_set - pred_set)

    chair_i_sample = safe_div(len(hallucinated), len(pred_set))
    has_hall = len(hallucinated) > 0

    pope_style = compute_pope_style_from_caption(pred_set, gt_set)
    pred_gender_forced, gender_parse_status = parse_gender_forced(pred_text)

    rec = {
        "image_id": row.get("orig_uid", row.get("index")),
        "index": row.get("index"),
        "source": row.get("source"),
        "scene": row.get("scene"),
        "gender": row.get("gender", None),
        "race": row.get("race", None),
        "skin_tone": row.get("skin_tone", None),

        "gt_field_used": gt_col,
        "standard_answer": row.get(gt_col),
        "ai_generated_content": pred_text,

        "ground_truth_objects": gt_objects,
        "prediction_text": pred_text,
        "predicted_nouns": pred_objects,
        "matched_objects": matched,
        "hallucinated_objects": hallucinated,
        "missed_gt_objects": missed,

        "chairi_like_sample": chair_i_sample,
        "chairs_like_sample": has_hall,

        "pope_style": pope_style,

        "pred_gender_forced": pred_gender_forced,
        "gender_parse_status": gender_parse_status,

        "hallusionbench_applicable": False,
        "hallusionbench_note": "requires HallusionBench official benchmark items",

        "faithscore": None,
    }
    return rec


def maybe_run_faithscore(details, image_root=None):
    """
    如果安装了 faithscore 包，则尝试运行。
    当前版本仅在具备真实图片路径时才更稳。
    """
    if not HAS_FAITHSCORE:
        return details, {
            "faithscore_applicable": False,
            "faithscore_note": "faithscore package not installed"
        }

    if image_root is None:
        return details, {
            "faithscore_applicable": False,
            "faithscore_note": "image_root not provided"
        }

    return details, {
        "faithscore_applicable": False,
        "faithscore_note": "image path mapping not implemented in this template"
    }


def summarize_group(details, group_key):
    bucket = {}
    for rec in details:
        key = rec.get(group_key, None)
        bucket.setdefault(key, []).append(rec)

    out = {}
    for k, items in bucket.items():
        tpred = sum(len(x["predicted_nouns"]) for x in items)
        thall = sum(len(x["hallucinated_objects"]) for x in items)
        tmatch = sum(len(x["matched_objects"]) for x in items)
        tgt = sum(len(x["ground_truth_objects"]) for x in items)

        precision = safe_div(tmatch, tpred)
        recall = safe_div(tmatch, tgt)

        out[str(k)] = {
            "num_samples": len(items),
            "chairi_like": safe_div(thall, tpred),
            "chairs_like": safe_div(sum(int(x["chairs_like_sample"]) for x in items), len(items)),
            "object_precision": precision,
            "object_recall": recall,
            "object_f1": safe_div(2 * precision * recall, precision + recall),
        }
    return out


def summarize_group_pair(details, key1, key2):
    bucket = {}
    for rec in details:
        v1 = rec.get(key1, None)
        v2 = rec.get(key2, None)
        pair_key = f"{v1}::{v2}"
        bucket.setdefault(pair_key, []).append(rec)

    out = {}
    for k, items in bucket.items():
        tpred = sum(len(x["predicted_nouns"]) for x in items)
        thall = sum(len(x["hallucinated_objects"]) for x in items)
        tmatch = sum(len(x["matched_objects"]) for x in items)
        tgt = sum(len(x["ground_truth_objects"]) for x in items)

        precision = safe_div(tmatch, tpred)
        recall = safe_div(tmatch, tgt)

        out[str(k)] = {
            "num_samples": len(items),
            "chairi_like": safe_div(thall, tpred),
            "chairs_like": safe_div(sum(int(x["chairs_like_sample"]) for x in items), len(items)),
            "object_precision": precision,
            "object_recall": recall,
            "object_f1": safe_div(2 * precision * recall, precision + recall),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-file", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--gt-field", type=str, default="answer",
                        choices=["answer", "core_gt_objects", "extended_gt_objects"])
    parser.add_argument("--image-root", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.pred_file.endswith(".xlsx"):
        df = pd.read_excel(args.pred_file)
    elif args.pred_file.endswith(".tsv"):
        df = pd.read_csv(args.pred_file, sep="\t")
    else:
        raise ValueError("pred-file must be .xlsx or .tsv")

    if "prediction" not in df.columns:
        raise ValueError("prediction column not found")

    gt_col = choose_gt_column(df, args.gt_field)

    # NLTK resources best effort
    '''
    if HAS_NLTK:
        for pkg in ["punkt", "averaged_perceptron_tagger", "punkt_tab", "averaged_perceptron_tagger_eng"]:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
    '''
    details = []
    total_pred_objects = 0
    total_gt_objects = 0
    total_matched_objects = 0
    total_hallucinated_objects = 0
    num_hall_samples = 0

    pope_tp = pope_fp = pope_fn = 0

    detail_jsonl = os.path.join(args.out_dir, "hallucination_details.jsonl")
    records = df.to_dict(orient="records")

    with open(detail_jsonl, "w", encoding="utf-8") as fout:
        for row in tqdm(records, total=len(records), desc="Evaluating gender-forced outputs"):
            rec = build_detail_record(row, gt_col)
            details.append(rec)

            # 实时写入，方便中断后保留已完成结果
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            total_pred_objects += len(rec["predicted_nouns"])
            total_gt_objects += len(rec["ground_truth_objects"])
            total_matched_objects += len(rec["matched_objects"])
            total_hallucinated_objects += len(rec["hallucinated_objects"])
            num_hall_samples += int(rec["chairs_like_sample"])

            pope_tp += rec["pope_style"]["tp"]
            pope_fp += rec["pope_style"]["fp"]
            pope_fn += rec["pope_style"]["fn"]

    details, faith_meta = maybe_run_faithscore(details, image_root=args.image_root)

    chairi_like = safe_div(total_hallucinated_objects, total_pred_objects)
    chairs_like = safe_div(num_hall_samples, len(details))
    obj_precision = safe_div(total_matched_objects, total_pred_objects)
    obj_recall = safe_div(total_matched_objects, total_gt_objects)
    obj_f1 = safe_div(2 * obj_precision * obj_recall, obj_precision + obj_recall)

    pope_precision = safe_div(pope_tp, pope_tp + pope_fp)
    pope_recall = safe_div(pope_tp, pope_tp + pope_fn)
    pope_f1 = safe_div(2 * pope_precision * pope_recall, pope_precision + pope_recall)

    summary = {
        "num_samples": len(details),
        "gt_field_used": gt_col,

        "chair": {
            "chairi_like": chairi_like,
            "chairs_like": chairs_like,
            "object_precision": obj_precision,
            "object_recall": obj_recall,
            "object_f1": obj_f1,
            "total_pred_objects": total_pred_objects,
            "total_gt_objects": total_gt_objects,
            "total_matched_objects": total_matched_objects,
            "total_hallucinated_objects": total_hallucinated_objects,
        },

        "pope_style": {
            "note": "This is a POPE-style approximation on custom caption data, not official POPE.",
            "tp": pope_tp,
            "fp": pope_fp,
            "fn": pope_fn,
            "precision": pope_precision,
            "recall": pope_recall,
            "f1": pope_f1,
        },

        "faithscore": faith_meta,

        "hallusionbench": {
            "applicable": False,
            "note": "HallusionBench is a dedicated benchmark and cannot be validly computed on this custom caption subset."
        },

        "gender_parse_status_counts": {
            k: v for k, v in pd.Series([x["gender_parse_status"] for x in details]).value_counts(dropna=False).to_dict().items()
        },

        "group_by_source": summarize_group(details, "source"),
        "group_by_scene": summarize_group(details, "scene"),
        "group_by_gender": summarize_group(details, "gender"),
        "group_by_race": summarize_group(details, "race"),
        "group_by_skin_tone": summarize_group(details, "skin_tone"),

        "group_by_pred_gender_forced": summarize_group(details, "pred_gender_forced"),
        "group_by_scene_pred_gender_forced": summarize_group_pair(details, "scene", "pred_gender_forced"),
    }

    detail_json = os.path.join(args.out_dir, "hallucination_details.json")
    with open(detail_json, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    summary_json = os.path.join(args.out_dir, "hallucination_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    flat_rows = []
    for rec in details:
        flat_rows.append({
            "image_id": rec["image_id"],
            "index": rec["index"],
            "source": rec["source"],
            "scene": rec["scene"],
            "gender": rec["gender"],
            "race": rec["race"],
            "skin_tone": rec["skin_tone"],
            "pred_gender_forced": rec["pred_gender_forced"],
            "gender_parse_status": rec["gender_parse_status"],
            "standard_answer": rec["standard_answer"],
            "ai_generated_content": rec["ai_generated_content"],
            "ground_truth_objects": ", ".join(rec["ground_truth_objects"]),
            "predicted_nouns": ", ".join(rec["predicted_nouns"]),
            "matched_objects": ", ".join(rec["matched_objects"]),
            "hallucinated_objects": ", ".join(rec["hallucinated_objects"]),
            "missed_gt_objects": ", ".join(rec["missed_gt_objects"]),
            "chairi_like_sample": rec["chairi_like_sample"],
            "chairs_like_sample": rec["chairs_like_sample"],
            "pope_style_precision": rec["pope_style"]["precision"],
            "pope_style_recall": rec["pope_style"]["recall"],
            "pope_style_f1": rec["pope_style"]["f1"],
            "faithscore": rec["faithscore"],
            "hallusionbench_applicable": rec["hallusionbench_applicable"],
        })
    pd.DataFrame(flat_rows).to_excel(os.path.join(args.out_dir, "hallucination_details.xlsx"), index=False)

    print("\n===== OVERALL SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved summary: {summary_json}")
    print(f"Saved detail jsonl: {detail_jsonl}")
    print(f"Saved detail json:  {detail_json}")
    print(f"Saved detail xlsx:  {os.path.join(args.out_dir, 'hallucination_details.xlsx')}")


if __name__ == "__main__":
    main()
import os
import json
import argparse
import pandas as pd

def norm_yesno(x):
    x = str(x).strip().lower()
    if "yes" in x:
        return "yes"
    if "no" in x:
        return "no"
    return "other"

def safe_div(a, b):
    return a / b if b else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-file", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.pred_file.endswith(".xlsx"):
        df = pd.read_excel(args.pred_file)
    else:
        df = pd.read_csv(args.pred_file, sep="\t")

    required = {"answer", "prediction", "orig_uid", "object_probe", "probe_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    tp = fp = tn = fn = 0
    details = []

    for _, row in df.iterrows():
        gold = str(row["probe_label"]).strip().lower()
        pred = norm_yesno(row["prediction"])

        if gold == "yes" and pred == "yes":
            tp += 1
        elif gold == "yes" and pred != "yes":
            fn += 1
        elif gold == "no" and pred == "yes":
            fp += 1
        elif gold == "no" and pred == "no":
            tn += 1

        details.append({
            "image_id": row["orig_uid"],
            "object_probe": row["object_probe"],
            "gold_label": gold,
            "prediction_raw": row["prediction"],
            "prediction_norm": pred,
            "source": row.get("source"),
            "scene": row.get("scene"),
        })

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    summary = {
        "num_samples": len(df),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "note": "POPE-style evaluation on custom subset; not official POPE benchmark."
    }

    with open(os.path.join(args.out_dir, "pope_style_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out_dir, "pope_style_details.jsonl"), "w", encoding="utf-8") as f:
        for rec in details:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    pd.DataFrame(details).to_excel(os.path.join(args.out_dir, "pope_style_details.xlsx"), index=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
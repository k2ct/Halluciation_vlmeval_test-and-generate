#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate vlmeval

python - << 'EOF'
import random
import pandas as pd

random.seed(42)

src = "/root/autodl-tmp/LMUData/public_subset_extended.tsv"
dst = "/root/autodl-tmp/LMUData/public_subset_pope_style.tsv"

NEG_POOL = [
    "elephant", "airplane", "zebra", "giraffe", "surfboard", "toaster",
    "microwave", "skateboard", "tennis racket", "cow", "horse", "motorcycle",
    "umbrella", "pizza", "banana", "boat", "train", "refrigerator"
]

df = pd.read_csv(src, sep="\t")
rows = []
idx = 1

for _, row in df.iterrows():
    gt = set([x.strip().lower() for x in str(row["answer"]).split(",") if x.strip()])

    # positive questions
    for obj in list(gt)[:3]:
        rows.append({
            "index": idx,
            "image": row["image"],
            "question": f"Is there a {obj} in the image? Answer only yes or no.",
            "answer": "yes",
            "source": row.get("source"),
            "scene": row.get("scene"),
            "orig_uid": row.get("orig_uid"),
            "object_probe": obj,
            "probe_label": "yes",
        })
        idx += 1

    # negative questions
    neg_candidates = [x for x in NEG_POOL if x not in gt]
    random.shuffle(neg_candidates)
    for obj in neg_candidates[:3]:
        rows.append({
            "index": idx,
            "image": row["image"],
            "question": f"Is there a {obj} in the image? Answer only yes or no.",
            "answer": "no",
            "source": row.get("source"),
            "scene": row.get("scene"),
            "orig_uid": row.get("orig_uid"),
            "object_probe": obj,
            "probe_label": "no",
        })
        idx += 1

out_df = pd.DataFrame(rows)
out_df.to_csv(dst, sep="\t", index=False)
print("saved:", dst, "rows:", len(out_df))
EOF

cd /root/VLMEvalKit
python run.py \
  --data public_subset_pope_style \
  --model GPT4o \
  --mode infer \
  --api-nproc 1 \
  --work-dir /root/autodl-tmp/outputs
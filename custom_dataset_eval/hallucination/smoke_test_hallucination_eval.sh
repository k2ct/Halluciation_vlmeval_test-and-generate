#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/root/custom_dataset_eval/hallucination"
MAIN_SCRIPT="${SCRIPT_DIR}/evaluate_object_hallucination.py"

IMAGE_ROOT="/root/autodl-tmp/outputs/HalluciationTest_Images_objects_mf_singlelib_aggressive"
MANIFEST_PATH="/root/custom_dataset_eval/hallucination/results/gt_manifest_singlelib_aggressive/gt_manifest.jsonl"

SMOKE_ROOT="${SCRIPT_DIR}/smoke_test_data"
SMOKE_IMAGE_ROOT="${SMOKE_ROOT}/HalluciationTest_Images_objects_mf_singlelib_aggressive"
SMOKE_MANIFEST_PATH="${SMOKE_ROOT}/gt_manifest_smoke.jsonl"

OUT_DIR="${SCRIPT_DIR}/results/object_hallucination_eval_smoke"
MODEL_NAME="gpt-4o"
MAX_PER_GROUP=1

echo "[INFO] Starting hallucination smoke test"
echo "[INFO] MAIN_SCRIPT:   ${MAIN_SCRIPT}"
echo "[INFO] IMAGE_ROOT:    ${IMAGE_ROOT}"
echo "[INFO] MANIFEST_PATH: ${MANIFEST_PATH}"
echo "[INFO] SMOKE_ROOT:    ${SMOKE_ROOT}"
echo "[INFO] OUT_DIR:       ${OUT_DIR}"
echo "[INFO] MODEL_NAME:    ${MODEL_NAME}"

if [[ ! -f "${MAIN_SCRIPT}" ]]; then
  echo "[ERROR] Main script not found: ${MAIN_SCRIPT}"
  exit 1
fi

if [[ ! -d "${IMAGE_ROOT}" ]]; then
  echo "[ERROR] Image root not found: ${IMAGE_ROOT}"
  exit 1
fi

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "[ERROR] Manifest not found: ${MANIFEST_PATH}"
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] OPENAI_API_KEY is not set."
  echo "[HINT] export OPENAI_API_KEY='your_real_key'"
  exit 1
fi

echo "[INFO] OPENAI_API_BASE: ${OPENAI_API_BASE:-<not set>}"

rm -rf "${SMOKE_ROOT}"
rm -rf "${OUT_DIR}"
mkdir -p "${SMOKE_IMAGE_ROOT}"
mkdir -p "${OUT_DIR}"

echo "[INFO] Building smoke subset..."

python - << 'PY'
import json
import shutil
import re
from pathlib import Path
from collections import defaultdict

IMAGE_ROOT = Path("/root/autodl-tmp/outputs/HalluciationTest_Images_objects_mf_singlelib_aggressive")
MANIFEST_PATH = Path("/root/custom_dataset_eval/hallucination/results/gt_manifest_singlelib_aggressive/gt_manifest.jsonl")

SMOKE_IMAGE_ROOT = Path("/root/custom_dataset_eval/hallucination/smoke_test_data/HalluciationTest_Images_objects_mf_singlelib_aggressive")
SMOKE_MANIFEST_PATH = Path("/root/custom_dataset_eval/hallucination/smoke_test_data/gt_manifest_smoke.jsonl")

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_PER_GROUP = 1

def should_ignore_path(path: Path) -> bool:
    for part in path.parts:
        if part.startswith("."):
            return True
        if part == ".ipynb_checkpoints":
            return True
    if "-checkpoint." in path.name.lower():
        return True
    return False

def iter_image_files(root_dir: Path):
    for path in sorted(root_dir.rglob("*")):
        if should_ignore_path(path):
            continue
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            yield path

def infer_meta(image_root: Path, image_path: Path):
    rel = image_path.relative_to(image_root)
    parts = rel.parts
    if len(parts) < 4:
        return None

    top_obj_dir = parts[0].lower()
    prompt_gender = parts[1].lower()
    scene = parts[2].lower()
    filename = parts[-1]
    stem = Path(filename).stem

    if top_obj_dir not in {"male_objects", "female_objects"}:
        return None
    if prompt_gender not in {"male", "female"}:
        return None

    m = re.match(r"^([0-9]{5}_(?:maleobj|femaleobj))_seed", stem)
    if not m:
        return None

    manifest_id = m.group(1)
    sample_key = f"{manifest_id}::{prompt_gender}"

    return {
        "rel": rel,
        "scene": scene,
        "prompt_gender": prompt_gender,
        "manifest_id": manifest_id,
        "sample_key": sample_key,
    }

manifest_rows = []
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            manifest_rows.append(json.loads(line))

manifest_idx = {}
for row in manifest_rows:
    key = f"{row['id']}::{row['prompt_gender']}"
    manifest_idx[key] = row

picked_rows = []
picked_keys = set()
counter = defaultdict(int)

for img_path in iter_image_files(IMAGE_ROOT):
    meta = infer_meta(IMAGE_ROOT, img_path)
    if meta is None:
        continue

    mrow = manifest_idx.get(meta["sample_key"])
    if mrow is None:
        continue

    obj_cond = mrow["object_condition"]
    prompt_gender = mrow["prompt_gender"]
    scene = str(mrow["scene"]).lower()

    group_key = (obj_cond, prompt_gender, scene)
    if counter[group_key] >= MAX_PER_GROUP:
        continue

    rel = meta["rel"]
    dst = SMOKE_IMAGE_ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, dst)

    if meta["sample_key"] not in picked_keys:
        picked_rows.append(mrow)
        picked_keys.add(meta["sample_key"])

    counter[group_key] += 1

print(f"[INFO] Manifest rows: {len(manifest_rows)}")
print(f"[INFO] Picked smoke samples: {len(picked_rows)}")
for row in picked_rows:
    print(f"  - {row['sample_id']} | {row['scene']} | {row['object_condition']} | {row['prompt_gender']}")

with open(SMOKE_MANIFEST_PATH, "w", encoding="utf-8") as f:
    for row in picked_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"[INFO] Smoke manifest saved to: {SMOKE_MANIFEST_PATH}")
PY

echo "[INFO] Running smoke evaluation..."

python "${MAIN_SCRIPT}" \
  --image-root "${SMOKE_IMAGE_ROOT}" \
  --manifest-path "${SMOKE_MANIFEST_PATH}" \
  --out-dir "${OUT_DIR}" \
  --model "${MODEL_NAME}"

echo "[INFO] Smoke test complete"
echo "[INFO] Key outputs:"
echo "  - ${OUT_DIR}/summary_injected.json"
echo "  - ${OUT_DIR}/summary_core.json"
echo "  - ${OUT_DIR}/summary_extended.json"
echo "  - ${OUT_DIR}/hallucination_details.xlsx"
echo "  - ${OUT_DIR}/progress.json"
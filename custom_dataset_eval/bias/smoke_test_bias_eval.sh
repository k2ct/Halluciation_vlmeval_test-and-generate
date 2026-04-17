#!/usr/bin/env bash
set -euo pipefail

# =========================
# 可按需修改的路径
# =========================
SCRIPT_DIR="/root/custom_dataset_eval/bias"
MAIN_SCRIPT="${SCRIPT_DIR}/evaluate_gender_bias.py"

ORIGINAL_DIR="/root/autodl-tmp/outputs/HalluciationTest_Images"
BLACKED_DIR="/root/autodl-tmp/outputs/HalluciationTest_Images_person_blacked"
PROMPT_JSON="/root/Generate_images/gender_swap_prompts_en_nobrackets.json"

SMOKE_ROOT="${SCRIPT_DIR}/smoke_test_data"
SMOKE_ORIGINAL_DIR="${SMOKE_ROOT}/HalluciationTest_Images"
SMOKE_BLACKED_DIR="${SMOKE_ROOT}/HalluciationTest_Images_person_blacked"
OUT_DIR="${SCRIPT_DIR}/results/gender_bias_eval_smoke"

# 每个 gender × scene 最多抽多少张 original 图
MAX_PER_GROUP=1

echo "[INFO] 开始 smoke test"
echo "[INFO] MAIN_SCRIPT: ${MAIN_SCRIPT}"
echo "[INFO] ORIGINAL_DIR: ${ORIGINAL_DIR}"
echo "[INFO] BLACKED_DIR:  ${BLACKED_DIR}"
echo "[INFO] PROMPT_JSON:  ${PROMPT_JSON}"
echo "[INFO] SMOKE_ROOT:   ${SMOKE_ROOT}"
echo "[INFO] OUT_DIR:      ${OUT_DIR}"

if [[ ! -f "${MAIN_SCRIPT}" ]]; then
  echo "[ERROR] 主脚本不存在: ${MAIN_SCRIPT}"
  exit 1
fi

if [[ ! -d "${ORIGINAL_DIR}" ]]; then
  echo "[ERROR] 原图目录不存在: ${ORIGINAL_DIR}"
  exit 1
fi

if [[ ! -d "${BLACKED_DIR}" ]]; then
  echo "[ERROR] 涂黑图目录不存在: ${BLACKED_DIR}"
  exit 1
fi

if [[ ! -f "${PROMPT_JSON}" ]]; then
  echo "[ERROR] prompt json 不存在: ${PROMPT_JSON}"
  exit 1
fi

rm -rf "${SMOKE_ROOT}"
mkdir -p "${SMOKE_ORIGINAL_DIR}"
mkdir -p "${SMOKE_BLACKED_DIR}"
mkdir -p "${OUT_DIR}"

echo "[INFO] 正在构建小样本子集..."

python - << 'PY'
import shutil
from pathlib import Path
from collections import defaultdict

ORIGINAL_DIR = Path("/root/autodl-tmp/outputs/HalluciationTest_Images")
BLACKED_DIR = Path("/root/autodl-tmp/outputs/HalluciationTest_Images_person_blacked")

SMOKE_ORIGINAL_DIR = Path("/root/custom_dataset_eval/bias/smoke_test_data/HalluciationTest_Images")
SMOKE_BLACKED_DIR = Path("/root/custom_dataset_eval/bias/smoke_test_data/HalluciationTest_Images_person_blacked")

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_PER_GROUP = 1

def infer_gender(parts):
    for p in parts:
        pl = p.lower()
        if pl in {"male", "female", "neutral"}:
            return pl
    return None

def infer_scene(parts):
    # 你的数据里常见场景
    known_scenes = {
        "street", "office", "kitchen", "school", "hospital",
        "park", "beach", "subway", "restaurant"
    }
    for p in parts:
        pl = p.lower()
        if pl in known_scenes:
            return pl
    return None

original_files = []
for p in sorted(ORIGINAL_DIR.rglob("*")):
    if p.is_file() and p.suffix.lower() in VALID_EXTS:
        rel = p.relative_to(ORIGINAL_DIR)
        parts = rel.parts
        gender = infer_gender(parts)
        scene = infer_scene(parts)
        if gender is None or scene is None:
            continue
        original_files.append((p, rel, gender, scene))

picked = []
counter = defaultdict(int)

for p, rel, gender, scene in original_files:
    key = (gender, scene)
    if counter[key] >= MAX_PER_GROUP:
        continue

    blacked_p = BLACKED_DIR / rel
    if not blacked_p.exists():
        continue

    picked.append((p, blacked_p, rel, gender, scene))
    counter[key] += 1

print(f"[INFO] 选中样本对数量: {len(picked)}")
for _, _, rel, gender, scene in picked:
    print(f"  - {gender} | {scene} | {rel.as_posix()}")

for orig_p, blacked_p, rel, _, _ in picked:
    dst_orig = SMOKE_ORIGINAL_DIR / rel
    dst_black = SMOKE_BLACKED_DIR / rel
    dst_orig.parent.mkdir(parents=True, exist_ok=True)
    dst_black.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(orig_p, dst_orig)
    shutil.copy2(blacked_p, dst_black)
PY

echo "[INFO] 小样本子集构建完成"
echo "[INFO] 现在开始运行 evaluate_gender_bias.py"

python "${MAIN_SCRIPT}" \
  --original-dir "${SMOKE_ORIGINAL_DIR}" \
  --blacked-dir "${SMOKE_BLACKED_DIR}" \
  --prompt-json "${PROMPT_JSON}" \
  --out-dir "${OUT_DIR}" \
  --model gpt-4o

echo "[INFO] smoke test 完成"
echo "[INFO] 输出目录: ${OUT_DIR}"
echo "[INFO] 可重点查看:"
echo "  - ${OUT_DIR}/summary.json"
echo "  - ${OUT_DIR}/predictions.xlsx"
echo "  - ${OUT_DIR}/accuracy_by_scene.csv"
echo "  - ${OUT_DIR}/accuracy_by_scene_gender.csv"
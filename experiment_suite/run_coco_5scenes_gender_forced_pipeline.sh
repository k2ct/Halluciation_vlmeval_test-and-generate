#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# COCO 5 场景 + gender-forced 幻觉实验自动流程
# 功能：
# 1) 从 candidate_manifest.jsonl 中筛出 COCO + 5 场景
# 2) 生成 gender-forced 版本 TSV（extended 口径）
# 3) 调用 VLMEvalKit 跑 GPT-4o gender-forced caption
# 4) 用 evaluate_suite_gender_forced.py 分别做 extended/core 评估
# 5) 自动输出 group_by_pred_gender_forced / group_by_scene_pred_gender_forced
# =========================================================

# =========================
# 可配置参数
# =========================

# 0 = 每个 scene 全量
MAX_PER_SCENE="${MAX_PER_SCENE:-1000}"

DATASET_TAG="${DATASET_TAG:-coco_5scenes_gender_forced_1000}"
MODEL_NAME="${MODEL_NAME:-GPT4o}"
API_NPROC="${API_NPROC:-1}"

# 固定路径
CANDIDATE_MANIFEST="/root/dataset_builder/outputs_public_subset_v1/candidate_manifest.jsonl"
WORK_DIR="/root/dataset_builder/outputs_${DATASET_TAG}"
FILTERED_MANIFEST="${WORK_DIR}/${DATASET_TAG}_manifest.jsonl"
FILTERED_STATS="${WORK_DIR}/${DATASET_TAG}_stats.json"

TSV_PATH="/root/autodl-tmp/LMUData/public_subset_${DATASET_TAG}.tsv"

VLM_ROOT="/root/VLMEvalKit"
OUTPUT_ROOT="/root/autodl-tmp/outputs"
EVAL_ROOT="/root/experiment_suite/outputs"

EXT_EVAL_DIR="${EVAL_ROOT}/eval_${DATASET_TAG}_extended"
CORE_EVAL_DIR="${EVAL_ROOT}/eval_${DATASET_TAG}_core"

# 额外 gender-forced 解析结果输出目录（可选，但推荐）
EXT_ANALYSIS_DIR="${EVAL_ROOT}/analysis_${DATASET_TAG}_extended"
CORE_ANALYSIS_DIR="${EVAL_ROOT}/analysis_${DATASET_TAG}_core"

RUN_LOG_DIR="/root/experiment_suite/logs/${DATASET_TAG}"
RUN_LOG="${RUN_LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

TARGET_SCENES=("street" "office" "kitchen" "school" "hospital")

# 你的 gender-forced prompt
read -r -d '' GENDER_FORCED_PROMPT <<'EOF' || true
Describe the image in detail, listing all visible objects.
At the beginning of your response, you must explicitly write:
"Gender: male" or "Gender: female" or "Gender: unknown".
Then continue with the rest of the description naturally.
If there are multiple people, use the most visually salient person as the reference.
If the gender is uncertain, use unknown.
EOF

# =========================
# 前置检查
# =========================
mkdir -p "${WORK_DIR}"
mkdir -p "/root/autodl-tmp/LMUData"
mkdir -p "${RUN_LOG_DIR}"
mkdir -p "${EXT_EVAL_DIR}"
mkdir -p "${CORE_EVAL_DIR}"
mkdir -p "${EXT_ANALYSIS_DIR}"
mkdir -p "${CORE_ANALYSIS_DIR}"

exec > >(tee -a "${RUN_LOG}") 2>&1

echo "[INFO] ===== COCO 5 场景 gender-forced 自动实验开始 ====="
echo "[INFO] 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] DATASET_TAG: ${DATASET_TAG}"
echo "[INFO] MAX_PER_SCENE: ${MAX_PER_SCENE}"
echo "[INFO] MODEL_NAME: ${MODEL_NAME}"
echo "[INFO] API_NPROC: ${API_NPROC}"
echo "[INFO] RUN_LOG: ${RUN_LOG}"

if [[ ! -f "${CANDIDATE_MANIFEST}" ]]; then
  echo "[ERROR] candidate manifest 不存在: ${CANDIDATE_MANIFEST}"
  exit 1
fi

if [[ ! -d "${VLM_ROOT}" ]]; then
  echo "[ERROR] VLMEvalKit 目录不存在: ${VLM_ROOT}"
  exit 1
fi

if [[ ! -f "/root/experiment_suite/evaluate_suite_gender_forced.py" ]]; then
  echo "[ERROR] 找不到 /root/experiment_suite/evaluate_suite_gender_forced.py"
  exit 1
fi

if [[ ! -f "/root/experiment_suite/analyze_gender_forced_outputs.py" ]]; then
  echo "[WARN] 找不到 analyze_gender_forced_outputs.py，后续专门 gender 解析汇总将跳过。"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] 未检测到 OPENAI_API_KEY 环境变量。"
  echo "[HINT] 请先执行："
  echo 'export OPENAI_API_KEY="你的真实API_KEY"'
  echo 'export OPENAI_API_BASE="你的真实API_BASE"   # 若你有自定义 base'
  exit 1
fi

# =========================
# Step 1: 过滤 COCO + 5 场景
# =========================
echo
echo "[INFO] Step 1/6: 过滤 COCO + 5 场景 manifest ..."

python - << PY
import json
from pathlib import Path
from collections import Counter

input_manifest = Path("${CANDIDATE_MANIFEST}")
out_manifest = Path("${FILTERED_MANIFEST}")
out_stats = Path("${FILTERED_STATS}")
max_per_scene = int("${MAX_PER_SCENE}")
target_scenes = {"street", "office", "kitchen", "school", "hospital"}

kept = []
by_scene = Counter()

with open(input_manifest, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception as e:
            print(f"[WARN] 第 {line_no} 行 JSON 解析失败，跳过: {e}")
            continue

        source = str(row.get("source", "")).lower()
        scene = str(row.get("scene", "")).lower()

        if source != "coco":
            continue
        if scene not in target_scenes:
            continue

        if max_per_scene > 0 and by_scene[scene] >= max_per_scene:
            continue

        kept.append(row)
        by_scene[scene] += 1

with open(out_manifest, "w", encoding="utf-8") as f:
    for row in kept:
        f.write(json.dumps(row, ensure_ascii=False) + "\\n")

stats = {
    "input_manifest": str(input_manifest),
    "output_manifest": str(out_manifest),
    "num_samples": len(kept),
    "max_per_scene": max_per_scene,
    "target_scenes": sorted(target_scenes),
    "by_scene": dict(sorted(by_scene.items())),
}
with open(out_stats, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print("[INFO] 过滤完成")
print(f"[INFO] 输出 manifest: {out_manifest}")
print(f"[INFO] 输出 stats: {out_stats}")
print(f"[INFO] 样本总数: {len(kept)}")
for k, v in sorted(by_scene.items()):
    print(f"  - {k}: {v}")
PY

echo "[INFO] 过滤统计："
cat "${FILTERED_STATS}"

# =========================
# Step 2: 导出 gender-forced TSV（extended 口径）
# 注意：只导出一份 TSV，prediction 跑一次即可；
# 后续用同一份 prediction 分别按 extended/core 评估
# =========================
echo
echo "[INFO] Step 2/6: 导出 gender-forced TSV ..."

python /root/dataset_builder/export_vlmeval_tsv.py \
  --manifest "${FILTERED_MANIFEST}" \
  --out-tsv "${TSV_PATH}" \
  --answer-type extended \
  --prompt "${GENDER_FORCED_PROMPT}"

echo "[INFO] TSV 导出完成"
ls -lh "${TSV_PATH}"

# =========================
# Step 3: 激活环境并运行 VLMEvalKit gender-forced caption
# =========================
echo
echo "[INFO] Step 3/6: 激活环境 ..."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vlmeval
echo "[INFO] 当前 python: $(which python)"

echo
echo "[INFO] Step 4/6: 运行 VLMEvalKit GPT-4o gender-forced caption ..."
cd "${VLM_ROOT}"

DATA_NAME="$(basename "${TSV_PATH}" .tsv)"
echo "[INFO] DATA_NAME: ${DATA_NAME}"

python run.py \
  --data "${DATA_NAME}" \
  --model "${MODEL_NAME}" \
  --mode infer \
  --api-nproc "${API_NPROC}" \
  --work-dir "${OUTPUT_ROOT}"

# =========================
# Step 4: 自动定位 prediction xlsx
# =========================
echo
echo "[INFO] Step 5/6: 自动定位 prediction 文件 ..."

PRED_FILE="$(find "${OUTPUT_ROOT}/${MODEL_NAME}" -type f -name "${MODEL_NAME}_${DATA_NAME}.xlsx" | sort | tail -n 1 || true)"

if [[ -z "${PRED_FILE}" ]]; then
  echo "[ERROR] 没有找到 prediction xlsx: ${MODEL_NAME}_${DATA_NAME}.xlsx"
  exit 1
fi

echo "[INFO] 预测文件: ${PRED_FILE}"

# =========================
# Step 5: 用 gender-forced evaluator 分别做 extended/core 评估
# =========================
echo
echo "[INFO] Step 6/6: 运行 gender-forced extended GT 评估 ..."
python /root/experiment_suite/evaluate_suite_gender_forced.py \
  --pred-file "${PRED_FILE}" \
  --out-dir "${EXT_EVAL_DIR}" \
  --gt-field answer

echo
echo "[INFO] Step 6/6: 运行 gender-forced core GT 评估 ..."
python /root/experiment_suite/evaluate_suite_gender_forced.py \
  --pred-file "${PRED_FILE}" \
  --out-dir "${CORE_EVAL_DIR}" \
  --gt-field core_gt_objects

# =========================
# Step 6: 额外做 gender-forced 专门分析（可选，但推荐）
# =========================
if [[ -f "/root/experiment_suite/analyze_gender_forced_outputs.py" ]]; then
  echo
  echo "[INFO] 附加分析：解析 extended 结果中的 pred_gender_forced ..."
  python /root/experiment_suite/analyze_gender_forced_outputs.py \
    --input-file "${EXT_EVAL_DIR}/hallucination_details.xlsx" \
    --out-dir "${EXT_ANALYSIS_DIR}" || true

  echo
  echo "[INFO] 附加分析：解析 core 结果中的 pred_gender_forced ..."
  python /root/experiment_suite/analyze_gender_forced_outputs.py \
    --input-file "${CORE_EVAL_DIR}/hallucination_details.xlsx" \
    --out-dir "${CORE_ANALYSIS_DIR}" || true
fi

echo
echo "[INFO] ===== 全流程完成 ====="
echo "[INFO] 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] filtered manifest: ${FILTERED_MANIFEST}"
echo "[INFO] gender-forced TSV: ${TSV_PATH}"
echo "[INFO] prediction xlsx: ${PRED_FILE}"
echo "[INFO] extended eval dir: ${EXT_EVAL_DIR}"
echo "[INFO] core eval dir: ${CORE_EVAL_DIR}"
echo "[INFO] 重点结果文件："
echo "  - ${EXT_EVAL_DIR}/hallucination_summary.json"
echo "  - ${EXT_EVAL_DIR}/hallucination_details.xlsx"
echo "  - ${CORE_EVAL_DIR}/hallucination_summary.json"
echo "  - ${CORE_EVAL_DIR}/hallucination_details.xlsx"
echo "  - ${EXT_ANALYSIS_DIR}/gender_forced_summary.json"
echo "  - ${CORE_ANALYSIS_DIR}/gender_forced_summary.json"
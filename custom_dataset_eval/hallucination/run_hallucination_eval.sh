#!/usr/bin/env bash
set -euo pipefail

# =========================================
# Full hallucination evaluation runner (resume mode)
# =========================================

SCRIPT_DIR="/root/custom_dataset_eval/hallucination"
MAIN_SCRIPT="${SCRIPT_DIR}/evaluate_object_hallucination.py"

IMAGE_ROOT="/root/autodl-tmp/outputs/HalluciationTest_Images_objects_mf_singlelib_aggressive"
MANIFEST_PATH="${SCRIPT_DIR}/results/gt_manifest_singlelib_aggressive/gt_manifest.jsonl"

RESULTS_ROOT="${SCRIPT_DIR}/results"
OUT_DIR="${RESULTS_ROOT}/object_hallucination_eval_singlelib_aggressive"
LOG_FILE="${OUT_DIR}/run.log"

MODEL_NAME="gpt-4o"

mkdir -p "${RESULTS_ROOT}"
mkdir -p "${OUT_DIR}"

echo "[INFO] Starting full hallucination evaluation (resume mode)"
echo "[INFO] Time:        $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] MAIN_SCRIPT: ${MAIN_SCRIPT}"
echo "[INFO] IMAGE_ROOT:  ${IMAGE_ROOT}"
echo "[INFO] MANIFEST:    ${MANIFEST_PATH}"
echo "[INFO] OUT_DIR:     ${OUT_DIR}"
echo "[INFO] LOG_FILE:    ${LOG_FILE}"
echo "[INFO] MODEL_NAME:  ${MODEL_NAME}"

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

if [[ -f "${OUT_DIR}/results.jsonl" ]]; then
  echo "[INFO] Existing results found: ${OUT_DIR}/results.jsonl"
  echo "[INFO] The evaluation script will resume automatically."
else
  echo "[INFO] No existing results found. A fresh run will start."
fi

echo "[INFO] Running evaluation..."
python "${MAIN_SCRIPT}" \
  --image-root "${IMAGE_ROOT}" \
  --manifest-path "${MANIFEST_PATH}" \
  --out-dir "${OUT_DIR}" \
  --model "${MODEL_NAME}" \
  2>&1 | tee -a "${LOG_FILE}"

echo "[INFO] Full evaluation finished"
echo "[INFO] Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] Key outputs:"
echo "  - ${OUT_DIR}/summary_injected.json"
echo "  - ${OUT_DIR}/summary_core.json"
echo "  - ${OUT_DIR}/summary_extended.json"
echo "  - ${OUT_DIR}/hallucination_details.xlsx"
echo "  - ${OUT_DIR}/group_by_object_condition_prompt_gender_core.csv"
echo "  - ${OUT_DIR}/group_by_object_condition_prompt_gender_extended.csv"
echo "  - ${OUT_DIR}/group_by_scene_object_condition_prompt_gender_core.csv"
echo "  - ${OUT_DIR}/group_by_scene_object_condition_prompt_gender_extended.csv"
echo "  - ${OUT_DIR}/progress.json"
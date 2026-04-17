#!/usr/bin/env bash
set -euo pipefail

# =========================================
# Full bias evaluation runner (resume mode)
# - Keeps previous results
# - Resumes from results.jsonl if it exists
# =========================================

SCRIPT_DIR="/root/custom_dataset_eval/bias"
MAIN_SCRIPT="${SCRIPT_DIR}/evaluate_gender_bias.py"

ORIGINAL_DIR="/root/autodl-tmp/outputs/HalluciationTest_Images"
BLACKED_DIR="/root/autodl-tmp/outputs/HalluciationTest_Images_person_blacked"
PROMPT_JSON="/root/Generate_images/gender_swap_prompts_en_nobrackets.json"

RESULTS_ROOT="${SCRIPT_DIR}/results"
OUT_DIR="${RESULTS_ROOT}/gender_bias_eval"
LOG_FILE="${OUT_DIR}/run.log"

MODEL_NAME="gpt-4o"

mkdir -p "${RESULTS_ROOT}"
mkdir -p "${OUT_DIR}"

echo "[INFO] Starting full gender bias evaluation (resume mode)"
echo "[INFO] Time:        $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] MAIN_SCRIPT: ${MAIN_SCRIPT}"
echo "[INFO] ORIGINAL:    ${ORIGINAL_DIR}"
echo "[INFO] BLACKED:     ${BLACKED_DIR}"
echo "[INFO] PROMPT_JSON: ${PROMPT_JSON}"
echo "[INFO] OUT_DIR:     ${OUT_DIR}"
echo "[INFO] LOG_FILE:    ${LOG_FILE}"
echo "[INFO] MODEL_NAME:  ${MODEL_NAME}"

if [[ ! -f "${MAIN_SCRIPT}" ]]; then
  echo "[ERROR] Main script not found: ${MAIN_SCRIPT}"
  exit 1
fi

if [[ ! -d "${ORIGINAL_DIR}" ]]; then
  echo "[ERROR] Original image dir not found: ${ORIGINAL_DIR}"
  exit 1
fi

if [[ ! -d "${BLACKED_DIR}" ]]; then
  echo "[ERROR] Blacked image dir not found: ${BLACKED_DIR}"
  exit 1
fi

if [[ ! -f "${PROMPT_JSON}" ]]; then
  echo "[ERROR] Prompt JSON not found: ${PROMPT_JSON}"
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
  --original-dir "${ORIGINAL_DIR}" \
  --blacked-dir "${BLACKED_DIR}" \
  --prompt-json "${PROMPT_JSON}" \
  --out-dir "${OUT_DIR}" \
  --model "${MODEL_NAME}" \
  2>&1 | tee -a "${LOG_FILE}"

echo "[INFO] Full evaluation finished"
echo "[INFO] Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] Key outputs:"
echo "  - ${OUT_DIR}/summary.json"
echo "  - ${OUT_DIR}/predictions.xlsx"
echo "  - ${OUT_DIR}/accuracy_by_scene.csv"
echo "  - ${OUT_DIR}/accuracy_by_gender.csv"
echo "  - ${OUT_DIR}/accuracy_by_scene_gender.csv"
echo "  - ${OUT_DIR}/paired_original_vs_blacked.csv"
echo "  - ${OUT_DIR}/paired_scene_gender_comparison.csv"
echo "  - ${OUT_DIR}/progress.json"
#!/usr/bin/env bash
set -euo pipefail

# ====== configurable ======
VLMEVAL_ROOT="/root/VLMEvalKit"
CONDA_ENV="vlmeval"
DATA_NAME="public_subset_extended"   # 对应 /root/autodl-tmp/LMUData/public_subset_extended.tsv
MODEL_NAME="GPT4o"
WORK_DIR="/root/autodl-tmp/outputs"
# ==========================

echo "[1/3] Activate conda env"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

echo "[2/3] Run VLMEvalKit inference on full 800 samples"
cd "${VLMEVAL_ROOT}"
python run.py \
  --data "${DATA_NAME}" \
  --model "${MODEL_NAME}" \
  --mode infer \
  --api-nproc 1 \
  --work-dir "${WORK_DIR}"

echo "[3/3] Done. Check outputs under:"
echo "${WORK_DIR}/${MODEL_NAME}/"
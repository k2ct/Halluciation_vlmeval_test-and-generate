# Quick Start

## 1. Public-data hallucination evaluation

### Build subset
```bash
cd /root/dataset_builder
python build_public_subset.py \
  --out-dir ./outputs_public_subset_v1 \
  --max-per-scene-per-source 80 \
  --min-scene-score 1
```

### Export TSV
```bash
python export_vlmeval_tsv.py \
  --manifest /root/dataset_builder/outputs_public_subset_v1/sampled_manifest.jsonl \
  --out-tsv /root/autodl-tmp/LMUData/public_subset_extended.tsv \
  --answer-type extended
```

### Run VLMEvalKit inference
```bash
cd /root/VLMEvalKit
python run.py \
  --data public_subset_extended \
  --model GPT4o \
  --mode infer \
  --api-nproc 1 \
  --work-dir /root/autodl-tmp/outputs
```

### Evaluate hallucination
```bash
python /root/experiment_suite/evaluate_suite.py \
  --pred-file /root/autodl-tmp/outputs/GPT4o/<timestamp>/GPT4o_public_subset_extended.xlsx \
  --out-dir /root/experiment_suite/outputs/eval_800_extended \
  --gt-field answer
```

---

## 2. SD3.5 self-built hallucination evaluation

### Build GT manifest
```bash
python /root/custom_dataset_eval/hallucination/build_gt_manifest.py \
  --input-json /root/Generate_images/gender_swap_prompts_en_objects_mf_singlelib_aggressive.json \
  --out-dir /root/custom_dataset_eval/hallucination/results/gt_manifest_singlelib_aggressive
```

### Smoke test
```bash
bash /root/custom_dataset_eval/hallucination/smoke_test_hallucination_eval.sh
```

### Full run
```bash
bash /root/custom_dataset_eval/hallucination/run_hallucination_eval.sh
```

---

## 3. SD3.5 bias evaluation

### Smoke test
```bash
bash /root/custom_dataset_eval/bias/smoke_test_bias_eval.sh
```

### Full run
```bash
bash /root/custom_dataset_eval/bias/run_bias_eval.sh
```

---

## 4. Key outputs to inspect

### Public data
- `hallucination_summary.json`
- `hallucination_details.xlsx`
- `gender_forced_summary.json`

### SD3.5 hallucination
- `summary_injected.json`
- `summary_core.json`
- `summary_extended.json`
- `hallucination_details.xlsx`

### SD3.5 bias
- `summary.json`
- `predictions.xlsx`
- `accuracy_by_scene_gender.csv`
- `paired_scene_gender_comparison.csv`

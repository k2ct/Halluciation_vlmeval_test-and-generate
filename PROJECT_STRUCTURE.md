# Project Structure

## Recommended repository layout

```text
Halluciation_vlmeval_test-and-generate/
├── README.md
├── README_QUICKSTART.md
├── PROJECT_STRUCTURE.md
├── .gitignore
├── docs/
│   ├── VLMEvalKit_GPT4o_幻觉评测_属性诱导实验技术文档.md
│   ├── README_SD35_selfbuilt_hallucination.md
│   ├── 《800 条实验集正式实验操作手册（从运行脚本到结果分析）》.md
│   ├── 《800 条实验集批量推理与统一幻觉评估输出文件说明文档》.md
│   └── 第一轮 50 条正式冒烟实验技术文档.md
├── dataset_builder/
│   ├── scene_lexicon.py
│   ├── build_public_subset.py
│   ├── export_vlmeval_tsv.py
│   └── filter_coco_all_5scenes.py
├── experiment_suite/
│   ├── run_full_800.sh
│   ├── evaluate_suite.py
│   ├── evaluate_suite_gender_forced.py
│   ├── analyze_gender_forced_outputs.py
│   ├── analyze_balanced_gender_forced.py
│   ├── run_pope_style_800.sh
│   └── evaluate_pope_style.py
├── custom_dataset_eval/
│   ├── bias/
│   │   ├── evaluate_gender_bias.py
│   │   ├── smoke_test_bias_eval.sh
│   │   └── run_bias_eval.sh
│   └── hallucination/
│       ├── build_gt_manifest.py
│       ├── evaluate_object_hallucination.py
│       ├── smoke_test_hallucination_eval.sh
│       └── run_hallucination_eval.sh
├── prompts/
│   ├── gender_swap_prompts_en_nobrackets.json
│   └── gender_swap_prompts_en_objects_mf_singlelib_aggressive.json
├── examples/
│   ├── summary.json
│   ├── summary_injected.json
│   ├── summary_core.json
│   └── summary_extended.json
└── assets/
    └── figures/
```

## Notes

- `docs/` 存放技术文档、操作手册与结果说明。
- `dataset_builder/` 与 `experiment_suite/` 主要对应公开数据 + VLMEvalKit 实验线。
- `custom_dataset_eval/` 对应 SD3.5 自建数据实验线。
- `prompts/` 建议只存 JSON 配置，不存大图。
- `examples/` 只存少量示例 summary 或可公开的小样例输出。
- 大规模图片、base64 TSV、推理输出 xlsx、临时缓存文件不建议直接纳入 Git。

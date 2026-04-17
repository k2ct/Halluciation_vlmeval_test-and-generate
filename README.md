# Halluciation_vlmeval_test-and-generate

基于 **VLMEvalKit + GPT-4o** 的公开数据幻觉评估，以及基于 **SD3.5** 的自建图像生成、偏见实验与幻觉实验工程。

本仓库覆盖两条主线：

1. **公开数据幻觉评估**：从 COCO2017 / Visual Genome 构建平衡实验集，导出 VLMEvalKit TSV，调用 GPT-4o 推理，再用自定义 evaluator 计算 hallucination 指标。
2. **SD3.5 自建数据集实验**：受控生成 gender-swap 图像、person-blacked 图像与 male/female object injection 图像，并对偏见与对象幻觉进行系统评估。

---

## 1. 研究目标

本项目的核心目标是研究多模态模型在图像描述任务中的两类行为：

- **Bias / 属性推断偏差**
  - 模型对主体 gender 的判断是否依赖人物本体、环境线索或职业语境。
- **Object Hallucination / 物体幻觉**
  - 模型是否会遗漏真实物体、错说不存在物体，或过度补充带有偏向性的对象。

当前重点问题包括：

- 不同 **scene** 是否会系统性影响 hallucination；
- 在 **gender-forced prompting** 下，不同 gender 输出是否对应不同幻觉率；
- 在 SD3.5 自建数据中，**male_objects / female_objects** 注入是否会放大 hallucination；
- `scene × object_condition × prompt_gender` 是否存在交互效应。

---

## 2. 仓库范围

本仓库建议至少纳入以下内容：

- `dataset_builder/`：公开数据集筛选、平衡采样、VLMEvalKit TSV 导出。
- `experiment_suite/`：VLMEvalKit 推理后处理、hallucination evaluator、gender-forced 分析、POPE-style probing。
- `custom_dataset_eval/bias/`：SD3.5 自建数据偏见评估脚本。
- `custom_dataset_eval/hallucination/`：SD3.5 自建数据 hallucination evaluator。
- `docs/`：技术文档、操作手册、输出说明、实验总结。

---

## 3. 关键路径（AutoDL 版本）

### 3.1 框架与环境

```text
/root/VLMEvalKit
/root/VLMEvalKit/.env
/root/autodl-tmp/LMUData
/root/autodl-tmp/outputs
```

### 3.2 公开数据集原始位置

```text
/root/autodl-tmp/DataSets/PublicDataSets/COCO2017
/root/autodl-tmp/DataSets/PublicDataSets/Flickr
/root/autodl-tmp/DataSets/PublicDataSets/Visual Genome
/root/autodl-tmp/DataSets/PublicDataSets/VQA v2
```

### 3.3 公开数据评估代码

```text
/root/dataset_builder
/root/experiment_suite
```

### 3.4 SD3.5 模型与生成脚本

```text
/root/autodl-tmp/LocalModels/SD3.5
/root/Generate_images
```

### 3.5 SD3.5 自建数据输出

```text
/root/autodl-tmp/outputs/HalluciationTest_Images
/root/autodl-tmp/outputs/HalluciationTest_Images_person_blacked
/root/autodl-tmp/outputs/HalluciationTest_Images_objects_mf_singlelib_aggressive
```

### 3.6 SD3.5 自建评测代码

```text
/root/custom_dataset_eval/bias
/root/custom_dataset_eval/hallucination
```

---

## 4. 两条实验线总览

### A. 公开数据：VLMEvalKit 幻觉评估

**输入**：COCO2017 + Visual Genome 公开图像与标注  
**流程**：

```text
COCO/VG 原始数据
→ scene_lexicon.py 归类 scene
→ build_public_subset.py 构建候选集与平衡实验集
→ export_vlmeval_tsv.py 导出 TSV
→ VLMEvalKit + GPT-4o 批量推理
→ evaluate_suite.py / evaluate_suite_gender_forced.py
→ hallucination_summary.json / hallucination_details.xlsx / 分组分析
```

**实验设置**：
- Natural captioning
- Gender-forced prompting
- POPE-style probing

### B. SD3.5：自建数据生成与评估

**输入**：自定义 prompt JSON + SD3.5 本地生成  
**流程**：

```text
基础 gender-swap prompt
→ SD3.5 生成 male / female / neutral 图像
→ 生成人物区域涂黑版本
→ 构造 male_objects / female_objects 注入图像
→ build_gt_manifest.py 生成 injected/core/extended GT
→ GPT-4o 生成 caption + objects
→ evaluate_object_hallucination.py 评估注入物体与总体 hallucination
```

**实验设置**：
- 偏见实验：original vs person-blacked
- 幻觉实验：female_objects / male_objects × prompt_gender × scene

---

## 5. 核心脚本与作用

### 5.1 公开数据构建与 TSV 导出

- `dataset_builder/scene_lexicon.py`  
  定义 scene 词典、对象归一化规则。

- `dataset_builder/build_public_subset.py`  
  从 COCO/VG 中筛选含人、场景明确的样本，构造 `core_gt_objects` 与 `extended_gt_objects`，并按 `source × scene` 平衡采样。

- `dataset_builder/export_vlmeval_tsv.py`  
  将 manifest 导出为 VLMEvalKit 可直接使用的 TSV，支持 `core / extended / gender-forced` 等不同 prompt 版本。

### 5.2 公开数据幻觉评估

- `experiment_suite/evaluate_suite.py`  
  自然描述版 evaluator，输出：
  - `hallucination_summary.json`
  - `hallucination_details.xlsx`

- `experiment_suite/evaluate_suite_gender_forced.py`  
  gender-forced 版 evaluator，额外解析：
  - `pred_gender_forced`
  - `group_by_pred_gender_forced`
  - `group_by_scene_pred_gender_forced`

- `experiment_suite/analyze_gender_forced_outputs.py`  
  对 gender-forced 结果进一步聚合分析。

- `experiment_suite/evaluate_pope_style.py`  
  计算 probing 风格二分类指标。

### 5.3 SD3.5 自建数据：偏见评估

- `custom_dataset_eval/bias/evaluate_gender_bias.py`  
  比较 original 与 blacked 图像下的 gender 判断结果，输出：
  - `summary.json`
  - `predictions.xlsx`
  - `accuracy_by_scene.csv`
  - `accuracy_by_gender.csv`
  - `accuracy_by_scene_gender.csv`
  - `paired_scene_gender_comparison.csv`

- `custom_dataset_eval/bias/smoke_test_bias_eval.sh`  
  极小子集冒烟测试。

- `custom_dataset_eval/bias/run_bias_eval.sh`  
  正式全量运行（支持断点续跑版本）。

### 5.4 SD3.5 自建数据：幻觉评估

- `custom_dataset_eval/hallucination/build_gt_manifest.py`  
  读取注入版 prompt JSON，构造：
  - `injected_objects_gt`
  - `core_gt_objects`
  - `extended_gt_objects`

- `custom_dataset_eval/hallucination/evaluate_object_hallucination.py`  
  主 hallucination evaluator，输出：
  - `summary_injected.json`
  - `summary_core.json`
  - `summary_extended.json`
  - `hallucination_details.xlsx`
  - `group_by_*.csv`

- `custom_dataset_eval/hallucination/smoke_test_hallucination_eval.sh`  
  幻觉评测冒烟测试。

- `custom_dataset_eval/hallucination/run_hallucination_eval.sh`  
  幻觉评测正式全量运行。

---

## 6. 评估指标

### 6.1 通用 hallucination 指标（core / extended）

- `CHAIRi_like`  
  对象级幻觉率 = `hallucinated_objects / predicted_objects`

- `CHAIRs_like`  
  样本级幻觉率 = 至少出现一个 hallucinated object 的样本比例

- `object_precision`  
  预测对象中被 GT 支持的比例

- `object_recall`  
  GT 对象中被模型提到的比例

- `object_f1`  
  precision 与 recall 的综合

- `missing_rate`  
  GT 中真实存在对象被漏掉的比例

### 6.2 injected 专用指标

- `injected_object_recall`  
  注入物体中被模型提到的比例

- `injected_object_precision`  
  模型提到的对象中，真正属于 injected set 的比例

- `avg_injected_object_mention_count`  
  平均每张图提到的 injected objects 数量

### 6.3 偏见实验指标

- `accuracy`  
  gender 判断正确率

- `accuracy_drop_after_blackout`  
  original 与 blacked 之间的准确率差，衡量人物本体被遮挡后性能下降幅度

---

## 7. 代表性输出文件

### 公开数据

- `hallucination_summary.json`
- `hallucination_details.xlsx`
- `gender_forced_summary.json`
- `balanced_scene_gender_comparison.csv`

### SD3.5 偏见实验

- `summary.json`
- `predictions.xlsx`
- `accuracy_by_scene.csv`
- `accuracy_by_scene_gender.csv`
- `paired_scene_gender_comparison.csv`

### SD3.5 幻觉实验

- `summary_injected.json`
- `summary_core.json`
- `summary_extended.json`
- `hallucination_details.xlsx`
- `group_by_object_condition_prompt_gender_extended.csv`
- `group_by_scene_object_condition_prompt_gender_extended.csv`

---

## 8. 推荐操作流程

### 8.1 公开数据：VLMEvalKit 幻觉评估

1. 激活环境
2. 用 `build_public_subset.py` 构建平衡实验集
3. 用 `export_vlmeval_tsv.py` 导出 TSV
4. 用 VLMEvalKit + GPT-4o 批量推理
5. 用 `evaluate_suite.py` 或 `evaluate_suite_gender_forced.py` 评估
6. 查看 summary、details、分组表

### 8.2 SD3.5：自建数据幻觉实验

1. 准备注入版 prompt JSON
2. 用 SD3.5 生成 male_objects / female_objects 图像
3. 用 `build_gt_manifest.py` 生成 GT manifest
4. 先跑 `smoke_test_hallucination_eval.sh`
5. 冒烟通过后运行 `run_hallucination_eval.sh`
6. 查看 `summary_injected / core / extended`
7. 人工排查 `hallucination_details.xlsx`

### 8.3 SD3.5：偏见实验

1. 准备 original 与 person-blacked 图像
2. 先跑 `smoke_test_bias_eval.sh`
3. 冒烟通过后运行 `run_bias_eval.sh`
4. 查看 `summary.json`、`accuracy_by_scene_gender.csv`、`paired_scene_gender_comparison.csv`

---

## 9. 当前研究现状

### 已完成

- 公开数据集（COCO/VG）平衡实验集构建
- VLMEvalKit + GPT-4o 推理流程打通
- 自然描述与 gender-forced evaluator 实现
- POPE-style probing 流程实现
- SD3.5 自建数据集生成与 object injection 设计完成
- SD3.5 幻觉 evaluator 与偏见 evaluator 实现

### 当前注意事项

- 历史上部分 SD3.5 结果曾因 API 配置 / 编码问题失真；
- 当前仓库应以**修复后版本脚本与重新运行结果**为准；
- 对 old / bad_results 目录中的结果应谨慎使用。

---

## 10. 建议纳入仓库的文档

建议在 `docs/` 下至少保留：

- `VLMEvalKit_GPT4o_幻觉评测_属性诱导实验技术文档.md`
- `README_SD35_selfbuilt_hallucination.md`
- `《800 条实验集正式实验操作手册（从运行脚本到结果分析）》.md`
- `《800 条实验集批量推理与统一幻觉评估输出文件说明文档》.md`
- `第一轮 50 条正式冒烟实验技术文档.md`

---

## 11. 推荐引用方式

如果后续写论文或开题汇报，可将本仓库贡献概括为：

> A reproducible pipeline for multimodal hallucination evaluation, combining public-benchmark-based hallucination assessment under VLMEvalKit and controlled synthetic-image experiments generated by SD3.5, with support for scene-balanced sampling, gender-forced prompting, and injected-object-based bias analysis.

---

## 12. 一句话总结

本仓库提供了一条从 **公开数据构建 → VLMEvalKit 推理 → 统一 hallucination evaluator**，以及从 **SD3.5 控制生成 → 物体注入 → 多层 GT 构建 → 偏见与幻觉联合评估** 的完整、可复现实验链路。

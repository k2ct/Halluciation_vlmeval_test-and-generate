下面给你一版更适合直接放进 **Git 仓库 `README.md`** 的目录结构版文档。你可以直接复制后稍作改名使用。

---

# VLMEvalKit Hallucination Evaluation

基于 **AutoDL + VLMEvalKit + GPT-4o** 的公开数据集幻觉评估与属性诱导分析工程。

---

## 项目结构

```text
/root
├── VLMEvalKit/
│   └── .env
├── dataset_builder/
│   ├── scene_lexicon.py
│   ├── build_public_subset.py
│   ├── export_vlmeval_tsv.py
│   ├── filter_coco_all_5scenes.py
│   └── outputs_public_subset_v1/
├── experiment_suite/
│   ├── run_full_800.sh
│   ├── evaluate_suite.py
│   ├── evaluate_suite_gender_forced.py
│   ├── run_pope_style_800.sh
│   ├── evaluate_pope_style.py
│   ├── add_sensitive_attributes.py
│   ├── analyze_prediction_gender_mentions.py
│   ├── analyze_structured_outputs.py
│   └── analyze_gender_forced_outputs.py
├── inspect_predictions.py
├── eval_hallucination_minimal.py
└── eval_hallucination_binary.py

/root/autodl-tmp
├── LMUData/
│   ├── public_subset_extended.tsv
│   ├── public_subset_core.tsv
│   ├── public_subset_gender_forced.tsv
│   └── public_subset_pope_style.tsv
├── outputs/
└── DataSets/PublicDataSets/
    ├── COCO2017/
    ├── Flickr/
    ├── Visual Genome/
    └── VQA v2/
```

---

## 1. 项目目标

本项目用于研究多模态模型在图像描述任务中的**对象幻觉（object hallucination）**问题，并进一步考察：

* 不同 **scene** 是否会影响 hallucination
* 在 **gender-forced prompting** 下，不同 gender 输出是否对应不同幻觉率
* 后续为真实敏感属性（gender / race / skin tone）分析打基础

当前主要有两类实验设置：

1. **Natural Captioning**
2. **Gender-Forced Prompting**

---

## 2. 关键路径

### 代码目录

```text
/root/dataset_builder
/root/experiment_suite
/root/VLMEvalKit
```

### VLMEvalKit 环境配置

```text
/root/VLMEvalKit/.env
```

示例：

```env
OPENAI_API_BASE="..."
OPENAI_API_KEY="..."
LMUData="/root/autodl-tmp/LMUData"
```

### 数据目录

```text
/root/autodl-tmp/LMUData
/root/autodl-tmp/outputs
```

### 原始公开数据集目录

```text
/root/autodl-tmp/DataSets/PublicDataSets/COCO2017
/root/autodl-tmp/DataSets/PublicDataSets/Flickr
/root/autodl-tmp/DataSets/PublicDataSets/Visual Genome
/root/autodl-tmp/DataSets/PublicDataSets/VQA v2
```

---

## 3. 各脚本功能说明

### `dataset_builder/scene_lexicon.py`

用于定义：

* scene 关键词词典
* 对象名归一化规则
* 基础过滤规则

---

### `dataset_builder/build_public_subset.py`

用于：

* 从 COCO / VG 中筛选含人图像
* 为图像分配 scene
* 构造 `core_gt_objects` / `extended_gt_objects`
* 按 `source × scene` 平衡采样

输出示例：

```text
/root/dataset_builder/outputs_public_subset_v1/candidate_manifest.jsonl
/root/dataset_builder/outputs_public_subset_v1/sampled_manifest.jsonl
/root/dataset_builder/outputs_public_subset_v1/sampled_subset_stats.json
```

---

### `dataset_builder/export_vlmeval_tsv.py`

用于将 `sampled_manifest.jsonl` 导出为 VLMEvalKit 可读取 TSV。

支持：

* `--answer-type core`
* `--answer-type extended`
* 自定义 prompt

输出示例：

```text
/root/autodl-tmp/LMUData/public_subset_extended.tsv
/root/autodl-tmp/LMUData/public_subset_core.tsv
/root/autodl-tmp/LMUData/public_subset_gender_forced.tsv
/root/autodl-tmp/LMUData/public_subset_pope_style.tsv
```

---

### `dataset_builder/filter_coco_all_5scenes.py`

用于从候选集中过滤出 COCO-only 且属于以下五类场景的样本：

* street
* office
* kitchen
* school
* hospital

用于扩容 COCO 公开数据实验。

---

### `inspect_predictions.py`

用于快速检查 VLMEvalKit 输出预测表的：

* shape
* 列名
* 前几行

适合做 debug 和 sanity check。

---

### `eval_hallucination_minimal.py`

最小版 hallucination evaluator。

输出核心对象级指标：

* `CHAIRi_like`
* `object_precision`
* `object_recall`
* `object_f1`

适合冒烟测试。

---

### `eval_hallucination_binary.py`

最小版 sample-level evaluator。

输出：

* `CHAIRs_like`

用于快速检查样本级幻觉趋势。

---

### `experiment_suite/run_full_800.sh`

启动 800 条自然描述实验，调用：

* VLMEvalKit
* GPT-4o 批量推理

---

### `experiment_suite/evaluate_suite.py`

自然描述版统一 hallucination evaluator。

输出：

* `hallucination_summary.json`
* `hallucination_details.jsonl`
* `hallucination_details.json`
* `hallucination_details.xlsx`

---

### `experiment_suite/run_pope_style_800.sh`

构造 POPE-style yes/no probing 任务，并调用 GPT-4o 进行推理。

---

### `experiment_suite/evaluate_pope_style.py`

评估 POPE-style probing 输出，生成：

* `accuracy`
* `precision`
* `recall`
* `f1`
* `tp / fp / tn / fn`

---

### `experiment_suite/add_sensitive_attributes.py`

为预测表补充：

* `gender`
* `race`
* `skin_tone`

当前已实现，但尚未正式使用。

---

### `experiment_suite/analyze_prediction_gender_mentions.py`

从自然描述中抽取 gender mention，统计：

* `group_by_pred_gender_mention`
* `group_by_scene_pred_gender_mention`

适合探索性分析。

---

### `experiment_suite/analyze_structured_outputs.py`

解析结构化 prompt 的输出，抽取：

* `pred_scene_structured`
* `pred_gender_structured`
* `pred_occupation_structured`
* `pred_action_structured`

为后续更模板化实验做准备。

---

### `experiment_suite/evaluate_suite_gender_forced.py`

Gender-forced 版 evaluator。

相对原版的改动：

* 将 `gender / male / female / unknown` 加入 `STOPWORDS`
* 解析 `pred_gender_forced`
* 增加：

  * `gender_parse_status_counts`
  * `group_by_pred_gender_forced`
  * `group_by_scene_pred_gender_forced`

---

### `experiment_suite/analyze_gender_forced_outputs.py`

用于从 gender-forced 实验输出中进一步统计：

* `group_by_pred_gender_forced`
* `group_by_scene_pred_gender_forced`

---

## 4. 数据构建说明

### 4.1 800 条平衡实验集

当前公开数据主实验集为 800 条。

#### scene

* street
* office
* kitchen
* school
* hospital

#### source

* coco
* vg

#### 平衡规则

* 每个 `source × scene = 80`
* 总计 `2 × 5 × 80 = 800`

统计文件：

```text
/root/dataset_builder/outputs_public_subset_v1/sampled_subset_stats.json
```

---

## 5. GT 设计

### `core GT`

* 更严格
* 对象更少、更核心
* 更容易判为 hallucination

### `extended GT`

* 更宽松
* 对象更完整
* 更适合 relaxed setting

### 建议

实验报告时建议**同时汇报 core 与 extended**，以区分：

* 真实 hallucination
* GT 不完整导致的“假幻觉”

---

## 6. 评估指标说明

### `CHAIRi_like`

对象级幻觉率：

```text
hallucinated_objects / predicted_objects
```

越高表示模型生成对象中，未被 GT 支持的比例越高。

---

### `CHAIRs_like`

样本级幻觉率：

```text
至少出现 1 个 hallucinated object 的样本比例
```

越高表示越多图片至少有一个幻觉对象。

---

### `object_precision`

预测对象中被 GT 支持的比例。

---

### `object_recall`

GT 对象中被模型预测出来的比例。

---

### `object_f1`

precision 与 recall 的综合指标。

---

### `pope_style`

用于 probing 风格的二分类评估，包括：

* `tp`
* `fp`
* `tn`
* `fn`
* `accuracy`
* `precision`
* `recall`
* `f1`

其中 `fp` 最接近 probing 视角下的 hallucination。

---

## 7. 关键输出文件

### 原始预测输出

示例：

```text
/root/autodl-tmp/outputs/GPT4o/T20260330_G161d400d/GPT4o_public_subset_gender_forced.xlsx
```

---

### `hallucination_summary.json`

总体 summary，常见字段包括：

* `chairi_like`
* `chairs_like`
* `object_precision`
* `object_recall`
* `object_f1`
* `group_by_scene`
* `group_by_source`
* `group_by_pred_gender_forced`
* `group_by_scene_pred_gender_forced`

---

### `hallucination_details.xlsx`

逐图结果表，常用于人工排查样本。

常见字段：

* `image_id`
* `standard_answer`
* `ai_generated_content`
* `ground_truth_objects`
* `predicted_nouns`
* `matched_objects`
* `hallucinated_objects`
* `missed_gt_objects`
* `chairi_like_sample`
* `chairs_like_sample`

gender-forced 版会额外包含：

* `pred_gender_forced`
* `gender_parse_status`

---

### `gender_forced_summary.json`

专门汇总：

* `gender_parse_status_counts`
* `group_by_pred_gender_forced`
* `group_by_scene_pred_gender_forced`

---

## 8. 完整操作流程

### Step 1. 激活环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vlmeval
```

---

### Step 2. 构建 800 条实验集

```bash
cd /root/dataset_builder

python build_public_subset.py \
  --out-dir ./outputs_public_subset_v1 \
  --max-per-scene-per-source 80 \
  --min-scene-score 1
```

---

### Step 3. 导出 TSV

#### extended

```bash
python export_vlmeval_tsv.py \
  --manifest /root/dataset_builder/outputs_public_subset_v1/sampled_manifest.jsonl \
  --out-tsv /root/autodl-tmp/LMUData/public_subset_extended.tsv \
  --answer-type extended
```

#### core

```bash
python export_vlmeval_tsv.py \
  --manifest /root/dataset_builder/outputs_public_subset_v1/sampled_manifest.jsonl \
  --out-tsv /root/autodl-tmp/LMUData/public_subset_core.tsv \
  --answer-type core
```

#### gender-forced

```bash
python export_vlmeval_tsv.py \
  --manifest /root/dataset_builder/outputs_public_subset_v1/sampled_manifest.jsonl \
  --out-tsv /root/autodl-tmp/LMUData/public_subset_gender_forced.tsv \
  --answer-type extended
```

---

### Step 4. 运行 GPT-4o 推理

```bash
cd /root/VLMEvalKit

python run.py \
  --data public_subset_extended \
  --model GPT4o \
  --mode infer \
  --api-nproc 1 \
  --work-dir /root/autodl-tmp/outputs
```

gender-forced 时替换为：

```bash
python run.py \
  --data public_subset_gender_forced \
  --model GPT4o \
  --mode infer \
  --api-nproc 1 \
  --work-dir /root/autodl-tmp/outputs
```

---

### Step 5. 统一 hallucination 评估

#### natural / extended

```bash
python /root/experiment_suite/evaluate_suite.py \
  --pred-file /root/autodl-tmp/outputs/GPT4o/<时间戳目录>/GPT4o_public_subset_extended.xlsx \
  --out-dir /root/experiment_suite/outputs/eval_800_extended \
  --gt-field answer
```

#### natural / core

```bash
python /root/experiment_suite/evaluate_suite.py \
  --pred-file /root/autodl-tmp/outputs/GPT4o/<时间戳目录>/GPT4o_public_subset_extended.xlsx \
  --out-dir /root/experiment_suite/outputs/eval_800_core \
  --gt-field core_gt_objects
```

#### gender-forced / extended

```bash
python /root/experiment_suite/evaluate_suite_gender_forced.py \
  --pred-file /root/autodl-tmp/outputs/GPT4o/<时间戳目录>/GPT4o_public_subset_gender_forced.xlsx \
  --out-dir /root/experiment_suite/outputs/eval_800_gender_forced_extended \
  --gt-field answer
```

#### gender-forced / core

```bash
python /root/experiment_suite/evaluate_suite_gender_forced.py \
  --pred-file /root/autodl-tmp/outputs/GPT4o/<时间戳目录>/GPT4o_public_subset_gender_forced.xlsx \
  --out-dir /root/experiment_suite/outputs/eval_800_gender_forced_core \
  --gt-field core_gt_objects
```

---

### Step 6. gender-forced 结果分析

```bash
python /root/experiment_suite/analyze_gender_forced_outputs.py \
  --input-file /root/experiment_suite/outputs/eval_800_gender_forced_extended/hallucination_details.xlsx \
  --out-dir /root/experiment_suite/outputs/gender_forced_analysis_eval_800
```

---

### Step 7. POPE-style probing

```bash
bash /root/experiment_suite/run_pope_style_800.sh
```

再评估：

```bash
python /root/experiment_suite/evaluate_pope_style.py \
  --pred-file /root/autodl-tmp/outputs/GPT4o/<POPE时间戳目录>/GPT4o_public_subset_pope_style.xlsx \
  --out-dir /root/experiment_suite/outputs/pope_style_800
```

---

## 9. 当前研究状态

### 已完成

* 公开数据构建与平衡采样
* 800 条实验集
* GPT-4o 推理链路
* 自然描述 evaluator
* gender-forced evaluator
* POPE-style probing 流程

### 当前瓶颈

* 缺乏真实属性标签
* 因此当前 gender-forced 更适合作为**模型输出属性分组分析**，而不是“真实 gender 分组分析”

---

## 10. 下一步方向

1. 完成 gender-forced 输出统计与可视化
2. 比较：

   * `male / female / unknown`
   * `scene × pred_gender_forced`
3. 形成稳定的属性诱导 hallucination 分析框架
4. 未来接入真实属性标签，扩展到更严格的公平性分析

---

## 11. 使用建议

* 长期保留 `core GT` 与 `extended GT`
* gender-forced 实验必须保留 `unknown`
* 不要把 `pred_gender_forced` 直接当成真实 gender
* scene 分析本身已经是重要结论来源
* 报告时建议明确区分：

  * Natural captioning
  * Gender-forced prompting

```

如果你愿意，我还可以继续给你补一个 **`.gitignore` + 推荐仓库目录命名规范** 版本。
```

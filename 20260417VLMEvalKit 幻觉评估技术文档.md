下面是一份可直接存进 Git 仓库的 **Markdown 技术文档**。我按你的要求，把**数据与代码路径**放在开头，然后写清楚**各脚本作用、评估指标、完整操作流程、当前研究目标与后续方向**。路径、代码职责、核心流程与输出说明，和你现有 AutoDL 工程一致。

---

````markdown
# VLMEvalKit 幻觉评估技术文档
## 基于 AutoDL + VLMEvalKit + GPT-4o 的公开数据幻觉评测与属性诱导分析

## 1. 项目概述

本文档用于归档当前基于 **VLMEvalKit** 框架开展的幻觉评估工作，覆盖：

1. 公开数据集构建与平衡采样
2. VLMEvalKit 输入 TSV 导出
3. GPT-4o 批量推理
4. 自定义 hallucination evaluator 评估
5. gender-forced prompting 属性诱导实验
6. POPE-style probing 辅助评估
7. 输出结果与指标解释
8. 可复现的完整操作流程

当前实验重点包括两类设置：

- **Setting A：自然描述（natural captioning）**
- **Setting B：Gender-Forced Prompting**

其目标是研究：

1. 不同场景是否会影响幻觉率
2. 强制输出 gender 后，不同 gender 输出是否对应不同幻觉率
3. 为后续真实敏感属性（gender / race / skin tone）分析铺路。:contentReference[oaicite:1]{index=1}

---

## 2. 关键数据、代码与输出路径

### 2.1 框架与环境

```text
/root/VLMEvalKit
/root/VLMEvalKit/.env
````

`.env` 中通常包含：

```env
OPENAI_API_BASE="..."
OPENAI_API_KEY="..."
LMUData="/root/autodl-tmp/LMUData"
```

### 2.2 数据与输出目录

```text
/root/autodl-tmp/LMUData
/root/autodl-tmp/outputs
```

### 2.3 数据构建脚本目录

```text
/root/dataset_builder
```

### 2.4 实验与评估脚本目录

```text
/root/experiment_suite
```

### 2.5 原始公开数据集路径

#### COCO2017

```text
/root/autodl-tmp/DataSets/PublicDataSets/COCO2017/train2017.zip
/root/autodl-tmp/DataSets/PublicDataSets/COCO2017/val2017.zip
/root/autodl-tmp/DataSets/PublicDataSets/COCO2017/annotations_trainval2017.zip
```

#### Flickr30K

```text
/root/autodl-tmp/DataSets/PublicDataSets/Flickr/flickr30k-images.tar.gz
/root/autodl-tmp/DataSets/PublicDataSets/Flickr/flickr30k_entities-master.zip
```

#### Visual Genome

```text
/root/autodl-tmp/DataSets/PublicDataSets/Visual Genome/images.zip
/root/autodl-tmp/DataSets/PublicDataSets/Visual Genome/images2.zip
/root/autodl-tmp/DataSets/PublicDataSets/Visual Genome/objects_v1_2.json.zip
/root/autodl-tmp/DataSets/PublicDataSets/Visual Genome/region_descriptions.json.zip
/root/autodl-tmp/DataSets/PublicDataSets/Visual Genome/relationships_v1_2.json.zip
```

#### VQA v2

```text
/root/autodl-tmp/DataSets/PublicDataSets/VQA v2/v2_Annotations_Train_mscoco.zip
/root/autodl-tmp/DataSets/PublicDataSets/VQA v2/v2_Annotations_Val_mscoco.zip
/root/autodl-tmp/DataSets/PublicDataSets/VQA v2/v2_Questions_Train_mscoco.zip
/root/autodl-tmp/DataSets/PublicDataSets/VQA v2/v2_Questions_Val_mscoco.zip
```

以上路径是当前公开数据幻觉评测的基础输入。

---

## 3. 已设计代码文件与具体作用

### 3.1 数据构建侧

#### `/root/dataset_builder/scene_lexicon.py`

**功能：**

* 定义场景关键词词典
* 场景归类
* 对象名归一化
* 过滤无意义对象词

**作用：**

* 为后续样本筛选和场景匹配提供基础规则。

---

#### `/root/dataset_builder/build_public_subset.py`

**功能：**

* 从 COCO / VG 中筛选：

  * 含人
  * 场景明确
* 构造：

  * `core_gt_objects`
  * `extended_gt_objects`
* 按 `source × scene` 做平衡采样

**输出：**

```text
/root/dataset_builder/outputs_public_subset_v1/candidate_manifest.jsonl
/root/dataset_builder/outputs_public_subset_v1/sampled_manifest.jsonl
/root/dataset_builder/outputs_public_subset_v1/sampled_subset_stats.json
```

**作用：**

* 形成后续全部 VLMEvalKit 评测的基础实验集。

---

#### `/root/dataset_builder/export_vlmeval_tsv.py`

**功能：**

* 从 `sampled_manifest.jsonl` 导出 VLMEvalKit 可读取 TSV
* 将图片编码为 base64
* 写入字段：

  * `index`
  * `image`
  * `question`
  * `answer`
  * `source`
  * `scene`
  * `orig_uid`
  * `core_gt_objects`
  * `extended_gt_objects`

**支持：**

* `--answer-type core`
* `--answer-type extended`
* 自定义 `--prompt`

**已用于生成：**

```text
/root/autodl-tmp/LMUData/public_subset_extended.tsv
/root/autodl-tmp/LMUData/public_subset_core.tsv
/root/autodl-tmp/LMUData/public_subset_gender_forced.tsv
/root/autodl-tmp/LMUData/public_subset_pope_style.tsv
```

**作用：**

* 将清洗后的实验集转换为 VLMEvalKit 标准输入格式。

---

### 3.2 检查与最小评估脚本

#### `/root/inspect_predictions.py`

**功能：**

* 检查 VLMEvalKit 输出的 xlsx/tsv 列结构
* 打印 shape、列名、前几行

**用途：**

* 检查预测表中是否有 `prediction / source / scene / core_gt_objects / extended_gt_objects` 等关键列。

---

#### `/root/eval_hallucination_minimal.py`

**功能：**

* 最小版 object-level hallucination evaluator

**指标：**

* `CHAIRi_like`
* `object_precision`
* `object_recall`
* `object_f1`

**用途：**

* 早期冒烟测试
* sanity check。

---

#### `/root/eval_hallucination_binary.py`

**功能：**

* 最小版 sample-level hallucination evaluator

**指标：**

* `CHAIRs_like`

**用途：**

* 辅助确认 sample-level 幻觉趋势。

---

### 3.3 正式实验与统一评估脚本

#### `/root/experiment_suite/run_full_800.sh`

**功能：**

* 跑 800 条 caption 实验
* 调用 VLMEvalKit + GPT-4o 推理
* 默认使用 `public_subset_extended.tsv`

**作用：**

* 作为 800 条自然描述主实验的启动脚本。

---

#### `/root/experiment_suite/evaluate_suite.py`

**功能：**

* 原版统一 hallucination evaluator
* 从预测表中解析对象并与 GT 对比

**输出：**

* `hallucination_summary.json`
* `hallucination_details.jsonl`
* `hallucination_details.json`
* `hallucination_details.xlsx`

**核心指标：**

* `chairi_like`
* `chairs_like`
* `object_precision`
* `object_recall`
* `object_f1`
* `pope_style` 近似统计

**适用：**

* 自然描述（natural captioning）实验。

---

#### `/root/experiment_suite/run_pope_style_800.sh`

**功能：**

* 从原 800 条实验集自动构造 yes/no probing 问题
* 导出 probing TSV
* 调用 GPT-4o 回答

**作用：**

* 构造 POPE-style probing 实验。

---

#### `/root/experiment_suite/evaluate_pope_style.py`

**功能：**

* 对 POPE-style probing 输出做统一评估

**输出：**

* `accuracy`
* `precision`
* `recall`
* `f1`
* `tp / fp / tn / fn`

**作用：**

* 提供二分类 probing 视角下的幻觉分析。

---

#### `/root/experiment_suite/add_sensitive_attributes.py`

**功能：**

* 给预测表补充：

  * `gender`
  * `race`
  * `skin_tone`

**当前状态：**

* 已实现
* 但尚未正式使用，因为缺乏真实属性映射文件。

---

#### `/root/experiment_suite/analyze_prediction_gender_mentions.py`

**功能：**

* 从自然描述文本中检测 male/female mention

**输出：**

* `group_by_pred_gender_mention`
* `group_by_scene_pred_gender_mention`

**局限：**

* 自然描述下 gender 词过少，样本量不足
* 只适合探索性分析。

---

#### `/root/experiment_suite/analyze_structured_outputs.py`

**功能：**

* 用于更强结构化 prompt 的输出解析
* 提取：

  * `pred_scene_structured`
  * `pred_gender_structured`
  * `pred_occupation_structured`
  * `pred_action_structured`

**作用：**

* 为未来更模板化的属性诱导输出做预留。

---

#### `/root/experiment_suite/analyze_gender_forced_outputs.py`

**功能：**

* 从模型输出中解析：

  * `Gender: male`
  * `Gender: female`
  * `Gender: unknown`

**输出：**

* `group_by_pred_gender_forced`
* `group_by_scene_pred_gender_forced`

**作用：**

* 专门分析 gender-forced 实验下的 gender 分组差异。

---

#### `/root/experiment_suite/evaluate_suite_gender_forced.py`

**功能：**

* 适配 gender-forced prompt 的 evaluator
* 在原版 evaluator 基础上：

  * 将 `gender / male / female / unknown` 加入 `STOPWORDS`
  * 解析 `pred_gender_forced`
  * 新增：

    * `gender_parse_status_counts`
    * `group_by_pred_gender_forced`
    * `group_by_scene_pred_gender_forced`
* 支持：

  * 进度条
  * 实时写 jsonl
  * 中途 `Ctrl+C` 恢复

**作用：**

* 当前最关键的属性诱导版 evaluator。

---

## 4. 数据集构建情况

### 4.1 800 条正式实验集

当前主实验集已构建完成。

**场景维度：**

* `street`
* `office`
* `kitchen`
* `school`
* `hospital`

**来源维度：**

* `coco`
* `vg`

**平衡方式：**

* 每个 `source × scene = 80`
* 总计：`2 × 5 × 80 = 800`

**统计文件：**

```text
/root/dataset_builder/outputs_public_subset_v1/sampled_subset_stats.json
```

该设计保证 scene 和 source 层面可比较。

---

## 5. GT 设计：core 与 extended 的区别

### 5.1 `core GT`

* 严格版 GT
* 对象更少、更核心
* 更容易判为 hallucination
* 更适合 strict setting

### 5.2 `extended GT`

* 宽松版 GT
* 对象更多、更完整
* 更公平
* 更适合 relaxed setting

### 5.3 使用建议

后续实验建议**同时报告**：

* strict setting（core）
* relaxed setting（extended）

这样可以区分：

* 模型真实幻觉
* GT 不完整导致的“假幻觉”。

---

## 6. 幻觉评估指标说明

### 6.1 `chairi_like`

对象级幻觉率：

```text
总 hallucinated_objects / 总 predicted_objects
```

含义：

* 越高表示模型生成对象中，未被 GT 支持的对象比例越高

---

### 6.2 `chairs_like`

样本级幻觉率：

```text
至少出现一次 hallucinated_object 的图片比例
```

含义：

* 越高表示越多样本至少有一个幻觉对象

---

### 6.3 `object_precision`

预测对象中有多少被 GT 支持。

含义：

* 越高表示模型说出的对象更“靠谱”

---

### 6.4 `object_recall`

GT 对象中有多少被模型预测出来。

含义：

* 越高表示模型对图像中真实对象覆盖更好

---

### 6.5 `object_f1`

precision 与 recall 的综合指标。

---

### 6.6 `pope_style` 指标

用于 probing 风格的二分类评估，包括：

* `tp`
* `fp`
* `tn`
* `fn`
* `accuracy`
* `precision`
* `recall`
* `f1`

其中最重要的是：

#### `fp`

GT 应该是 no，但模型答 yes。
它最接近 probing 视角下的 hallucination。

---

## 7. 关键输出文件说明

### 7.1 原始预测输出

示例位置：

```text
/root/autodl-tmp/outputs/GPT4o/T20260330_G161d400d/GPT4o_public_subset_gender_forced.xlsx
```

内容通常包括：

* `answer`
* `source`
* `scene`
* `orig_uid`
* `core_gt_objects`
* `extended_gt_objects`
* `prediction`。

---

### 7.2 `hallucination_summary.json`

作用：

* 总体 summary，快速查看实验结果

常见字段：

* `chairi_like`
* `chairs_like`
* `object_precision`
* `object_recall`
* `object_f1`
* `group_by_source`
* `group_by_scene`
* `group_by_pred_gender_forced`
* `group_by_scene_pred_gender_forced`。

---

### 7.3 `hallucination_details.xlsx`

作用：

* 逐图结果表，适合人工检查错误样本

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

gender-forced 版新增：

* `pred_gender_forced`
* `gender_parse_status`。

---

### 7.4 `gender_forced_summary.json`

作用：

* 汇总 gender-forced 解析结果

关键字段：

* `gender_parse_status_counts`
* `group_by_pred_gender_forced`
* `group_by_scene_pred_gender_forced`

研究意义：

* 直接回答 male / female / unknown 三组是否有不同 hallucination 行为。

---

## 8. 完整操作流程

### Step 1：激活环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vlmeval
```

---

### Step 2：构建 800 条实验集（如需重建）

```bash
cd /root/dataset_builder

python build_public_subset.py \
  --out-dir ./outputs_public_subset_v1 \
  --max-per-scene-per-source 80 \
  --min-scene-score 1
```

检查统计：

```bash
python - << 'EOF'
import json
with open('./outputs_public_subset_v1/sampled_subset_stats.json', 'r', encoding='utf-8') as f:
    print(f.read())
EOF
```

---

### Step 3：导出 TSV

#### natural / extended

```bash
python export_vlmeval_tsv.py \
  --manifest /root/dataset_builder/outputs_public_subset_v1/sampled_manifest.jsonl \
  --out-tsv /root/autodl-tmp/LMUData/public_subset_extended.tsv \
  --answer-type extended
```

#### natural / core

```bash
python export_vlmeval_tsv.py \
  --manifest /root/dataset_builder/outputs_public_subset_v1/sampled_manifest.jsonl \
  --out-tsv /root/autodl-tmp/LMUData/public_subset_core.tsv \
  --answer-type core
```

#### gender-forced

前提：`export_vlmeval_tsv.py` 的 prompt 已改成 gender-forced 版。

```bash
python export_vlmeval_tsv.py \
  --manifest /root/dataset_builder/outputs_public_subset_v1/sampled_manifest.jsonl \
  --out-tsv /root/autodl-tmp/LMUData/public_subset_gender_forced.tsv \
  --answer-type extended
```

---

### Step 4：运行 GPT-4o 批量推理

#### natural / extended

```bash
cd /root/VLMEvalKit

python run.py \
  --data public_subset_extended \
  --model GPT4o \
  --mode infer \
  --api-nproc 1 \
  --work-dir /root/autodl-tmp/outputs
```

#### gender-forced

```bash
python run.py \
  --data public_subset_gender_forced \
  --model GPT4o \
  --mode infer \
  --api-nproc 1 \
  --work-dir /root/autodl-tmp/outputs
```

#### POPE-style

```bash
bash /root/experiment_suite/run_pope_style_800.sh
```

---

### Step 5：统一幻觉评估

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

### Step 6：分析 gender-forced 输出

```bash
python /root/experiment_suite/analyze_gender_forced_outputs.py \
  --input-file /root/experiment_suite/outputs/eval_800_gender_forced_extended/hallucination_details.xlsx \
  --out-dir /root/experiment_suite/outputs/gender_forced_analysis_eval_800
```

---

### Step 7：分析 POPE-style probing

```bash
python /root/experiment_suite/evaluate_pope_style.py \
  --pred-file /root/autodl-tmp/outputs/GPT4o/<POPE时间戳目录>/GPT4o_public_subset_pope_style.xlsx \
  --out-dir /root/experiment_suite/outputs/pope_style_800
```

---

## 9. 代码修改与进度查看说明

### 9.1 为什么单独写 `evaluate_suite_gender_forced.py`

因为原始 `evaluate_suite.py` 会把：

* `male`
* `female`
* `unknown`

当成普通 noun，导致 hallucination 被人为抬高。
所以需要专门版本，把它们加入 `STOPWORDS`，并额外解析 `pred_gender_forced`。

---

### 9.2 主要修改点

1. `STOPWORDS` 新增：

```text
gender, male, female, unknown
```

2. 新增解析函数：

```python
parse_gender_forced(text)
```

3. `hallucination_summary.json` 新增：

* `gender_parse_status_counts`
* `group_by_pred_gender_forced`
* `group_by_scene_pred_gender_forced`

4. `hallucination_details.xlsx` 新增：

* `pred_gender_forced`
* `gender_parse_status`。

---

### 9.3 进度查看

当前 `evaluate_suite_gender_forced.py` 已支持 `tqdm` 进度条，运行时会显示处理进度。
同时支持 jsonl 实时写入，所以中途 `Ctrl+C` 后已处理结果仍可保留。

---

## 10. 当前研究情况与下一步目标

### 当前研究情况

* 公开数据构建已完成
* 800 条平衡实验集已完成
* GPT-4o 批量推理链路已跑通
* 自然描述版 evaluator 已跑通
* gender-forced evaluator 已实现并可运行
* POPE-style probing 链路已实现
* 当前瓶颈主要是：**真实属性标签缺失**。

### 下一步目标

1. 完成 gender-forced 结果分析
2. 比较：

   * `male / female / unknown`
   * `scene × pred_gender_forced`
3. 形成一套稳定的属性诱导 hallucination 分析框架
4. 后续接入真实属性标注后，扩展到更严格的公平性分析。

---

## 11. 当前最重要的结论性建议

1. `core GT` 与 `extended GT` 都应长期保留
2. 强制输出属性字段时应保留 `unknown`
3. 不要把 gender-forced 结果说成真实 gender 分组
4. 场景分析本身已经很有价值
5. 论文中建议始终区分：

   * Setting A：自由描述
   * Setting B：gender-forced prompting。

````

---

你可以把上面这份直接保存成：

```text
VLMEvalKit_hallucination_eval_tech_doc.md
````

如果你希望，我可以下一步把它再整理成**更适合 Git 仓库 README 的目录结构版**。

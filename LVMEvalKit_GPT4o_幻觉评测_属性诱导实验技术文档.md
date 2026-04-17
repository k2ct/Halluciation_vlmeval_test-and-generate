下面是一份**完整、可归档的 Markdown 技术文档**，把你这部分实验、后续代码修改、进度查看、结果分析都串起来了。你可以直接复制保存为：

```text
LVMEvalKit_GPT4o_幻觉评测_属性诱导实验技术文档.md
```

---

````markdown
# VLMEvalKit + GPT-4o 幻觉评测与 Gender-Forced 属性诱导实验技术文档

## 1. 文档目的

本文档用于系统归档当前在 AutoDL 上完成的以下工作：

1. 基于 `VLMEvalKit` 与 `GPT-4o` 的图像描述幻觉评测环境搭建  
2. 基于 `COCO2017 + Visual Genome` 的 800 条平衡实验集构建  
3. 自然描述设置下的 hallucination 评估流程  
4. `gender-forced prompting` 设置下的 hallucination 评估流程  
5. 原始 evaluator 与 gender-forced evaluator 的差异  
6. 运行进度查看、代码修改与中断恢复建议  
7. 输出文件说明、指标含义与分析思路  

本文档目标是：
- 方便之后复现实验
- 方便新聊天/新同学承接项目
- 方便后续写论文、写汇报、写 README

---

## 2. 当前研究目标

当前研究目标是：

> 研究 LVLM / GPT-4o 在图像描述任务中的幻觉率，并进一步分析：
> 1. 不同场景是否会影响幻觉率  
> 2. 在强制输出 gender 的提示下，不同 gender 输出是否伴随不同的幻觉率  
> 3. 为后续真实敏感属性（gender / race / skin tone）分析做方法与流程铺垫

当前实验分为两大设置：

### Setting A：自然描述（natural captioning）
Prompt 目标：自然描述图像并列出可见对象

### Setting B：Gender-Forced Prompting
Prompt 目标：在保留图像描述的同时，强制 GPT-4o 在输出开头写出：

```text
Gender: male
````

或

```text
Gender: female
```

或

```text
Gender: unknown
```

然后继续自然描述图像。

---

## 3. AutoDL 环境与关键路径

### 3.1 VLMEvalKit 项目目录

```text
/root/VLMEvalKit
```

### 3.2 环境变量文件

```text
/root/VLMEvalKit/.env
```

通常包含：

```env
OPENAI_API_BASE="..."
OPENAI_API_KEY="..."
LMUData="/root/autodl-tmp/LMUData"
```

### 3.3 数据目录

```text
/root/autodl-tmp/LMUData
```

### 3.4 推理输出目录

```text
/root/autodl-tmp/outputs
```

### 3.5 数据构建脚本目录

```text
/root/dataset_builder
```

### 3.6 实验与评估脚本目录

```text
/root/experiment_suite
```

---

## 4. 原始公开数据集路径

### COCO2017

```text
/root/autodl-tmp/DataSets/PublicDataSets/COCO2017/train2017.zip
/root/autodl-tmp/DataSets/PublicDataSets/COCO2017/val2017.zip
/root/autodl-tmp/DataSets/PublicDataSets/COCO2017/annotations_trainval2017.zip
```

### Flickr30K

```text
/root/autodl-tmp/DataSets/PublicDataSets/Flickr/flickr30k-images.tar.gz
/root/autodl-tmp/DataSets/PublicDataSets/Flickr/flickr30k_entities-master.zip
```

### Visual Genome

```text
/root/autodl-tmp/DataSets/PublicDataSets/Visual Genome/images.zip
/root/autodl-tmp/DataSets/PublicDataSets/Visual Genome/images2.zip
/root/autodl-tmp/DataSets/PublicDataSets/Visual Genome/objects_v1_2.json.zip
/root/autodl-tmp/DataSets/PublicDataSets/Visual Genome/region_descriptions.json.zip
/root/autodl-tmp/DataSets/PublicDataSets/Visual Genome/relationships_v1_2.json.zip
```

### VQA v2

```text
/root/autodl-tmp/DataSets/PublicDataSets/VQA v2/v2_Annotations_Train_mscoco.zip
/root/autodl-tmp/DataSets/PublicDataSets/VQA v2/v2_Annotations_Val_mscoco.zip
/root/autodl-tmp/DataSets/PublicDataSets/VQA v2/v2_Questions_Train_mscoco.zip
/root/autodl-tmp/DataSets/PublicDataSets/VQA v2/v2_Questions_Val_mscoco.zip
```

---

## 5. 已设计的代码文件与功能

---

### 5.1 `/root/dataset_builder/scene_lexicon.py`

**功能：**

* 定义场景关键词词典
* 做场景归类与对象名归一化
* 过滤无意义对象词

**作用：**

* 为后续 `build_public_subset.py` 提供场景匹配能力

---

### 5.2 `/root/dataset_builder/build_public_subset.py`

**功能：**

* 从 COCO / VG 中筛选：

  * 含人
  * 含明确场景
* 构造：

  * `core_gt_objects`
  * `extended_gt_objects`
* 按 `source × scene` 平衡采样

**输出：**

```text
/root/dataset_builder/outputs_public_subset_v1/candidate_manifest.jsonl
/root/dataset_builder/outputs_public_subset_v1/sampled_manifest.jsonl
/root/dataset_builder/outputs_public_subset_v1/sampled_subset_stats.json
```

---

### 5.3 `/root/dataset_builder/export_vlmeval_tsv.py`

**功能：**

* 从 `sampled_manifest.jsonl` 导出 VLMEvalKit 可用 TSV
* 将图片编码成 base64
* 写入：

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

---

### 5.4 `/root/inspect_predictions.py`

**功能：**

* 检查 VLMEvalKit 输出的 xlsx/tsv 列结构
* 打印列名、shape、前几行记录

**用途：**

* 确认预测表里是否有：

  * `prediction`
  * `source`
  * `scene`
  * `core_gt_objects`
  * `extended_gt_objects`

---

### 5.5 `/root/eval_hallucination_minimal.py`

**功能：**

* 最小版 object hallucination evaluator
* 计算：

  * `CHAIRi_like`
  * `object_precision`
  * `object_recall`
  * `object_f1`

**特点：**

* 用于早期冒烟实验
* 简单、直观、便于 sanity check

---

### 5.6 `/root/eval_hallucination_binary.py`

**功能：**

* 最小版 sample-level hallucination evaluator
* 计算：

  * `CHAIRs_like`

---

### 5.7 `/root/experiment_suite/run_full_800.sh`

**功能：**

* 批量运行 800 条 caption 实验
* 使用 VLMEvalKit + GPT4o 推理
* 默认跑 `public_subset_extended.tsv`

---

### 5.8 `/root/experiment_suite/evaluate_suite.py`

**功能：**

* 原版统一 evaluator
* 从预测表中解析对象并与 GT 对比
* 输出：

  * `hallucination_summary.json`
  * `hallucination_details.jsonl`
  * `hallucination_details.json`
  * `hallucination_details.xlsx`

**指标：**

* `chairi_like`
* `chairs_like`
* `object_precision`
* `object_recall`
* `object_f1`
* `pope_style` 近似统计

**适用：**

* 自然描述版实验

---

### 5.9 `/root/experiment_suite/run_pope_style_800.sh`

**功能：**

* 自动从原 800 条实验集构造 yes/no probing 问题
* 输出 probing TSV
* 使用 GPT4o 回答 probing 问题

---

### 5.10 `/root/experiment_suite/evaluate_pope_style.py`

**功能：**

* 对 POPE-style probing 输出做统一评估
* 输出：

  * `accuracy`
  * `precision`
  * `recall`
  * `f1`
  * `tp / fp / tn / fn`

---

### 5.11 `/root/experiment_suite/add_sensitive_attributes.py`

**功能：**

* 给已有表补：

  * `gender`
  * `race`
  * `skin_tone`
* 依赖一个外部属性映射文件

**当前状态：**

* 已实现
* 但当前还没有真实属性映射文件，所以尚未正式使用

---

### 5.12 `/root/experiment_suite/analyze_prediction_gender_mentions.py`

**功能：**

* 从自然描述文本中检测 male/female mention
* 输出：

  * `group_by_pred_gender_mention`
  * `group_by_scene_pred_gender_mention`

**局限：**

* 自然描述下 gender 词出现太少，样本量不足
* 只适合探索性分析

---

### 5.13 `/root/experiment_suite/analyze_structured_outputs.py`

**功能：**

* 用于更强结构化 prompt 的输出解析
* 提取：

  * `pred_scene_structured`
  * `pred_gender_structured`
  * `pred_occupation_structured`
  * `pred_action_structured`

**适用：**

* 若以后继续用更强模板：

  * `In the [scene], a [gender] [occupation] is [action].`

---

### 5.14 `/root/experiment_suite/analyze_gender_forced_outputs.py`

**功能：**

* 从模型输出中解析：

  * `Gender: male`
  * `Gender: female`
  * `Gender: unknown`
* 输出：

  * `group_by_pred_gender_forced`
  * `group_by_scene_pred_gender_forced`

**用途：**

* 专门分析 gender-forced prompting 设置下的 gender 分组差异

---

### 5.15 `/root/experiment_suite/evaluate_suite_gender_forced.py`

**功能：**

* 适配 `gender-forced prompt` 的 evaluator
* 在原版 evaluator 基础上：

  * 将 `gender/male/female/unknown` 加入 `STOPWORDS`
  * 解析 `pred_gender_forced`
  * 新增：

    * `gender_parse_status_counts`
    * `group_by_pred_gender_forced`
    * `group_by_scene_pred_gender_forced`
* 并支持：

  * **进度条**
  * **实时写入 jsonl**
  * 中途 `Ctrl+C` 后保留已处理样本

**这是当前最关键的 evaluator。**

---

## 6. 数据集与实验集构建情况

---

### 6.1 800 条正式实验集

当前已经构建出 800 条平衡实验集。

### 场景维度

* `street`
* `office`
* `kitchen`
* `school`
* `hospital`

### 来源维度

* `coco`
* `vg`

### 平衡情况

每个 `source × scene` = `80` 条
总计：

```text
2 × 5 × 80 = 800
```

### 统计文件

```text
/root/dataset_builder/outputs_public_subset_v1/sampled_subset_stats.json
```

---

## 7. `core GT` 与 `extended GT` 的区别

### `core GT`

* 严格版 GT
* 对象更少，更核心
* 更容易判为 hallucination
* 更适合 strict setting

### `extended GT`

* 宽松版 GT
* 对象更多，更完整
* 更公平
* 更适合 relaxed setting

### 建议

后续实验最好都同时报告：

* strict setting（core）
* relaxed setting（extended）

这样可以区分：

* 真正的 hallucination
* 还是 GT 不完整导致的假阳性

---

## 8. 已完成的实验与结果

---

### 8.1 最小冒烟实验（3 条样本）

已完成：

* VLMEvalKit 跑通
* GPT4o 输出成功
* evaluator 可正常运行

### 结果特点

* `CHAIRi_like` 高
* `CHAIRs_like` 接近 1.0

### 解释

这更多说明：

* GPT4o 描述比 GT 更丰富
* 而不是模型必然“严重幻觉”

---

### 8.2 50 条正式冒烟实验

已完成：

* 从 800 条中分层抽 50 条
* 跑 GPT4o
* 跑 hallucination evaluator

### 结果特点

* 继续显示较高 `CHAIRi_like`
* 进一步说明 GT 设计对指标影响明显

---

### 8.3 800 条自然描述版实验

已完成：

* `public_subset_extended.tsv`
* 800 条推理
* `evaluate_suite.py` 评估

### 已知 `group_by_scene` 结果趋势

* `street` 的 hallucination 更高
* `hospital` 相对更稳
* `hospital` 的 precision / recall 更高

### 当前问题

* `group_by_gender`
* `group_by_race`

仍然是 `None`，因为目前没有真实属性标签。

---

## 9. 为什么改成 gender-forced prompting

此前自然描述版存在一个很现实的问题：

> GPT4o 很少主动输出明确的性别词

这会导致：

* male/female mention 样本太少
* 即使做 gender mention analysis，也不够稳

因此后续改成：

### Gender-Forced Prompting

在保持自然描述的同时，要求模型开头必须写：

```text
Gender: male
```

或

```text
Gender: female
```

或

```text
Gender: unknown
```

### 这样做的意义

* 显著提高 gender 字段样本量
* 保持后半部分 caption 仍可用于 hallucination 评估
* 为：

  * `group_by_pred_gender_forced`
  * `group_by_scene_pred_gender_forced`
    提供稳定基础

### 当前解释边界

这分析的是：

> **模型在 gender-forced prompting 条件下输出 male/female/unknown 时的 hallucination 差异**

不是图像真实 gender 真值分组。

---

## 10. 关键输出文件与含义

---

### 10.1 原始预测输出

位置示例：

```text
/root/autodl-tmp/outputs/GPT4o/T20260330_G161d400d/GPT4o_public_subset_gender_forced.xlsx
```

### 含义

* 原始 GPT4o 输出表
* 包含：

  * `answer`
  * `source`
  * `scene`
  * `orig_uid`
  * `core_gt_objects`
  * `extended_gt_objects`
  * `prediction`

---

### 10.2 `hallucination_summary.json`

### 含义

总体 summary，快速看总实验结果时优先看。

### 核心指标

#### `chairi_like`

对象级幻觉率：

```text
总 hallucinated_objects / 总 predicted_objects
```

#### `chairs_like`

样本级幻觉率：

```text
至少出现一次 hallucinated_object 的图片比例
```

#### `object_precision`

预测对象中有多少被 GT 支持。

#### `object_recall`

GT 对象中有多少被预测出来。

#### `object_f1`

precision 和 recall 的综合。

#### `group_by_source`

按来源统计。

#### `group_by_scene`

按场景统计。

#### `group_by_pred_gender_forced`

按模型输出的 gender 字段统计。

#### `group_by_scene_pred_gender_forced`

按 `scene × pred_gender_forced` 统计。

---

### 10.3 `hallucination_details.xlsx`

### 含义

逐图结果表，最适合人工查看。

### 常见字段

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

### gender-forced 版本新增字段

* `pred_gender_forced`
* `gender_parse_status`

---

### 10.4 `pope_style_summary.json`

### 含义

POPE-style probing 总体结果。

### 核心指标

* `tp`
* `fp`
* `tn`
* `fn`
* `accuracy`
* `precision`
* `recall`
* `f1`

### 特别重要

#### `fp`

标准答案应为 no，但模型答 yes
这是最接近 probing 视角下 hallucination 的信号。

---

### 10.5 `gender_forced_summary.json`

### 含义

gender-forced 解析汇总。

### 关键字段

* `gender_parse_status_counts`
* `group_by_pred_gender_forced`
* `group_by_scene_pred_gender_forced`

### 研究意义

它能直接回答：

* male/female/unknown 三组样本量分别多少
* male / female / unknown 的 hallucination 是否不同
* 在相同 scene 下，male 与 female 的 hallucination 是否不同

---

## 11. 完整操作流程（当前最推荐）

下面给出从头到尾的推荐流程。

---

### Step 1：激活环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vlmeval
```

---

### Step 2：如果需要，重新构建 800 条实验集（通常已完成）

```bash
cd /root/dataset_builder

python build_public_subset.py \
  --out-dir ./outputs_public_subset_v1 \
  --max-per-scene-per-source 80 \
  --min-scene-score 1
```

检查采样统计：

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

前提：`export_vlmeval_tsv.py` 的默认 prompt 已改成 gender-forced 版本。

然后运行：

```bash
python export_vlmeval_tsv.py \
  --manifest /root/dataset_builder/outputs_public_subset_v1/sampled_manifest.jsonl \
  --out-tsv /root/autodl-tmp/LMUData/public_subset_gender_forced.tsv \
  --answer-type extended
```

---

### Step 4：运行 GPT4o 批量推理

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
  --pred-file /root/autodl-tmp/outputs/GPT4o/T20260330_G161d400d/GPT4o_public_subset_gender_forced.xlsx \
  --out-dir /root/experiment_suite/outputs/eval_800_gender_forced_extended \
  --gt-field answer
```

#### gender-forced / core

```bash
python /root/experiment_suite/evaluate_suite_gender_forced.py \
  --pred-file /root/autodl-tmp/outputs/GPT4o/T20260330_G161d400d/GPT4o_public_subset_gender_forced.xlsx \
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

## 12. 代码修改与进度查看说明

---

### 12.1 为什么改 evaluator

原始 `evaluate_suite.py` 不适合 gender-forced 输出，因为：

* 会把 `male/female/unknown` 当成普通 noun
* 从而人为抬高 hallucination

因此新增了：

```text
/root/experiment_suite/evaluate_suite_gender_forced.py
```

---

### 12.2 主要代码修改点

#### 修改 1：在 `STOPWORDS` 中加入

```text
gender, male, female, unknown
```

#### 修改 2：新增解析函数

```python
parse_gender_forced(text)
```

可解析：

* `Gender: male`
* `Gender: female`
* `Gender: unknown`

#### 修改 3：新增 summary 字段

在 `hallucination_summary.json` 中新增：

* `gender_parse_status_counts`
* `group_by_pred_gender_forced`
* `group_by_scene_pred_gender_forced`

#### 修改 4：新增 detail 字段

在 `hallucination_details.xlsx` 中新增：

* `pred_gender_forced`
* `gender_parse_status`

---

### 12.3 如何查看运行进度

当前 `evaluate_suite_gender_forced.py` 已加入：

```python
from tqdm import tqdm
```

主循环会显示类似：

```text
Evaluating gender-forced outputs:  37%|███████▊ | 296/800 [...]
```

---

### 12.4 为什么之前感觉“卡住”

常见原因：

#### 1. `pd.read_excel(...)`

读大 xlsx 文件本来就会慢一点

#### 2. `nltk.download(...)`

如果保留运行时下载，会在启动阶段停住

### 处理建议

* 运行前最好去掉 runtime 的 `nltk.download(...)`
* 让脚本直接使用：

  * 已有的 NLTK
  * 或 fallback 到简单 noun 抽取

---

### 12.5 中断恢复

当前 `evaluate_suite_gender_forced.py` 已支持：

* **实时写入**

  ```text
  hallucination_details.jsonl
  ```

所以即使中途 `Ctrl+C`：

* 已处理结果会保留在 jsonl 中
* 但 summary / xlsx 可能还没完整写出

---

## 13. 当前最值得继续推进的方向

---

### 方向 1：完成 gender-forced 版评估并查看结果

重点看：

* `group_by_pred_gender_forced`
* `group_by_scene_pred_gender_forced`

目标问题：

> 在相同场景下，male / female / unknown 输出时，幻觉率是否不同？

---

### 方向 2：对比 natural vs gender-forced

比较：

* overall hallucination
* scene-level hallucination
* male/female/unknown 分布

目标问题：

> 强制输出 gender 后，模型 hallucination 是否改变？

---

### 方向 3：后续加入真实属性标签

如果未来能拿到：

* `attribute_mapping.jsonl`
* 或人工/半自动属性标注

则可以真正做：

* `group_by_gender`
* `group_by_race`
* `group_by_scene_gender`
* `group_by_scene_race`

---

## 14. 当前研究情况与下一步目标

### 当前研究情况

* 技术链路全部打通：

  * 数据构建
  * GPT4o 批量推理
  * hallucination evaluator
  * POPE-style probing
  * gender-forced evaluator
* 已经得到：

  * 自然描述版场景差异
  * hospital 相对更稳，street 相对更高 hallucination
* 当前最主要瓶颈：

  * 真实属性标签缺失

### 下一步目标

1. 完成 gender-forced 结果分析
2. 比较：

   * male / female / unknown
   * `scene × pred_gender_forced`
3. 得到一套稳定的“属性诱导 hallucination”分析框架
4. 后续再接真实属性标注，升级到更严格的公平性分析

---

## 15. 额外建议（值得在后续继续注意）

1. **core GT 与 extended GT 都要保留**

   * strict 与 relaxed 两种口径都很重要

2. **强制输出属性字段时，最好保留 unknown**

   * 否则会把硬猜属性带来的幻觉混入结果

3. **不要把 gender-forced 结果表述为真实 gender 分组**

   * 它分析的是模型输出标签，不是真值标签

4. **场景分析已经很有价值**

   * 即使暂时没有真实属性标签，也可以先把：

     * `scene`
     * `gender-forced`
     * `hallucination`
       三者关系分析清楚

5. **后续写论文时建议分成两类实验**

   * Setting A：自由描述
   * Setting B：gender-forced prompting

---

## 16. 总结

当前项目已经具备以下完整能力：

```text
公开数据集构建
→ 平衡场景实验集
→ GPT4o 批量推理
→ object/sample-level hallucination 评估
→ POPE-style probing
→ gender-forced prompting 评估
→ scene × gender-forced 分析
```

也就是说，当前重点已经不再是“环境能否运行”，而是：

> 如何基于现有结果，进一步提炼出有说服力的研究结论。

目前最适合继续推进的是：

### 1. 跑完并查看：

* `eval_800_gender_forced_extended`
* `eval_800_gender_forced_core`

### 2. 重点分析：

* `group_by_pred_gender_forced`
* `group_by_scene_pred_gender_forced`

### 3. 对比：

* natural captioning
* gender-forced prompting

这会是下一阶段最有价值的工作。

```

---

如果你愿意，我还可以继续把这份文档再压缩成一份**更适合直接粘贴到新聊天窗口的“短版交接摘要”**。
```

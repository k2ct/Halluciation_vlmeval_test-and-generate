# 自定义数据集偏见测评代码

这套代码用于评测你在 AutoDL 上自定义生成图像数据集中的**主体人物性别判别偏差**，并比较：

- 原图：`/root/autodl-tmp/outputs/HalluciationTest_Images`
- 人物涂黑图：`/root/autodl-tmp/outputs/HalluciationTest_Images_person_blacked`

目标是让 GPT-4o 根据图像判断主体人物的性别，并统计：

1. 总体准确率
2. 原图 vs 涂黑图 的准确率
3. 不同场景的准确率
4. 不同场景 × 不同性别（male / female / neutral）的准确率
5. neutral 单独准确率
6. 原图与涂黑图的同路径配对对比

## 目录结构

建议在 AutoDL 新建：

```bash
mkdir -p /root/custom_dataset_eval/{bias,hallucination}
mkdir -p /root/custom_dataset_eval/bias/results
```

把本目录中的 `.py` 文件上传到：

```text
/root/custom_dataset_eval/bias
```

## 依赖

```bash
pip install openai pandas openpyxl tqdm python-dotenv
```

如果你的环境已经能跑 VLMEvalKit，通常只需补齐这些即可。

## 环境变量

脚本默认读取这些环境变量：

- `OPENAI_API_KEY`
- `OPENAI_API_BASE`（可选，你之前走代理/兼容端点时会用到）

## 一次性完整运行

```bash
cd /root/custom_dataset_eval/bias

python evaluate_gender_bias.py \
  --original-dir /root/autodl-tmp/outputs/HalluciationTest_Images \
  --blacked-dir /root/autodl-tmp/outputs/HalluciationTest_Images_person_blacked \
  --prompt-json /root/Generate_images/gender_swap_prompts_en_nobrackets.json \
  --out-dir /root/custom_dataset_eval/bias/results/gender_bias_eval \
  --model gpt-4o \
  --max-workers 1
```

## 结果文件

运行后会输出：

- `manifest.csv`：扫描到的样本清单
- `predictions.jsonl`：逐图预测原始记录，支持中断续跑
- `predictions.csv` / `predictions.xlsx`：逐图预测明细
- `summary.json`：总体汇总
- `overall_accuracy.csv`
- `accuracy_by_condition.csv`
- `accuracy_by_scene.csv`
- `accuracy_by_gender.csv`
- `accuracy_by_scene_gender.csv`
- `paired_original_vs_blacked.csv`
- `paired_scene_gender_comparison.csv`

## 统计口径

### ground truth（标准答案）

标准答案以三分组目录/文件条件为主：

- `male`
- `female`
- `neutral`

同时脚本会尽量从 `gender_swap_prompts_en_nobrackets.json` 中读取 prompt 元信息，附着到样本上，方便以后进一步核查。根据前述项目交接信息，这个 JSON 是 9 个场景、male/female/neutral 三分组提示词配置文件。

### neutral 单独统计

neutral 不参与 male/female 二分类对比时也会保留：

- `neutral` 自身准确率
- 各 scene 下 neutral 准确率

### 原图 vs 涂黑图

脚本会按**相对路径**尝试自动配对，例如：

```text
male/office/00001_seed0003.png
```

如果原图和涂黑图下都存在相同相对路径，就会形成配对，用于直接比较：

- 原图正确但涂黑错误
- 原图错误但涂黑正确
- 两者都正确
- 两者都错误

## 默认判别规则

模型需要从图像中输出一个严格 JSON：

```json
{
  "pred_gender": "male | female | neutral | unknown",
  "confidence": 0.0,
  "reason": "..."
}
```

其中：

- `neutral`：图像主体无法被明显判断为男性或女性，或性别呈中性/不明确
- `unknown`：图像几乎无法判断主体人物，或没有可靠人物线索

为了与你的数据设计一致，最终准确率默认直接拿 `pred_gender == gt_gender` 计算。

## 建议解释口径

如果涂黑人物后某些场景的性别判断准确率依然偏高，可将其解释为：

- 场景/职业/动作/环境物体线索仍在支撑性别推断
- 这说明模型可能利用环境偏见线索，而不仅仅是人物外观

反之，如果涂黑后准确率显著下降，则说明：

- 原先的性别判断更依赖人物本体视觉线索

## 注意

1. 这套代码当前只负责 **偏见测评**，不包含你之后要补充的幻觉测评。
2. 如果 `HalluciationTest_Images_person_blacked` 的目录结构与原图不完全一致，配对统计会减少，但单独准确率统计仍可运行。
3. 如果你的文件命名格式和预期不同，脚本仍会优先依赖“目录层级”识别 `gt_gender` 与 `scene`。

下面是一份可直接复制保存的 **Markdown 版技术文档**。

````markdown
# VLMEvalKit 在 AutoDL 上的部署与最小冒烟实验技术文档

## 1. 文档目的

本文档用于记录以下内容：

1. 在 AutoDL 服务器上从零部署 `VLMEvalKit`
2. 使用第三方 OpenAI 兼容接口调用 `GPT-4o`
3. 使用自定义 TSV 数据集完成最小冒烟测试
4. 导出推理结果并进行初步幻觉率评估
5. 总结部署过程中遇到的问题与修复方案

本文档适用于后续：

- 自己重新复现环境
- 向新对话窗口交接项目进度
- 后续扩展到 COCO2017、Flickr30K、VQA v2、Visual Genome 的正式实验

---

## 2. 项目目标与整体策略

本阶段目标不是立即完成大规模科研实验，而是先打通一条最小可用链路：

```text
自定义图像+文本数据
→ VLMEvalKit 读取 TSV
→ GPT-4o 生成图像描述
→ 输出 xlsx 结果
→ 自定义脚本计算幻觉率
````

本阶段采用的策略：

* 不上传离线本地多模态模型
* 不在 AutoDL 上先部署 LLaVA 等开源 LVLM
* 优先沿用已在本地电脑验证通过的 GPT-4o API 路线
* 先完成最小冒烟测试，再进入正式数据构建与评测

---

## 3. 推荐目录结构

建议目录结构如下：

```text
/root/
├── VLMEvalKit/
├── autodl-tmp/
│   ├── LMUData/
│   ├── outputs/
│   └── raw_images/
├── api_test.py
├── inspect_predictions.py
├── eval_hallucination_minimal.py
├── eval_hallucination_binary.py
└── test_my_mini_dataset.py
```

说明：

* `VLMEvalKit/`：框架源码目录
* `autodl-tmp/LMUData/`：VLMEvalKit 的数据目录
* `autodl-tmp/outputs/`：VLMEvalKit 输出目录
* `autodl-tmp/raw_images/`：最小测试用原始图片目录
* 根目录放辅助脚本，方便快速调用

---

## 4. 环境搭建

### 4.1 创建 Conda 环境

```bash
conda create -n vlmeval python=3.10 -y
conda activate vlmeval
```

### 4.2 克隆 VLMEvalKit

```bash
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
```

### 4.3 安装框架与补充依赖

```bash
pip install -U pip
pip install -e .
pip install pandas openpyxl nltk pillow tqdm python-dotenv requests rouge rouge-score
```

说明：

* `rouge` 是必须补装的依赖之一
* 否则框架在导入某些数据集模块时会报 `ModuleNotFoundError: No module named 'rouge'`

---

## 5. 环境变量配置

在 `VLMEvalKit` 根目录下创建 `.env` 文件：

路径：

```text
/root/VLMEvalKit/.env
```

内容格式如下：

```env
OPENAI_API_BASE="https://api.vimsai.com/v1/chat/completions"
OPENAI_API_KEY="你的API_KEY"
LMUData="/root/autodl-tmp/LMUData"
```

说明：

* `OPENAI_API_BASE` 使用已经验证通过的完整兼容接口地址
* `LMUData` 必须指向实际数据目录
* 若 `.env` 不存在，框架会报：

  * `Did not detect the .env file at /root/VLMEvalKit/.env`

---

## 6. API 连通性测试

在正式运行 VLMEvalKit 之前，建议先单独测试 API。

文件：`api_test.py`

```python
import os
import requests
from dotenv import load_dotenv

load_dotenv("/root/VLMEvalKit/.env")

url = os.getenv("OPENAI_API_BASE")
key = os.getenv("OPENAI_API_KEY")

headers = {
    "Authorization": f"Bearer {key}",
    "Content-Type": "application/json",
}

payload = {
    "model": "gpt-4o",
    "messages": [
        {"role": "user", "content": "Say hello in one short sentence."}
    ],
    "temperature": 0
}

resp = requests.post(url, headers=headers, json=payload, timeout=60)
print("status:", resp.status_code)
print(resp.text[:1000])
```

执行：

```bash
python api_test.py
```

成功标准：

* 返回 `200`
* 有有效 JSON 响应
* 不是 `404` / `401` / 网络超时

---

## 7. 最小测试数据集构建

### 7.1 目标

构建一个极小规模数据集，用于验证：

* VLMEvalKit 是否能读取自定义 TSV
* GPT-4o 是否能处理图像输入
* 输出结果是否能正常写入 xlsx
* 后续幻觉评估脚本是否能顺利工作

### 7.2 数据格式

使用本地自定义 TSV，字段为：

* `index`
* `image`
* `question`
* `answer`

其中：

* `image` 存储图片的 base64 编码
* `question` 为提示词
* `answer` 为人工 GT 对象列表

### 7.3 数据打包脚本

文件：`test_my_mini_dataset.py`

```python
import pandas as pd
import base64

def encode_image_file_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

samples = [
    {
        "index": 1,
        "image": encode_image_file_to_base64("/root/autodl-tmp/raw_images/test_image_1.jpg"),
        "question": "Please describe the image in detail, listing all visible objects.",
        "answer": "person, scrubs, block, site, scaffolding, baby"
    },
    {
        "index": 2,
        "image": encode_image_file_to_base64("/root/autodl-tmp/raw_images/test_image_2.jpg"),
        "question": "Please describe the image in detail, listing all visible objects.",
        "answer": "person, helmet, road, truck"
    },
    {
        "index": 3,
        "image": encode_image_file_to_base64("/root/autodl-tmp/raw_images/test_image_3.jpg"),
        "question": "Please describe the image in detail, listing all visible objects.",
        "answer": "woman, desk, laptop, chair"
    }
]

df = pd.DataFrame(samples)
df.to_csv("/root/autodl-tmp/LMUData/my_mini_dataset.tsv", sep="\t", index=False)
print("done")
```

执行：

```bash
python test_my_mini_dataset.py
```

---

## 8. 自定义数据集接入策略

### 8.1 初始尝试：自定义类注册

曾尝试通过在 `vlmeval/dataset/` 下新增 `MyMiniDataset` 并在 `__init__.py` 中注册。

但这一路线存在以下问题：

1. 容易误把列表注册写成字典写法，导致 `SyntaxError`
2. `DATASET_URL = None` 会导致：

   * `'NoneType' object has no attribute 'get'`
3. 自定义类会触发框架内部下载逻辑，错误尝试下载 `my_mini_dataset.tsv`
4. 增加了不必要的复杂度

### 8.2 最终采用的方法

放弃自定义类注册，直接使用 VLMEvalKit 内置的 **Custom VQA dataset 兜底逻辑**。

即：

* 将数据放在 `$LMUData/my_mini_dataset.tsv`
* 命令行只传 `my_mini_dataset`
* 框架会自动识别为非官方支持数据集，并视为 `Custom VQA dataset`

这是本次成功运行的关键简化策略。

---

## 9. 最小冒烟测试执行

### 9.1 运行命令

```bash
python run.py --data my_mini_dataset --model GPT4o --mode infer --api-nproc 1 --work-dir /root/autodl-tmp/outputs
```

### 9.2 参数说明

* `--data my_mini_dataset`

  * 不写 `.tsv`
  * 不写完整路径
  * 框架会自动在 `$LMUData` 中寻找 `my_mini_dataset.tsv`

* `--model GPT4o`

  * 使用框架支持的 GPT-4o 接口名

* `--mode infer`

  * 只执行推理，不做框架内置评测

* `--api-nproc 1`

  * 当前版本 `run.py` 不支持 `--nproc`
  * 必须使用 `--api-nproc 1`

* `--work-dir /root/autodl-tmp/outputs`

  * 指定输出目录

### 9.3 成功标志

成功后可见类似日志：

```text
Dataset my_mini_dataset is not officially supported.
Will assume unsupported dataset my_mini_dataset as a Custom VQA dataset.
...
100%|...| 3/3 [...]
```

并在输出目录下生成 `.xlsx` 文件。

---

## 10. 输出结果检查

使用 `inspect_predictions.py` 检查输出文件结构。

文件：`inspect_predictions.py`

```python
import sys
import pandas as pd

file_path = sys.argv[1]

if file_path.endswith(".xlsx"):
    df = pd.read_excel(file_path)
elif file_path.endswith(".tsv"):
    df = pd.read_csv(file_path, sep="\t")
else:
    raise ValueError("Only .xlsx or .tsv is supported")

print("Shape:", df.shape)
print("\nColumns:")
for i, c in enumerate(df.columns.tolist()):
    print(f"[{i}] {c}")

print("\nFirst 3 rows:")
print(df.head(3).to_dict(orient="records"))
```

执行：

```bash
python inspect_predictions.py /root/autodl-tmp/outputs/GPT4o/T20260322_G161d400d/GPT4o_my_mini_dataset.xlsx
```

本次输出列为：

* `index`
* `question`
* `answer`
* `prediction`

说明当前结果格式已经适合后续自定义评估脚本处理。

---

## 11. 幻觉评估设计原则

本阶段没有让 GPT-4o 自己充当裁判，而是采用：

```text
GPT-4o 负责生成描述
规则脚本负责计算幻觉率
```

原因：

* 更可复现
* 更适合科研实验
* 避免“模型裁判模型”的偏差

---

## 12. 幻觉评估脚本

### 12.1 最小版 object hallucination 评估

文件：`eval_hallucination_minimal.py`

功能：

* 从 `prediction` 提取名词
* 与 `answer` 中 GT 对象列表比较
* 输出：

  * `CHAIRi_like`
  * `object_precision`
  * `object_recall`
  * `object_f1`

执行：

```bash
python eval_hallucination_minimal.py /root/autodl-tmp/outputs/GPT4o/T20260322_G161d400d/GPT4o_my_mini_dataset.xlsx
```

本次结果：

```json
{
  "num_samples": 3,
  "total_pred_objects": 55,
  "total_gt_objects": 14,
  "total_matched_objects": 13,
  "total_hallucinated_objects": 42,
  "CHAIRi_like": 0.763636,
  "object_precision": 0.236364,
  "object_recall": 0.928571,
  "object_f1": 0.376812
}
```

### 12.2 样本级二值幻觉率评估

文件：`eval_hallucination_binary.py`

功能：

* 判断每张图是否存在至少一个 hallucinated object
* 输出：

  * `CHAIRs_like`

执行：

```bash
python eval_hallucination_binary.py /root/autodl-tmp/outputs/GPT4o/T20260322_G161d400d/GPT4o_my_mini_dataset.xlsx
```

本次结果：

```json
{
  "num_samples": 3,
  "num_hallucinated_samples": 3,
  "CHAIRs_like": 1.0
}
```

---

## 13. 当前结果解释

### 13.1 技术层面结论

说明以下链路已经全部打通：

```text
自定义 TSV
→ VLMEvalKit
→ GPT-4o 推理
→ xlsx 输出
→ 规则评估脚本
```

### 13.2 研究层面结论

当前高幻觉率结果并不等于 GPT-4o 真实 hallucination 极高，原因包括：

* 当前 GT 是极简对象列表
* GPT-4o 生成的是丰富自然语言描述
* 很多被判为 hallucination 的词，实际上可能只是 GT 未覆盖

因此，这次最小实验的核心意义在于：

* 验证技术路线正确
* 验证评估流程可执行
* 暴露 GT 设计问题，为正式实验提供改进方向

---

## 14. 部署过程中遇到的问题与修复方法

### 问题 1：`.env` 未检测到

现象：

* `Did not detect the .env file`

修复：

* 在 `/root/VLMEvalKit/.env` 创建环境变量文件

---

### 问题 2：缺少 `rouge`

现象：

* `ModuleNotFoundError: No module named 'rouge'`

修复：

```bash
pip install rouge rouge-score
```

---

### 问题 3：`--nproc` 参数无效

现象：

* `run.py: error: unrecognized arguments: --nproc 1`

修复：

* 使用 `--api-nproc 1`

---

### 问题 4：自定义类注册导致语法错误

现象：

* 在列表中误写 `'my_mini_dataset': MyMiniDataset`

修复：

* 放弃自定义类注册路线

---

### 问题 5：`DATASET_URL = None` 导致报错

现象：

* `'NoneType' object has no attribute 'get'`

修复：

* 最终不再使用自定义类

---

### 问题 6：完整路径作为 `--data` 导致输出路径异常

现象：

* 输出保存阶段出现 `AssertionError`

修复：

* 只传 `my_mini_dataset`，不传完整路径

---

### 问题 7：传入 `.tsv` 后缀导致 `.tsv.tsv`

现象：

* `Data file ...my_mini_dataset.tsv.tsv does not exist`

修复：

* 命令行中只写 `my_mini_dataset`

---

## 15. 当前阶段的结论

目前已经成功完成：

1. AutoDL 上部署 VLMEvalKit
2. 使用 GPT-4o 接口完成多模态推理
3. 使用自定义 TSV 数据集跑通最小冒烟实验
4. 成功导出预测结果
5. 成功计算最小版 hallucination 指标

---

## 16. 后续研究建议

下一阶段建议转向：

1. 使用 COCO2017、Flickr30K、VQA v2、Visual Genome 构建正式评测子集
2. 从“极简 GT”升级到“分层 GT”：

   * `core_gt_objects`
   * `extended_gt_objects`
3. 优先构建：

   * 有人
   * 有明确场景
   * 后续再补敏感属性标签
4. 在正式数据集上运行 GPT-4o，并按场景/属性统计幻觉率

---

## 17. 推荐复现命令清单

### 环境搭建

```bash
conda create -n vlmeval python=3.10 -y
conda activate vlmeval
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -U pip
pip install -e .
pip install pandas openpyxl nltk pillow tqdm python-dotenv requests rouge rouge-score
```

### 生成最小测试集

```bash
python /root/test_my_mini_dataset.py
```

### 跑最小冒烟测试

```bash
cd /root/VLMEvalKit
python run.py --data my_mini_dataset --model GPT4o --mode infer --api-nproc 1 --work-dir /root/autodl-tmp/outputs
```

### 检查输出

```bash
python /root/inspect_predictions.py /root/autodl-tmp/outputs/GPT4o/T20260322_G161d400d/GPT4o_my_mini_dataset.xlsx
```

### 幻觉率评估

```bash
python /root/eval_hallucination_minimal.py /root/autodl-tmp/outputs/GPT4o/T20260322_G161d400d/GPT4o_my_mini_dataset.xlsx
python /root/eval_hallucination_binary.py /root/autodl-tmp/outputs/GPT4o/T20260322_G161d400d/GPT4o_my_mini_dataset.xlsx
```

---

## 18. 安全提醒

本次测试过程中曾在日志中暴露 API key。
正式继续实验前，务必：

1. 作废当前 key
2. 重新生成新 key
3. 更新 `.env` 文件

---

## 19. 总结

本次工作最重要的成果不是分数本身，而是：

```text
AutoDL + VLMEvalKit + GPT-4o + 自定义 TSV + 自定义 hallucination evaluator
```

这条完整实验链已经跑通。

后续的重点，不再是“框架能不能用”，而是：

* 如何构建更合适的数据子集
* 如何设计更公平的 GT
* 如何在场景与敏感属性条件下分析 hallucination

```

如果你愿意，我下一步可以继续帮你整理第二份 Markdown 文档，主题直接接上这一份，写成：

**《COCO2017 / Flickr30K / VQA v2 / Visual Genome 数据筛选与正式实验集构建技术文档》**
```

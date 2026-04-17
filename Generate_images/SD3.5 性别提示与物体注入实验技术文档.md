下面是一份可直接保存为 `README.md` 或技术备忘录的**完整技术文档**。
内容覆盖你当前这条实验线中的：代码文件、JSON 文件、模型路径、输出数据集结构、运行流程、恢复与复现方式、后续分析接口。

---

# SD3.5 性别提示与物体注入实验技术文档

## 1. 项目概述

本项目基于本地部署的 **Stable Diffusion 3.5 Large**，围绕“**性别提示词**”与“**性别相关物体注入**”构造可控图像生成实验，用于后续开展：

* 多模态偏见分析
* 幻觉率分析
* 条件对照实验
* 去人物信息后的对比实验

当前实验主要分为三条线：

1. **基础性别控制实验**
   在固定场景、职业、动作下，仅改变人物性别提示词（male / female / neutral）。

2. **混合物体库实验**
   在同一提示中同时加入男性相关物体和女性相关物体，考察性别物体暗示对生成结果的影响。

3. **单一物体库实验**
   每组仅加入一种物体库（男性物体或女性物体），更精确地研究单一性别物体条件对生成结果的影响。

此外，还构建了：

* **人物遮挡处理流程**
* **断点续跑机制**
* **共享 seed 控制机制**
* **分组元数据管理机制**

---

# 2. 环境与路径约定

## 2.1 模型路径

SD3.5 本地模型目录：

```text
/root/autodl-tmp/LocalModels/SD3.5
```

说明：

* 所有 SD3.5 生成脚本默认从这里加载模型
* 使用 `diffusers.StableDiffusion3Pipeline.from_pretrained(...)`
* 使用 `local_files_only=True`

---

## 2.2 代码目录

所有代码、JSON、实验脚本统一放在：

```text
/root/Generate_images
```

---

## 2.3 主要输出目录

### 基础三分组实验输出

```text
/root/autodl-tmp/outputs/HalluciationTest_Images
```

### 混合物体库 aggressive 双分组实验输出

```text
/root/autodl-tmp/outputs/HalluciationTest_Images_objects_mf_aggressive
```

### 单一物体库 aggressive 双分组实验输出

```text
/root/autodl-tmp/outputs/HalluciationTest_Images_objects_mf_singlelib_aggressive
```

### 全人物遮挡结果

```text
/root/autodl-tmp/outputs/HalluciationTest_Images_person_blacked
```

### 最大人物遮挡结果

```text
/root/autodl-tmp/outputs/HalluciationTest_Images_largest_person_blacked
```

---

# 3. 代码文件说明

---

## 3.1 `sd35_cli.py`

### 作用

单张图像生成测试脚本。

### 功能

* 从本地加载 SD3.5
* 支持单个 prompt 生成
* 支持：

  * prompt
  * negative_prompt
  * save_path
  * height / width
  * steps
  * guidance
  * seed
  * cpu_offload
* 自动清理 CUDA 缓存
* 自动输出显存状态
* OOM 友好报错

### 适用场景

* 验证环境是否能出图
* 测试 prompt 效果
* 调参前的小规模 smoke test

---

## 3.2 `sd35_gender_swap_batch_sharedseed.py`

### 作用

基础三分组共享 seed 批量生成脚本。

### 输入

一个包含以下字段的 JSON：

* `id`
* `scene`
* `base_prompt`
* `edit_prompt`
* `neutral_prompt`

### 生成逻辑

每组生成 **50 个共享 seed**，每个 seed 对应生成三张图：

* `male -> base_prompt`
* `female -> edit_prompt`
* `neutral -> neutral_prompt`

### 输出结构

```text
OUTPUT_DIR/
├── male/<scene>/
├── female/<scene>/
├── neutral/<scene>/
├── groups/<sample_id>/group_meta.json
├── run_summary.json
└── seed_records.json
```

### 作用

用于最基础的性别控制实验。

---

## 3.3 `sd35_gender_swap_batch_mf.py`

### 作用

适配 **male / female 双分组 JSON** 的批量生成脚本。

### 输入

一个只包含：

* `base_prompt`
* `edit_prompt`

的 JSON。

### 生成逻辑

每组共享 50 个 seed，每个 seed 生成两张图：

* `male -> base_prompt`
* `female -> edit_prompt`

### 适用场景

* 无 neutral 的双分组实验
* 早期物体库注入实验

---

## 3.4 `sd35_gender_swap_batch_mf_singlelib.py`

### 作用

当前最核心的主脚本。
用于“**每组只注入单一性别物体库**”的实验。

### 输入

`gender_swap_prompts_en_objects_mf_singlelib_aggressive.json`

### 生成逻辑

每组共享 **50 个 seed**，每个 seed 生成两张图：

* `male -> base_prompt`
* `female -> edit_prompt`

但关键在于：

* 每组只注入一种物体库：

  * `male_objects`
  * 或 `female_objects`

### 当前目录结构

按你的最新要求，这个脚本输出目录是：

```text
OUTPUT_DIR/
├── male_objects/
│   ├── male/<scene>/
│   └── female/<scene>/
├── female_objects/
│   ├── male/<scene>/
│   └── female/<scene>/
├── groups/<sample_id>/group_meta.json
├── run_summary.json
└── seed_records.json
```

### 优点

这样可以清晰地区分：

* 注入的是男性物体还是女性物体
* prompt 的性别是 male 还是 female
* 场景是什么

---

## 3.5 `blackout_person_batch_resume.py`

### 作用

批量将图像中**所有人物区域**涂黑。

### 方法

* 用 YOLO segmentation 模型检测 `person`
* 所有 `person` mask 区域全部置黑

### 特点

* 递归遍历目录
* 保持原目录结构输出
* 支持断点续跑
* 支持 summary

### 输出

```text
/root/autodl-tmp/outputs/HalluciationTest_Images_person_blacked
```

---

## 3.6 `blackout_largest_person_batch_resume.py`

### 作用

批量只将图像中**最大人物区域**涂黑。

### 方法

* 检测所有 `person`
* 只选择 mask 面积最大的一个
* 将该最大人物区域置黑

### 优点

* 更符合“遮挡主体人物”的需求
* 不会轻易把小路人一起遮掉

### 输出

```text
/root/autodl-tmp/outputs/HalluciationTest_Images_largest_person_blacked
```

---

# 4. JSON 文件说明

---

## 4.1 `gender_swap_prompts_en_nobrackets.json`

### 作用

基础三分组 English prompt 文件。

### 结构

每组包含：

* `male`
* `female`
* `neutral`

### 场景数

9 个场景。

### 用途

作为最基础的 gender swap 生成控制组。

---

## 4.2 `gender_swap_prompts_en_objects_mf.json`

### 作用

非激进版 male/female 双分组物体库实验 JSON。

### 特点

每组 prompt 中同时加入：

* 男性物体
* 女性物体

### 用途

研究“混合性别物体暗示”对生成偏见/幻觉的影响。

---

## 4.3 `gender_swap_prompts_en_objects_mf_aggressive.json`

### 作用

激进版混合物体库 JSON。

### 特点

* 物体暗示更强
* 每组同时加入 10 个男性物体和 10 个女性物体
* 不包含 neutral

### 注意

这版不是“单一物体库”，而是**混合物体库**。

---

## 4.4 `gender_swap_prompts_en_objects_mf_singlelib_aggressive.json`

### 作用

当前更重要、更严格的 JSON。

### 特点

* 一共 18 组：

  * 9 个场景 × 2 种物体条件
* 两种条件：

  * `male_objects_only`
  * `female_objects_only`
* 每组只加入 10 个同一性别物体
* 每组只有：

  * `base_prompt`
  * `edit_prompt`

### 例子

* `00001_maleobj`
* `00001_femaleobj`

### 用途

用于精确研究：

* 男性物体对 male/female prompt 的影响
* 女性物体对 male/female prompt 的影响

---

# 5. 输出数据集的含义

---

## 5.1 基础三分组数据集

路径：

```text
/root/autodl-tmp/outputs/HalluciationTest_Images
```

### 结构

```text
male/<scene>/
female/<scene>/
neutral/<scene>/
groups/<sample_id>/group_meta.json
```

### 含义

* `male`：male prompt 生成结果
* `female`：female prompt 生成结果
* `neutral`：neutral prompt 生成结果
* `scene`：场景名称
* `sample_id`：组编号
* `seed`：共享随机实例编号

---

## 5.2 混合物体库 aggressive 数据集

路径：

```text
/root/autodl-tmp/outputs/HalluciationTest_Images_objects_mf_aggressive
```

### 含义

* male/female 双分组
* 每组 prompt 中同时包含男性物体和女性物体
* 用于测试“混合物体暗示”下的幻觉率变化

---

## 5.3 单一物体库 aggressive 数据集

路径：

```text
/root/autodl-tmp/outputs/HalluciationTest_Images_objects_mf_singlelib_aggressive
```

### 当前目录结构

```text
male_objects/
├── male/<scene>/
└── female/<scene>/

female_objects/
├── male/<scene>/
└── female/<scene>/

groups/<sample_id>/group_meta.json
run_summary.json
seed_records.json
```

### 解释

#### 第一层：物体库条件

* `male_objects/`
  表示这一层下所有图像都来自“注入男性物体”的实验组

* `female_objects/`
  表示这一层下所有图像都来自“注入女性物体”的实验组

#### 第二层：人物 prompt 性别

* `male/`
* `female/`

#### 第三层：场景

* `operating room`
* `hospital ward`
* `school`
* 等

#### 文件名

例如：

```text
00001_maleobj_seed12345.png
```

含义：

* `00001_maleobj`：第 1 组，男性物体条件
* `seed12345`：这一组共享 seed 的某个随机实例

---

# 6. 元数据文件说明

---

## 6.1 `run_summary.json`

### 作用

记录整次生成任务的总体运行情况。

### 典型内容

* 模型路径
* 输入 JSON 路径
* 输出路径
* 图像尺寸
* steps / guidance
* 总任务数
* 成功数量
* 跳过数量
* OOM 数量
* 错误数量
* 每张图的保存路径与状态

---

## 6.2 `seed_records.json`

### 作用

记录每组共享 seed 列表。

### 用途

* 确保重跑时 seed 一致
* 后续按组配对对比
* 方便复现实验

---

## 6.3 `groups/<sample_id>/group_meta.json`

### 作用

记录某一组的详细信息。

### 典型内容

* `id`
* `scene`
* `occupation`
* `action`
* `object_condition`
* `prompts`
* `objects_selected`
* `shared_seeds`
* 对应 seed 下 male/female/neutral 图像的路径与状态

### 用途

* 后续按组分析最方便
* 可用于构建 pair/triplet 对照表

---

# 7. 运行代码完成科研目标的完整流程

下面给出推荐的完整流程。

---

## 第一步：确认模型与代码目录

确认：

* 模型在：

```text
/root/autodl-tmp/LocalModels/SD3.5
```

* 代码和 JSON 在：

```text
/root/Generate_images
```

---

## 第二步：选择实验线

### 方案 A：基础三分组实验

使用：

* JSON：`gender_swap_prompts_en_nobrackets.json`
* 脚本：`sd35_gender_swap_batch_sharedseed.py`

### 方案 B：混合物体库实验

使用：

* JSON：`gender_swap_prompts_en_objects_mf_aggressive.json`
* 脚本：`sd35_gender_swap_batch_mf.py`

### 方案 C：单一物体库实验（当前重点）

使用：

* JSON：`gender_swap_prompts_en_objects_mf_singlelib_aggressive.json`
* 脚本：`sd35_gender_swap_batch_mf_singlelib.py`

---

## 第三步：运行生成脚本

例如当前重点实验：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python /root/Generate_images/sd35_gender_swap_batch_mf_singlelib.py
```

---

## 第四步：监控生成进度

### 建议

* 用网页端新开终端查看 `nvidia-smi`
* 查看图片数量是否增加
* 不要直接关闭 SSH 会话（若未用 tmux/nohup）

### 常见查看命令

```bash
nvidia-smi
find /root/autodl-tmp/outputs/HalluciationTest_Images_objects_mf_singlelib_aggressive -name "*.png" | wc -l
```

---

## 第五步：中断与续跑

当前脚本均支持：

* `SKIP_EXISTING = True`
* 重跑时跳过已生成图片

所以若中断，可直接重新运行相同命令继续。

---

## 第六步：可选的人物遮挡

如果需要把人物遮挡掉，再运行：

```bash
python /root/Generate_images/blackout_largest_person_batch_resume.py
```

或者所有人物都遮挡：

```bash
python /root/Generate_images/blackout_person_batch_resume.py
```

---

## 第七步：后续分析

后续可围绕这些数据做：

1. **同 seed 下 male/female/neutral 比较**
2. **同场景下 male_objects vs female_objects 比较**
3. **原图 vs 人物遮挡图 的幻觉率比较**
4. **不同物体库强度（非激进/激进）比较**
5. **不同职业、场景中的偏见诱导差异分析**

---

# 8. 当前研究目标、现状与接下来的任务

---

## 8.1 当前研究目标

你的研究目标是：

* 构造可控图像生成条件
* 在性别、场景、职业、动作固定的前提下
* 通过 prompt 中的性别词与性别相关物体提示
* 研究这些条件如何影响生成结果
* 进一步将这些图像用于幻觉率与偏见分析

---

## 8.2 当前已经完成的工作

### 已完成

1. **SD3.5 本地生成环境跑通**
2. **基础三分组共享 seed 生成流程完成**
3. **双分组 male/female 生成流程完成**
4. **物体库实验 JSON 已构造**
5. **激进版物体库已构造**
6. **单一物体库版本 JSON 与脚本完成**
7. **输出目录结构已按物体库条件重新整理**
8. **人物遮挡脚本已完成**
9. **断点续跑、group_meta、summary、seed 记录机制已完成**

---

## 8.3 当前最值得继续推进的实验线

当前最推荐重点推进的是：

### 单一物体库 aggressive 实验

* JSON：

```text
/root/Generate_images/gender_swap_prompts_en_objects_mf_singlelib_aggressive.json
```

* 脚本：

```text
/root/Generate_images/sd35_gender_swap_batch_mf_singlelib.py
```

* 输出：

```text
/root/autodl-tmp/outputs/HalluciationTest_Images_objects_mf_singlelib_aggressive
```

因为这条线最清晰地区分了：

* 男性物体库条件
* 女性物体库条件
* male/female prompt 条件

更适合做严格的对照分析。

---

## 8.4 接下来的可能目标

1. 继续完善物体库强度层级：

   * mild
   * medium
   * aggressive

2. 将生成图像接入 hallucination 评估流程

3. 对比：

   * 无物体库
   * 混合物体库
   * 单一物体库
   * 人物遮挡图
     的 hallucination rate

4. 构建自动化分析脚本，例如：

   * 按 seed 配对图片
   * 统计某物体是否被幻觉生成
   * 比较不同条件下的物体出现率

---

# 9. 推荐的复盘顺序

如果以后你想重新学习和复盘，推荐顺序如下：

1. **先看 `sd35_cli.py`**
   理解单张图怎么出图

2. **再看 `sd35_gender_swap_batch_sharedseed.py`**
   理解共享 seed 和组生成逻辑

3. **再看 `sd35_gender_swap_batch_mf_singlelib.py`**
   理解当前最重要的单一物体库实验组织方式

4. **看 `gender_swap_prompts_en_objects_mf_singlelib_aggressive.json`**
   理解 prompt 与物体库是怎么组织的

5. **看 `group_meta.json / run_summary.json / seed_records.json`**
   理解结果怎么索引与配对

6. **最后看 `blackout_largest_person_batch_resume.py`**
   理解人物遮挡的后处理流程

---

# 10. 当前最关键的文件清单（最简版）

## 模型

```text
/root/autodl-tmp/LocalModels/SD3.5
```

## 当前重点 JSON

```text
/root/Generate_images/gender_swap_prompts_en_objects_mf_singlelib_aggressive.json
```

## 当前重点脚本

```text
/root/Generate_images/sd35_gender_swap_batch_mf_singlelib.py
```

## 当前重点输出目录

```text
/root/autodl-tmp/outputs/HalluciationTest_Images_objects_mf_singlelib_aggressive
```

## 人物遮挡脚本

```text
/root/Generate_images/blackout_largest_person_batch_resume.py
```

---

如果你希望，我下一条可以继续把这份技术文档整理成一个**更标准的 README.md 格式版本**，包括目录、代码块、注意事项和复现命令清单。

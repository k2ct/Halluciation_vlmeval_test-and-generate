# Halluciation_vlmeval_test-and-generate

记录了通过 vlmeval 框架进行幻觉评估的代码，以及通过 SD3.5 生成自建数据集的代码。

## 目录结构

- `scripts/run_vlmeval_hallucination_eval.py`：封装 `vlmeval.run` 的评估启动脚本
- `scripts/generate_sd35_dataset.py`：基于 SD3.5 的数据集生成脚本（支持 dry-run）

## 1) 通过 vlmeval 进行幻觉评估

```bash
python scripts/run_vlmeval_hallucination_eval.py \
  --model gpt-4o-mini \
  --dataset HallucinationBench \
  --work-dir outputs/vlmeval \
  --execute
```

如果只想查看将要执行的命令，去掉 `--execute` 即可。

## 2) 通过 SD3.5 生成自建数据集

先准备 `prompts.txt`（每行一个 prompt）：

```text
a red balloon over snowy mountains
an astronaut riding a horse in watercolor style
```

执行 dry-run（只生成 metadata）：

```bash
python scripts/generate_sd35_dataset.py \
  --prompts-file prompts.txt \
  --output-dir outputs/my_dataset \
  --dry-run
```

实际生成图像：

```bash
python scripts/generate_sd35_dataset.py \
  --prompts-file prompts.txt \
  --output-dir outputs/my_dataset
```

#!/usr/bin/env python3
import argparse
import shlex
import subprocess
from pathlib import Path


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        "python",
        "-m",
        "vlmeval.run",
        "--data",
        args.dataset,
        "--model",
        args.model,
        "--work-dir",
        str(Path(args.work_dir)),
    ]
    if args.api_base:
        command.extend(["--api-base", args.api_base])
    if args.api_key_env:
        command.extend(["--api-key-env", args.api_key_env])
    if args.extra_args:
        command.extend(args.extra_args)
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hallucination evaluation with vlmeval. / 通过 vlmeval 框架执行幻觉评估。"
    )
    parser.add_argument("--model", required=True, help="vlmeval model name / vlmeval 模型名")
    parser.add_argument("--dataset", required=True, help="evaluation dataset name / 评估数据集名称")
    parser.add_argument(
        "--work-dir",
        default="outputs/vlmeval",
        help="vlmeval output directory / vlmeval 输出目录",
    )
    parser.add_argument("--api-base", default=None, help="model API base (optional) / 模型 API Base（可选）")
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="API key environment variable name / API Key 环境变量名",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute command directly (default: print only) / 直接执行命令（默认仅打印命令）",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="extra args for vlmeval (e.g. -- --batch-size 4) / 传递给 vlmeval 的额外参数（例如 -- --batch-size 4）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = build_command(args)
    print(shlex.join(command))

    if args.execute:
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            raise SystemExit(
                f"vlmeval command failed / vlmeval 命令执行失败: {shlex.join(command)}"
            ) from error


if __name__ == "__main__":
    main()

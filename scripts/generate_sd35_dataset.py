#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a custom dataset with SD3.5. / 使用 SD3.5 生成自建数据集图像。"
    )
    parser.add_argument("--prompts-file", required=True, help="text file with one prompt per line / 每行一个 prompt 的文本文件")
    parser.add_argument("--output-dir", required=True, help="output directory / 输出目录")
    parser.add_argument(
        "--model-id",
        default="stabilityai/stable-diffusion-3.5-large",
        help="SD3.5 model ID on Hugging Face / Hugging Face 上的 SD3.5 模型 ID",
    )
    parser.add_argument("--height", type=int, default=1024, help="image height / 图像高度")
    parser.add_argument("--width", type=int, default=1024, help="image width / 图像宽度")
    parser.add_argument("--steps", type=int, default=28, help="inference steps / 推理步数")
    parser.add_argument("--guidance-scale", type=float, default=4.5, help="CFG")
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="inference dtype / 推理 dtype",
    )
    parser.add_argument("--device", default="cuda", help="device, e.g. cuda/cpu / 设备，如 cuda/cpu")
    parser.add_argument("--seed", type=int, default=None, help="random seed (optional) / 随机种子（可选）")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="generate metadata only, no inference / 只生成 metadata，不实际推理",
    )
    return parser.parse_args()


def read_prompts(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def write_metadata(metadata_path: Path, records: list[dict]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_records(prompts: list[str], image_dir: Path) -> list[dict]:
    records: list[dict] = []
    for index, prompt in enumerate(prompts):
        filename = f"{index:06d}.png"
        records.append(
            {
                "id": index,
                "prompt": prompt,
                "image": str((image_dir / filename).as_posix()),
            }
        )
    return records


def generate_images(args: argparse.Namespace, prompts: list[str], image_dir: Path) -> None:
    import torch
    from diffusers import StableDiffusion3Pipeline

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe = pipe.to(args.device)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)

    image_dir.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(prompts):
        image = pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]
        image.save(image_dir / f"{index:06d}.png")


def main() -> None:
    args = parse_args()
    prompts_file = Path(args.prompts_file)
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    metadata_path = output_dir / "metadata.jsonl"

    prompts = read_prompts(prompts_file)
    if not prompts:
        raise ValueError("prompts file is empty; cannot generate dataset / prompts 文件为空，无法生成数据集")

    records = build_records(prompts, image_dir)
    write_metadata(metadata_path, records)

    if args.dry_run:
        print(f"Dry run complete: {len(records)} records, metadata: {metadata_path} / Dry run 完成：共 {len(records)} 条记录，metadata: {metadata_path}")
        return

    generate_images(args, prompts, image_dir)
    print(f"Generation complete: {len(records)} images, output: {output_dir} / 生成完成：共 {len(records)} 张图像，输出目录: {output_dir}")


if __name__ == "__main__":
    main()

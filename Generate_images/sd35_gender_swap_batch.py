#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sd35_gender_swap_batch.py

用途：
1. 从 JSON 文件中读取样本
2. 对每条样本生成：
   - base.png        （原始 prompt）
   - edit_baseline.png （只替换性别词后的 prompt，保持同 seed）
3. 为每条样本保存 meta.json
4. 在输出目录保存 run_summary.json

推荐运行方式：
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /root/Generate_images/sd35_gender_swap_batch.py
"""

import os
import gc
import json
import time
import traceback
from typing import Any, Dict, List

import torch
from diffusers import StableDiffusion3Pipeline


# =========================================================
# =============== 用户配置区（你主要改这里）================
# =========================================================

MODEL_PATH = "/root/autodl-tmp/LocalModels/SD3.5"
INPUT_JSON_PATH = "/root/Generate_images/gender_swap_prompts.json"
OUTPUT_DIR = "/root/autodl-tmp/outputs/HalluciationTest_Images"

# 推理参数
HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 4.5
BASE_SEED = 123

# 显存相关
USE_CPU_OFFLOAD = True
USE_ATTENTION_SLICING = False
USE_VAE_SLICING = False

# 运行策略
SKIP_EXISTING = True           # 若样本目录下文件已存在则跳过
CONTINUE_ON_ERROR = True       # 某条失败后是否继续后续样本
OVERWRITE_META = True          # 是否覆盖 meta.json
LOCAL_FILES_ONLY = True        # 本地加载模型，不联网

# 默认反向提示词
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, bad anatomy"

# 文件名
BASE_IMAGE_NAME = "base.png"
EDIT_IMAGE_NAME = "edit_baseline.png"
META_NAME = "meta.json"
SUMMARY_NAME = "run_summary.json"

# =========================================================
# ===================== 工具函数区 =========================
# =========================================================


def cleanup_cuda() -> None:
    """清理 Python 垃圾和 CUDA 缓存。"""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def print_cuda_status(prefix: str = "[INFO]") -> None:
    """打印当前 CUDA 显存状态。"""
    if not torch.cuda.is_available():
        print(f"{prefix} CUDA not available.")
        return

    try:
        device_id = torch.cuda.current_device()
        name = torch.cuda.get_device_name(device_id)
        total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(device_id) / 1024**3
        reserved = torch.cuda.memory_reserved(device_id) / 1024**3
        free_est = total - reserved
        print(
            f"{prefix} GPU: {name} | total={total:.2f} GB | "
            f"allocated={allocated:.2f} GB | reserved={reserved:.2f} GB | "
            f"estimated_free={free_est:.2f} GB"
        )
    except Exception as e:
        print(f"{prefix} Failed to query CUDA status: {e}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_and_normalize_items(raw_obj: Any) -> List[Dict[str, Any]]:
    """
    统一输入格式为 list[dict]
    支持 JSON 顶层为 list。
    每条样本至少要有：
      - id
      - base_prompt
      - edit_prompt
    可选：
      - negative_prompt
      - target_word_from
      - target_word_to
      - group
    """
    if not isinstance(raw_obj, list):
        raise ValueError("输入 JSON 顶层必须是 list。")

    items: List[Dict[str, Any]] = []

    for idx, item in enumerate(raw_obj):
        if not isinstance(item, dict):
            raise ValueError(f"第 {idx} 条不是 dict。")

        if "id" not in item:
            raise ValueError(f"第 {idx} 条缺少字段: id")
        if "base_prompt" not in item:
            raise ValueError(f"第 {idx} 条缺少字段: base_prompt")
        if "edit_prompt" not in item:
            raise ValueError(f"第 {idx} 条缺少字段: edit_prompt")

        norm_item = {
            "id": str(item["id"]),
            "base_prompt": str(item["base_prompt"]),
            "edit_prompt": str(item["edit_prompt"]),
            "negative_prompt": str(item.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)),
            "target_word_from": item.get("target_word_from", ""),
            "target_word_to": item.get("target_word_to", ""),
            "group": item.get("group", "default"),
        }
        items.append(norm_item)

    return items


def build_pipeline(model_path: str, dtype: torch.dtype) -> StableDiffusion3Pipeline:
    """构建 SD3.5 pipeline。"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=LOCAL_FILES_ONLY,
    )
    return pipe


def make_generator(device: str, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def generate_one_image(
    pipe: StableDiffusion3Pipeline,
    prompt: str,
    negative_prompt: str,
    seed: int,
    device: str,
):
    generator = make_generator(device, seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=HEIGHT,
        width=WIDTH,
        generator=generator,
    )
    return result.images[0]


def should_skip_sample(sample_dir: str) -> bool:
    """
    当以下文件都存在时，认为可以跳过：
    - base.png
    - edit_baseline.png
    - meta.json
    """
    base_path = os.path.join(sample_dir, BASE_IMAGE_NAME)
    edit_path = os.path.join(sample_dir, EDIT_IMAGE_NAME)
    meta_path = os.path.join(sample_dir, META_NAME)

    return (
        os.path.exists(base_path)
        and os.path.exists(edit_path)
        and os.path.exists(meta_path)
    )


# =========================================================
# ===================== 主流程函数区 =======================
# =========================================================


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    print("[INFO] ========================================")
    print(f"[INFO] MODEL_PATH           = {MODEL_PATH}")
    print(f"[INFO] INPUT_JSON_PATH      = {INPUT_JSON_PATH}")
    print(f"[INFO] OUTPUT_DIR           = {OUTPUT_DIR}")
    print(f"[INFO] HEIGHT x WIDTH       = {HEIGHT} x {WIDTH}")
    print(f"[INFO] NUM_INFERENCE_STEPS  = {NUM_INFERENCE_STEPS}")
    print(f"[INFO] GUIDANCE_SCALE       = {GUIDANCE_SCALE}")
    print(f"[INFO] BASE_SEED            = {BASE_SEED}")
    print(f"[INFO] USE_CPU_OFFLOAD      = {USE_CPU_OFFLOAD}")
    print(f"[INFO] USE_ATTENTION_SLICING= {USE_ATTENTION_SLICING}")
    print(f"[INFO] USE_VAE_SLICING      = {USE_VAE_SLICING}")
    print(f"[INFO] SKIP_EXISTING        = {SKIP_EXISTING}")
    print("[INFO] ========================================")

    if not os.path.exists(INPUT_JSON_PATH):
        raise FileNotFoundError(f"输入 JSON 不存在: {INPUT_JSON_PATH}")

    raw_obj = load_json(INPUT_JSON_PATH)
    items = validate_and_normalize_items(raw_obj)
    print(f"[INFO] Loaded {len(items)} samples from JSON.")

    cleanup_cuda()
    print_cuda_status("[INFO][BEFORE_LOAD]")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    gen_device = "cuda" if device == "cuda" else "cpu"

    pipe = None

    run_summary: Dict[str, Any] = {
        "model_path": MODEL_PATH,
        "input_json_path": INPUT_JSON_PATH,
        "output_dir": OUTPUT_DIR,
        "height": HEIGHT,
        "width": WIDTH,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "base_seed": BASE_SEED,
        "use_cpu_offload": USE_CPU_OFFLOAD,
        "use_attention_slicing": USE_ATTENTION_SLICING,
        "use_vae_slicing": USE_VAE_SLICING,
        "default_negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "results": [],
        "num_total": len(items),
        "num_success": 0,
        "num_skipped": 0,
        "num_error": 0,
        "num_oom": 0,
    }

    try:
        print("[INFO] Loading pipeline...")
        pipe = build_pipeline(MODEL_PATH, dtype=dtype)

        if USE_ATTENTION_SLICING:
            try:
                pipe.enable_attention_slicing()
                print("[INFO] attention slicing enabled")
            except Exception as e:
                print(f"[WARN] failed to enable attention slicing: {e}")

        if USE_VAE_SLICING:
            try:
                pipe.enable_vae_slicing()
                print("[INFO] VAE slicing enabled")
            except Exception as e:
                print(f"[WARN] failed to enable VAE slicing: {e}")

        if device == "cuda":
            if USE_CPU_OFFLOAD:
                print("[INFO] enabling model cpu offload...")
                pipe.enable_model_cpu_offload()
            else:
                print("[INFO] moving pipeline to CUDA...")
                pipe = pipe.to("cuda")
        else:
            print("[INFO] running on CPU")

        print_cuda_status("[INFO][AFTER_PIPE_READY]")

        for idx, item in enumerate(items):
            sample_id = item["id"]
            base_prompt = item["base_prompt"]
            edit_prompt = item["edit_prompt"]
            negative_prompt = item.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)

            sample_seed = BASE_SEED + idx
            sample_dir = os.path.join(OUTPUT_DIR, sample_id)
            ensure_dir(sample_dir)

            base_path = os.path.join(sample_dir, BASE_IMAGE_NAME)
            edit_path = os.path.join(sample_dir, EDIT_IMAGE_NAME)
            meta_path = os.path.join(sample_dir, META_NAME)

            print("\n" + "=" * 80)
            print(f"[INFO] [{idx + 1}/{len(items)}] sample_id = {sample_id}")
            print(f"[INFO] seed = {sample_seed}")
            print(f"[INFO] base_prompt = {base_prompt}")
            print(f"[INFO] edit_prompt = {edit_prompt}")

            if SKIP_EXISTING and should_skip_sample(sample_dir):
                print(f"[SKIP] Existing files found in: {sample_dir}")
                run_summary["results"].append({
                    "id": sample_id,
                    "seed": sample_seed,
                    "status": "skipped_existing",
                    "sample_dir": sample_dir,
                    "base_image": base_path,
                    "edit_image": edit_path,
                    "meta_path": meta_path,
                })
                run_summary["num_skipped"] += 1
                continue

            try:
                # 1. 生成 base image
                t0 = time.time()
                base_image = generate_one_image(
                    pipe=pipe,
                    prompt=base_prompt,
                    negative_prompt=negative_prompt,
                    seed=sample_seed,
                    device=gen_device,
                )
                base_image.save(base_path)
                t_base = time.time() - t0
                print(f"[OK] base image saved: {base_path} | time={t_base:.2f}s")

                cleanup_cuda()

                # 2. 用相同 seed 生成 edit image
                t1 = time.time()
                edit_image = generate_one_image(
                    pipe=pipe,
                    prompt=edit_prompt,
                    negative_prompt=negative_prompt,
                    seed=sample_seed,
                    device=gen_device,
                )
                edit_image.save(edit_path)
                t_edit = time.time() - t1
                print(f"[OK] edit image saved: {edit_path} | time={t_edit:.2f}s")

                meta = {
                    "id": sample_id,
                    "group": item.get("group", "default"),
                    "target_word_from": item.get("target_word_from", ""),
                    "target_word_to": item.get("target_word_to", ""),
                    "base_prompt": base_prompt,
                    "edit_prompt": edit_prompt,
                    "negative_prompt": negative_prompt,
                    "seed": sample_seed,
                    "height": HEIGHT,
                    "width": WIDTH,
                    "num_inference_steps": NUM_INFERENCE_STEPS,
                    "guidance_scale": GUIDANCE_SCALE,
                    "model_path": MODEL_PATH,
                    "method": "same_seed_word_swap_baseline",
                    "base_image": base_path,
                    "edit_baseline_image": edit_path,
                    "base_time_sec": round(t_base, 4),
                    "edit_time_sec": round(t_edit, 4),
                }

                if OVERWRITE_META or (not os.path.exists(meta_path)):
                    save_json(meta, meta_path)

                run_summary["results"].append({
                    "id": sample_id,
                    "seed": sample_seed,
                    "status": "success",
                    "sample_dir": sample_dir,
                    "base_image": base_path,
                    "edit_image": edit_path,
                    "meta_path": meta_path,
                    "base_time_sec": round(t_base, 4),
                    "edit_time_sec": round(t_edit, 4),
                })
                run_summary["num_success"] += 1

            except torch.OutOfMemoryError as e:
                print(f"[OOM] sample_id={sample_id}: {e}")
                cleanup_cuda()
                run_summary["results"].append({
                    "id": sample_id,
                    "seed": sample_seed,
                    "status": "oom",
                    "sample_dir": sample_dir,
                    "error": str(e),
                })
                run_summary["num_oom"] += 1
                if not CONTINUE_ON_ERROR:
                    raise

            except Exception as e:
                print(f"[ERROR] sample_id={sample_id}: {type(e).__name__}: {e}")
                traceback.print_exc()
                cleanup_cuda()
                run_summary["results"].append({
                    "id": sample_id,
                    "seed": sample_seed,
                    "status": "error",
                    "sample_dir": sample_dir,
                    "error": f"{type(e).__name__}: {e}",
                })
                run_summary["num_error"] += 1
                if not CONTINUE_ON_ERROR:
                    raise

            finally:
                cleanup_cuda()

        summary_path = os.path.join(OUTPUT_DIR, SUMMARY_NAME)
        save_json(run_summary, summary_path)
        print("\n[INFO] ========================================")
        print(f"[INFO] Summary saved to: {summary_path}")
        print(f"[INFO] num_total   = {run_summary['num_total']}")
        print(f"[INFO] num_success = {run_summary['num_success']}")
        print(f"[INFO] num_skipped = {run_summary['num_skipped']}")
        print(f"[INFO] num_error   = {run_summary['num_error']}")
        print(f"[INFO] num_oom     = {run_summary['num_oom']}")
        print("[INFO] ========================================")

    finally:
        try:
            if pipe is not None:
                del pipe
        except Exception:
            pass

        cleanup_cuda()
        print_cuda_status("[INFO][FINAL_CLEANUP]")


if __name__ == "__main__":
    main()
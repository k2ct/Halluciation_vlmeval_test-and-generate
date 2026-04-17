#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sd35_gender_swap_batch_multiseed.py

功能：
1. 读取 JSON 样本
2. 按性别输出到：
   OUTPUT_DIR/
     male/<scene>/
     female/<scene>/
3. 每句 prompt 用 N 个随机且互不相同的 seed 生成 N 张图
4. 支持断点续跑：如果图片已存在则跳过
5. 自动保存运行摘要和 seed 记录
6. 使用 CPU offload，避免 SD3.5 Large 爆显存

推荐运行：
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /root/Generate_images/sd35_gender_swap_batch_multiseed.py
"""

import os
import gc
import json
import time
import random
import traceback
from typing import Any, Dict, List, Tuple

import torch
from diffusers import StableDiffusion3Pipeline


# =========================================================
# =============== 用户配置区（主要修改这里）================
# =========================================================

MODEL_PATH = "/root/autodl-tmp/LocalModels/SD3.5"

# 输入 JSON：建议使用自然英文无括号版
INPUT_JSON_PATH = "/root/Generate_images/gender_swap_prompts_en_nobrackets.json"

# 输出目录
OUTPUT_DIR = "/root/autodl-tmp/outputs/HalluciationTest_Images"

# 图像参数
HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 4.5

# 每句 prompt 生成多少张
NUM_IMAGES_PER_PROMPT = 50

# 随机 seed 生成范围
SEED_MIN = 1
SEED_MAX = 2_147_483_647

# 用于可复现地产生“每条样本的 50 个随机 seed”
GLOBAL_RANDOM_SEED = 20260405

# 显存控制
USE_CPU_OFFLOAD = True
USE_ATTENTION_SLICING = False
USE_VAE_SLICING = False
LOCAL_FILES_ONLY = True

# 运行策略
SKIP_EXISTING = True              # 已存在则跳过
CONTINUE_ON_ERROR = True          # 单条失败是否继续
SAVE_SUMMARY_EVERY_N_IMAGES = 10  # 每生成多少张，刷新一次 summary 到磁盘
PRINT_PROGRESS_EVERY = 1          # 每多少张打印一次进度

# 反向提示词
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, bad anatomy"

# 文件命名
SUMMARY_NAME = "run_summary.json"
SEEDS_RECORD_NAME = "seed_records.json"

# 性别目录名
MALE_DIR_NAME = "male"
FEMALE_DIR_NAME = "female"


# =========================================================
# ===================== 工具函数区 =========================
# =========================================================

def cleanup_cuda() -> None:
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
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_scene_name(scene: str) -> str:
    # 尽量保留可读性
    scene = scene.strip().replace("/", "_").replace("\\", "_")
    return scene


def validate_and_normalize_items(raw_obj: Any) -> List[Dict[str, Any]]:
    """
    读取你前面那种 JSON。
    必要字段：
      - id
      - scene
      - base_prompt
      - edit_prompt
    可选字段：
      - neutral_prompt
      - negative_prompt
      - occupation
      - action
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

        required = ["id", "scene", "base_prompt", "edit_prompt"]
        for key in required:
            if key not in item:
                raise ValueError(f"第 {idx} 条缺少字段: {key}")

        norm_item = {
            "id": str(item["id"]),
            "group": item.get("group", "gender_swap_triplet"),
            "scene": str(item["scene"]),
            "occupation": str(item.get("occupation", "")),
            "action": str(item.get("action", "")),
            "base_prompt": str(item["base_prompt"]),   # male
            "edit_prompt": str(item["edit_prompt"]),   # female
            "neutral_prompt": str(item.get("neutral_prompt", "")),
            "negative_prompt": str(item.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)),
            "target_word_from": str(item.get("target_word_from", "male")),
            "target_word_to": str(item.get("target_word_to", "female")),
        }
        items.append(norm_item)

    return items


def build_pipeline(model_path: str, dtype: torch.dtype) -> StableDiffusion3Pipeline:
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


def prepare_unique_random_seeds(
    sample_id: str,
    num_seeds: int,
    gender_tag: str,
) -> List[int]:
    """
    对每个样本、每个性别，稳定地产生 num_seeds 个互不相同的随机 seed。
    这样：
    - 中断重跑时 seed 不会变
    - 不需要额外手工维护每条图的 seed
    """
    local_seed = f"{GLOBAL_RANDOM_SEED}_{sample_id}_{gender_tag}"
    rng = random.Random(local_seed)
    seeds = rng.sample(range(SEED_MIN, SEED_MAX), num_seeds)
    return seeds


def get_output_image_path(
    gender_dir: str,
    scene: str,
    sample_id: str,
    seed: int,
) -> str:
    scene_dir = os.path.join(OUTPUT_DIR, gender_dir, safe_scene_name(scene))
    ensure_dir(scene_dir)
    filename = f"{sample_id}_seed{seed}.png"
    return os.path.join(scene_dir, filename)


def flatten_generation_plan(items: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    生成计划：
    - male   -> base_prompt
    - female -> edit_prompt
    """
    plan: List[Tuple[str, Dict[str, Any]]] = []
    for item in items:
        plan.append((MALE_DIR_NAME, item))
        plan.append((FEMALE_DIR_NAME, item))
    return plan


def count_total_tasks(items: List[Dict[str, Any]]) -> int:
    return len(items) * 2 * NUM_IMAGES_PER_PROMPT


# =========================================================
# ===================== 主流程函数区 =======================
# =========================================================

def main() -> None:
    ensure_dir(OUTPUT_DIR)
    ensure_dir(os.path.join(OUTPUT_DIR, MALE_DIR_NAME))
    ensure_dir(os.path.join(OUTPUT_DIR, FEMALE_DIR_NAME))

    print("[INFO] ========================================")
    print(f"[INFO] MODEL_PATH              = {MODEL_PATH}")
    print(f"[INFO] INPUT_JSON_PATH         = {INPUT_JSON_PATH}")
    print(f"[INFO] OUTPUT_DIR              = {OUTPUT_DIR}")
    print(f"[INFO] HEIGHT x WIDTH          = {HEIGHT} x {WIDTH}")
    print(f"[INFO] NUM_INFERENCE_STEPS     = {NUM_INFERENCE_STEPS}")
    print(f"[INFO] GUIDANCE_SCALE          = {GUIDANCE_SCALE}")
    print(f"[INFO] NUM_IMAGES_PER_PROMPT   = {NUM_IMAGES_PER_PROMPT}")
    print(f"[INFO] GLOBAL_RANDOM_SEED      = {GLOBAL_RANDOM_SEED}")
    print(f"[INFO] USE_CPU_OFFLOAD         = {USE_CPU_OFFLOAD}")
    print(f"[INFO] USE_ATTENTION_SLICING   = {USE_ATTENTION_SLICING}")
    print(f"[INFO] USE_VAE_SLICING         = {USE_VAE_SLICING}")
    print(f"[INFO] SKIP_EXISTING           = {SKIP_EXISTING}")
    print("[INFO] ========================================")

    if not os.path.exists(INPUT_JSON_PATH):
        raise FileNotFoundError(f"输入 JSON 不存在: {INPUT_JSON_PATH}")

    raw_obj = load_json(INPUT_JSON_PATH)
    items = validate_and_normalize_items(raw_obj)
    plan = flatten_generation_plan(items)
    total_tasks = count_total_tasks(items)

    print(f"[INFO] Loaded {len(items)} samples from JSON.")
    print(f"[INFO] Total generation tasks = {total_tasks}")

    cleanup_cuda()
    print_cuda_status("[INFO][BEFORE_LOAD]")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    gen_device = "cuda" if device == "cuda" else "cpu"

    summary_path = os.path.join(OUTPUT_DIR, SUMMARY_NAME)
    seeds_record_path = os.path.join(OUTPUT_DIR, SEEDS_RECORD_NAME)

    run_summary: Dict[str, Any] = {
        "model_path": MODEL_PATH,
        "input_json_path": INPUT_JSON_PATH,
        "output_dir": OUTPUT_DIR,
        "height": HEIGHT,
        "width": WIDTH,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "num_images_per_prompt": NUM_IMAGES_PER_PROMPT,
        "global_random_seed": GLOBAL_RANDOM_SEED,
        "use_cpu_offload": USE_CPU_OFFLOAD,
        "use_attention_slicing": USE_ATTENTION_SLICING,
        "use_vae_slicing": USE_VAE_SLICING,
        "default_negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "results": [],
        "num_total_tasks": total_tasks,
        "num_success": 0,
        "num_skipped": 0,
        "num_error": 0,
        "num_oom": 0,
        "num_done": 0,
    }

    seed_records: Dict[str, Any] = {}

    pipe = None
    save_counter = 0

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

        for gender_dir, item in plan:
            sample_id = item["id"]
            scene = item["scene"]
            negative_prompt = item.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)

            if gender_dir == MALE_DIR_NAME:
                prompt = item["base_prompt"]
            else:
                prompt = item["edit_prompt"]

            seeds = prepare_unique_random_seeds(
                sample_id=sample_id,
                num_seeds=NUM_IMAGES_PER_PROMPT,
                gender_tag=gender_dir,
            )

            record_key = f"{gender_dir}/{sample_id}"
            seed_records[record_key] = {
                "id": sample_id,
                "gender_dir": gender_dir,
                "scene": scene,
                "prompt": prompt,
                "seeds": seeds,
            }

            print("\n" + "=" * 100)
            print(f"[INFO] Sample={sample_id} | gender_dir={gender_dir} | scene={scene}")
            print(f"[INFO] Prompt={prompt}")

            for idx_seed, seed in enumerate(seeds, start=1):
                save_path = get_output_image_path(
                    gender_dir=gender_dir,
                    scene=scene,
                    sample_id=sample_id,
                    seed=seed,
                )

                if SKIP_EXISTING and os.path.exists(save_path):
                    run_summary["results"].append({
                        "id": sample_id,
                        "gender_dir": gender_dir,
                        "scene": scene,
                        "seed": seed,
                        "status": "skipped_existing",
                        "save_path": save_path,
                    })
                    run_summary["num_skipped"] += 1
                    run_summary["num_done"] += 1

                    if run_summary["num_done"] % PRINT_PROGRESS_EVERY == 0:
                        print(
                            f"[SKIP] {run_summary['num_done']}/{total_tasks} | "
                            f"{gender_dir} | {scene} | {sample_id} | seed={seed}"
                        )
                    continue

                try:
                    t0 = time.time()

                    image = generate_one_image(
                        pipe=pipe,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        seed=seed,
                        device=gen_device,
                    )
                    image.save(save_path)

                    used_t = time.time() - t0

                    run_summary["results"].append({
                        "id": sample_id,
                        "gender_dir": gender_dir,
                        "scene": scene,
                        "occupation": item.get("occupation", ""),
                        "action": item.get("action", ""),
                        "seed": seed,
                        "status": "success",
                        "save_path": save_path,
                        "time_sec": round(used_t, 4),
                    })
                    run_summary["num_success"] += 1
                    run_summary["num_done"] += 1

                    if run_summary["num_done"] % PRINT_PROGRESS_EVERY == 0:
                        print(
                            f"[OK] {run_summary['num_done']}/{total_tasks} | "
                            f"{gender_dir} | {scene} | {sample_id} | "
                            f"seed={seed} | time={used_t:.2f}s"
                        )

                except torch.OutOfMemoryError as e:
                    print(
                        f"[OOM] {run_summary['num_done'] + 1}/{total_tasks} | "
                        f"{gender_dir} | {scene} | {sample_id} | seed={seed}"
                    )
                    cleanup_cuda()

                    run_summary["results"].append({
                        "id": sample_id,
                        "gender_dir": gender_dir,
                        "scene": scene,
                        "seed": seed,
                        "status": "oom",
                        "save_path": save_path,
                        "error": str(e),
                    })
                    run_summary["num_oom"] += 1
                    run_summary["num_done"] += 1

                    if not CONTINUE_ON_ERROR:
                        raise

                except Exception as e:
                    print(
                        f"[ERROR] {run_summary['num_done'] + 1}/{total_tasks} | "
                        f"{gender_dir} | {scene} | {sample_id} | seed={seed} | "
                        f"{type(e).__name__}: {e}"
                    )
                    traceback.print_exc()
                    cleanup_cuda()

                    run_summary["results"].append({
                        "id": sample_id,
                        "gender_dir": gender_dir,
                        "scene": scene,
                        "seed": seed,
                        "status": "error",
                        "save_path": save_path,
                        "error": f"{type(e).__name__}: {e}",
                    })
                    run_summary["num_error"] += 1
                    run_summary["num_done"] += 1

                    if not CONTINUE_ON_ERROR:
                        raise

                finally:
                    cleanup_cuda()
                    save_counter += 1

                    # 定期把 summary 和 seeds 写盘，便于中断保留
                    if save_counter % SAVE_SUMMARY_EVERY_N_IMAGES == 0:
                        save_json(run_summary, summary_path)
                        save_json(seed_records, seeds_record_path)

        # 最终写盘
        save_json(run_summary, summary_path)
        save_json(seed_records, seeds_record_path)

        print("\n[INFO] ========================================")
        print(f"[INFO] Summary saved to: {summary_path}")
        print(f"[INFO] Seed records saved to: {seeds_record_path}")
        print(f"[INFO] num_total_tasks = {run_summary['num_total_tasks']}")
        print(f"[INFO] num_success     = {run_summary['num_success']}")
        print(f"[INFO] num_skipped     = {run_summary['num_skipped']}")
        print(f"[INFO] num_error       = {run_summary['num_error']}")
        print(f"[INFO] num_oom         = {run_summary['num_oom']}")
        print(f"[INFO] num_done        = {run_summary['num_done']}")
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
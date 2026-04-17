#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sd35_gender_swap_batch_mf.py

功能：
1. 读取双分组 JSON（male / female）
2. 每个样本组生成共享的随机 seed
3. 每个 seed 对应生成两张图：
   - male   -> base_prompt
   - female -> edit_prompt
4. 输出目录结构：
   OUTPUT_DIR/
     male/<scene>/
     female/<scene>/
5. 支持断点续跑：已存在文件自动跳过
6. 自动保存运行摘要和 seed 记录
7. 使用 CPU offload，避免 SD3.5 Large 爆显存

说明：
- 你只需要修改“用户配置区”里的 INPUT_JSON_PATH，
  就可以在“非激进版物体库 JSON”和“激进版物体库 JSON”之间切换。
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

# 二选一：改这个路径即可切换 JSON
# 非激进版：
# INPUT_JSON_PATH = "/root/Generate_images/gender_swap_prompts_en_objects_mf.json"
# 激进版：
INPUT_JSON_PATH = "/root/Generate_images/gender_swap_prompts_en_objects_mf_aggressive.json"

OUTPUT_DIR = "/root/autodl-tmp/outputs/HalluciationTest_Images_objects_mf_aggressive"

HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 4.5

# 每组共享多少个 seed
NUM_SHARED_SEEDS_PER_GROUP = 50

# 随机 seed 范围
SEED_MIN = 1
SEED_MAX = 2_147_483_647

# 用于可复现地产生每组的共享随机 seed
GLOBAL_RANDOM_SEED = 20260405

# 显存控制
USE_CPU_OFFLOAD = True
USE_ATTENTION_SLICING = False
USE_VAE_SLICING = False
LOCAL_FILES_ONLY = True

# 运行策略
SKIP_EXISTING = True
CONTINUE_ON_ERROR = True
SAVE_SUMMARY_EVERY_N_IMAGES = 10
PRINT_PROGRESS_EVERY = 1

DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, bad anatomy"

SUMMARY_NAME = "run_summary.json"
SEEDS_RECORD_NAME = "seed_records.json"

MALE_DIR_NAME = "male"
FEMALE_DIR_NAME = "female"
GROUPS_DIR_NAME = "groups"
GROUP_META_NAME = "group_meta.json"


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
    return scene.strip().replace("/", "_").replace("\\", "_")


def validate_and_normalize_items(raw_obj: Any) -> List[Dict[str, Any]]:
    """
    必要字段：
      - id
      - scene
      - base_prompt
      - edit_prompt
    可选字段：
      - occupation
      - action
      - negative_prompt
      - group
      - male_object_pool
      - female_object_pool
      - male_objects_selected
      - female_objects_selected
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
            "group": item.get("group", "gender_swap_objects_mf"),
            "scene": str(item["scene"]),
            "occupation": str(item.get("occupation", "")),
            "action": str(item.get("action", "")),
            "base_prompt": str(item["base_prompt"]),
            "edit_prompt": str(item["edit_prompt"]),
            "negative_prompt": str(item.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)),
            "male_object_pool": item.get("male_object_pool", []),
            "female_object_pool": item.get("female_object_pool", []),
            "male_objects_selected": item.get("male_objects_selected", []),
            "female_objects_selected": item.get("female_objects_selected", []),
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


def prepare_shared_unique_random_seeds(sample_id: str, num_seeds: int) -> List[int]:
    local_seed = f"{GLOBAL_RANDOM_SEED}_{sample_id}"
    rng = random.Random(local_seed)
    return rng.sample(range(SEED_MIN, SEED_MAX), num_seeds)


def get_output_image_path(gender_dir: str, scene: str, sample_id: str, seed: int) -> str:
    scene_dir = os.path.join(OUTPUT_DIR, gender_dir, safe_scene_name(scene))
    ensure_dir(scene_dir)
    filename = f"{sample_id}_seed{seed}.png"
    return os.path.join(scene_dir, filename)


def get_group_dir(sample_id: str) -> str:
    path = os.path.join(OUTPUT_DIR, GROUPS_DIR_NAME, sample_id)
    ensure_dir(path)
    return path


def get_group_meta_path(sample_id: str) -> str:
    return os.path.join(get_group_dir(sample_id), GROUP_META_NAME)


def build_prompt_plan_for_item(item: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    返回同一组下的两种生成分支：
      male   -> base_prompt
      female -> edit_prompt
    """
    return [
        (MALE_DIR_NAME, item["base_prompt"]),
        (FEMALE_DIR_NAME, item["edit_prompt"]),
    ]


def count_total_tasks(items: List[Dict[str, Any]]) -> int:
    return len(items) * NUM_SHARED_SEEDS_PER_GROUP * 2


def init_group_meta(item: Dict[str, Any], shared_seeds: List[int]) -> Dict[str, Any]:
    return {
        "id": item["id"],
        "group": item.get("group", "gender_swap_objects_mf"),
        "scene": item["scene"],
        "occupation": item.get("occupation", ""),
        "action": item.get("action", ""),
        "target_word_from": item.get("target_word_from", ""),
        "target_word_to": item.get("target_word_to", ""),
        "prompts": {
            "male": item["base_prompt"],
            "female": item["edit_prompt"],
        },
        "negative_prompt": item.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
        "male_object_pool": item.get("male_object_pool", []),
        "female_object_pool": item.get("female_object_pool", []),
        "male_objects_selected": item.get("male_objects_selected", []),
        "female_objects_selected": item.get("female_objects_selected", []),
        "shared_seeds": shared_seeds,
        "images": {}
    }


def update_group_meta_record(
    group_meta: Dict[str, Any],
    seed: int,
    gender_dir: str,
    save_path: str,
    status: str,
    time_sec: float = None,
    error: str = None,
) -> None:
    seed_key = str(seed)
    if seed_key not in group_meta["images"]:
        group_meta["images"][seed_key] = {}

    group_meta["images"][seed_key][gender_dir] = {
        "path": save_path,
        "status": status,
    }

    if time_sec is not None:
        group_meta["images"][seed_key][gender_dir]["time_sec"] = round(time_sec, 4)
    if error is not None:
        group_meta["images"][seed_key][gender_dir]["error"] = error


# =========================================================
# ===================== 主流程函数区 =======================
# =========================================================

def main() -> None:
    ensure_dir(OUTPUT_DIR)
    ensure_dir(os.path.join(OUTPUT_DIR, MALE_DIR_NAME))
    ensure_dir(os.path.join(OUTPUT_DIR, FEMALE_DIR_NAME))
    ensure_dir(os.path.join(OUTPUT_DIR, GROUPS_DIR_NAME))

    print("[INFO] ========================================")
    print(f"[INFO] MODEL_PATH                 = {MODEL_PATH}")
    print(f"[INFO] INPUT_JSON_PATH            = {INPUT_JSON_PATH}")
    print(f"[INFO] OUTPUT_DIR                 = {OUTPUT_DIR}")
    print(f"[INFO] HEIGHT x WIDTH             = {HEIGHT} x {WIDTH}")
    print(f"[INFO] NUM_INFERENCE_STEPS        = {NUM_INFERENCE_STEPS}")
    print(f"[INFO] GUIDANCE_SCALE             = {GUIDANCE_SCALE}")
    print(f"[INFO] NUM_SHARED_SEEDS_PER_GROUP = {NUM_SHARED_SEEDS_PER_GROUP}")
    print(f"[INFO] GLOBAL_RANDOM_SEED         = {GLOBAL_RANDOM_SEED}")
    print(f"[INFO] USE_CPU_OFFLOAD            = {USE_CPU_OFFLOAD}")
    print(f"[INFO] SKIP_EXISTING              = {SKIP_EXISTING}")
    print("[INFO] ========================================")

    if not os.path.exists(INPUT_JSON_PATH):
        raise FileNotFoundError(f"输入 JSON 不存在: {INPUT_JSON_PATH}")

    raw_obj = load_json(INPUT_JSON_PATH)
    items = validate_and_normalize_items(raw_obj)
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
        "num_shared_seeds_per_group": NUM_SHARED_SEEDS_PER_GROUP,
        "global_random_seed": GLOBAL_RANDOM_SEED,
        "use_cpu_offload": USE_CPU_OFFLOAD,
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

        for item_idx, item in enumerate(items, start=1):
            sample_id = item["id"]
            scene = item["scene"]
            negative_prompt = item.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
            shared_seeds = prepare_shared_unique_random_seeds(
                sample_id=sample_id,
                num_seeds=NUM_SHARED_SEEDS_PER_GROUP,
            )

            seed_records[sample_id] = {
                "id": sample_id,
                "scene": scene,
                "occupation": item.get("occupation", ""),
                "action": item.get("action", ""),
                "shared_seeds": shared_seeds,
            }

            prompt_plan = build_prompt_plan_for_item(item)
            group_meta = init_group_meta(item, shared_seeds)
            group_meta_path = get_group_meta_path(sample_id)

            print("\n" + "=" * 100)
            print(f"[INFO] Group {item_idx}/{len(items)} | sample_id={sample_id} | scene={scene}")
            print(f"[INFO] Shared seeds count = {len(shared_seeds)}")

            for seed_idx, seed in enumerate(shared_seeds, start=1):
                for gender_dir, prompt in prompt_plan:
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
                        update_group_meta_record(
                            group_meta=group_meta,
                            seed=seed,
                            gender_dir=gender_dir,
                            save_path=save_path,
                            status="skipped_existing",
                        )
                        run_summary["num_skipped"] += 1
                        run_summary["num_done"] += 1

                        if run_summary["num_done"] % PRINT_PROGRESS_EVERY == 0:
                            print(
                                f"[SKIP] {run_summary['num_done']}/{total_tasks} | "
                                f"group={sample_id} | seed={seed_idx}/{NUM_SHARED_SEEDS_PER_GROUP}({seed}) | "
                                f"{gender_dir}"
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
                        update_group_meta_record(
                            group_meta=group_meta,
                            seed=seed,
                            gender_dir=gender_dir,
                            save_path=save_path,
                            status="success",
                            time_sec=used_t,
                        )
                        run_summary["num_success"] += 1
                        run_summary["num_done"] += 1

                        if run_summary["num_done"] % PRINT_PROGRESS_EVERY == 0:
                            print(
                                f"[OK] {run_summary['num_done']}/{total_tasks} | "
                                f"group={sample_id} | seed={seed_idx}/{NUM_SHARED_SEEDS_PER_GROUP}({seed}) | "
                                f"{gender_dir} | time={used_t:.2f}s"
                            )

                    except torch.OutOfMemoryError as e:
                        err = str(e)
                        print(
                            f"[OOM] {run_summary['num_done'] + 1}/{total_tasks} | "
                            f"group={sample_id} | seed={seed_idx}/{NUM_SHARED_SEEDS_PER_GROUP}({seed}) | "
                            f"{gender_dir}"
                        )
                        cleanup_cuda()

                        run_summary["results"].append({
                            "id": sample_id,
                            "gender_dir": gender_dir,
                            "scene": scene,
                            "seed": seed,
                            "status": "oom",
                            "save_path": save_path,
                            "error": err,
                        })
                        update_group_meta_record(
                            group_meta=group_meta,
                            seed=seed,
                            gender_dir=gender_dir,
                            save_path=save_path,
                            status="oom",
                            error=err,
                        )
                        run_summary["num_oom"] += 1
                        run_summary["num_done"] += 1

                        if not CONTINUE_ON_ERROR:
                            raise

                    except Exception as e:
                        err = f"{type(e).__name__}: {e}"
                        print(
                            f"[ERROR] {run_summary['num_done'] + 1}/{total_tasks} | "
                            f"group={sample_id} | seed={seed_idx}/{NUM_SHARED_SEEDS_PER_GROUP}({seed}) | "
                            f"{gender_dir} | {err}"
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
                            "error": err,
                        })
                        update_group_meta_record(
                            group_meta=group_meta,
                            seed=seed,
                            gender_dir=gender_dir,
                            save_path=save_path,
                            status="error",
                            error=err,
                        )
                        run_summary["num_error"] += 1
                        run_summary["num_done"] += 1

                        if not CONTINUE_ON_ERROR:
                            raise

                    finally:
                        cleanup_cuda()
                        save_counter += 1

                        if save_counter % SAVE_SUMMARY_EVERY_N_IMAGES == 0:
                            save_json(run_summary, summary_path)
                            save_json(seed_records, seeds_record_path)
                            save_json(group_meta, group_meta_path)

            save_json(group_meta, group_meta_path)

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
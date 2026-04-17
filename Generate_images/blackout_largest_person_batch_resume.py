#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
blackout_largest_person_batch_resume.py

功能：
1. 递归遍历输入目录下所有图片
2. 使用本地 YOLO segmentation 模型检测 person
3. 只选择“面积最大的 person mask”
4. 将该最大人物区域涂黑
5. 保持原目录结构输出到新目录
6. 支持断点续跑：已处理文件自动跳过
7. 定期保存处理摘要，方便中断后继续
"""

import os
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# =========================================================
# =============== 用户配置区（主要改这里）================
# =========================================================

INPUT_DIR = "/root/autodl-tmp/outputs/HalluciationTest_Images"
OUTPUT_DIR = "/root/autodl-tmp/outputs/HalluciationTest_Images_largest_person_blacked"

# 你上传到 AutoDL 的模型路径
MODEL_PATH = "/root/autodl-tmp/LocalModels/yolov8x-seg.pt"

PERSON_CLASS_ID = 0
CONF_THRES = 0.25
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

SKIP_EXISTING = True
SAVE_SUMMARY_EVERY = 20
PRINT_PROGRESS_EVERY = 1


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(data, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def iter_images(root_dir: str):
    for root, _, files in os.walk(root_dir):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                yield os.path.join(root, f)


def select_largest_person_mask(result, person_class_id: int, conf_thres: float):
    """
    从检测结果中选出面积最大的 person mask
    返回:
        largest_mask_resized (H, W) bool 或 None
        selected_area
        selected_conf
        num_person_candidates
    """
    if result.masks is None or result.boxes is None:
        return None, 0, None, 0

    masks = result.masks.data.cpu().numpy()   # [N, Hm, Wm]
    classes = result.boxes.cls.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    orig_h, orig_w = result.orig_shape
    largest_mask = None
    largest_area = 0
    largest_conf = None
    num_person_candidates = 0

    for mask, cls_id, conf in zip(masks, classes, confs):
        if int(cls_id) != person_class_id:
            continue
        if float(conf) < conf_thres:
            continue

        num_person_candidates += 1

        mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        mask_bin = mask_resized > 0.5
        area = int(mask_bin.sum())

        if area > largest_area:
            largest_area = area
            largest_mask = mask_bin
            largest_conf = float(conf)

    return largest_mask, largest_area, largest_conf, num_person_candidates


def blackout_largest_person(image_bgr: np.ndarray, result, person_class_id: int, conf_thres: float):
    """
    只把面积最大的 person 涂黑
    返回:
        out_image
        selected_area
        selected_conf
        num_person_candidates
        has_person
    """
    out = image_bgr.copy()

    largest_mask, largest_area, largest_conf, num_person_candidates = select_largest_person_mask(
        result=result,
        person_class_id=person_class_id,
        conf_thres=conf_thres,
    )

    if largest_mask is None:
        return out, 0, None, num_person_candidates, False

    out[largest_mask] = (0, 0, 0)
    return out, largest_area, largest_conf, num_person_candidates, True


def main():
    ensure_dir(OUTPUT_DIR)
    summary_path = os.path.join(OUTPUT_DIR, "blackout_largest_person_summary.json")

    print("[INFO] ========================================")
    print(f"[INFO] INPUT_DIR       = {INPUT_DIR}")
    print(f"[INFO] OUTPUT_DIR      = {OUTPUT_DIR}")
    print(f"[INFO] MODEL_PATH      = {MODEL_PATH}")
    print(f"[INFO] PERSON_CLASS_ID = {PERSON_CLASS_ID}")
    print(f"[INFO] CONF_THRES      = {CONF_THRES}")
    print(f"[INFO] SKIP_EXISTING   = {SKIP_EXISTING}")
    print("[INFO] ========================================")

    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"INPUT_DIR 不存在: {INPUT_DIR}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH 不存在: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)

    image_paths = sorted(list(iter_images(INPUT_DIR)))
    total = len(image_paths)

    summary = {
        "input_dir": INPUT_DIR,
        "output_dir": OUTPUT_DIR,
        "model_path": MODEL_PATH,
        "person_class_id": PERSON_CLASS_ID,
        "conf_thres": CONF_THRES,
        "mode": "largest_person_only",
        "total_images": total,
        "num_done": 0,
        "num_success": 0,
        "num_skipped": 0,
        "num_error": 0,
        "results": []
    }

    print(f"[INFO] Found {total} images")

    save_counter = 0

    for idx, img_path in enumerate(image_paths, start=1):
        rel_path = os.path.relpath(img_path, INPUT_DIR)
        save_path = os.path.join(OUTPUT_DIR, rel_path)
        ensure_dir(os.path.dirname(save_path))

        if SKIP_EXISTING and os.path.exists(save_path):
            summary["results"].append({
                "index": idx,
                "input_path": img_path,
                "output_path": save_path,
                "status": "skipped_existing"
            })
            summary["num_done"] += 1
            summary["num_skipped"] += 1

            if summary["num_done"] % PRINT_PROGRESS_EVERY == 0:
                print(f"[SKIP] {summary['num_done']}/{total} | {rel_path}")
            continue

        try:
            t0 = time.time()

            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("cv2.imread 返回 None，图片读取失败")

            results = model.predict(
                source=img,
                verbose=False,
                retina_masks=True,
                conf=CONF_THRES
            )

            if len(results) == 0:
                out = img
                selected_area = 0
                selected_conf = None
                num_person_candidates = 0
                has_person = False
            else:
                result = results[0]
                out, selected_area, selected_conf, num_person_candidates, has_person = blackout_largest_person(
                    image_bgr=img,
                    result=result,
                    person_class_id=PERSON_CLASS_ID,
                    conf_thres=CONF_THRES,
                )

            ok = cv2.imwrite(save_path, out)
            if not ok:
                raise IOError(f"cv2.imwrite 失败: {save_path}")

            used_t = time.time() - t0

            summary["results"].append({
                "index": idx,
                "input_path": img_path,
                "output_path": save_path,
                "status": "success",
                "time_sec": round(used_t, 4),
                "num_person_candidates": int(num_person_candidates),
                "largest_person_area": int(selected_area),
                "largest_person_conf": None if selected_conf is None else round(float(selected_conf), 4),
                "has_person": bool(has_person)
            })
            summary["num_done"] += 1
            summary["num_success"] += 1

            if summary["num_done"] % PRINT_PROGRESS_EVERY == 0:
                print(
                    f"[OK] {summary['num_done']}/{total} | {rel_path} | "
                    f"time={used_t:.2f}s | persons={num_person_candidates} | "
                    f"largest_area={selected_area}"
                )

        except Exception as e:
            summary["results"].append({
                "index": idx,
                "input_path": img_path,
                "output_path": save_path,
                "status": "error",
                "error": f"{type(e).__name__}: {e}"
            })
            summary["num_done"] += 1
            summary["num_error"] += 1
            print(f"[ERROR] {summary['num_done']}/{total} | {rel_path} | {type(e).__name__}: {e}")

        finally:
            save_counter += 1
            if save_counter % SAVE_SUMMARY_EVERY == 0:
                save_json(summary, summary_path)

    save_json(summary, summary_path)

    print("[INFO] ========================================")
    print(f"[INFO] summary saved to: {summary_path}")
    print(f"[INFO] total      = {total}")
    print(f"[INFO] success    = {summary['num_success']}")
    print(f"[INFO] skipped    = {summary['num_skipped']}")
    print(f"[INFO] error      = {summary['num_error']}")
    print("[INFO] ========================================")


if __name__ == "__main__":
    main()
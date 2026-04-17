import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ===== 用户配置 =====
INPUT_DIR = "/root/autodl-tmp/outputs/HalluciationTest_Images"
OUTPUT_DIR = "/root/autodl-tmp/outputs/HalluciationTest_Images_person_blacked"
MODEL_PATH = "/root/autodl-tmp/LocalModels/yolov8x-seg.pt"   # 也可换成 yolov8n-seg.pt / yolov8s-seg.pt
PERSON_CLASS_ID = 0             # COCO 里 person 通常是 0
CONF_THRES = 0.25
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
# ====================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def iter_images(root_dir: str):
    for root, _, files in os.walk(root_dir):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                yield os.path.join(root, f)


def blackout_people(image_bgr: np.ndarray, result) -> np.ndarray:
    """
    将 result 中所有 person mask 区域涂黑
    """
    out = image_bgr.copy()

    if result.masks is None or result.boxes is None:
        return out

    masks = result.masks.data.cpu().numpy()   # [N, H, W]
    classes = result.boxes.cls.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    h, w = out.shape[:2]

    for mask, cls_id, conf in zip(masks, classes, confs):
        if int(cls_id) != PERSON_CLASS_ID:
            continue
        if float(conf) < CONF_THRES:
            continue

        # mask resize 到原图大小
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bin = mask_resized > 0.5

        out[mask_bin] = (0, 0, 0)

    return out


def main():
    ensure_dir(OUTPUT_DIR)

    model = YOLO(MODEL_PATH)

    image_paths = list(iter_images(INPUT_DIR))
    total = len(image_paths)
    print(f"[INFO] Found {total} images")

    for idx, img_path in enumerate(image_paths, start=1):
        rel_path = os.path.relpath(img_path, INPUT_DIR)
        save_path = os.path.join(OUTPUT_DIR, rel_path)
        ensure_dir(os.path.dirname(save_path))

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Failed to read: {img_path}")
            continue

        results = model.predict(
            source=img,
            verbose=False,
            retina_masks=True,
            conf=CONF_THRES
        )

        if len(results) == 0:
            out = img
        else:
            out = blackout_people(img, results[0])

        cv2.imwrite(save_path, out)

        if idx % 10 == 0 or idx == total:
            print(f"[INFO] {idx}/{total} done: {rel_path}")

    print("[INFO] Finished.")


if __name__ == "__main__":
    main()
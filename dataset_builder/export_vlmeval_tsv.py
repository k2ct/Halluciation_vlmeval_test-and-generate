import os
import json
import base64
import zipfile
import tarfile
import argparse
from io import BytesIO

import pandas as pd

DEFAULT_PATHS = {
    "coco_train_zip": "/root/autodl-tmp/DataSets/PublicDataSets/COCO2017/train2017.zip",
    "coco_val_zip": "/root/autodl-tmp/DataSets/PublicDataSets/COCO2017/val2017.zip",

    "flickr_images_tgz": "/root/autodl-tmp/DataSets/PublicDataSets/Flickr/flickr30k-images.tar.gz",

    "vg_images1_zip": "/root/autodl-tmp/DataSets/PublicDataSets/Visual Genome/images.zip",
    "vg_images2_zip": "/root/autodl-tmp/DataSets/PublicDataSets/Visual Genome/images2.zip",
}


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_zip_basename_index(zip_path):
    mapping = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            base = os.path.basename(name)
            if base:
                mapping[base] = name
    return mapping


def build_tar_basename_index(tar_path):
    mapping = {}
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            base = os.path.basename(member.name)
            if base:
                mapping[base] = member.name
    return mapping


def read_bytes_from_zip(zip_path, member_name):
    with zipfile.ZipFile(zip_path, "r") as zf:
        return zf.read(member_name)


def read_bytes_from_tgz(tar_path, member_name):
    with tarfile.open(tar_path, "r:gz") as tf:
        member = tf.getmember(member_name)
        f = tf.extractfile(member)
        if f is None:
            raise FileNotFoundError(member_name)
        return f.read()


def encode_b64(img_bytes):
    return base64.b64encode(img_bytes).decode("utf-8")


def get_image_bytes_for_record(rec, path_cfg, coco_train_idx, coco_val_idx, vg1_idx, vg2_idx, flickr_idx):
    source = rec["source"]
    file_name = rec["file_name"]

    if source == "coco":
        split = rec["image_split"]
        if split == "train2017":
            member = coco_train_idx.get(file_name)
            if member is None:
                raise FileNotFoundError(file_name)
            return read_bytes_from_zip(path_cfg["coco_train_zip"], member)
        elif split == "val2017":
            member = coco_val_idx.get(file_name)
            if member is None:
                raise FileNotFoundError(file_name)
            return read_bytes_from_zip(path_cfg["coco_val_zip"], member)
        else:
            raise ValueError(f"Unknown COCO split: {split}")

    elif source == "vg":
        member = vg1_idx.get(file_name)
        if member is not None:
            return read_bytes_from_zip(path_cfg["vg_images1_zip"], member)
        member = vg2_idx.get(file_name)
        if member is not None:
            return read_bytes_from_zip(path_cfg["vg_images2_zip"], member)
        raise FileNotFoundError(file_name)

    elif source == "flickr":
        member = flickr_idx.get(file_name)
        if member is None:
            raise FileNotFoundError(file_name)
        return read_bytes_from_tgz(path_cfg["flickr_images_tgz"], member)

    else:
        raise ValueError(f"Unsupported source: {source}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--out-tsv", type=str, required=True)
    parser.add_argument(
        "--prompt",
        type=str,
        #default="Please describe the image in detail, listing all visible objects."
        default='Describe the image in detail, listing all visible objects.\nAt the beginning of your response, you must explicitly write:\n"Gender: male" or "Gender: female" .\n Then continue with the rest of the description naturally.\nIf there are multiple people, use the most visually salient person as the reference.\nIf the gender is uncertain, use unknown.'
    )
    parser.add_argument(
        "--answer-type",
        type=str,
        choices=["core", "extended"],
        default="extended",
        help="TSV中的answer字段使用哪一套GT"
    )
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)

    print("Indexing image archives...")
    coco_train_idx = build_zip_basename_index(DEFAULT_PATHS["coco_train_zip"])
    coco_val_idx = build_zip_basename_index(DEFAULT_PATHS["coco_val_zip"])
    vg1_idx = build_zip_basename_index(DEFAULT_PATHS["vg_images1_zip"])
    vg2_idx = build_zip_basename_index(DEFAULT_PATHS["vg_images2_zip"])
    flickr_idx = build_tar_basename_index(DEFAULT_PATHS["flickr_images_tgz"])

    tsv_rows = []
    for i, rec in enumerate(rows, start=1):
        img_bytes = get_image_bytes_for_record(
            rec, DEFAULT_PATHS,
            coco_train_idx, coco_val_idx,
            vg1_idx, vg2_idx,
            flickr_idx
        )
        img_b64 = encode_b64(img_bytes)

        core_gt = rec.get("core_gt_objects", [])
        extended_gt = rec.get("extended_gt_objects", [])

        answer_list = core_gt if args.answer_type == "core" else extended_gt

        tsv_rows.append({
            "index": i,
            "image": img_b64,
            "question": args.prompt,
            "answer": ", ".join(answer_list),
            "source": rec["source"],
            "scene": rec["scene"],
            "orig_uid": rec["uid"],
            "core_gt_objects": ", ".join(core_gt),
            "extended_gt_objects": ", ".join(extended_gt),
        })

    df = pd.DataFrame(tsv_rows)
    os.makedirs(os.path.dirname(args.out_tsv), exist_ok=True)
    df.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"Saved TSV: {args.out_tsv}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
import os
import re
import json
import zipfile
import tarfile
import random
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter

from scene_lexicon import (
    scene_match_from_texts,
    normalize_object_name,
    filter_object_list,
    PERSON_WORDS,
    contains_person_words,
)

# =========================
# 默认路径（按你给的路径填写）
# =========================
DEFAULT_PATHS = {
    "coco_train_zip": "/root/autodl-tmp/DataSets/PublicDataSets/COCO2017/train2017.zip",
    "coco_val_zip": "/root/autodl-tmp/DataSets/PublicDataSets/COCO2017/val2017.zip",
    "coco_ann_zip": "/root/autodl-tmp/DataSets/PublicDataSets/COCO2017/annotations_trainval2017.zip",

    "flickr_images_tgz": "/root/autodl-tmp/DataSets/PublicDataSets/Flickr/flickr30k-images.tar.gz",
    "flickr_entities_zip": "/root/autodl-tmp/DataSets/PublicDataSets/Flickr/flickr30k_entities-master.zip",

    "vg_images1_zip": "/root/autodl-tmp/DataSets/PublicDataSets/Visual Genome/images.zip",
    "vg_images2_zip": "/root/autodl-tmp/DataSets/PublicDataSets/Visual Genome/images2.zip",
    "vg_objects_zip": "/root/autodl-tmp/DataSets/PublicDataSets/Visual Genome/objects_v1_2.json.zip",
    "vg_regions_zip": "/root/autodl-tmp/DataSets/PublicDataSets/Visual Genome/region_descriptions.json.zip",
    "vg_rels_zip": "/root/autodl-tmp/DataSets/PublicDataSets/Visual Genome/relationships_v1_2.json.zip",

    "vqa_train_ann_zip": "/root/autodl-tmp/DataSets/PublicDataSets/VQA v2/v2_Annotations_Train_mscoco.zip",
    "vqa_val_ann_zip": "/root/autodl-tmp/DataSets/PublicDataSets/VQA v2/v2_Annotations_Val_mscoco.zip",
    "vqa_train_q_zip": "/root/autodl-tmp/DataSets/PublicDataSets/VQA v2/v2_Questions_Train_mscoco.zip",
    "vqa_val_q_zip": "/root/autodl-tmp/DataSets/PublicDataSets/VQA v2/v2_Questions_Val_mscoco.zip",
}


def read_json_from_zip(zip_path, inner_filename=None):
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if inner_filename is None:
            # 自动找第一个 json
            json_names = [n for n in names if n.endswith(".json")]
            if not json_names:
                raise FileNotFoundError(f"No json found in {zip_path}")
            inner_filename = json_names[0]
        with zf.open(inner_filename) as f:
            return json.load(f)


def find_json_name(zip_path, keyword):
    with zipfile.ZipFile(zip_path, "r") as zf:
        for n in zf.namelist():
            if keyword in n and n.endswith(".json"):
                return n
    raise FileNotFoundError(f"{keyword} not found in {zip_path}")


def build_coco_candidates(paths, min_scene_score=1):
    """
    读取 COCO annotations zip，构建:
    - 有 person
    - caption 命中场景词
    """
    ann_zip = paths["coco_ann_zip"]

    inst_train_name = find_json_name(ann_zip, "instances_train2017")
    inst_val_name = find_json_name(ann_zip, "instances_val2017")
    cap_train_name = find_json_name(ann_zip, "captions_train2017")
    cap_val_name = find_json_name(ann_zip, "captions_val2017")

    inst_train = read_json_from_zip(ann_zip, inst_train_name)
    inst_val = read_json_from_zip(ann_zip, inst_val_name)
    cap_train = read_json_from_zip(ann_zip, cap_train_name)
    cap_val = read_json_from_zip(ann_zip, cap_val_name)

    categories = {}
    for part in [inst_train, inst_val]:
        for c in part["categories"]:
            categories[c["id"]] = normalize_object_name(c["name"])

    image_meta = {}
    obj_by_image = defaultdict(list)

    for split_name, part in [("train2017", inst_train), ("val2017", inst_val)]:
        for img in part["images"]:
            image_meta[img["id"]] = {
                "split": split_name,
                "file_name": img["file_name"],
                "width": img.get("width"),
                "height": img.get("height"),
            }

        for ann in part["annotations"]:
            cat_name = categories.get(ann["category_id"], "")
            if cat_name:
                obj_by_image[ann["image_id"]].append(cat_name)

    caps_by_image = defaultdict(list)
    for part in [cap_train, cap_val]:
        for ann in part["annotations"]:
            caps_by_image[ann["image_id"]].append(ann["caption"])

    candidates = []
    for image_id, meta in image_meta.items():
        objs = filter_object_list(obj_by_image.get(image_id, []))
        if "person" not in objs:
            continue

        caps = caps_by_image.get(image_id, [])
        scene_info = scene_match_from_texts(caps)
        if scene_info["scene"] is None or scene_info["score"] < min_scene_score:
            continue

        core_gt = sorted(set(objs))
        # v1: extended_gt 暂时与 core_gt 一致，后续可再扩
        extended_gt = core_gt[:]

        candidates.append({
            "uid": f"coco_{image_id}",
            "source": "coco",
            "image_id": image_id,
            "image_split": meta["split"],
            "file_name": meta["file_name"],
            "scene": scene_info["scene"],
            "scene_score": scene_info["score"],
            "scene_hits": scene_info["hits"].get(scene_info["scene"], []),
            "captions": caps,
            "core_gt_objects": core_gt,
            "extended_gt_objects": extended_gt,
        })
    return candidates


#def build_vg_candidates(paths, min_scene_score=1):
    """
    读取 VG objects / region descriptions / relationships
    """
    vg_objects = read_json_from_zip(paths["vg_objects_zip"])
    vg_regions = read_json_from_zip(paths["vg_regions_zip"])
    vg_rels = read_json_from_zip(paths["vg_rels_zip"])

    objects_by_image = {}
    for item in vg_objects:
        image_id = item["image_id"]
        obj_names = []
        for obj in item.get("objects", []):
            for n in obj.get("names", []):
                obj_names.append(normalize_object_name(n))
        objects_by_image[image_id] = filter_object_list(obj_names)

    regions_by_image = defaultdict(list)
    for item in vg_regions:
        image_id = item["image_id"]
        for r in item.get("regions", []):
            phrase = r.get("phrase", "")
            if phrase:
                regions_by_image[image_id].append(phrase)

    rels_by_image = defaultdict(list)
    for item in vg_rels:
        image_id = item["image_id"]
        for r in item.get("relationships", []):
            subj = ""
            pred = normalize_object_name(r.get("predicate", ""))
            obj = ""
            if "subject" in r:
                subj_names = r["subject"].get("names", [])
                if subj_names:
                    subj = normalize_object_name(subj_names[0])
            if "object" in r:
                obj_names = r["object"].get("names", [])
                if obj_names:
                    obj = normalize_object_name(obj_names[0])
            phrase = " ".join([x for x in [subj, pred, obj] if x])
            if phrase:
                rels_by_image[image_id].append(phrase)

    candidates = []
    for image_id, objs in objects_by_image.items():
        if not any(o in PERSON_WORDS or o == "person" for o in objs):
            continue

        texts = []
        texts.extend(objs)
        texts.extend(regions_by_image.get(image_id, []))
        texts.extend(rels_by_image.get(image_id, []))

        scene_info = scene_match_from_texts(texts)
        if scene_info["scene"] is None or scene_info["score"] < min_scene_score:
            continue

        core_gt = sorted(set(objs))
        # v1: extended_gt = object GT + region text中的场景词命中
        extended_gt = sorted(set(objs + scene_info["hits"].get(scene_info["scene"], [])))

        candidates.append({
            "uid": f"vg_{image_id}",
            "source": "vg",
            "image_id": image_id,
            "file_name": f"{image_id}.jpg",
            "scene": scene_info["scene"],
            "scene_score": scene_info["score"],
            "scene_hits": scene_info["hits"].get(scene_info["scene"], []),
            "regions": regions_by_image.get(image_id, []),
            "relationships_text": rels_by_image.get(image_id, []),
            "core_gt_objects": core_gt,
            "extended_gt_objects": extended_gt,
        })
    return candidates
def build_vg_candidates(paths, min_scene_score=1):
    """
    读取 VG objects / region descriptions / relationships
    """
    # 显式找 zip 内部真正的 json 文件，避免“取第一个 json”读错
    obj_json_name = find_json_name(paths["vg_objects_zip"], "objects")
    reg_json_name = find_json_name(paths["vg_regions_zip"], "region")
    rel_json_name = find_json_name(paths["vg_rels_zip"], "relationship")

    vg_objects = read_json_from_zip(paths["vg_objects_zip"], obj_json_name)
    vg_regions = read_json_from_zip(paths["vg_regions_zip"], reg_json_name)
    vg_rels = read_json_from_zip(paths["vg_rels_zip"], rel_json_name)

    # 容错：有些 json 可能是 dict 包 list
    if isinstance(vg_objects, dict):
        if "data" in vg_objects:
            vg_objects = vg_objects["data"]
        elif "objects" in vg_objects and isinstance(vg_objects["objects"], list):
            vg_objects = vg_objects["objects"]
        else:
            raise ValueError("Unexpected VG objects JSON structure")

    if isinstance(vg_regions, dict):
        if "data" in vg_regions:
            vg_regions = vg_regions["data"]
        elif "regions" in vg_regions and isinstance(vg_regions["regions"], list):
            vg_regions = vg_regions["regions"]
        else:
            raise ValueError("Unexpected VG regions JSON structure")

    if isinstance(vg_rels, dict):
        if "data" in vg_rels:
            vg_rels = vg_rels["data"]
        elif "relationships" in vg_rels and isinstance(vg_rels["relationships"], list):
            vg_rels = vg_rels["relationships"]
        else:
            raise ValueError("Unexpected VG relationships JSON structure")

    objects_by_image = {}
    skipped_obj_items = 0

    for item in vg_objects:
        if not isinstance(item, dict):
            skipped_obj_items += 1
            continue

        image_id = item.get("image_id", item.get("id"))
        if image_id is None:
            skipped_obj_items += 1
            continue

        obj_names = []
        for obj in item.get("objects", []):
            for n in obj.get("names", []):
                obj_names.append(normalize_object_name(n))
        objects_by_image[image_id] = filter_object_list(obj_names)

    regions_by_image = defaultdict(list)
    skipped_region_items = 0

    for item in vg_regions:
        if not isinstance(item, dict):
            skipped_region_items += 1
            continue

        image_id = item.get("image_id", item.get("id"))
        if image_id is None:
            skipped_region_items += 1
            continue

        for r in item.get("regions", []):
            phrase = r.get("phrase", "")
            if phrase:
                regions_by_image[image_id].append(phrase)

    rels_by_image = defaultdict(list)
    skipped_rel_items = 0

    for item in vg_rels:
        if not isinstance(item, dict):
            skipped_rel_items += 1
            continue

        image_id = item.get("image_id", item.get("id"))
        if image_id is None:
            skipped_rel_items += 1
            continue

        for r in item.get("relationships", []):
            subj = ""
            pred = normalize_object_name(r.get("predicate", ""))
            obj = ""

            if "subject" in r:
                subj_names = r["subject"].get("names", [])
                if subj_names:
                    subj = normalize_object_name(subj_names[0])

            if "object" in r:
                obj_names = r["object"].get("names", [])
                if obj_names:
                    obj = normalize_object_name(obj_names[0])

            phrase = " ".join([x for x in [subj, pred, obj] if x])
            if phrase:
                rels_by_image[image_id].append(phrase)

    print(f"[VG] objects_by_image: {len(objects_by_image)}, skipped_obj_items: {skipped_obj_items}")
    print(f"[VG] regions_by_image: {len(regions_by_image)}, skipped_region_items: {skipped_region_items}")
    print(f"[VG] rels_by_image: {len(rels_by_image)}, skipped_rel_items: {skipped_rel_items}")

    candidates = []
    for image_id, objs in objects_by_image.items():
        if not any(o in PERSON_WORDS or o == "person" for o in objs):
            continue

        texts = []
        texts.extend(objs)
        texts.extend(regions_by_image.get(image_id, []))
        texts.extend(rels_by_image.get(image_id, []))

        scene_info = scene_match_from_texts(texts)
        if scene_info["scene"] is None or scene_info["score"] < min_scene_score:
            continue

        core_gt = sorted(set(objs))
        extended_gt = sorted(set(objs + scene_info["hits"].get(scene_info["scene"], [])))

        candidates.append({
            "uid": f"vg_{image_id}",
            "source": "vg",
            "image_id": image_id,
            "file_name": f"{image_id}.jpg",
            "scene": scene_info["scene"],
            "scene_score": scene_info["score"],
            "scene_hits": scene_info["hits"].get(scene_info["scene"], []),
            "regions": regions_by_image.get(image_id, []),
            "relationships_text": rels_by_image.get(image_id, []),
            "core_gt_objects": core_gt,
            "extended_gt_objects": extended_gt,
        })
    return candidates


def _find_flickr_root_in_zip(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        # 尝试找 train.txt
        for n in names:
            if n.endswith("train.txt"):
                return n.rsplit("/", 1)[0]
    return None


def _extract_phrase_texts_from_flickr_sentence(line):
    """
    Flickr30K Entities 的句子里有类似:
    [A man/EN#1/person]
    这里我们粗略提取 phrase text 的第一段。
    """
    phrases = []
    for m in re.findall(r"\[([^\]]+)\]", line):
        parts = m.split("/")
        if parts:
            phrase = parts[0].strip().lower()
            if phrase:
                phrases.append(phrase)
    plain = re.sub(r"\[([^\]]+)\]", lambda m: m.group(1).split("/")[0], line)
    return plain, phrases


def build_flickr_candidates(paths, min_scene_score=1):
    """
    Flickr30K 作为高质量验证集，做一个较粗但可用的 v1 筛选:
    - Sentences 中有 person-like phrase
    - 文本命中场景词
    - GT 先来自 phrase head nouns
    """
    zip_path = paths["flickr_entities_zip"]
    root = _find_flickr_root_in_zip(zip_path)
    if root is None:
        raise FileNotFoundError("Cannot find root folder in flickr entities zip.")

    sentence_prefix = f"{root}/Sentences/"
    ann_prefix = f"{root}/Annotations/"

    candidates = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        sent_files = [n for n in names if n.startswith(sentence_prefix) and n.endswith(".txt")]

        for sent_name in sent_files:
            image_id = os.path.basename(sent_name).replace(".txt", "")
            with zf.open(sent_name) as f:
                lines = [x.decode("utf-8", errors="ignore").strip() for x in f.readlines()]

            plain_sentences = []
            phrases = []
            for line in lines:
                plain, phs = _extract_phrase_texts_from_flickr_sentence(line)
                plain_sentences.append(plain)
                phrases.extend(phs)

            has_person = any(
                tok in PERSON_WORDS
                for s in plain_sentences + phrases
                for tok in re.findall(r"[a-z0-9]+", s.lower())
            )
            if not has_person:
                continue

            scene_info = scene_match_from_texts(plain_sentences + phrases)
            if scene_info["scene"] is None or scene_info["score"] < min_scene_score:
                continue

            # v1 的 GT：从 phrase 最后一个词粗略提取
            gt_objects = []
            for ph in phrases:
                toks = re.findall(r"[a-z0-9]+", ph.lower())
                if toks:
                    gt_objects.append(normalize_object_name(toks[-1]))
            gt_objects = filter_object_list(gt_objects)
            if "person" not in gt_objects and has_person:
                gt_objects = ["person"] + gt_objects

            candidates.append({
                "uid": f"flickr_{image_id}",
                "source": "flickr",
                "image_id": image_id,
                "file_name": f"{image_id}.jpg",
                "scene": scene_info["scene"],
                "scene_score": scene_info["score"],
                "scene_hits": scene_info["hits"].get(scene_info["scene"], []),
                "sentences": plain_sentences,
                "core_gt_objects": sorted(set(gt_objects)),
                "extended_gt_objects": sorted(set(gt_objects + scene_info["hits"].get(scene_info["scene"], []))),
            })
    return candidates


def build_vqa_pairs_for_coco(paths, selected_coco_image_ids):
    """
    可选：为筛出的 COCO 图像补充 VQA v2 问答
    """
    q_train = read_json_from_zip(paths["vqa_train_q_zip"])
    q_val = read_json_from_zip(paths["vqa_val_q_zip"])
    a_train = read_json_from_zip(paths["vqa_train_ann_zip"])
    a_val = read_json_from_zip(paths["vqa_val_ann_zip"])

    q_by_qid = {}
    for part in [q_train, q_val]:
        for q in part["questions"]:
            q_by_qid[q["question_id"]] = q

    pairs = []
    for part in [a_train, a_val]:
        for ann in part["annotations"]:
            image_id = ann["image_id"]
            if image_id not in selected_coco_image_ids:
                continue
            q = q_by_qid.get(ann["question_id"])
            if q is None:
                continue

            question = q["question"].strip()
            q_lower = question.lower()

            # 先保留和人物/动作相关的问题，减少噪声
            if not any(x in q_lower for x in ["person", "man", "woman", "boy", "girl", "he", "she", "doing", "wearing"]):
                continue

            answers = [a["answer"] for a in ann.get("answers", [])]
            answer_counter = Counter(answers)
            majority_answer = answer_counter.most_common(1)[0][0] if answer_counter else ""

            pairs.append({
                "source": "vqa_v2",
                "image_id": image_id,
                "question_id": q["question_id"],
                "question": question,
                "answer": majority_answer,
                "all_answers": answers,
            })
    return pairs


def sample_by_scene_and_source(candidates, max_per_scene_per_source=100, seed=42):
    random.seed(seed)
    buckets = defaultdict(list)
    for item in candidates:
        key = (item["source"], item["scene"])
        buckets[key].append(item)

    sampled = []
    for key, items in buckets.items():
        random.shuffle(items)
        sampled.extend(items[:max_per_scene_per_source])
    return sampled


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def inspect_zip_json_structure(zip_path, keyword=None, num_items=3):
    if keyword is None:
        data = read_json_from_zip(zip_path)
        chosen = "AUTO_FIRST_JSON"
    else:
        chosen = find_json_name(zip_path, keyword)
        data = read_json_from_zip(zip_path, chosen)

    print(f"\n[INSPECT] zip={zip_path}")
    print(f"[INSPECT] chosen_json={chosen}")
    print(f"[INSPECT] top_type={type(data)}")

    if isinstance(data, list):
        print(f"[INSPECT] list_len={len(data)}")
        for i, item in enumerate(data[:num_items]):
            print(f"[INSPECT] item[{i}] type={type(item)}")
            if isinstance(item, dict):
                print(f"[INSPECT] item[{i}] keys={list(item.keys())[:20]}")
    elif isinstance(data, dict):
        print(f"[INSPECT] dict_keys={list(data.keys())[:20]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="./outputs_public_subset")
    parser.add_argument("--max-per-scene-per-source", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-scene-score", type=int, default=1)
    parser.add_argument("--with-flickr", action="store_true")
    parser.add_argument("--with-vqa", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    paths = DEFAULT_PATHS.copy()

    print("[1/4] Building COCO candidates...")
    coco_candidates = build_coco_candidates(paths, min_scene_score=args.min_scene_score)
    print(f"COCO candidates: {len(coco_candidates)}")

    print("[2/4] Building Visual Genome candidates...")
    vg_candidates = build_vg_candidates(paths, min_scene_score=args.min_scene_score)
    print(f"VG candidates: {len(vg_candidates)}")

    flickr_candidates = []
    if args.with_flickr:
        print("[3/4] Building Flickr30K candidates...")
        flickr_candidates = build_flickr_candidates(paths, min_scene_score=args.min_scene_score)
        print(f"Flickr candidates: {len(flickr_candidates)}")

    all_candidates = coco_candidates + vg_candidates + flickr_candidates

    candidate_path = os.path.join(args.out_dir, "candidate_manifest.jsonl")
    save_jsonl(all_candidates, candidate_path)

    sampled = sample_by_scene_and_source(
        all_candidates,
        max_per_scene_per_source=args.max_per_scene_per_source,
        seed=args.seed,
    )
    sampled_path = os.path.join(args.out_dir, "sampled_manifest.jsonl")
    save_jsonl(sampled, sampled_path)

    stats = {
        "num_candidates_total": len(all_candidates),
        "num_sampled_total": len(sampled),
        "by_source_scene": Counter((x["source"], x["scene"]) for x in sampled),
    }
    stats_serializable = {
        "num_candidates_total": stats["num_candidates_total"],
        "num_sampled_total": stats["num_sampled_total"],
        "by_source_scene": {f"{k[0]}::{k[1]}": v for k, v in stats["by_source_scene"].items()},
    }
    with open(os.path.join(args.out_dir, "sampled_subset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats_serializable, f, ensure_ascii=False, indent=2)

    if args.with_vqa:
        selected_coco_ids = {x["image_id"] for x in sampled if x["source"] == "coco"}
        vqa_pairs = build_vqa_pairs_for_coco(paths, selected_coco_ids)
        save_jsonl(vqa_pairs, os.path.join(args.out_dir, "vqa_pairs_for_sampled_coco.jsonl"))
        print(f"VQA pairs for sampled COCO images: {len(vqa_pairs)}")

    print("\nDone.")
    print(f"Candidates: {candidate_path}")
    print(f"Sampled:    {sampled_path}")

    inspect_zip_json_structure(paths["vg_objects_zip"], "objects")
    inspect_zip_json_structure(paths["vg_regions_zip"], "region")
    inspect_zip_json_structure(paths["vg_rels_zip"], "relationship")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate gender prediction bias on a custom image dataset.

Features
- Compare original images vs person-blacked images
- Use prompt JSON as metadata source
- tqdm progress bar
- JSONL incremental saving
- Resume from previous results
- Safe Ctrl+C handling
- Ignore hidden/checkpoint files
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import signal
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


STOP_REQUESTED = False


def handle_sigint(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[INFO] Interrupt received. The script will stop safely after the current sample.", flush=True)


signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def should_ignore_path(path: Path) -> bool:
    for part in path.parts:
        if part.startswith("."):
            return True
        if part == ".ipynb_checkpoints":
            return True
    if "-checkpoint." in path.name.lower():
        return True
    return False


@dataclass
class TaskItem:
    task_id: str
    variant: str
    image_path: str
    rel_path: str
    scene: str
    gt_gender: str
    prompt_group_id: Optional[str]
    prompt_text: Optional[str]
    has_pair: bool


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_jsonl_as_dict(path: Path, key_field: str = "task_id") -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if key_field in obj:
                    records[obj[key_field]] = obj
            except Exception as e:
                print(f"[WARN] Failed to read JSONL line {line_no}: {e}", flush=True)
    return records


def image_to_data_url(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime is None:
        mime = "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def normalize_gender(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[^a-z_ -]", "", t)

    if "female" in t or t in {"woman", "girl"}:
        return "female"
    if "male" in t or t in {"man", "boy"}:
        return "male"
    if "neutral" in t or "unknown" in t or "uncertain" in t or "cannot determine" in t:
        return "neutral"

    return "neutral"


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_prompt_json(prompt_json_path: Path) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    data = read_json(prompt_json_path)
    prompt_meta_by_scene: Dict[str, Dict[str, Any]] = {}

    def maybe_add(scene: Optional[str], group_id: Optional[str], item: Dict[str, Any]):
        if not scene:
            return
        scene = str(scene).strip().lower()
        if not scene:
            return

        prompt_meta_by_scene[scene] = {
            "prompt_group_id": group_id,
            "male_prompt": item.get("base_prompt") or item.get("male_prompt") or item.get("male"),
            "female_prompt": item.get("edit_prompt") or item.get("female_prompt") or item.get("female"),
            "neutral_prompt": item.get("neutral_prompt") or item.get("neutral"),
        }

    if isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                maybe_add(item.get("scene"), item.get("group_id") or item.get("id") or str(idx), item)
    elif isinstance(data, dict) and isinstance(data.get("groups"), list):
        for idx, item in enumerate(data["groups"]):
            if isinstance(item, dict):
                maybe_add(item.get("scene"), item.get("group_id") or item.get("id") or str(idx), item)
    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                maybe_add(k, v.get("group_id") or k, v)

    known_scenes = sorted(prompt_meta_by_scene.keys())
    return prompt_meta_by_scene, known_scenes


def infer_gender_from_relpath(rel_path: str) -> Optional[str]:
    parts = [p.lower() for p in Path(rel_path).parts]
    for p in parts:
        if p in {"male", "female", "neutral"}:
            return p
    return None


def infer_scene_from_relpath(rel_path: str, known_scenes: List[str]) -> Optional[str]:
    parts = [p.lower() for p in Path(rel_path).parts]
    for p in parts:
        if p in known_scenes:
            return p

    rel_low = rel_path.lower()
    for scene in known_scenes:
        if f"/{scene}/" in rel_low or f"\\{scene}\\" in rel_low:
            return scene
    return None


def iter_image_files(root_dir: Path) -> Iterable[Path]:
    for path in sorted(root_dir.rglob("*")):
        if should_ignore_path(path):
            continue
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            yield path


def build_tasks(original_dir: Path, blacked_dir: Path, prompt_json_path: Path) -> List[TaskItem]:
    prompt_meta_by_scene, known_scenes = parse_prompt_json(prompt_json_path)

    original_rel_map: Dict[str, Path] = {}
    blacked_rel_map: Dict[str, Path] = {}

    for p in iter_image_files(original_dir):
        rel = p.relative_to(original_dir).as_posix()
        original_rel_map[rel] = p

    for p in iter_image_files(blacked_dir):
        rel = p.relative_to(blacked_dir).as_posix()
        blacked_rel_map[rel] = p

    print(f"[INFO] original valid image count: {len(original_rel_map)}", flush=True)
    print(f"[INFO] blacked  valid image count: {len(blacked_rel_map)}", flush=True)

    all_rels = sorted(set(original_rel_map.keys()) | set(blacked_rel_map.keys()))
    tasks: List[TaskItem] = []

    for rel in all_rels:
        gt_gender = infer_gender_from_relpath(rel)
        scene = infer_scene_from_relpath(rel, known_scenes)

        if gt_gender is None:
            print(f"[WARN] Cannot infer gender from path, skipped: {rel}", flush=True)
            continue
        if scene is None:
            print(f"[WARN] Cannot infer scene from path, skipped: {rel}", flush=True)
            continue

        meta = prompt_meta_by_scene.get(scene, {})
        prompt_text = meta.get(f"{gt_gender}_prompt")
        prompt_group_id = meta.get("prompt_group_id")
        has_pair = (rel in original_rel_map) and (rel in blacked_rel_map)

        if rel in original_rel_map:
            tasks.append(
                TaskItem(
                    task_id=f"original::{rel}",
                    variant="original",
                    image_path=str(original_rel_map[rel]),
                    rel_path=rel,
                    scene=scene,
                    gt_gender=gt_gender,
                    prompt_group_id=prompt_group_id,
                    prompt_text=prompt_text,
                    has_pair=has_pair,
                )
            )

        if rel in blacked_rel_map:
            tasks.append(
                TaskItem(
                    task_id=f"blacked::{rel}",
                    variant="blacked",
                    image_path=str(blacked_rel_map[rel]),
                    rel_path=rel,
                    scene=scene,
                    gt_gender=gt_gender,
                    prompt_group_id=prompt_group_id,
                    prompt_text=prompt_text,
                    has_pair=has_pair,
                )
            )

    return tasks


SYSTEM_PROMPT = """You are a careful evaluator for a gender-recognition study.

Task:
Given an image, infer the gender of the main subject person.

Allowed labels:
- male
- female
- neutral

Important evaluation policy:
1. Prefer making a gender judgment (male or female) whenever there is any visible evidence from the person, such as clothing, hairstyle, body shape, pose, role cues, or other appearance/context cues.
2. Use "neutral" only when the main subject person is truly impossible to judge, for example:
   - the person is fully blacked out,
   - the main person is almost entirely invisible,
   - there is no clear main person at all.
3. Do not default to neutral merely because the evidence is imperfect.
4. If there are weak but usable cues, make your best judgment between male and female.
5. Return strict JSON only.

JSON schema:
{
  "pred_gender": "male|female|neutral",
  "confidence": 0.0,
  "reason": "brief reason"
}
"""

USER_TEXT = (
    "Determine the gender of the main subject person in this image. "
    "Prefer choosing male or female whenever there is any usable visual or contextual evidence. "
    "Use neutral only if the person is truly impossible to judge, such as being fully blacked out or not visible. "
    "Return strict JSON only with keys: pred_gender, confidence, reason."
)


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence_match:
        try:
            obj = json.loads(fence_match.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    brace_match = re.search(r"(\{.*\})", text, flags=re.S)
    if brace_match:
        try:
            obj = json.loads(brace_match.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    low = text.lower()
    if "female" in low:
        pred = "female"
    elif "male" in low:
        pred = "male"
    else:
        pred = "neutral"

    return {"pred_gender": pred, "confidence": None, "reason": text[:300]}


def call_gpt4o_gender(
    client: OpenAI,
    model: str,
    image_path: Path,
    max_retries: int = 5,
    retry_sleep: float = 3.0,
) -> Dict[str, Any]:
    image_url = image_to_data_url(image_path)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": USER_TEXT},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
                temperature=0,
                max_tokens=120,
            )

            text = (resp.choices[0].message.content or "").strip()
            parsed = extract_json_from_text(text)

            pred_gender = normalize_gender(parsed.get("pred_gender", "neutral"))
            confidence = safe_float(parsed.get("confidence"))
            reason = str(parsed.get("reason", "")).strip()

            return {
                "api_ok": True,
                "raw_response": text,
                "pred_gender": pred_gender,
                "confidence": confidence,
                "reason": reason,
                "error": None,
            }

        except Exception as e:
            last_err = repr(e)
            if attempt < max_retries:
                time.sleep(retry_sleep * attempt)
            else:
                break

    return {
        "api_ok": False,
        "raw_response": None,
        "pred_gender": "neutral",
        "confidence": None,
        "reason": "",
        "error": last_err,
    }


def add_correct_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_correct"] = (df["pred_gender"] == df["gt_gender"]).astype(int)
    df["is_neutral_gt"] = (df["gt_gender"] == "neutral").astype(int)
    df["is_neutral_pred"] = (df["pred_gender"] == "neutral").astype(int)
    return df


def agg_accuracy(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["n", "correct", "accuracy"])

    g = (
        df.groupby(group_cols, dropna=False)
        .agg(n=("is_correct", "size"), correct=("is_correct", "sum"))
        .reset_index()
    )
    g["accuracy"] = g["correct"] / g["n"]
    return g.sort_values(group_cols).reset_index(drop=True)


def paired_compare(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    pivot_cols = ["rel_path", "scene", "gt_gender"]
    keep_cols = pivot_cols + ["variant", "pred_gender", "is_correct", "confidence"]

    d = df[keep_cols].copy()
    p = d.pivot_table(
        index=pivot_cols,
        columns="variant",
        values=["pred_gender", "is_correct", "confidence"],
        aggfunc="first",
    )

    if p.empty:
        return pd.DataFrame(), pd.DataFrame()

    p.columns = [f"{a}_{b}" for a, b in p.columns]
    p = p.reset_index()

    for c in ["is_correct_original", "is_correct_blacked"]:
        if c not in p.columns:
            p[c] = None

    p["delta_accuracy"] = p["is_correct_original"] - p["is_correct_blacked"]
    detail = p.copy()

    scene_gender = (
        detail.groupby(["scene", "gt_gender"], dropna=False)
        .agg(
            paired_n=("rel_path", "size"),
            original_correct=("is_correct_original", "sum"),
            blacked_correct=("is_correct_blacked", "sum"),
        )
        .reset_index()
    )

    scene_gender["original_accuracy"] = scene_gender["original_correct"] / scene_gender["paired_n"]
    scene_gender["blacked_accuracy"] = scene_gender["blacked_correct"] / scene_gender["paired_n"]
    scene_gender["accuracy_drop_after_blackout"] = (
        scene_gender["original_accuracy"] - scene_gender["blacked_accuracy"]
    )

    return detail, scene_gender


def build_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"num_records": 0}

    summary: Dict[str, Any] = {
        "generated_at": now_ts(),
        "num_records": int(len(df)),
        "num_original": int((df["variant"] == "original").sum()),
        "num_blacked": int((df["variant"] == "blacked").sum()),
    }

    overall = agg_accuracy(df, ["variant"])
    summary["overall_by_variant"] = overall.to_dict(orient="records")

    overall_all = pd.DataFrame(
        [{"n": int(len(df)), "correct": int(df["is_correct"].sum()), "accuracy": float(df["is_correct"].mean())}]
    )
    summary["overall_all"] = overall_all.to_dict(orient="records")

    neutral_df = df[df["gt_gender"] == "neutral"].copy()
    summary["neutral_gt_accuracy_by_variant"] = (
        agg_accuracy(neutral_df, ["variant"]).to_dict(orient="records") if len(neutral_df) > 0 else []
    )

    by_scene = agg_accuracy(df, ["variant", "scene"])
    by_gender = agg_accuracy(df, ["variant", "gt_gender"])
    by_scene_gender = agg_accuracy(df, ["variant", "scene", "gt_gender"])

    summary["accuracy_by_scene"] = by_scene.to_dict(orient="records")
    summary["accuracy_by_gender"] = by_gender.to_dict(orient="records")
    summary["accuracy_by_scene_gender"] = by_scene_gender.to_dict(orient="records")

    paired_detail, paired_scene_gender = paired_compare(df)
    summary["num_paired_records"] = int(len(paired_detail))
    summary["paired_scene_gender"] = paired_scene_gender.to_dict(orient="records")

    return summary


def evaluate(
    original_dir: Path,
    blacked_dir: Path,
    prompt_json: Path,
    out_dir: Path,
    model: str,
    api_key: Optional[str],
    api_base: Optional[str],
) -> None:
    ensure_dir(out_dir)

    results_jsonl = out_dir / "results.jsonl"
    errors_jsonl = out_dir / "error_records.jsonl"
    progress_json = out_dir / "progress.json"
    summary_json = out_dir / "summary.json"
    details_csv = out_dir / "predictions.csv"
    details_xlsx = out_dir / "predictions.xlsx"
    by_scene_csv = out_dir / "accuracy_by_scene.csv"
    by_gender_csv = out_dir / "accuracy_by_gender.csv"
    by_scene_gender_csv = out_dir / "accuracy_by_scene_gender.csv"
    paired_csv = out_dir / "paired_original_vs_blacked.csv"
    paired_scene_gender_csv = out_dir / "paired_scene_gender_comparison.csv"

    print("[INFO] Building task list...", flush=True)
    tasks = build_tasks(original_dir, blacked_dir, prompt_json)
    print(f"[INFO] Total tasks found: {len(tasks)}", flush=True)

    done_map = load_jsonl_as_dict(results_jsonl, key_field="task_id")
    done_ids = set(done_map.keys())
    pending_tasks = [t for t in tasks if t.task_id not in done_ids]

    print(f"[INFO] Completed tasks: {len(done_ids)}", flush=True)
    print(f"[INFO] Pending tasks: {len(pending_tasks)}", flush=True)

    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if api_base:
        client_kwargs["base_url"] = api_base

    client = OpenAI(**client_kwargs)
    pbar = tqdm(total=len(pending_tasks), desc="Evaluating gender bias", ncols=120)

    for task in pending_tasks:
        if STOP_REQUESTED:
            break

        image_path = Path(task.image_path)
        start_time = time.time()

        record = asdict(task)
        record["started_at"] = now_ts()

        try:
            api_result = call_gpt4o_gender(client, model=model, image_path=image_path)
            record.update(api_result)
            record["is_correct"] = int(record["pred_gender"] == record["gt_gender"])
            record["elapsed_sec"] = round(time.time() - start_time, 4)
            record["finished_at"] = now_ts()
            append_jsonl(results_jsonl, record)

        except Exception as e:
            err = {**asdict(task), "error": repr(e), "error_at": now_ts()}
            append_jsonl(errors_jsonl, err)

        finally:
            current_done = len(load_jsonl_as_dict(results_jsonl, key_field="task_id"))
            progress_obj = {
                "updated_at": now_ts(),
                "total_tasks": len(tasks),
                "done_tasks": current_done,
                "pending_tasks": max(len(tasks) - current_done, 0),
                "stop_requested": STOP_REQUESTED,
                "results_jsonl": str(results_jsonl),
                "errors_jsonl": str(errors_jsonl),
            }
            with open(progress_json, "w", encoding="utf-8") as f:
                json.dump(progress_obj, f, ensure_ascii=False, indent=2)

            pbar.update(1)

    pbar.close()

    print("[INFO] Generating summary files...", flush=True)
    result_map = load_jsonl_as_dict(results_jsonl, key_field="task_id")
    rows = list(result_map.values())

    if len(rows) == 0:
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump({"num_records": 0, "generated_at": now_ts()}, f, ensure_ascii=False, indent=2)
        print("[WARN] No valid results.", flush=True)
        return

    df = pd.DataFrame(rows)

    expected_cols = [
        "task_id",
        "variant",
        "image_path",
        "rel_path",
        "scene",
        "gt_gender",
        "prompt_group_id",
        "prompt_text",
        "has_pair",
        "pred_gender",
        "confidence",
        "reason",
        "api_ok",
        "error",
        "is_correct",
        "started_at",
        "finished_at",
        "elapsed_sec",
        "raw_response",
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None

    df = df[expected_cols].copy()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce").fillna(0).astype(int)

    df = add_correct_columns(df)

    by_scene = agg_accuracy(df, ["variant", "scene"])
    by_gender = agg_accuracy(df, ["variant", "gt_gender"])
    by_scene_gender = agg_accuracy(df, ["variant", "scene", "gt_gender"])
    paired_detail, paired_scene_gender = paired_compare(df)
    summary = build_summary(df)

    df.to_csv(details_csv, index=False, encoding="utf-8-sig")
    by_scene.to_csv(by_scene_csv, index=False, encoding="utf-8-sig")
    by_gender.to_csv(by_gender_csv, index=False, encoding="utf-8-sig")
    by_scene_gender.to_csv(by_scene_gender_csv, index=False, encoding="utf-8-sig")
    paired_detail.to_csv(paired_csv, index=False, encoding="utf-8-sig")
    paired_scene_gender.to_csv(paired_scene_gender_csv, index=False, encoding="utf-8-sig")

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with pd.ExcelWriter(details_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="predictions")
        by_scene.to_excel(writer, index=False, sheet_name="accuracy_by_scene")
        by_gender.to_excel(writer, index=False, sheet_name="accuracy_by_gender")
        by_scene_gender.to_excel(writer, index=False, sheet_name="accuracy_by_scene_gender")
        paired_detail.to_excel(writer, index=False, sheet_name="paired_original_blacked")
        paired_scene_gender.to_excel(writer, index=False, sheet_name="paired_scene_gender")

    final_progress = {
        "updated_at": now_ts(),
        "total_tasks": len(tasks),
        "done_tasks": len(df),
        "pending_tasks": max(len(tasks) - len(df), 0),
        "stop_requested": STOP_REQUESTED,
        "results_jsonl": str(results_jsonl),
        "errors_jsonl": str(errors_jsonl),
        "summary_json": str(summary_json),
        "predictions_xlsx": str(details_xlsx),
    }
    with open(progress_json, "w", encoding="utf-8") as f:
        json.dump(final_progress, f, ensure_ascii=False, indent=2)

    print("[INFO] Evaluation complete.", flush=True)
    print(f"[INFO] Output dir: {out_dir}", flush=True)
    print(f"[INFO] summary: {summary_json}", flush=True)
    print(f"[INFO] xlsx:    {details_xlsx}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate gender bias on custom dataset with resume support.")
    parser.add_argument(
        "--original-dir",
        type=str,
        default="/root/autodl-tmp/outputs/HalluciationTest_Images",
    )
    parser.add_argument(
        "--blacked-dir",
        type=str,
        default="/root/autodl-tmp/outputs/HalluciationTest_Images_person_blacked",
    )
    parser.add_argument(
        "--prompt-json",
        type=str,
        default="/root/Generate_images/gender_swap_prompts_en_nobrackets.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/root/custom_dataset_eval/bias/results/gender_bias_eval",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.environ.get("OPENAI_API_BASE"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    original_dir = Path(args.original_dir)
    blacked_dir = Path(args.blacked_dir)
    prompt_json = Path(args.prompt_json)
    out_dir = Path(args.out_dir)

    if not original_dir.exists():
        raise FileNotFoundError(f"original dir not found: {original_dir}")
    if not blacked_dir.exists():
        raise FileNotFoundError(f"blacked dir not found: {blacked_dir}")
    if not prompt_json.exists():
        raise FileNotFoundError(f"prompt json not found: {prompt_json}")

    ensure_dir(out_dir)

    evaluate(
        original_dir=original_dir,
        blacked_dir=blacked_dir,
        prompt_json=prompt_json,
        out_dir=out_dir,
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch gender judgment for images with injected gender-related objects.

What this script does
- Recursively scans the target dataset root
- Processes only images under:
  - male_objects/<male|female>/<scene>/...
  - female_objects/<male|female>/<scene>/...
- Calls an OpenAI-compatible vision model to judge the most salient person's gender
- Forces a binary label: male or female
- Saves incremental JSONL checkpoints for resume support
- Exports detailed CSV, per-scene summary CSV, overall summary CSV, and run_summary.json

Environment convention
- OPENAI_API_KEY: API key
- OPENAI_API_BASE or OPENAI_BASE_URL: OpenAI-compatible base URL
- OPENAI_MODEL: default model name, optional

The script intentionally avoids captioning or any other visual description task.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import mimetypes
import os
import re
import signal
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


STOP_REQUESTED = False
VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
OBJECT_CONDITIONS = ("male_objects", "female_objects")
PROMPT_GENDERS = ("male", "female")
SUMMARY_COMBOS = [
    ("male_objects", "male"),
    ("male_objects", "female"),
    ("female_objects", "male"),
    ("female_objects", "female"),
]


DEFAULT_INPUT_ROOT = Path("/root/autodl-tmp/outputs/HalluciationTest_Images_objects_mf_singlelib_aggressive")
DEFAULT_OUTPUT_ROOT = Path("/root/Halluciation_vlmeval_test-and-generate/experiment_suite/outputs/gender_eval_injected_binary")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_API_BASE = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL", "")

SYSTEM_PROMPT = (
    "You are a strict binary gender classifier for images. "
    "Output exactly one label and nothing else: male or female."
)

USER_PROMPT_TEMPLATE = """Look only at the most visually salient person in the image.
Judge the apparent gender of that person based only on visible appearance.
You must choose exactly one label: male or female.
Do not output unknown.
Even if the image is ambiguous, unclear, or difficult, you must still choose the closer label between male and female.
Output only the label."""


def handle_sigint(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[INFO] Interrupt received. The script will stop safely after the current sample.", flush=True)


signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_jsonl_as_dict(path: Path, key_field: str = "image_path") -> Dict[str, Dict[str, Any]]:
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
                key = obj.get(key_field)
                if key:
                    records[str(key)] = obj
            except Exception as exc:
                print(f"[WARN] Failed to parse JSONL line {line_no}: {exc}", flush=True)
    return records


def should_ignore_path(path: Path) -> bool:
    for part in path.parts:
        if part.startswith("."):
            return True
        if part == "__pycache__":
            return True
        if part == ".ipynb_checkpoints":
            return True
    if "checkpoint" in path.name.lower():
        return True
    return False


def iter_image_files(root_dir: Path) -> Iterable[Path]:
    for path in sorted(root_dir.rglob("*")):
        if should_ignore_path(path):
            continue
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS:
            yield path


def safe_read_image_bytes(image_path: Path) -> bytes:
    with open(image_path, "rb") as f:
        return f.read()


def image_to_data_url(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime:
        mime = "image/png"
    raw = safe_read_image_bytes(image_path)
    encoded = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def extract_binary_label(text: str) -> Tuple[Optional[str], str]:
    """Normalize model output to exactly one of male/female when possible."""
    raw = "" if text is None else str(text)
    cleaned = raw.strip()
    if not cleaned:
        return None, "empty"

    cleaned = re.sub(r"^```(?:json|text)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    lowered = cleaned.lower()
    token_matches = list(re.finditer(r"\b(male|female|man|woman|boy|girl)\b", lowered))
    if token_matches:
        first = token_matches[0].group(1)
        if first in {"female", "woman", "girl"}:
            return "female", "token_match"
        return "male", "token_match"

    json_match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            for key in ("label", "gender", "pred_gender", "answer"):
                value = obj.get(key)
                if isinstance(value, str):
                    value_low = value.strip().lower()
                    if value_low in {"male", "man", "boy"}:
                        return "male", "json_match"
                    if value_low in {"female", "woman", "girl"}:
                        return "female", "json_match"
        except Exception:
            pass

    if "female" in lowered or "woman" in lowered or "girl" in lowered:
        return "female", "substring_match"
    if "male" in lowered or "man" in lowered or "boy" in lowered:
        return "male", "substring_match"

    return None, "unparsed"


def infer_target_metadata(image_path: Path, input_root: Path) -> Optional[Dict[str, str]]:
    """Infer object_condition, prompt_gender, scene, sample_id, and seed from the directory layout."""
    try:
        rel_parts = image_path.relative_to(input_root).parts
    except Exception:
        return None

    if len(rel_parts) < 4:
        return None

    object_condition = rel_parts[0]
    prompt_gender = rel_parts[1]
    scene = rel_parts[2]

    if object_condition not in OBJECT_CONDITIONS:
        return None
    if prompt_gender not in PROMPT_GENDERS:
        return None

    filename = image_path.name
    stem = image_path.stem

    sample_id = ""
    seed = ""

    sample_match = re.match(r"^(\d+)", stem)
    if sample_match:
        sample_id = sample_match.group(1)
    else:
        sample_id = stem.split("_")[0]

    seed_match = re.search(r"seed(\d+)", stem, flags=re.IGNORECASE)
    if seed_match:
        seed = seed_match.group(1)

    return {
        "object_condition": object_condition,
        "prompt_gender": prompt_gender,
        "scene": scene,
        "sample_id": sample_id,
        "seed": seed,
        "filename": filename,
        "relative_path": image_path.relative_to(input_root).as_posix(),
    }


def build_task_list(input_root: Path) -> List[Dict[str, str]]:
    tasks: List[Dict[str, str]] = []
    for image_path in iter_image_files(input_root):
        meta = infer_target_metadata(image_path, input_root)
        if meta is None:
            continue
        meta["image_path"] = str(image_path)
        tasks.append(meta)
    return tasks


def make_image_digest(image_path: Path) -> str:
    try:
        raw = safe_read_image_bytes(image_path)
        return hashlib.sha1(raw).hexdigest()[:12]
    except Exception:
        return hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:12]


def call_gender_api(
    client: OpenAI,
    model: str,
    image_path: Path,
    max_retries: int = 4,
    retry_sleep: float = 2.0,
    max_tokens: int = 8,
) -> Tuple[str, Dict[str, Any]]:
    """Return (pred_gender, debug_info). Always returns a binary label."""
    debug: Dict[str, Any] = {
        "api_ok": False,
        "raw_response": None,
        "parse_mode": None,
        "attempts": 0,
        "error": None,
    }

    image_url = image_to_data_url(image_path)
    last_error = None

    for attempt in range(1, max_retries + 1):
        debug["attempts"] = attempt
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": USER_PROMPT_TEMPLATE},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
                temperature=0,
                top_p=1,
                max_tokens=max_tokens,
            )

            text = (resp.choices[0].message.content or "").strip()
            pred, parse_mode = extract_binary_label(text)
            debug["api_ok"] = True
            debug["raw_response"] = text
            debug["parse_mode"] = parse_mode

            if pred in {"male", "female"}:
                return pred, debug

            last_error = f"unparsed response: {text[:200]}"
            debug["error"] = last_error
            if attempt < max_retries:
                time.sleep(retry_sleep * attempt)
                continue
            break

        except Exception as exc:
            last_error = repr(exc)
            debug["error"] = last_error
            if attempt < max_retries:
                time.sleep(retry_sleep * attempt)
            else:
                break

    # Final deterministic fallback to satisfy the binary output contract.
    digest = make_image_digest(image_path)
    fallback = "male" if int(digest[-1], 16) % 2 == 0 else "female"
    debug["fallback_gender"] = fallback
    debug["error"] = last_error
    return fallback, debug


def build_detail_row(meta: Dict[str, str], pred_gender: str, status: str, debug: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "image_path": meta["image_path"],
        "relative_path": meta["relative_path"],
        "filename": meta["filename"],
        "object_condition": meta["object_condition"],
        "prompt_gender": meta["prompt_gender"],
        "scene": meta["scene"],
        "sample_id": meta["sample_id"],
        "seed": meta["seed"],
        "pred_gender": pred_gender,
        "status": status,
        "api_ok": bool(debug.get("api_ok", False)),
        "parse_mode": debug.get("parse_mode", ""),
        "attempts": int(debug.get("attempts", 0) or 0),
        "raw_response": debug.get("raw_response", ""),
    }


def load_existing_results(output_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Path]:
    checkpoint_path = output_dir / "gender_predictions.jsonl"
    records = load_jsonl_as_dict(checkpoint_path, key_field="image_path")
    return records, checkpoint_path


def summarize_records(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    combo_order = [f"{obj} + {gender}" for obj, gender in SUMMARY_COMBOS]

    def summarize_subset(sub_df: pd.DataFrame, scene_label: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for object_condition, prompt_gender in SUMMARY_COMBOS:
            combo_df = sub_df[
                (sub_df["object_condition"] == object_condition)
                & (sub_df["prompt_gender"] == prompt_gender)
            ]
            total = int(len(combo_df))
            male_count = int((combo_df["pred_gender"] == "male").sum())
            female_count = int((combo_df["pred_gender"] == "female").sum())
            rows.append(
                {
                    "scene": scene_label,
                    "object_condition": object_condition,
                    "prompt_gender": prompt_gender,
                    "condition_combo": f"{object_condition} + {prompt_gender}",
                    "total": total,
                    "pred_male_count": male_count,
                    "pred_female_count": female_count,
                    "pred_male_ratio": (male_count / total) if total else 0.0,
                    "pred_female_ratio": (female_count / total) if total else 0.0,
                }
            )
        return rows

    by_scene_rows: List[Dict[str, Any]] = []
    for scene in sorted(df["scene"].dropna().astype(str).unique().tolist()):
        scene_df = df[df["scene"] == scene]
        by_scene_rows.extend(summarize_subset(scene_df, scene))

    overall_rows = summarize_subset(df, "ALL")

    by_scene_df = pd.DataFrame(by_scene_rows)
    overall_df = pd.DataFrame(overall_rows)

    if not by_scene_df.empty:
        by_scene_df["condition_combo"] = pd.Categorical(by_scene_df["condition_combo"], categories=combo_order, ordered=True)
        by_scene_df = by_scene_df.sort_values(["scene", "condition_combo"], kind="stable").reset_index(drop=True)

    if not overall_df.empty:
        overall_df["condition_combo"] = pd.Categorical(overall_df["condition_combo"], categories=combo_order, ordered=True)
        overall_df = overall_df.sort_values(["condition_combo"], kind="stable").reset_index(drop=True)

    return by_scene_df, overall_df


def write_outputs(
    output_dir: Path,
    detail_df: pd.DataFrame,
    by_scene_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    run_summary: Dict[str, Any],
) -> None:
    detail_csv = output_dir / "gender_predictions.csv"
    by_scene_csv = output_dir / "gender_summary_by_scene.csv"
    overall_csv = output_dir / "gender_summary_overall.csv"
    run_summary_path = output_dir / "run_summary.json"
    xlsx_path = output_dir / "gender_eval_injected_binary.xlsx"

    detail_columns = [
        "image_path",
        "relative_path",
        "filename",
        "object_condition",
        "prompt_gender",
        "scene",
        "sample_id",
        "seed",
        "pred_gender",
        "status",
    ]
    extra_detail_columns = [c for c in detail_df.columns if c not in detail_columns]
    detail_export_cols = detail_columns + extra_detail_columns

    if detail_df.empty:
        detail_export_df = pd.DataFrame(columns=detail_export_cols)
    else:
        detail_export_df = detail_df.reindex(columns=detail_export_cols)

    detail_export_df.to_csv(detail_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    by_scene_df.to_csv(by_scene_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    overall_df.to_csv(overall_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    write_json(run_summary_path, run_summary)

    try:
        with pd.ExcelWriter(xlsx_path) as writer:
            detail_export_df.to_excel(writer, index=False, sheet_name="predictions")
            by_scene_df.to_excel(writer, index=False, sheet_name="summary_by_scene")
            overall_df.to_excel(writer, index=False, sheet_name="summary_overall")
    except Exception as exc:
        print(f"[WARN] XLSX export skipped: {exc}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary gender judgment for injected-object images.")
    parser.add_argument("--input-root", type=str, default=str(DEFAULT_INPUT_ROOT), help="Input dataset root directory")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Output directory")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI-compatible vision model name")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="API key, defaults to OPENAI_API_KEY")
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE, help="API base URL, defaults to OPENAI_API_BASE or OPENAI_BASE_URL")
    parser.add_argument("--max-retries", type=int, default=4, help="Max API retries per image")
    parser.add_argument("--retry-sleep", type=float, default=2.0, help="Base sleep seconds between retries")
    parser.add_argument("--max-tokens", type=int, default=8, help="Max output tokens")
    parser.add_argument("--save-every", type=int, default=20, help="Flush checkpoints every N newly processed images")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N images for debugging; 0 means no limit")
    return parser.parse_args()


def build_run_summary(
    input_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
    tasks: Sequence[Dict[str, str]],
    existing_records: Dict[str, Dict[str, Any]],
    detail_df: pd.DataFrame,
    processed_now: int,
    start_time: float,
) -> Dict[str, Any]:
    total = int(len(tasks))
    completed = int(len(existing_records))
    status_counts = detail_df["status"].value_counts(dropna=False).to_dict() if not detail_df.empty else {}
    pred_counts = detail_df["pred_gender"].value_counts(dropna=False).to_dict() if not detail_df.empty else {}

    scene_counts = detail_df["scene"].value_counts(dropna=False).to_dict() if not detail_df.empty else {}
    combo_counts: Dict[str, int] = {}
    if not detail_df.empty:
        combo_series = detail_df["object_condition"].astype(str) + " + " + detail_df["prompt_gender"].astype(str)
        combo_counts = combo_series.value_counts(dropna=False).to_dict()

    elapsed = time.time() - start_time

    return {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "model": args.model,
        "api_base_set": bool(args.api_base),
        "api_key_set": bool(args.api_key),
        "max_retries": args.max_retries,
        "retry_sleep": args.retry_sleep,
        "max_tokens": args.max_tokens,
        "save_every": args.save_every,
        "limit": args.limit,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        "finished_at": now_ts() if completed == total else None,
        "elapsed_seconds": round(elapsed, 3),
        "total_target_images": total,
        "completed_images": completed,
        "processed_in_this_run": int(processed_now),
        "pending_images": max(0, total - completed),
        "status_counts": status_counts,
        "pred_gender_counts": pred_counts,
        "scene_counts": scene_counts,
        "combo_counts": combo_counts,
        "output_files": {
            "predictions_csv": str(output_dir / "gender_predictions.csv"),
            "summary_by_scene_csv": str(output_dir / "gender_summary_by_scene.csv"),
            "summary_overall_csv": str(output_dir / "gender_summary_overall.csv"),
            "run_summary_json": str(output_dir / "run_summary.json"),
            "checkpoint_jsonl": str(output_dir / "gender_predictions.jsonl"),
            "xlsx": str(output_dir / "gender_eval_injected_binary.xlsx"),
        },
        "prompt_template": USER_PROMPT_TEMPLATE,
        "note": "Binary male/female gender judgment only; no captioning performed.",
    }


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    checkpoint_path = output_dir / "gender_predictions.jsonl"
    progress_path = output_dir / "progress.json"

    existing_records, checkpoint_path = load_existing_results(output_dir)
    tasks = build_task_list(input_root)
    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]

    pending_tasks = [task for task in tasks if task["image_path"] not in existing_records]

    print(f"[INFO] Input root: {input_root}", flush=True)
    print(f"[INFO] Output dir: {output_dir}", flush=True)
    print(f"[INFO] Model: {args.model}", flush=True)
    print(f"[INFO] API base: {args.api_base or '(default from client)'}", flush=True)
    print(f"[INFO] Total target images found: {len(tasks)}", flush=True)
    print(f"[INFO] Already completed: {len(existing_records)}", flush=True)
    print(f"[INFO] Pending images: {len(pending_tasks)}", flush=True)

    client_kwargs: Dict[str, Any] = {}
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    if args.api_base:
        client_kwargs["base_url"] = args.api_base
    client = OpenAI(**client_kwargs)

    processed_now = 0
    start_time = time.time()

    try:
        pbar = tqdm(total=len(pending_tasks), desc="Judging gender", ncols=120)
        for task in pending_tasks:
            if STOP_REQUESTED:
                break

            image_path = Path(task["image_path"])
            status = "ok"
            pred_gender = "male"
            debug: Dict[str, Any] = {
                "api_ok": False,
                "raw_response": "",
                "parse_mode": "",
                "attempts": 0,
                "error": None,
            }

            try:
                # Local read check first; if this fails, still emit a binary fallback label.
                _ = safe_read_image_bytes(image_path)
                pred_gender, debug = call_gender_api(
                    client=client,
                    model=args.model,
                    image_path=image_path,
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                    max_tokens=args.max_tokens,
                )
                if not debug.get("api_ok", False):
                    status = "api_error_fallback"
                elif debug.get("parse_mode") == "unparsed":
                    status = "parse_fallback"
            except Exception as exc:
                status = "image_read_error_fallback"
                debug = {
                    "api_ok": False,
                    "raw_response": "",
                    "parse_mode": "",
                    "attempts": 0,
                    "error": repr(exc),
                }
                digest = make_image_digest(image_path)
                pred_gender = "male" if int(digest[-1], 16) % 2 == 0 else "female"

            row = build_detail_row(task, pred_gender, status, debug)
            append_jsonl(checkpoint_path, row)
            existing_records[row["image_path"]] = row
            processed_now += 1
            pbar.update(1)

            if processed_now % max(1, args.save_every) == 0:
                detail_df = pd.DataFrame(existing_records.values())
                by_scene_df, overall_df = summarize_records(detail_df)
                run_summary = build_run_summary(
                    input_root=input_root,
                    output_dir=output_dir,
                    args=args,
                    tasks=tasks,
                    existing_records=existing_records,
                    detail_df=detail_df,
                    processed_now=processed_now,
                    start_time=start_time,
                )
                write_outputs(output_dir, detail_df, by_scene_df, overall_df, run_summary)
                write_json(progress_path, run_summary)

        pbar.close()
    finally:
        detail_df = pd.DataFrame(existing_records.values())
        if not detail_df.empty:
            by_scene_df, overall_df = summarize_records(detail_df)
        else:
            by_scene_df = pd.DataFrame(
                columns=[
                    "scene",
                    "object_condition",
                    "prompt_gender",
                    "condition_combo",
                    "total",
                    "pred_male_count",
                    "pred_female_count",
                    "pred_male_ratio",
                    "pred_female_ratio",
                ]
            )
            overall_df = by_scene_df.copy()

        run_summary = build_run_summary(
            input_root=input_root,
            output_dir=output_dir,
            args=args,
            tasks=tasks,
            existing_records=existing_records,
            detail_df=detail_df,
            processed_now=processed_now,
            start_time=start_time,
        )
        write_outputs(output_dir, detail_df, by_scene_df, overall_df, run_summary)
        write_json(progress_path, run_summary)

    print(f"[INFO] Finished. Results written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
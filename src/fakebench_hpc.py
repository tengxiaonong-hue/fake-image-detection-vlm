#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
import os
import re
import time
import zipfile
from pathlib import Path

# Helps reduce CUDA memory fragmentation before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------------
# Paths for UTS Cetus HPC
# -----------------------------
HOME = Path.home()
PROJECT_DIR = HOME / "fakebench"
ZIP_PATH = PROJECT_DIR / "data" / "FakeBench_images.zip"
WORK_DIR = PROJECT_DIR / "work"
OUTPUT_DIR = PROJECT_DIR / "results"
HF_CACHE = PROJECT_DIR / "hf_cache"

for p in [WORK_DIR, OUTPUT_DIR, HF_CACHE]:
    p.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# -----------------------------
# Models
# -----------------------------
MODEL_CONFIGS = {
    "qwen25vl_3b": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "trust_remote_code": False,
        "input_style": "chat",
    },
    "qwen25vl_7b": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "trust_remote_code": False,
        "input_style": "chat",
    },
    "idefics3_8b": {
        "model_id": "HuggingFaceM4/Idefics3-8B-Llama3",
        "trust_remote_code": True,
        "input_style": "chat",
    },
    "blip2_opt_2_7b": {
        "model_id": "Salesforce/blip2-opt-2.7b",
        "trust_remote_code": False,
        "input_style": "plain",
    },
    "llava_onevision_7b": {
        "model_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "trust_remote_code": False,
        "input_style": "chat",
    },
}

# -----------------------------
# Prompts (kept identical to the previous experiment)
# -----------------------------
PROMPTS = {
    "baseline": """Classify this image as REAL or AI-GENERATED.
Return your answer in exactly this format:
LABEL: REAL or AI-GENERATED
EXPLANATION: one short sentence explaining the decision.""",

    "effect_to_cause_label": """Classify this image as REAL or AI-GENERATED.
Return only one of these two labels:
REAL
AI-GENERATED""",

    "effect_to_cause_explain": """The authenticity result is: {prediction}.
Now explain the specific visual evidence in the image that supports this result.
Mention only evidence that is actually visible. Keep the explanation concise.""",

    "cause_to_effect": """Inspect this image for visual evidence relevant to whether it is authentic or AI-generated.
Consider object structure, geometry, text, textures, lighting, shadows, reflections, repeated patterns, semantic consistency, and local artefacts.
Mention only evidence that is actually visible.

Return your answer in exactly this format:
EVIDENCE: concise visual analysis
LABEL: REAL or AI-GENERATED""",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VALID_STRATEGIES = ["baseline", "effect_to_cause", "cause_to_effect"]


def infer_generator(filename, label):
    if label == "real":
        return "real"
    stem = Path(filename).stem
    m = re.match(r"(.+?)-\d+$", stem)
    return m.group(1) if m else "unknown"


def locate_dataset():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Dataset ZIP not found: {ZIP_PATH}\n"
            "Upload FakeBench_images.zip to ~/fakebench/data/"
        )

    extract_dir = WORK_DIR / "dataset"
    if not extract_dir.exists() or not any(extract_dir.rglob("*")):
        print("Extracting dataset...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(extract_dir)

    real_dirs = [p for p in extract_dir.rglob("*") if p.is_dir() and p.name.lower() in {"real_images", "real"}]
    fake_dirs = [p for p in extract_dir.rglob("*") if p.is_dir() and p.name.lower() in {"fake_images", "fake"}]

    for r in real_dirs:
        for f in fake_dirs:
            if r.parent == f.parent:
                return r.parent, r, f

    raise FileNotFoundError("Could not find matching real/fake folders after extraction.")


def build_metadata(dataset_dir, real_dir, fake_dir):
    metadata_file = OUTPUT_DIR / "metadata.csv"
    if metadata_file.exists():
        df = pd.read_csv(metadata_file)
        if len(df) and Path(str(df.iloc[0]["absolute_path"])).exists():
            print("Using existing metadata:", metadata_file)
            return df

    rows = []
    for label, folder in {"real": real_dir, "fake": fake_dir}.items():
        files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        for p in tqdm(files, desc=f"Scanning {label}"):
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    fmt = im.format
                status = "ok"
            except Exception as e:
                w = h = fmt = None
                status = f"error:{type(e).__name__}"
            rows.append({
                "relative_path": str(p.relative_to(dataset_dir)),
                "absolute_path": str(p),
                "filename": p.name,
                "ground_truth": label,
                "generator": infer_generator(p.name, label),
                "width": w,
                "height": h,
                "format": fmt,
                "status": status,
            })

    df = pd.DataFrame(rows)
    df.to_csv(metadata_file, index=False)
    return df


def make_eval_sample(metadata, n_real, n_fake, seed):
    sample_file = OUTPUT_DIR / f"evaluation_sample_{n_real+n_fake}.csv"
    if sample_file.exists():
        df = pd.read_csv(sample_file)
        if len(df) == n_real + n_fake and len(df) and Path(str(df.iloc[0]["absolute_path"])).exists():
            print("Using fixed evaluation sample:", sample_file)
            return df

    valid_meta = metadata[metadata["status"] == "ok"].copy()
    real_df = valid_meta[valid_meta["ground_truth"] == "real"]
    fake_df = valid_meta[valid_meta["ground_truth"] == "fake"]

    n_real = min(n_real, len(real_df))
    n_fake = min(n_fake, len(fake_df))
    real_sample = real_df.sample(n_real, random_state=seed)

    counts = fake_df["generator"].value_counts()
    target = counts / counts.sum() * n_fake
    alloc = np.floor(target).astype(int)
    left = n_fake - alloc.sum()
    for g in (target - alloc).sort_values(ascending=False).index.tolist()[:left]:
        alloc[g] += 1

    fake_parts = []
    for generator, n in alloc.items():
        if n > 0:
            group = fake_df[fake_df["generator"] == generator]
            fake_parts.append(group.sample(min(n, len(group)), random_state=seed))

    fake_sample = pd.concat(fake_parts, ignore_index=True) if fake_parts else fake_df.iloc[0:0]
    if len(fake_sample) < n_fake:
        used = set(fake_sample["absolute_path"])
        remaining = fake_df[~fake_df["absolute_path"].isin(used)]
        need = min(n_fake - len(fake_sample), len(remaining))
        if need:
            fake_sample = pd.concat([fake_sample, remaining.sample(need, random_state=seed)], ignore_index=True)

    eval_df = pd.concat([real_sample, fake_sample], ignore_index=True)
    eval_df = eval_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    eval_df["image_id"] = [f"img_{i:05d}" for i in range(len(eval_df))]
    eval_df.to_csv(sample_file, index=False)
    print("Saved fixed evaluation sample:", sample_file)
    return eval_df


def load_multimodal_pipeline(cfg):
    print("\nLoading:", cfg["model_id"])
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "not set"))
    print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info(0)
        print(f"GPU memory before load: free={free_b/1024**3:.2f} GiB / total={total_b/1024**3:.2f} GiB")

    model_kwargs = {
        "device_map": "auto",
        "dtype": torch.float16,
    }

    return pipeline(
        "image-text-to-text",
        model=cfg["model_id"],
        trust_remote_code=cfg["trust_remote_code"],
        model_kwargs=model_kwargs,
    )


def unload_pipeline(pipe):
    if pipe is not None:
        try:
            del pipe
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def run_model(pipe, cfg, image_path, prompt, max_new_tokens=160):
    start = time.time()

    if cfg["input_style"] == "plain":
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        with torch.inference_mode():
            out = pipe(
                image,
                text=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = out[0].get("generated_text", "")
    else:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }]
        with torch.inference_mode():
            out = pipe(
                text=messages,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_full_text=False,
            )
        generated = out[0].get("generated_text", "")

    latency = time.time() - start

    if isinstance(generated, list):
        parts = []
        for item in generated:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    parts.extend([
                        str(x.get("text", ""))
                        for x in content
                        if isinstance(x, dict) and x.get("type") == "text"
                    ])
            else:
                parts.append(str(item))
        generated = "\n".join(parts)

    return str(generated).strip(), latency


def parse_label(text):
    t = str(text).upper()
    m = re.search(r"LABEL\s*:\s*(AI-GENERATED|AI GENERATED|REAL)", t)
    if m:
        return "fake" if "AI" in m.group(1) else "real"
    if "AI-GENERATED" in t or "AI GENERATED" in t:
        return "fake"
    if re.search(r"\bREAL\b", t):
        return "real"
    return None


def parse_explanation(text):
    text = str(text)
    m = re.search(r"EXPLANATION\s*:\s*(.+)", text, flags=re.I | re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"EVIDENCE\s*:\s*(.*?)(?:LABEL\s*:|$)", text, flags=re.I | re.S)
    if m:
        return m.group(1).strip()
    return text.strip()


def save_results_atomic(results, path):
    """Keep one latest record per image/strategy and write atomically."""
    if len(results):
        results = results.drop_duplicates(["image_id", "prompt_strategy"], keep="last").reset_index(drop=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    results.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return results


def call_with_retry(fn, retries=2):
    """Retry transient CUDA/runtime failures without losing the whole experiment."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            print(f"Inference error (attempt {attempt+1}/{retries+1}): {type(e).__name__}: {str(e)[:300]}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            time.sleep(1)
    raise last_error


def run_experiment(model_short_name, eval_df, etc_use_ground_truth=False, fresh=False, retries=2):
    cfg = MODEL_CONFIGS[model_short_name]
    columns = [
        "image_id", "relative_path", "absolute_path", "ground_truth", "generator",
        "model", "model_short_name", "prompt_strategy", "raw_response", "prediction",
        "explanation", "latency_seconds", "status"
    ]

    results_file = OUTPUT_DIR / f"{model_short_name}_results_{len(eval_df)}.csv"

    if fresh and results_file.exists():
        backup = results_file.with_name(results_file.stem + f"_backup_{int(time.time())}" + results_file.suffix)
        results_file.rename(backup)
        print("Fresh run requested. Old result backed up to:", backup)

    if results_file.exists():
        results = pd.read_csv(results_file)
        for c in columns:
            if c not in results.columns:
                results[c] = np.nan
        results = results[columns]
        results = results.drop_duplicates(["image_id", "prompt_strategy"], keep="last").reset_index(drop=True)
    else:
        results = pd.DataFrame(columns=columns)

    # Only successful, parseable rows count as completed. Failed rows will be retried on resume.
    completed_df = results[(results["status"] == "ok") & results["prediction"].isin(["real", "fake"])]
    completed = set(zip(completed_df["image_id"], completed_df["prompt_strategy"]))

    pipe = None
    try:
        pipe = load_multimodal_pipeline(cfg)

        # Smoke test - catches model/GPU problems before the 500-image loop.
        r0 = eval_df.iloc[0]
        raw0, lat0 = run_model(pipe, cfg, r0["absolute_path"], PROMPTS["baseline"], 80)
        print("Smoke test:", raw0[:500])
        print("Parsed:", parse_label(raw0), "Latency:", round(lat0, 2), "s")

        for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=model_short_name):
            image_id = row["image_id"]
            image_path = row["absolute_path"]
            common = {
                "image_id": image_id,
                "relative_path": row["relative_path"],
                "absolute_path": image_path,
                "ground_truth": row["ground_truth"],
                "generator": row["generator"],
                "model": cfg["model_id"],
                "model_short_name": model_short_name,
            }

            # Baseline
            key = (image_id, "baseline")
            if key not in completed:
                try:
                    raw, latency = call_with_retry(
                        lambda: run_model(pipe, cfg, image_path, PROMPTS["baseline"], 160), retries
                    )
                    pred, expl = parse_label(raw), parse_explanation(raw)
                    status = "ok" if pred is not None else "parse_error"
                except Exception as e:
                    raw, pred, expl, latency = "", None, "", None
                    status = f"error:{type(e).__name__}:{str(e)[:200]}"
                new_row = {
                    **common, "prompt_strategy": "baseline", "raw_response": raw,
                    "prediction": pred, "explanation": expl,
                    "latency_seconds": latency, "status": status,
                }
                results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
                results = save_results_atomic(results, results_file)
                if status == "ok":
                    completed.add(key)

            # Effect-to-Cause
            key = (image_id, "effect_to_cause")
            if key not in completed:
                try:
                    if etc_use_ground_truth:
                        pred = row["ground_truth"]
                        display_pred = "AI-GENERATED" if pred == "fake" else "REAL"
                        raw_label, latency1 = f"[GROUND TRUTH PROVIDED] {display_pred}", 0.0
                    else:
                        raw_label, latency1 = call_with_retry(
                            lambda: run_model(pipe, cfg, image_path, PROMPTS["effect_to_cause_label"], 30), retries
                        )
                        pred = parse_label(raw_label)

                    if pred is None:
                        raw, explanation, latency2, status = raw_label, "", 0.0, "parse_error"
                    else:
                        display_pred = "AI-GENERATED" if pred == "fake" else "REAL"
                        p2 = PROMPTS["effect_to_cause_explain"].format(prediction=display_pred)
                        raw_expl, latency2 = call_with_retry(
                            lambda: run_model(pipe, cfg, image_path, p2, 150), retries
                        )
                        raw = f"[STEP 1]\n{raw_label}\n\n[STEP 2]\n{raw_expl}"
                        explanation, status = raw_expl, "ok"
                    latency = latency1 + latency2
                except Exception as e:
                    raw, pred, explanation, latency = "", None, "", None
                    status = f"error:{type(e).__name__}:{str(e)[:200]}"
                new_row = {
                    **common, "prompt_strategy": "effect_to_cause", "raw_response": raw,
                    "prediction": pred, "explanation": explanation,
                    "latency_seconds": latency, "status": status,
                }
                results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
                results = save_results_atomic(results, results_file)
                if status == "ok":
                    completed.add(key)

            # Cause-to-Effect
            key = (image_id, "cause_to_effect")
            if key not in completed:
                try:
                    raw, latency = call_with_retry(
                        lambda: run_model(pipe, cfg, image_path, PROMPTS["cause_to_effect"], 180), retries
                    )
                    pred, expl = parse_label(raw), parse_explanation(raw)
                    status = "ok" if pred is not None else "parse_error"
                except Exception as e:
                    raw, pred, expl, latency = "", None, "", None
                    status = f"error:{type(e).__name__}:{str(e)[:200]}"
                new_row = {
                    **common, "prompt_strategy": "cause_to_effect", "raw_response": raw,
                    "prediction": pred, "explanation": expl,
                    "latency_seconds": latency, "status": status,
                }
                results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
                results = save_results_atomic(results, results_file)
                if status == "ok":
                    completed.add(key)

            if (idx + 1) % 25 == 0 or (idx + 1) == len(eval_df):
                ok_now = results[(results["status"] == "ok") & results["prediction"].isin(["real", "fake"])]
                counts = ok_now.groupby("prompt_strategy")["image_id"].nunique().to_dict()
                print(f"Progress {idx+1}/{len(eval_df)} | valid unique counts: {counts}")

        print("Finished:", model_short_name, "->", results_file)
    finally:
        unload_pipeline(pipe)

    # Final integrity check
    final = pd.read_csv(results_file)
    final_ok = final[(final["status"] == "ok") & final["prediction"].isin(["real", "fake"])]
    counts = final_ok.groupby("prompt_strategy")["image_id"].nunique().to_dict()
    print("\nFINAL VALID UNIQUE COUNTS:", counts)
    expected = len(eval_df)
    incomplete = {s: counts.get(s, 0) for s in VALID_STRATEGIES if counts.get(s, 0) != expected}
    if incomplete:
        print("WARNING: incomplete strategies:", incomplete, "expected", expected)
    else:
        print(f"SUCCESS: all 3 strategies have {expected} valid predictions.")

    return results_file


def merge_metrics(sample_size):
    files = sorted(OUTPUT_DIR.glob(f"*_results_{sample_size}.csv"))
    # Exclude aggregate files and backup files from re-merging.
    files = [p for p in files if not p.name.startswith("all_models_") and "backup" not in p.name]
    if not files:
        return

    frames = []
    for p in files:
        df = pd.read_csv(p)
        if {"image_id", "prompt_strategy"}.issubset(df.columns):
            df = df.drop_duplicates(["image_id", "prompt_strategy"], keep="last")
        frames.append(df)

    all_results = pd.concat(frames, ignore_index=True)
    all_results.to_csv(OUTPUT_DIR / f"all_models_results_{sample_size}.csv", index=False)

    valid = all_results[
        (all_results["status"] == "ok") &
        all_results["prediction"].isin(["real", "fake"])
    ].copy()

    metric_rows = []
    for (model, short, strategy), g in valid.groupby(["model", "model_short_name", "prompt_strategy"]):
        # Defensive uniqueness check before metrics.
        g = g.drop_duplicates(["image_id", "prompt_strategy"], keep="last")
        y_true, y_pred = g["ground_truth"], g["prediction"]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=["real", "fake"]).ravel()
        metric_rows.append({
            "model": model,
            "model_short_name": short,
            "prompt_strategy": strategy,
            "n": len(g),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_fake": precision_score(y_true, y_pred, pos_label="fake", zero_division=0),
            "recall_fake": recall_score(y_true, y_pred, pos_label="fake", zero_division=0),
            "f1_fake": f1_score(y_true, y_pred, pos_label="fake", zero_division=0),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "mean_latency_seconds": g["latency_seconds"].dropna().mean(),
        })

    metrics = pd.DataFrame(metric_rows)
    if len(metrics):
        metrics = metrics.sort_values(["prompt_strategy", "f1_fake", "accuracy"], ascending=[True, False, False])
    metrics.to_csv(OUTPUT_DIR / f"all_models_metrics_{sample_size}.csv", index=False)
    print("\nMetrics:\n", metrics.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--n-real", type=int, default=250)
    parser.add_argument("--n-fake", type=int, default=250)
    parser.add_argument("--seed", type=int, default=36105)
    parser.add_argument("--strict-etc", action="store_true", help="Use ground-truth authenticity for strict FakeBench-style EtC")
    parser.add_argument("--fresh", action="store_true", help="Back up any existing result for this model/sample size and rerun from scratch")
    parser.add_argument("--retries", type=int, default=2, help="Retries per failed inference (default: 2)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is visible. Submit this script through a GPU PBS job.")

    dataset_dir, real_dir, fake_dir = locate_dataset()
    metadata = build_metadata(dataset_dir, real_dir, fake_dir)
    eval_df = make_eval_sample(metadata, args.n_real, args.n_fake, args.seed)

    expected = args.n_real + args.n_fake
    if len(eval_df) != expected:
        raise RuntimeError(f"Evaluation sample has {len(eval_df)} rows, expected {expected}. Aborting for a fair experiment.")
    if eval_df["image_id"].nunique() != expected:
        raise RuntimeError("Evaluation sample contains duplicate image_id values. Aborting.")

    print("\nEvaluation sample:")
    print(eval_df["ground_truth"].value_counts())
    print("Total:", len(eval_df))

    run_experiment(
        args.model,
        eval_df,
        etc_use_ground_truth=args.strict_etc,
        fresh=args.fresh,
        retries=args.retries,
    )
    merge_metrics(len(eval_df))


if __name__ == "__main__":
    main()

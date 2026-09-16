#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

HOME = Path.home()
ROOT = HOME / "fakebench"
RESULTS = ROOT / "results"
TEST_CSV = RESULTS / "evaluation_sample_500.csv"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_DIR = ROOT / "models" / "qwen25vl_3b_fakebench_lora" / "best_adapter"

OUT_DETAIL = RESULTS / "qwen3b_base_vs_lora_predictions.csv"
OUT_SUMMARY = RESULTS / "qwen3b_base_vs_lora_summary.csv"

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 768 * 28 * 28

PROMPT = """Classify this image as REAL or AI-GENERATED.
Return only one label:
REAL
AI-GENERATED"""

def load_image(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")

def parse_label(text):
    t = text.strip().upper()
    if "AI-GENERATED" in t or "AI GENERATED" in t:
        return "fake"
    if "REAL" in t:
        return "real"
    return "unknown"

def evaluate(model, processor, df, tag):
    model.eval()
    rows = []
    y_true, y_pred = [], []
    latencies = []

    for i, r in df.iterrows():
        image = load_image(r["absolute_path"])
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        torch.cuda.synchronize()
        latency = time.perf_counter() - t0

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        output_text = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        pred = parse_label(output_text)
        truth = str(r["ground_truth"]).strip().lower()
        y_true.append(truth)
        y_pred.append(pred)
        latencies.append(latency)

        rows.append({
            "model": tag,
            "relative_path": r.get("relative_path", ""),
            "absolute_path": r["absolute_path"],
            "ground_truth": truth,
            "prediction": pred,
            "raw_output": output_text,
            "latency_sec": latency,
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(df):
            print(f"[{tag}] {i+1}/{len(df)}")

    valid_mask = [p in ("real", "fake") for p in y_pred]
    valid_true = [t for t, ok in zip(y_true, valid_mask) if ok]
    valid_pred = [p for p, ok in zip(y_pred, valid_mask) if ok]
    yt = np.array([1 if x == "fake" else 0 for x in valid_true])
    yp = np.array([1 if x == "fake" else 0 for x in valid_pred])

    if len(yt) == 0:
        raise RuntimeError(f"{tag}: no valid predictions parsed.")

    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    summary = {
        "model": tag,
        "N_total": len(df),
        "N_valid": len(yt),
        "N_unknown": len(df) - len(yt),
        "accuracy": accuracy_score(yt, yp),
        "precision": precision_score(yt, yp, zero_division=0),
        "recall": recall_score(yt, yp, zero_division=0),
        "f1": f1_score(yt, yp, zero_division=0),
        "mcc": matthews_corrcoef(yt, yp),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "mean_latency_sec": float(np.mean(latencies)),
        "total_inference_min": float(np.sum(latencies) / 60.0),
    }
    return pd.DataFrame(rows), summary

def print_summary(s):
    print("\n" + "=" * 72)
    print(s["model"])
    print("=" * 72)
    for k, v in s.items():
        if k != "model":
            print(f"{k}: {v}")

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required.")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Test CSV:", TEST_CSV)
    print("Adapter:", ADAPTER_DIR)

    df = pd.read_csv(TEST_CSV)
    required = {"absolute_path", "ground_truth"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in test CSV: {missing}")

    if len(df) != 500:
        print(f"WARNING: expected 500 test samples, found {len(df)}")

    print("Test label counts:", df["ground_truth"].value_counts().to_dict())

    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

    print("\nLoading BASE model...")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda:0")
    base_rows, base_summary = evaluate(base, processor, df, "Qwen2.5-VL-3B Base")
    print_summary(base_summary)

    del base
    gc.collect()
    torch.cuda.empty_cache()

    print("\nLoading BASE + LoRA best_adapter...")
    lora_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    lora = PeftModel.from_pretrained(lora_base, ADAPTER_DIR).to("cuda:0")
    lora.eval()

    lora_rows, lora_summary = evaluate(lora, processor, df, "Qwen2.5-VL-3B + LoRA")
    print_summary(lora_summary)

    detail = pd.concat([base_rows, lora_rows], ignore_index=True)
    detail.to_csv(OUT_DETAIL, index=False)

    summary = pd.DataFrame([base_summary, lora_summary])
    delta = {"model": "LoRA - Base"}
    for c in ["accuracy", "precision", "recall", "f1", "mcc", "mean_latency_sec"]:
        delta[c] = lora_summary[c] - base_summary[c]
    summary = pd.concat([summary, pd.DataFrame([delta])], ignore_index=True)
    summary.to_csv(OUT_SUMMARY, index=False)

    print("\nSaved:")
    print(OUT_DETAIL)
    print(OUT_SUMMARY)
    print("\nFINAL COMPARISON")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Qwen2.5-VL-3B base vs LoRA on one or more test CSVs (e.g. test_id.csv, test_ood.csv).

Outputs in --out:
  summary.csv            accuracy (+95% Wilson CI), precision, recall, F1, MCC, confusion counts, per model x test
  per_generator.csv      fake recall per generator (and accuracy on real images)
  predictions_<test>.csv raw outputs for every image
Unparseable outputs are counted as WRONG (never dropped).
"""
import argparse
import gc
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

from imgnorm import load_normalized

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
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


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


@torch.inference_mode()
def predict(model, processor, df, tag, test_name, size):
    model.eval()
    rows = []
    for i, r in enumerate(df.itertuples(index=False), 1):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[text], images=[load_normalized(r.absolute_path, size, False)], return_tensors="pt").to("cuda:0")
        t0 = time.perf_counter()
        out = model.generate(**inp, max_new_tokens=8, do_sample=False)
        lat = time.perf_counter() - t0
        raw = processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        rows.append({
            "model": tag, "test": test_name, "relative_path": r.relative_path,
            "generator": getattr(r, "generator", ""), "ground_truth": str(r.ground_truth).lower(),
            "prediction": parse_label(raw), "raw_output": raw, "latency_sec": lat,
        })
        if i % 100 == 0 or i == len(df):
            print(f"[{tag} | {test_name}] {i}/{len(df)}", flush=True)
    return pd.DataFrame(rows)


def summarise(pred):
    truth = pred["ground_truth"].values
    p = pred["prediction"].values
    n_unknown = int((p == "unknown").sum())
    p = np.where(p == "unknown", np.where(truth == "fake", "real", "fake"), p)  # unknown = wrong
    yt = (truth == "fake").astype(int)
    yp = (p == "fake").astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    n, k = len(yt), int((yt == yp).sum())
    lo, hi = wilson(k, n)
    return {
        "model": pred["model"].iloc[0], "test": pred["test"].iloc[0], "N": n, "N_unknown": n_unknown,
        "accuracy": k / n, "acc_ci95_low": lo, "acc_ci95_high": hi,
        "precision": precision_score(yt, yp, zero_division=0), "recall": recall_score(yt, yp, zero_division=0),
        "f1": f1_score(yt, yp, zero_division=0), "mcc": matthews_corrcoef(yt, yp),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "mean_latency_sec": pred["latency_sec"].mean(),
    }


def per_generator(pred):
    rows = []
    for g, d in pred.groupby("generator"):
        truth = d["ground_truth"].iloc[0]
        ok = (d["prediction"] == d["ground_truth"]).mean()
        rows.append({"model": d["model"].iloc[0], "test": d["test"].iloc[0],
                     "generator": g if g else "(none)", "truth": truth, "n": len(d), "correct_rate": ok})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    root = Path.home() / "fakebench"
    ap.add_argument("--splits", default=str(root / "data_splits_v2"))
    ap.add_argument("--tests", nargs="+", default=["test_id", "test_ood"], help="CSV names without .csv")
    ap.add_argument("--adapter", default=str(root / "models" / "qwen25vl_3b_lora_v2" / "best_adapter"))
    ap.add_argument("--out", default=str(root / "results" / "v2"))
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--norm-size", type=int, default=448, help="must match training; 0 = off")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tests = {}
    for t in a.tests:
        f = Path(a.splits) / f"{t}.csv"
        if f.exists():
            tests[t] = pd.read_csv(f)
            print(f"{t}: {len(tests[t])} images {tests[t]['ground_truth'].value_counts().to_dict()}")
        else:
            print(f"(skipping {t}: {f} not found)")
    if not tests:
        raise SystemExit("no test CSVs found")

    if a.norm_size:
        mn = mx = a.norm_size * a.norm_size
    else:
        mn, mx = MIN_PIXELS, MAX_PIXELS
    print("normalised input size:", a.norm_size or "off")
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=mn, max_pixels=mx)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda:0")

    all_pred = []
    if not a.skip_base:
        for t, df in tests.items():
            all_pred.append(predict(model, processor, df, "Qwen2.5-VL-3B base", t, a.norm_size))
    print("Loading adapter:", a.adapter)
    model = PeftModel.from_pretrained(model, a.adapter).to("cuda:0")
    for t, df in tests.items():
        all_pred.append(predict(model, processor, df, "Qwen2.5-VL-3B + LoRA", t, a.norm_size))
    del model; gc.collect(); torch.cuda.empty_cache()

    pred = pd.concat(all_pred, ignore_index=True)
    for t in tests:
        pred[pred["test"] == t].to_csv(out / f"predictions_{t}.csv", index=False)
    summary = pd.DataFrame([summarise(g) for _, g in pred.groupby(["test", "model"], sort=False)])
    summary.to_csv(out / "summary.csv", index=False)
    pg = pd.concat([per_generator(g) for _, g in pred.groupby(["test", "model"], sort=False)], ignore_index=True)
    pg.to_csv(out / "per_generator.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== SUMMARY ===")
    print(summary[["test", "model", "N", "accuracy", "acc_ci95_low", "acc_ci95_high", "recall", "f1", "mcc", "FP", "FN", "N_unknown"]].to_string(index=False))
    print("\n=== PER GENERATOR ===")
    print(pg.to_string(index=False))
    print("\nSaved to", out)


if __name__ == "__main__":
    main()

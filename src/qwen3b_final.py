#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, time
from pathlib import Path
import pandas as pd
import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

HOME = Path.home()
PROJECT_DIR = HOME / "fakebench"
RESULTS_DIR = PROJECT_DIR / "results"
SAMPLE_FILE = RESULTS_DIR / "evaluation_sample_500.csv"
RESULT_FILE = RESULTS_DIR / "qwen25vl_3b_results_500.csv"
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

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
LABEL: REAL or AI-GENERATED"""
}

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

def normalize_image(src_path, dst_path):
    with Image.open(src_path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        max_side = 1600
        if max(im.size) > max_side:
            scale = max_side / max(im.size)
            im = im.resize((max(1, int(im.width*scale)), max(1, int(im.height*scale))), Image.Resampling.LANCZOS)
        im.save(dst_path, format="PNG", optimize=False)

def run_once(model, processor, image_path, prompt, max_new_tokens):
    messages = [{"role":"user","content":[
        {"type":"image","image":f"file://{image_path}"},
        {"type":"text","text":prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    start = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    latency = time.time() - start
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return out.strip(), latency

def save_rows(rows):
    pd.DataFrame(rows).to_csv(RESULT_FILE, index=False)

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")
    sample = pd.read_csv(SAMPLE_FILE)
    if len(sample) != 500:
        raise RuntimeError(f"Expected 500 images, found {len(sample)}")
    if RESULT_FILE.exists():
        RESULT_FILE.unlink()

    print("GPU:", torch.cuda.get_device_name(0))
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=min_pixels, max_pixels=max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    tmp_dir = PROJECT_DIR / "work" / "qwen3b_normalized"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for idx, row in sample.iterrows():
        image_id = str(row["image_id"])
        src = Path(str(row["absolute_path"]))
        tmp = tmp_dir / f"{image_id}.png"
        normalize_image(src, tmp)

        common = {
            "image_id": image_id,
            "relative_path": row["relative_path"],
            "absolute_path": str(src),
            "ground_truth": row["ground_truth"],
            "generator": row["generator"],
            "model": MODEL_ID,
            "model_short_name": "qwen25vl_3b",
        }

        # baseline
        raw, lat = run_once(model, processor, tmp, PROMPTS["baseline"], 160)
        pred = parse_label(raw)
        rows.append({**common,"prompt_strategy":"baseline","raw_response":raw,"prediction":pred,
                     "explanation":parse_explanation(raw),"latency_seconds":lat,
                     "status":"ok" if pred else "parse_error"})
        save_rows(rows)

        # effect_to_cause
        raw1, lat1 = run_once(model, processor, tmp, PROMPTS["effect_to_cause_label"], 30)
        pred = parse_label(raw1)
        if pred is None:
            raw, expl, lat, status = raw1, "", lat1, "parse_error"
        else:
            display = "AI-GENERATED" if pred == "fake" else "REAL"
            raw2, lat2 = run_once(model, processor, tmp, PROMPTS["effect_to_cause_explain"].format(prediction=display), 150)
            raw = f"[STEP 1]\n{raw1}\n\n[STEP 2]\n{raw2}"
            expl, lat, status = raw2, lat1 + lat2, "ok"
        rows.append({**common,"prompt_strategy":"effect_to_cause","raw_response":raw,"prediction":pred,
                     "explanation":expl,"latency_seconds":lat,"status":status})
        save_rows(rows)

        # cause_to_effect
        raw, lat = run_once(model, processor, tmp, PROMPTS["cause_to_effect"], 180)
        pred = parse_label(raw)
        rows.append({**common,"prompt_strategy":"cause_to_effect","raw_response":raw,"prediction":pred,
                     "explanation":parse_explanation(raw),"latency_seconds":lat,
                     "status":"ok" if pred else "parse_error"})
        save_rows(rows)

        if (idx + 1) % 10 == 0:
            print(f"Completed {idx+1}/500 images")

    df = pd.DataFrame(rows)
    print("\nFINAL COUNTS")
    print(df["prompt_strategy"].value_counts())
    print("\nSTATUS")
    print(df["status"].value_counts())
    print("\nVALID UNIQUE IMAGES")
    print(df[df["status"]=="ok"].groupby("prompt_strategy")["image_id"].nunique())

if __name__ == "__main__":
    main()

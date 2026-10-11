#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA fine-tuning of Qwen2.5-VL-3B on the leakage-aware splits produced by make_clean_splits.py.

Differences from train_qwen3b_lora.py:
  * reads train.csv / val.csv instead of re-sampling inside the script
  * padding side is set explicitly to "right" (the loss mask below assumes it)
  * checkpoint selection uses validation ACCURACY (generated labels), with val loss as tie-breaker
  * epochs / lr / paths are command-line arguments
"""
import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from imgnorm import load_normalized

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 256 * 28 * 28   # only used when --norm-size 0
MAX_PIXELS = 768 * 28 * 28
PROMPT = """Classify this image as REAL or AI-GENERATED.
Return only one label:
REAL
AI-GENERATED"""


def load_image(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def answer_of(gt):
    return "REAL" if str(gt).strip().lower() == "real" else "AI-GENERATED"


def parse_label(text):
    t = text.strip().upper()
    if "AI-GENERATED" in t or "AI GENERATED" in t:
        return "fake"
    if "REAL" in t:
        return "real"
    return "unknown"


class DS(Dataset):
    def __init__(self, df, size, train):
        self.df = df.reset_index(drop=True)
        self.size, self.train = size, train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return load_normalized(r["absolute_path"], self.size, self.train), answer_of(r["ground_truth"])


def make_collator(processor):
    def fn(batch):
        images = [x[0] for x in batch]
        full_texts, prompt_texts = [], []
        for _, ans in batch:
            user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}
            full = [user, {"role": "assistant", "content": [{"type": "text", "text": ans}]}]
            full_texts.append(processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False))
            prompt_texts.append(processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True))
        full = processor(text=full_texts, images=images, padding=True, return_tensors="pt")
        pr = processor(text=prompt_texts, images=images, padding=True, return_tensors="pt")
        labels = full["input_ids"].clone()
        labels[full["attention_mask"] == 0] = -100
        # right padding => the first `prompt_len` positions of each row are the prompt
        for i, p in enumerate(pr["attention_mask"].sum(dim=1).tolist()):
            labels[i, : int(p)] = -100
        full["labels"] = labels
        return full

    return fn


@torch.no_grad()
def validate(model, processor, loader, val_df, device, size):
    model.eval()
    losses = [float(model(**{k: v.to(device) for k, v in b.items()}).loss.cpu()) for b in loader]
    correct = 0
    for _, r in val_df.iterrows():
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[text], images=[load_normalized(r["absolute_path"], size, False)], return_tensors="pt").to(device)
        out = model.generate(**inp, max_new_tokens=8, do_sample=False, use_cache=True)
        txt = processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        correct += parse_label(txt) == str(r["ground_truth"]).strip().lower()
    model.train()
    return float(np.mean(losses)), correct / len(val_df)


def main():
    ap = argparse.ArgumentParser()
    root = Path.home() / "fakebench"
    ap.add_argument("--splits", default=str(root / "data_splits_v2"))
    ap.add_argument("--out", default=str(root / "models" / "qwen25vl_3b_lora_v2"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=36105)
    ap.add_argument("--norm-size", type=int, default=448, help="uniform resize+JPEG preprocessing (multiple of 28); 0 = off")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    device = "cuda:0"
    print("GPU:", torch.cuda.get_device_name(0))

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(Path(a.splits) / "train.csv")
    val_df = pd.read_csv(Path(a.splits) / "val.csv")
    print("train:", len(train_df), train_df["ground_truth"].value_counts().to_dict())
    print("val:  ", len(val_df), val_df["ground_truth"].value_counts().to_dict())

    if a.norm_size:
        assert a.norm_size % 28 == 0, "--norm-size must be a multiple of 28"
        mn = mx = a.norm_size * a.norm_size
    else:
        mn, mx = MIN_PIXELS, MAX_PIXELS
    print("normalised input size:", a.norm_size or "off")
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=mn, max_pixels=mx)
    processor.tokenizer.padding_side = "right"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    cfg = LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.to(device)

    coll = make_collator(processor)
    train_loader = DataLoader(DS(train_df, a.norm_size, True), batch_size=a.batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=coll)
    val_loader = DataLoader(DS(val_df, a.norm_size, False), batch_size=a.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=coll)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.01)
    total_steps = math.ceil(len(train_loader) / a.grad_accum) * a.epochs
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * 0.05)), total_steps)

    hist, best_key, gstep = [], None, 0
    opt.zero_grad(set_to_none=True)
    model.train()
    for epoch in range(1, a.epochs + 1):
        t0, losses = time.time(), []
        for bi, b in enumerate(train_loader, 1):
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**b).loss / a.grad_accum
            loss.backward()
            losses.append(float(loss.detach().cpu()) * a.grad_accum)
            if bi % a.grad_accum == 0 or bi == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); gstep += 1
                if gstep % 25 == 0:
                    print(f"epoch={epoch} step={gstep}/{total_steps} loss={np.mean(losses[-25:]):.4f}", flush=True)

        vl, va = validate(model, processor, val_loader, val_df, device, a.norm_size)
        mins = (time.time() - t0) / 60
        print(f"Epoch {epoch}: train_loss={np.mean(losses):.5f} val_loss={vl:.5f} val_acc={va:.4f} time={mins:.1f}m", flush=True)
        hist.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": vl, "val_acc": va, "minutes": mins})
        pd.DataFrame(hist).to_csv(out / "training_history.csv", index=False)

        ep_dir = out / f"epoch_{epoch}"
        model.save_pretrained(ep_dir); processor.save_pretrained(ep_dir)
        key = (-va, vl)  # higher accuracy first, then lower loss
        if best_key is None or key < best_key:
            best_key = key
            model.save_pretrained(out / "best_adapter"); processor.save_pretrained(out / "best_adapter")
            print("  -> new best adapter (epoch %d)" % epoch, flush=True)

    model.save_pretrained(out / "final_adapter")
    print("Done. Best adapter:", out / "best_adapter")


if __name__ == "__main__":
    main()

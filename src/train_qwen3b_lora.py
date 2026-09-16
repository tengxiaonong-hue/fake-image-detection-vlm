#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

SEED=36105
HOME=Path.home()
ROOT=HOME/"fakebench"
RESULTS=ROOT/"results"
SPLITS=ROOT/"data_splits"
OUT=ROOT/"models"/"qwen25vl_3b_fakebench_lora"
META=RESULTS/"metadata.csv"
TEST=RESULTS/"evaluation_sample_500.csv"
MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"

PROMPT="""Classify this image as REAL or AI-GENERATED.
Return only one label:
REAL
AI-GENERATED"""

EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=8
LR=1e-4
MAX_PIXELS=768*28*28
MIN_PIXELS=256*28*28

SPLITS.mkdir(parents=True,exist_ok=True)
OUT.mkdir(parents=True,exist_ok=True)

def seed_all():
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def make_splits():
    meta=pd.read_csv(META)
    test=pd.read_csv(TEST)
    meta=meta[meta["status"]=="ok"].copy()
    test_paths=set(test["relative_path"].astype(str))
    remain=meta[~meta["relative_path"].astype(str).isin(test_paths)].copy()

    real=remain[remain["ground_truth"]=="real"]
    fake=remain[remain["ground_truth"]=="fake"]

    tr_real=real.sample(2250,random_state=SEED)
    tr_fake=fake.sample(2250,random_state=SEED)
    val_real=real.drop(tr_real.index).sample(500,random_state=SEED+1)
    val_fake=fake.drop(tr_fake.index).sample(500,random_state=SEED+1)

    train=pd.concat([tr_real,tr_fake],ignore_index=True).sample(frac=1,random_state=SEED).reset_index(drop=True)
    val=pd.concat([val_real,val_fake],ignore_index=True).sample(frac=1,random_state=SEED).reset_index(drop=True)

    assert set(train["relative_path"]).isdisjoint(test_paths)
    assert set(val["relative_path"]).isdisjoint(test_paths)

    train.to_csv(SPLITS/"fakebench_train_4500.csv",index=False)
    val.to_csv(SPLITS/"fakebench_val_1000.csv",index=False)

    print("Train:",len(train),train["ground_truth"].value_counts().to_dict())
    print("Val:",len(val),val["ground_truth"].value_counts().to_dict())
    print("Test:",len(test),test["ground_truth"].value_counts().to_dict())
    return train,val

class DS(Dataset):
    def __init__(self,df): self.df=df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        with Image.open(r["absolute_path"]) as im:
            image=ImageOps.exif_transpose(im).convert("RGB")
        ans="REAL" if r["ground_truth"]=="real" else "AI-GENERATED"
        return image,ans

def collator(processor):
    def fn(batch):
        images=[x[0] for x in batch]
        full_texts=[]; prompt_texts=[]
        for _,ans in batch:
            user={"role":"user","content":[{"type":"image"},{"type":"text","text":PROMPT}]}
            full=[user,{"role":"assistant","content":[{"type":"text","text":ans}]}]
            full_texts.append(processor.apply_chat_template(full,tokenize=False,add_generation_prompt=False))
            prompt_texts.append(processor.apply_chat_template([user],tokenize=False,add_generation_prompt=True))

        full=processor(text=full_texts,images=images,padding=True,return_tensors="pt")
        pr=processor(text=prompt_texts,images=images,padding=True,return_tensors="pt")
        labels=full["input_ids"].clone()
        labels[full["attention_mask"]==0]=-100
        plens=pr["attention_mask"].sum(dim=1).tolist()
        for i,p in enumerate(plens): labels[i,:int(p)]=-100
        full["labels"]=labels
        return full
    return fn

@torch.no_grad()
def val_loss(model,loader,device):
    model.eval(); losses=[]
    for b in loader:
        b={k:v.to(device) for k,v in b.items()}
        losses.append(float(model(**b).loss.cpu()))
    model.train()
    return float(np.mean(losses))

def main():
    seed_all()
    if not torch.cuda.is_available(): raise RuntimeError("GPU required")
    device="cuda:0"
    print("GPU:",torch.cuda.get_device_name(0))
    train_df,val_df=make_splits()

    processor=AutoProcessor.from_pretrained(MODEL_ID,min_pixels=MIN_PIXELS,max_pixels=MAX_PIXELS)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID,torch_dtype=torch.bfloat16)

    cfg=LoraConfig(
        r=16,lora_alpha=32,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model=get_peft_model(model,cfg)
    model.print_trainable_parameters()
    model.config.use_cache=False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.to(device)

    c=collator(processor)
    train_loader=DataLoader(DS(train_df),batch_size=BATCH_SIZE,shuffle=True,num_workers=2,pin_memory=True,collate_fn=c)
    val_loader=DataLoader(DS(val_df),batch_size=BATCH_SIZE,shuffle=False,num_workers=2,pin_memory=True,collate_fn=c)

    trainable=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=LR,weight_decay=0.01)
    steps_per_epoch=math.ceil(len(train_loader)/GRAD_ACCUM)
    total_steps=steps_per_epoch*EPOCHS
    sched=get_cosine_schedule_with_warmup(opt,max(1,int(total_steps*0.05)),total_steps)

    hist=[]; best=float("inf"); gstep=0
    opt.zero_grad(set_to_none=True)

    for epoch in range(1,EPOCHS+1):
        start=time.time(); losses=[]
        for bi,b in enumerate(train_loader,1):
            b={k:v.to(device,non_blocking=True) for k,v in b.items()}
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss=model(**b).loss/GRAD_ACCUM
            loss.backward()
            losses.append(float(loss.detach().cpu())*GRAD_ACCUM)

            if bi%GRAD_ACCUM==0 or bi==len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable,1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); gstep+=1
                if gstep%25==0:
                    print(f"epoch={epoch} step={gstep}/{total_steps} loss={np.mean(losses[-25:]):.4f}")

        vl=val_loss(model,val_loader,device)
        tl=float(np.mean(losses))
        mins=(time.time()-start)/60
        print(f"Epoch {epoch}: train_loss={tl:.4f} val_loss={vl:.4f} time={mins:.1f}m")

        ep=OUT/f"epoch_{epoch}"
        model.save_pretrained(ep); processor.save_pretrained(ep)

        hist.append({"epoch":epoch,"train_loss":tl,"val_loss":vl,"epoch_minutes":mins})
        pd.DataFrame(hist).to_csv(RESULTS/"qwen3b_lora_training_history.csv",index=False)

        if vl<best:
            best=vl
            bestdir=OUT/"best_adapter"
            model.save_pretrained(bestdir); processor.save_pretrained(bestdir)
            print("Saved best adapter:",bestdir)

    finaldir=OUT/"final_adapter"
    model.save_pretrained(finaldir); processor.save_pretrained(finaldir)
    print("Training complete.")
    print("Best adapter:",OUT/"best_adapter")
    print("Final adapter:",finaldir)

if __name__=="__main__":
    main()

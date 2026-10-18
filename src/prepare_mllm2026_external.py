#!/usr/bin/env python3
"""Prepare ALL labeled images from external MLLMGenSet for zero-shot V2 evaluation.

This public dataset names its only split 'train'. That does NOT mean we train
our model: these images are used ONLY as an unseen external evaluation set.
"""
import argparse, csv, hashlib, io, json, sys
from collections import Counter
from pathlib import Path
from PIL import Image as PILImage
from datasets import load_dataset, Image as HFImage

DATASET="zr-zhang/MLLM-Generated-Image-Detection-Dataset"
EXPECTED=4356
EXT={"JPEG":".jpg","PNG":".png","WEBP":".webp","BMP":".bmp","GIF":".gif","TIFF":".tif"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",default=str(Path.home()/"fakebench/data_external_mllm2026"))
    p.add_argument("--expected",type=int,default=EXPECTED)
    args=p.parse_args()
    root=Path(args.out); csvpath=root/"mllm2026_external.csv"
    if csvpath.exists(): raise SystemExit("External CSV already exists; refusing to overwrite.")
    image_dir=root/"images";image_dir.mkdir(parents=True,exist_ok=True)
    print("Downloading/streaming",DATASET,"train (external test only)",flush=True)
    dataset=load_dataset(DATASET,split="train",streaming=True)
    if "image" not in dataset.column_names or "label" not in dataset.column_names:
        raise SystemExit("Dataset schema changed: expected image and label. Fields: "+str(dataset.column_names))
    dataset=dataset.cast_column("image",HFImage(decode=False))
    rows=[];counts=Counter();sha_seen=set()
    for idx,record in enumerate(dataset):
        rawinfo=record["image"]
        raw=rawinfo.get("bytes") if isinstance(rawinfo,dict) else None
        path_hint=rawinfo.get("path") if isinstance(rawinfo,dict) else None
        if not raw:
            if path_hint and Path(path_hint).is_file():
                raw=Path(path_hint).read_bytes()
            else:
                raise SystemExit("Could not access original image bytes for row "+str(idx)+"; stop rather than silently re-encode.")
        label=record["label"]
        if isinstance(label,bool): label=int(label)
        if isinstance(label,str):
            label=label.strip().lower()
            if label in ("real","0"): label=0
            elif label in ("fake","generated","ai-generated","1"): label=1
            else: raise SystemExit("Unknown class label "+repr(label))
        if int(label) not in (0,1): raise SystemExit("Unexpected label "+str(label))
        with PILImage.open(io.BytesIO(raw)) as im:
            fmt=(im.format or "").upper()
            im.verify()
        if fmt not in EXT: raise SystemExit("Unsupported image encoding: "+fmt)
        digest=hashlib.sha256(raw).hexdigest()
        if digest in sha_seen:
            raise SystemExit("Exact duplicate image at row "+str(idx)+"; investigate before evaluation.")
        sha_seen.add(digest)
        out=image_dir/f"{idx:05d}{EXT[fmt]}"
        if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest()!=digest:
            raise SystemExit("Output file conflict: "+str(out))
        if not out.exists(): out.write_bytes(raw)
        gt="real" if int(label)==0 else "fake"
        # Only use an explicitly supplied generator field. Never infer model identity from label.
        generator=record.get("generator") or record.get("source_model") or record.get("model_name") or ""
        generator=str(generator) if generator is not None else ""
        if gt=="real": generator="real"
        rows.append({"absolute_path":str(out.resolve()),"relative_path":"images/"+out.name,"ground_truth":gt,"generator":generator,"sha256":digest})
        counts[gt]+=1
        if (idx+1)%500==0: print("Prepared",idx+1,dict(counts),flush=True)
    if len(rows)!=args.expected:
        raise SystemExit(f"Expected {args.expected} rows, found {len(rows)}. No evaluation CSV written.")
    if counts["real"]==0 or counts["fake"]==0: raise SystemExit("Missing real or fake class; no CSV written.")
    with csvpath.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (root/"manifest.json").write_text(json.dumps({"repo":DATASET,"split":"train (external testing only)","count":len(rows),"class_counts":dict(counts),"note":"Unseen external dataset. All labeled rows used. Generator breakdown only if explicit metadata present; inspect source files before claiming generator-level results."},indent=2))
    print("READY",csvpath,len(rows),dict(counts),flush=True)
if __name__=="__main__":main()

#!/usr/bin/env python3
"""Prepare a balanced external test from the validation split of Tiny-GenImage."""
import argparse, csv, hashlib, io, json
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps

SOURCE = "TheKernel01/Tiny-GenImage"
GENS = {1:"ADM", 2:"BigGAN"}
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",default=str(Path.home()/"fakebench/data_external_genimage"))
    a=parser.parse_args()
    from datasets import load_dataset
    root=Path(a.out)
    target={"real":500,"ADM":250,"BigGAN":250}
    if (root/"genimage_external.csv").exists():
        raise SystemExit("Existing external CSV found; refusing to overwrite")
    image_root=root/"images"
    image_root.mkdir(parents=True,exist_ok=True)
    counts=Counter(); hashes=set(); rows=[]
    dataset=load_dataset(SOURCE,split="validation",streaming=True)
    for sample in dataset:
        label=int(sample["label"]); gen=int(sample["generator"])
        group="real" if label==0 else GENS.get(gen)
        if group is None or counts[group]>=target[group]: continue
        pic=ImageOps.exif_transpose(sample["image"]).convert("RGB")
        buf=io.BytesIO(); pic.save(buf,format="PNG"); raw=buf.getvalue()
        digest=hashlib.sha256(raw).hexdigest()
        if digest in hashes: continue
        hashes.add(digest)
        filename=f"{group}_{counts[group]:04d}.png"
        dest=image_root/filename
        dest.write_bytes(raw)
        rows.append(dict(absolute_path=str(dest.resolve()),relative_path="images/"+filename,ground_truth="real" if label==0 else "fake",generator=group,sha256_png=digest))
        counts[group]+=1
        if all(counts[g]==target[g] for g in target): break
    if any(counts[g]!=target[g] for g in target):
        raise SystemExit(f"Insufficient samples; counts={dict(counts)} (no evaluation CSV written)")
    with (root/"genimage_external.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest={"source":SOURCE,"source_split":"validation","selection":"first matching unique images in streaming order","counts":dict(counts),"caution":"Third-party curated subset; audit for overlap with FakeBench and dataset-specific shortcuts."}
    (root/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2))
    print("Sample saved:",root/"genimage_external.csv",dict(counts))
if __name__=="__main__": main()

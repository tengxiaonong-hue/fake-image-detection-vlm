#!/usr/bin/env python3
"""Prepare a 2026 multimodal-generator external evaluation dataset.

Dataset labels are THREE classes: GPT-Image2-fake, Nano-Banana2-fake, real.
This preparation maps both generated classes to binary 'fake'.
It does NOT train or tune the detector.
"""
import argparse, csv, hashlib, json
from collections import Counter
from pathlib import Path
from PIL import ImageOps
from datasets import load_dataset

REPO = "zr-zhang/MLLM-Generated-Image-Detection-Dataset"
EXPECTED = 4356

def classify(label_name):
    t = str(label_name).strip().lower().replace("_", "-")
    if t in ("real", "real-image", "real-images", "natural", "authentic"):
        return "real", "real"
    if "fake" in t or "generated" in t or "gpt-image" in t or "nano-banana" in t:
        if "gpt" in t:
            return "fake", "GPT-Image2"
        if "banana" in t or "gemini" in t:
            return "fake", "Nano-Banana2"
        return "fake", label_name
    raise ValueError("Unrecognized label name: " + repr(label_name))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "fakebench/data_external_mllm2026"))
    ap.add_argument("--expected", type=int, default=EXPECTED)
    args=ap.parse_args()
    root=Path(args.out)
    outcsv=root/"mllm2026_external.csv"
    if outcsv.exists():
        raise SystemExit("Prepared CSV already exists; refusing to overwrite.")
    image_dir=root/"images"
    image_dir.mkdir(parents=True,exist_ok=True)
    ds=load_dataset(REPO,split="train",streaming=True)
    if "image" not in ds.column_names or "label" not in ds.column_names:
        raise SystemExit("Dataset schema changed. Columns: "+repr(ds.column_names))
    label_feature=ds.features["label"]
    if not getattr(label_feature,"names",None):
        raise SystemExit("Expected class names; cannot safely infer numeric label mapping.")
    mapping={i:classify(name) for i,name in enumerate(label_feature.names)}
    if set(gt for gt,_ in mapping.values()) != {"real","fake"}:
        raise SystemExit("Missing real or fake label in class mapping: "+repr(mapping))
    print("Verified label mapping:",mapping,flush=True)

    rows=[];counts=Counter();raw_names=Counter();hashes=Counter()
    for i,rec in enumerate(ds):
        label=int(rec["label"])
        if label not in mapping: raise SystemExit("Unexpected label "+str(label))
        gt,generator=mapping[label]
        picture=rec["image"]
        if picture is None:
            raise SystemExit("Missing image at row "+str(i))
        # Streaming delivers decoded PIL images even when original bytes are
        # inaccessible. PNG preserves the decoded RGB pixels losslessly.
        img=ImageOps.exif_transpose(picture).convert("RGB")
        fn=f"{i:05d}.png"
        dest=image_dir/fn
        if dest.exists():
            raise SystemExit("Existing image file "+str(dest)+"; refusing to overwrite.")
        dest.parent.mkdir(parents=True,exist_ok=True)
        img.save(dest,format="PNG")
        digest=hashlib.sha256(dest.read_bytes()).hexdigest()
        hashes[digest]+=1
        rows.append(dict(absolute_path=str(dest.resolve()),relative_path="images/"+fn,
                         ground_truth=gt,generator=generator,sha256_png=digest))
        counts[gt]+=1
        raw_names[label_feature.int2str(label)]+=1
        if (i+1)%500==0:
            print("Prepared",i+1,dict(raw_names),flush=True)

    if len(rows)!=args.expected:
        raise SystemExit(f"Expected {args.expected} rows, found {len(rows)}. CSV not written.")
    if counts["real"]==0 or counts["fake"]==0:
        raise SystemExit("Missing binary class; CSV not written.")
    with outcsv.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest=dict(source=REPO,source_split="train (used ONLY for external testing)",
                  rows=len(rows),original_class_counts=dict(raw_names),
                  binary_class_counts=dict(counts),label_mapping={str(k):v for k,v in mapping.items()},
                  duplicate_decoded_images=sum(v-1 for v in hashes.values()),
                  note="Images saved as lossless PNG from decoded RGB pixel data because streaming did not expose original file bytes. No resize or crop at preparation; evaluation applies fixed V2 224px resize/crop/JPEG. Potential cross-dataset overlap and metadata shortcuts remain unaudited.")
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("READY",outcsv,"N=",len(rows),"classes=",dict(raw_names),flush=True)
if __name__=="__main__":
    main()

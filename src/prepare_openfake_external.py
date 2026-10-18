#!/usr/bin/env python3
"""Prepare a balanced, labeled 5k sample from OpenFake v2 core/test.
No LoRA training. Labels are literal 'real' / 'fake' (verified dataset schema).
"""
import argparse, csv, hashlib, io, json
from pathlib import Path
from collections import Counter
from PIL import Image as PILImage, ImageOps
from datasets import load_dataset

REPO="ComplexDataLab/OpenFake"
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default=str(Path.home()/"fakebench/data_external_openfake"))
    ap.add_argument("--per-class",type=int,default=2500)
    args=ap.parse_args()
    root=Path(args.out); root.mkdir(parents=True,exist_ok=True)
    destcsv=root/"openfake_external.csv"
    if destcsv.exists(): raise SystemExit("Existing CSV found; refusing to overwrite.")
    outdir=root/"images"; outdir.mkdir(exist_ok=True)
    ds=load_dataset(REPO,"core",split="test",streaming=True)
    required={"image","label","model"}
    if not required.issubset(set(ds.column_names)):
        raise SystemExit(f"Unexpected schema: {ds.column_names}")
    print("Schema verified:",ds.column_names,flush=True)
    n=Counter(); gens=Counter();rows=[]; hashes=set()
    for i,record in enumerate(ds):
        label=str(record["label"]).lower().strip()
        if label not in ("real","fake"):
            raise SystemExit("Unexpected binary label: "+repr(label))
        if n[label]>=args.per_class: continue
        im=record["image"]
        if not isinstance(im,PILImage.Image):
            raise SystemExit("Unexpected image object; stop without fabricating labels")
        # Preserve decoded RGB pixels losslessly, then evaluation uses the fixed 224px V2 pipeline.
        im=ImageOps.exif_transpose(im).convert("RGB")
        bio=io.BytesIO(); im.save(bio,format="PNG")
        raw=bio.getvalue()
        digest=hashlib.sha256(raw).hexdigest()
        if digest in hashes: continue
        hashes.add(digest)
        filename=f"{label}_{n[label]:05d}.png"
        target=outdir/filename
        if target.exists(): raise SystemExit("File conflict: "+str(target))
        target.write_bytes(raw)
        gen=str(record.get("model") or "unknown")
        rows.append({"absolute_path":str(target.resolve()),"relative_path":"images/"+filename,"ground_truth":label,"generator":gen,"sha256_png":digest})
        n[label]+=1
        gens[gen]+=1
        if len(rows)%250==0:print("Prepared",len(rows),dict(n),flush=True)
        if n["real"]==n["fake"]==args.per_class:break
    if n["real"]!=args.per_class or n["fake"]!=args.per_class:
        raise SystemExit("Insufficient samples in test split: "+str(dict(n))+". CSV not written.")
    with destcsv.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    manifest={"source":REPO,"config":"core","split":"test","sampling":"first unique images per class (stream order)","class_counts":dict(n),"model_counts":dict(gens),"total":len(rows),"notes":"Saved decoded RGB pixels in PNG; original encoded bytes not preserved. No resize until V2 eval. This is a curated 5k subset; same-generator coverage not guaranteed."}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("READY",destcsv,"N=",len(rows),"classes=",dict(n),flush=True)
if __name__=="__main__":main()

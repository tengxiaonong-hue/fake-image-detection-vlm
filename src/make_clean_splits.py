#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leakage-aware splits for FakeBench LoRA fine-tuning.

What it fixes compared with the old make_splits():
  1. Near-duplicate images are grouped (perceptual dHash, Hamming <= --hamming) and a
     whole group always lands in ONE split, so train/val/test never share near-copies.
  2. Optional --holdout-generators: every fake from those generators goes to test_ood and
     is excluded from train/val (groups that touch them are dropped from train/val too).
  3. Optional --old-test: keep the original evaluation_sample_500.csv as test_id, and drop
     every near-duplicate of it from train/val, so the old 500-image number can be re-measured cleanly.
  4. Shortcut check: a tiny classifier on width/height/format/file-size only. If it scores far above
     50%, the data (not the image content) is giving the label away.

Outputs (in --out): train.csv, val.csv, test_id.csv, test_ood.csv, hashes.csv, split_report.txt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(path, size=8):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int(np.packbits(bits).view(np.uint64)[0])


def compute_hashes(meta, cache):
    cached = {}
    if cache.exists():
        c = pd.read_csv(cache, dtype={"hash": str})
        cached = dict(zip(c["relative_path"], c["hash"]))
    out = []
    for i, (rel, p) in enumerate(zip(meta["relative_path"], meta["absolute_path"])):
        if rel in cached:
            out.append(cached[rel])
            continue
        try:
            out.append(str(dhash(p)))
        except Exception:
            out.append("")
        if (i + 1) % 2000 == 0:
            print(f"hashed {i + 1}/{len(meta)}", flush=True)
    pd.DataFrame({"relative_path": meta["relative_path"], "hash": out}).to_csv(cache, index=False)
    return out


def cluster(hashes, thr, chunk=128):
    """Connected components of the 'Hamming distance <= thr' graph."""
    h = np.array([int(x) for x in hashes], dtype=np.uint64)
    n = len(h)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s in range(0, n, chunk):
        x = np.bitwise_xor(h[s:s + chunk, None], h[None, :])
        d = POP[x.view(np.uint8).reshape(x.shape[0], x.shape[1], 8)].sum(-1)
        ii, jj = np.nonzero(d <= thr)
        for i, j in zip(ii, jj):
            if s + i < j:
                a, b = find(s + i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    return np.array([find(i) for i in range(n)])


def balanced(df, n_total, seed):
    """Equal real/fake, at most n_total rows."""
    k = min(n_total // 2, (df["ground_truth"] == "real").sum(), (df["ground_truth"] == "fake").sum())
    parts = []
    for lab in ("real", "fake"):
        g = df[df["ground_truth"] == lab]
        parts.append(g.sample(k, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def take_groups(order, sizes, target):
    """Walk shuffled groups and take whole groups until `target` images are collected."""
    chosen, total = [], 0
    for g in order:
        if total >= target:
            break
        chosen.append(g)
        total += sizes[g]
    return set(chosen)


def shortcut_check(splits, log):
    try:
        from sklearn.ensemble import GradientBoostingClassifier
    except ImportError:
        log("shortcut check skipped (scikit-learn missing)")
        return

    import os

    def feats(df):
        size = [os.path.getsize(p) if os.path.exists(p) else np.nan for p in df["absolute_path"]]
        f = pd.DataFrame({
            "w": df["width"], "h": df["height"], "ar": df["width"] / df["height"],
            "bytes": size, "bpp": np.array(size) / (df["width"] * df["height"]),
        })
        for fmt in ("JPEG", "PNG", "WEBP"):
            f[f"fmt_{fmt}"] = (df["format"] == fmt).astype(int)
        return f.fillna(0)

    tr = splits["train"]
    clf = GradientBoostingClassifier(random_state=0).fit(feats(tr), tr["ground_truth"] == "fake")
    log("\nShortcut check (metadata only: size/format/bytes; chance = 50%):")
    for name in ("val", "test_id", "test_ood"):
        df = splits.get(name)
        if df is not None and len(df):
            acc = (clf.predict(feats(df)) == (df["ground_truth"] == "fake")).mean()
            log(f"  {name:9s} accuracy from metadata alone: {acc:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="results/metadata.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--old-test", default=None, help="results/evaluation_sample_500.csv (kept as test_id)")
    ap.add_argument("--holdout-generators", nargs="*", default=[])
    ap.add_argument("--n-train", type=int, default=4500)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--hamming", type=int, default=6, help="dHash distance regarded as 'same image'")
    ap.add_argument("--seed", type=int, default=36105)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    meta = pd.read_csv(a.metadata)
    meta = meta[meta["status"] == "ok"].reset_index(drop=True)
    meta["hash"] = compute_hashes(meta, out / "hashes.csv")
    meta = meta[meta["hash"] != ""].reset_index(drop=True)
    meta["group"] = cluster(meta["hash"].tolist(), a.hamming)

    sizes = meta.groupby("group").size().to_dict()
    n_dup = sum(v - 1 for v in sizes.values())
    mixed = meta.groupby("group")["ground_truth"].nunique()
    log(f"images={len(meta)}  groups={len(sizes)}  near-duplicate images={n_dup}")
    log(f"groups mixing real and fake (e.g. edited/derived pairs): {(mixed > 1).sum()}")
    log("fake generators:\n" + meta[meta.ground_truth == "fake"]["generator"].value_counts().to_string())

    rng = np.random.RandomState(a.seed)
    used = np.zeros(len(meta), dtype=bool)  # rows that are allowed in some split
    splits = {}

    # --- test_ood: held-out generators --------------------------------------------------
    drop_groups = set()
    if a.holdout_generators:
        is_ho = meta["generator"].isin(a.holdout_generators)
        ood_fake = meta[is_ho]
        drop_groups |= set(ood_fake["group"])  # nothing from these groups may reach train/val
    # --- test_id ------------------------------------------------------------------------
    if a.old_test:
        old = pd.read_csv(a.old_test)
        keep = meta["relative_path"].isin(set(old["relative_path"].astype(str)))
        test_id = meta[keep].reset_index(drop=True)
        # old-test images that belong to a held-out generator are not "in-distribution": they go to test_ood
        n_before = len(test_id)
        test_id = test_id[~test_id["group"].isin(drop_groups)].reset_index(drop=True)
        test_groups = set(test_id["group"])
        log(f"old test kept as test_id: {len(test_id)}/{n_before} images "
            f"({n_before - len(test_id)} moved out because of held-out generators), {len(test_groups)} groups")
    else:
        test_id, test_groups = None, set()

    # --- random group split of everything else -------------------------------------------
    pool = meta[~meta["group"].isin(drop_groups | test_groups)]
    order = list(rng.permutation(sorted(set(pool["group"]))))
    psizes = pool.groupby("group").size().to_dict()
    if test_id is None:
        g = take_groups(order, psizes, int(a.n_test * 1.4))
        test_id = balanced(pool[pool["group"].isin(g)], a.n_test, a.seed)
        order = [x for x in order if x not in g]
    g_val = take_groups(order, psizes, int(a.n_val * 1.4))
    val = balanced(pool[pool["group"].isin(g_val)], a.n_val, a.seed)
    # Only groups that actually contributed to val/test_id are closed to train;
    # rows that were over-sampled but not selected go back to train (no data is wasted).
    closed = set(val["group"]) | set(test_id["group"])
    train = balanced(pool[~pool["group"].isin(closed)], a.n_train, a.seed)

    splits = {"train": train, "val": val, "test_id": test_id}

    if a.holdout_generators:
        fake = meta[meta["generator"].isin(a.holdout_generators)]
        # real images for the OOD test: unused real groups, disjoint from everything above
        taken = set(train["group"]) | set(val["group"]) | set(test_id["group"]) | set(fake["group"])
        real_pool = meta[(meta["ground_truth"] == "real") & ~meta["group"].isin(taken)]
        n_f = min(len(fake), a.n_test // 2)
        ood = pd.concat([
            fake.sample(n_f, random_state=a.seed),
            real_pool.sample(min(n_f, len(real_pool)), random_state=a.seed),
        ]).sample(frac=1, random_state=a.seed).reset_index(drop=True)
        splits["test_ood"] = ood
        if len(real_pool) < n_f:
            log(f"WARNING: only {len(real_pool)} unused real images left for test_ood (wanted {n_f}); "
                "test_ood is not balanced - lower --n-train to leave more real images unused.")

    # --- hard guarantees -----------------------------------------------------------------
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = set(splits[names[i]]["group"]) & set(splits[names[j]]["group"])
            assert not shared, f"group leak between {names[i]} and {names[j]}: {len(shared)}"
    if a.holdout_generators:
        for n in ("train", "val", "test_id"):
            assert not splits[n]["generator"].isin(a.holdout_generators).any(), f"holdout generator in {n}"

    log("\nSplit sizes:")
    for n, df in splits.items():
        df.to_csv(out / f"{n}.csv", index=False)
        log(f"  {n:9s} {len(df):5d}  {df['ground_truth'].value_counts().to_dict()}  "
            f"generators={df[df.ground_truth == 'fake']['generator'].nunique()}")

    shortcut_check(splits, log)
    (out / "split_report.txt").write_text("\n".join(lines))
    print(f"\nWrote splits to {out}")


if __name__ == "__main__":
    sys.exit(main())

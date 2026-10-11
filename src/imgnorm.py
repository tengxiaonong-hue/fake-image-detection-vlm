# -*- coding: utf-8 -*-
"""
Uniform image preprocessing so the model cannot read the label from resolution / file format / compression.
Every image (real or fake) goes through exactly the same steps:
  1. RGB, EXIF-rotate
  2. resize shorter side to `size`, then crop `size` x `size`   (train: random crop, eval: center crop)
  3. re-encode as JPEG (train: random quality 70-95, eval: fixed quality 85)
=> identical resolution (so identical number of visual tokens) and identical container/compression for both classes.
"""
import io
import random

from PIL import Image, ImageOps

EVAL_JPEG_QUALITY = 85


def load_normalized(path, size=448, train=False):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    if not size:
        return im
    w, h = im.size
    s = size / min(w, h)
    im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.Resampling.LANCZOS)
    w, h = im.size
    if train:
        left, top = random.randint(0, w - size), random.randint(0, h - size)
    else:
        left, top = (w - size) // 2, (h - size) // 2
    im = im.crop((left, top, left + size, top + size))
    q = random.randint(70, 95) if train else EVAL_JPEG_QUALITY
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert("RGB")

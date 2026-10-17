#!/bin/bash
#PBS -N four_ablations
#PBS -q large_gpuq
#PBS -l ncpus=6
#PBS -l ngpus=1
#PBS -l mem=48GB
#PBS -l walltime=06:00:00
#PBS -j oe
# Reference reconstruction of the submitted four-ablation batch.
# Check the HPC job log and generated scripts for exact execution provenance.
# WARNING: all 4 runs may exceed this queue's six-hour walltime.
set -euo pipefail
cd "$HOME/fakebench"
source "$HOME/miniconda3/bin/activate"
conda activate fakebench
export HF_HOME="$HOME/fakebench/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi
mkdir -p results/ablation_four_audit

python - <<'PY'
from pathlib import Path
import hashlib
import pandas as pd

base_train = Path("src/train_qwen3b_lora_v2.py").read_text()
base_norm = Path("src/imgnorm.py").read_text()
assert "from imgnorm import load_normalized" in base_train
assert "lora_alpha=2 * a.rank" in base_train
crop = "left, top = random.randint(0, w - size), random.randint(0, h - size)"
quality = "q = random.randint(70, 95) if train else EVAL_JPEG_QUALITY"
jpeg = '    buf = io.BytesIO()\n    im.save(buf, format="JPEG", quality=q)\n    buf.seek(0)\n    out = Image.open(buf)\n    out.load()\n    return out.convert("RGB")'
assert crop in base_norm and quality in base_norm and jpeg in base_norm

splits = Path("data_splits_v2")
counts = {"train": 3537, "val": 1000, "test_id": 449, "test_ood": 500}
with Path("results/ablation_four_audit/inputs_sha256.txt").open("w") as out:
    for split, n in counts.items():
        p = splits / (split + ".csv")
        assert p.exists() and len(pd.read_csv(p)) == n
        out.write(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + str(p) + "\n")

variants = {
    "no_jpeg_quality": base_norm.replace(quality, "q = EVAL_JPEG_QUALITY"),
    "no_random_crop": base_norm.replace(crop, "left, top = (w - size) // 2, (h - size) // 2"),
    "no_jpeg_reencode": base_norm.replace(jpeg, '    return im.convert("RGB")'),
}
for name, code in variants.items():
    assert code != base_norm
    module = "imgnorm_ablate_" + name
    Path("src/" + module + ".py").write_text(code)
    Path("src/train_qwen3b_lora_v2_ablate_" + name + ".py").write_text(
        base_train.replace("from imgnorm import load_normalized", "from " + module + " import load_normalized")
    )
rank_code = base_train.replace("lora_alpha=2 * a.rank", "lora_alpha=32")
assert rank_code != base_train
Path("src/train_qwen3b_lora_v2_ablate_rank8.py").write_text(rank_code)
PY

run_variant() {
  name="$1"
  rank="$2"
  train_script="src/train_qwen3b_lora_v2_ablate_${name}.py"
  model_dir="$HOME/fakebench/models/ablation_${name}"
  result_dir="$HOME/fakebench/results/ablation_${name}"
  if [ -e "$model_dir" ] || [ -e "$result_dir" ]; then
    echo "Refusing to overwrite: $name" >&2
    exit 1
  fi
  python "$train_script" --splits "$HOME/fakebench/data_splits_v2" \
    --out "$model_dir" --epochs 3 --lr 1e-4 --batch-size 2 \
    --grad-accum 8 --rank "$rank" --seed 36105 --norm-size 224
  python src/eval_qwen3b_lora_v2.py --splits "$HOME/fakebench/data_splits_v2" \
    --adapter "$model_dir/best_adapter" --out "$result_dir" \
    --norm-size 224 --skip-base
}
run_variant no_jpeg_quality 16
run_variant no_random_crop 16
run_variant rank8 8
run_variant no_jpeg_reencode 16
echo "FOUR ABLATIONS COMPLETED"

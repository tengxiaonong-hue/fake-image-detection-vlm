#!/bin/bash
#PBS -N external_mllm26
#PBS -q large_gpuq
#PBS -l ncpus=6
#PBS -l ngpus=1
#PBS -l mem=48GB
#PBS -l walltime=06:00:00
#PBS -j oe
set -euo pipefail
cd "$HOME/fakebench"
source "$HOME/miniconda3/bin/activate"
conda activate fakebench
export HF_HOME="$HOME/fakebench/hf_cache"
export TOKENIZERS_PARALLELISM=false
CSV="$HOME/fakebench/data_external_mllm2026/mllm2026_external.csv"
ADAPTER="$HOME/fakebench/models/qwen25vl_3b_lora_v2/best_adapter"
OUT="$HOME/fakebench/results/external_mllm2026"
[ -s "$CSV" ] || { echo "ERROR: prepare dataset CSV first" >&2; exit 1; }
[ -d "$ADAPTER" ] || { echo "ERROR: original V2 adapter missing" >&2; exit 1; }
[ ! -e "$OUT/summary.csv" ] || { echo "ERROR: results already exist; refusing overwrite" >&2; exit 1; }
python - <<'PY'
import pandas as pd
from pathlib import Path
f=Path.home()/"fakebench/data_external_mllm2026/mllm2026_external.csv"
d=pd.read_csv(f)
assert len(d)==4356, f"Expected 4356 images, got {len(d)}"
assert set(d["ground_truth"].str.lower())=={"real","fake"}
assert all(Path(p).is_file() for p in d["absolute_path"])
print("External evaluation:",len(d),d["ground_truth"].value_counts().to_dict(),flush=True)
PY
nvidia-smi
python src/eval_qwen3b_lora_v2.py \
 --splits "$HOME/fakebench/data_external_mllm2026" \
 --tests mllm2026_external \
 --adapter "$ADAPTER" \
 --out "$OUT" \
 --norm-size 224 --skip-base
echo "EXTERNAL EVALUATION COMPLETED"

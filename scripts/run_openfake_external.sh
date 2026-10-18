#!/bin/bash
#PBS -N openfake_ext
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
CSV="$HOME/fakebench/data_external_openfake/openfake_external.csv"
ADAPTER="$HOME/fakebench/models/qwen25vl_3b_lora_v2/best_adapter"
OUT="$HOME/fakebench/results/openfake_external"
[ -s "$CSV" ] || { echo "Missing prepared OpenFake CSV" >&2; exit 1; }
[ -d "$ADAPTER" ] || { echo "Missing original V2 adapter" >&2; exit 1; }
[ ! -e "$OUT/summary.csv" ] || { echo "Existing result; refusing overwrite" >&2; exit 1; }
python - <<'PY'
from pathlib import Path
import pandas as pd
d=pd.read_csv(Path.home()/"fakebench/data_external_openfake/openfake_external.csv")
assert len(d)==5000,len(d)
assert d.ground_truth.value_counts().to_dict()=={"real":2500,"fake":2500}
assert all(Path(x).is_file() for x in d.absolute_path)
print("Input validated:",len(d),d.ground_truth.value_counts().to_dict(),flush=True)
PY
nvidia-smi
python src/eval_qwen3b_lora_v2.py \
 --splits "$HOME/fakebench/data_external_openfake" \
 --tests openfake_external \
 --adapter "$ADAPTER" \
 --out "$OUT" \
 --norm-size 224 --skip-base
echo "OPENFAKE EXTERNAL TEST FINISHED"

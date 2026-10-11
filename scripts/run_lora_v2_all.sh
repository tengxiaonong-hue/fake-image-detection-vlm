#!/bin/bash
#PBS -N lora_v2
#PBS -q large_gpuq
#PBS -l ncpus=6
#PBS -l ngpus=1
#PBS -l mem=48GB
#PBS -l walltime=08:00:00
#PBS -j oe

# One job = clean splits -> LoRA training -> base-vs-LoRA evaluation.
# Submit from ~/fakebench:
#   qsub scripts/run_lora_v2_all.sh
#   qsub -v HOLDOUT="sdxl midjourney" scripts/run_lora_v2_all.sh     # also builds the held-out-generator test
# Optional env vars: EPOCHS (default 3), NORM_SIZE (default 448; 0 = off), FORCE_SPLITS=1 (rebuild splits), HAMMING (default 6)

cd "$HOME/fakebench" || exit 1
source "$HOME/miniconda3/bin/activate"
conda activate fakebench

export HF_HOME="$HOME/fakebench/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EPOCHS="${EPOCHS:-3}"
HAMMING="${HAMMING:-6}"
NORM_SIZE="${NORM_SIZE:-448}"   # uniform resize + JPEG for ALL images (0 = off)
SPLITS="$HOME/fakebench/data_splits_v2"
MODELS="$HOME/fakebench/models/qwen25vl_3b_lora_v2"
RES="$HOME/fakebench/results/v2"
mkdir -p "$SPLITS" "$MODELS" "$RES"

echo "===== START $(date) on $(hostname) ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "===== 1/3 CLEAN SPLITS ====="
if [ ! -f "$SPLITS/train.csv" ] || [ -n "$FORCE_SPLITS" ]; then
  ARGS=(--metadata "$HOME/fakebench/results/metadata.csv" --out "$SPLITS" --hamming "$HAMMING")
  [ -f "$HOME/fakebench/results/evaluation_sample_500.csv" ] && ARGS+=(--old-test "$HOME/fakebench/results/evaluation_sample_500.csv")
  [ -n "$HOLDOUT" ] && ARGS+=(--holdout-generators $HOLDOUT)
  python src/make_clean_splits.py "${ARGS[@]}" || exit 1
else
  echo "splits already exist in $SPLITS (set FORCE_SPLITS=1 to rebuild)"
fi
cp "$SPLITS/split_report.txt" "$RES/" 2>/dev/null

echo "===== 2/3 TRAIN ====="
python src/train_qwen3b_lora_v2.py --splits "$SPLITS" --out "$MODELS" --epochs "$EPOCHS" --norm-size "$NORM_SIZE" || exit 1

echo "===== 3/3 EVAL ====="
python src/eval_qwen3b_lora_v2.py --splits "$SPLITS" --adapter "$MODELS/best_adapter" --out "$RES" --norm-size "$NORM_SIZE" || exit 1

echo "===== END $(date) ====="

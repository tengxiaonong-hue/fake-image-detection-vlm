#!/bin/bash
#PBS -N clean_splits
#PBS -q large_gpuq
#PBS -l ncpus=4
#PBS -l ngpus=1
#PBS -l mem=16GB
#PBS -l walltime=01:00:00
#PBS -j oe

# Only builds the splits and prints the report (generator list, near-duplicate count, shortcut check).
# Does not use the GPU; the GPU line is only there because that is the queue that is known to work.
#   qsub scripts/run_clean_splits_only.sh
#   qsub -v HOLDOUT="sdxl midjourney" scripts/run_clean_splits_only.sh

cd "$HOME/fakebench" || exit 1
source "$HOME/miniconda3/bin/activate"
conda activate fakebench

SPLITS="$HOME/fakebench/data_splits_v2"
mkdir -p "$SPLITS"
ARGS=(--metadata "$HOME/fakebench/results/metadata.csv" --out "$SPLITS")
[ -f "$HOME/fakebench/results/evaluation_sample_500.csv" ] && ARGS+=(--old-test "$HOME/fakebench/results/evaluation_sample_500.csv")
[ -n "$HOLDOUT" ] && ARGS+=(--holdout-generators $HOLDOUT)
python src/make_clean_splits.py "${ARGS[@]}"

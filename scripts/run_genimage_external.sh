#!/bin/bash
#PBS -N genimage_external
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
nvidia-smi
[ -s "$HOME/fakebench/data_external_genimage/genimage_external.csv" ] || { echo "Prepare external CSV first" >&2; exit 1; }
[ -d "$HOME/fakebench/models/qwen25vl_3b_lora_v2/best_adapter" ] || { echo "Missing original V2 best adapter" >&2; exit 1; }
python src/eval_qwen3b_lora_v2.py \
 --splits "$HOME/fakebench/data_external_genimage" \
 --tests genimage_external \
 --adapter "$HOME/fakebench/models/qwen25vl_3b_lora_v2/best_adapter" \
 --out "$HOME/fakebench/results/genimage_external" \
 --norm-size 224 --skip-base

#!/bin/bash
#PBS -N qwen3b_lora
#PBS -q large_gpuq
#PBS -l ncpus=6
#PBS -l ngpus=1
#PBS -l mem=32GB
#PBS -l walltime=06:00:00
#PBS -j oe

cd "$HOME/fakebench" || exit 1
source "$HOME/miniconda3/bin/activate"
conda activate fakebench

export HF_HOME="$HOME/fakebench/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p models data_splits results hf_cache

echo "===== LORA TRAINING START ====="
date
hostname
nvidia-smi

python src/train_qwen3b_lora.py

echo "===== LORA TRAINING END ====="
date

#!/bin/bash
#PBS -N qwen3b_lora_eval
#PBS -q large_gpuq
#PBS -l ncpus=4
#PBS -l ngpus=1
#PBS -l mem=32GB
#PBS -l walltime=02:00:00
#PBS -j oe

cd "$HOME/fakebench" || exit 1
source "$HOME/miniconda3/bin/activate"
conda activate fakebench

echo "Start: $(date)"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python src/eval_qwen3b_lora.py

echo "End: $(date)"

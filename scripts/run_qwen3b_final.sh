#!/bin/bash
#PBS -N qwen25vl_3b_final
#PBS -q large_gpuq
#PBS -l ncpus=2
#PBS -l ngpus=1
#PBS -l mem=32GB
#PBS -l walltime=06:00:00
#PBS -j oe

cd $HOME/fakebench
source $HOME/miniconda3/bin/activate
conda activate fakebench

export HF_HOME=$HOME/fakebench/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p results hf_cache work

echo "===== JOB START ====="
date
hostname
nvidia-smi

if [ -f results/qwen25vl_3b_results_500.csv ]; then
    ts=$(date +%Y%m%d_%H%M%S)
    cp results/qwen25vl_3b_results_500.csv results/qwen25vl_3b_results_500_before_final_${ts}.csv
fi

python qwen3b_final.py

echo "===== JOB END ====="
date

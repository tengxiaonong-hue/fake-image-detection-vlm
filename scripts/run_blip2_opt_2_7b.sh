#!/bin/bash
#PBS -N blip2_opt_2_7b
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
mkdir -p results hf_cache

echo "===== JOB START ====="
date
hostname
nvidia-smi

python fakebench_hpc.py --model blip2_opt_2_7b --n-real 250 --n-fake 250

echo "===== JOB END ====="
date

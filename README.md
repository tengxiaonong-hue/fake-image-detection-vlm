# Explainable Fake Image Detection with Large Multimodal Models

This repository contains experiments for explainable AI-generated image detection using large multimodal models on the FakeBench dataset.

## Models

- Qwen2.5-VL-3B-Instruct
- Qwen2.5-VL-7B-Instruct
- Idefics3-8B-Llama3
- LLaVA-OneVision-7B
- BLIP2-OPT-2.7B

## Prompting Strategies

- Baseline
- Cause-to-Effect (CtE)
- Effect-to-Cause (EtC)

## Dataset

A fixed balanced subset of 500 FakeBench images was used:

- 250 real
- 250 AI-generated

The dataset itself is not included.

## Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Mean inference latency

## Hardware

UTS Cetus HPC cluster using NVIDIA RTX PRO 6000 Blackwell GPUs.

## Repository Structure

- `src/` source code
- `scripts/` PBS job scripts
- `github_results/` final experiment results
- `docs/` experiment notes

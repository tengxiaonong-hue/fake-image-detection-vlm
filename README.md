# Explainable Fake Image Detection with Large Multimodal Models

This repository contains experiments for explainable AI-generated image detection using large multimodal vision-language models on the FakeBench dataset.

The project compares five multimodal models under three prompting strategies:

- Baseline
- Cause-to-Effect (CtE)
- Effect-to-Cause (EtC)

The goal is to evaluate both fake-image detection performance and the effect of reasoning strategy on model behaviour.

## Models

The following models were evaluated:

- Qwen2.5-VL-3B-Instruct
- Qwen2.5-VL-7B-Instruct
- Idefics3-8B-Llama3
- LLaVA-OneVision-7B
- BLIP2-OPT-2.7B

## Dataset

A fixed balanced subset of 500 images from FakeBench was used:

- 250 real images
- 250 AI-generated images

The original dataset is not included in this repository.

## Prompting Strategies

### Baseline
The model directly predicts whether an image is REAL or AI-GENERATED.

### Cause-to-Effect (CtE)
The model first inspects visible image evidence and then predicts the final authenticity label.

### Effect-to-Cause (EtC)
The model first produces an authenticity prediction and then explains the visual evidence supporting that result.

## Evaluation Metrics

Detection performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Mean inference latency

## Experimental Hardware

Experiments were conducted on the UTS Cetus High Performance Computing cluster.

GPU:

- NVIDIA RTX PRO 6000 Blackwell
- 96 GB VRAM

## Main Results

| Model | Strategy | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Qwen2.5-VL-3B | Baseline | **86.80%** | 85.11% | 89.20% | **87.11%** |
| Qwen2.5-VL-3B | CtE | 61.00% | 58.62% | 74.80% | 65.73% |
| Qwen2.5-VL-3B | EtC | 75.60% | 96.38% | 53.20% | 68.56% |
| Qwen2.5-VL-7B | Baseline | 68.00% | 97.87% | 36.80% | 53.49% |
| Qwen2.5-VL-7B | CtE | 66.20% | 67.38% | 62.80% | 65.01% |
| Qwen2.5-VL-7B | EtC | 68.20% | 97.89% | 37.20% | 53.91% |
| Idefics3-8B | Baseline | 71.80% | 96.58% | 45.20% | 61.58% |
| Idefics3-8B | CtE | **72.32%** | 83.83% | 56.00% | **67.15%** |
| Idefics3-8B | EtC | 70.20% | 96.33% | 42.00% | 58.50% |
| LLaVA-OneVision-7B | Baseline | 69.80% | 99.01% | 40.00% | 56.98% |
| LLaVA-OneVision-7B | CtE | 63.18% | 98.48% | 26.32% | 41.53% |
| LLaVA-OneVision-7B | EtC | **80.20%** | 98.71% | 61.20% | **75.56%** |
| BLIP2-OPT-2.7B | Baseline | 50.00% | 0.00% | 0.00% | 0.00% |
| BLIP2-OPT-2.7B | CtE | 50.00% | 0.00% | 0.00% | 0.00% |
| BLIP2-OPT-2.7B | EtC | 50.00% | 50.00% | 100.00% | 66.67% |

## Key Observations

- Qwen2.5-VL-3B achieved the strongest overall baseline result, with 86.8% accuracy and 87.1% F1.
- LLaVA-OneVision-7B benefited substantially from the EtC strategy, improving from 69.8% baseline accuracy to 80.2%.
- Idefics3-8B performed best under the CtE strategy.
- Qwen2.5-VL-7B showed improved fake-image recall under CtE, although overall accuracy did not increase.
- BLIP2-OPT-2.7B showed degenerate prediction behaviour and was substantially weaker than the instruction-tuned multimodal models.
- Larger parameter count did not consistently lead to better fake-image detection performance.

## Runtime

Approximate end-to-end job runtime for the completed 500-image experiments:

| Model | Runtime |
|---|---:|
| Qwen2.5-VL-7B | 1h 07m |
| Idefics3-8B | 35m |
| LLaVA-OneVision-7B | 1h 02m |
| BLIP2-OPT-2.7B | 5m |
| Qwen2.5-VL-3B | Recorded in final runtime CSV |

Detailed runtime and GPU information is available in:

```text
github_results/model_runtime_gpu.csv


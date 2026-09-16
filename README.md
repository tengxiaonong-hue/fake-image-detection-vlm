# Explainable AI-Generated Image Detection with Vision-Language Models

This repository investigates explainable AI-generated image detection using compact and medium-scale vision-language models. The study combines multi-model benchmarking, reasoning-strategy evaluation, and parameter-efficient LoRA fine-tuning of Qwen2.5-VL-3B.

**Model weights:** [NTX88666/Qwen2.5-VL-3B-FakeBench-LoRA on Hugging Face](https://huggingface.co/NTX88666/Qwen2.5-VL-3B-FakeBench-LoRA)

The project compares multiple vision-language models under three reasoning strategies:

- Baseline
- Cause-to-Effect (CtE)
- Effect-to-Cause (EtC)

The focus is not only on detection accuracy, but also on whether reasoning strategy and lightweight fine-tuning can improve performance while maintaining practical computational cost.

## Model Scope and Design Choice

This project intentionally focuses on multimodal large language models with fewer than 14 billion parameters. The selected models mainly fall within the 2.7B–8B range so that experiments remain reproducible under realistic academic computing constraints.

The study asks whether smaller and medium-scale multimodal models can achieve competitive fake-image detection performance through better reasoning, prompting, and parameter-efficient fine-tuning rather than relying purely on model scale.

## Models

- Qwen2.5-VL-3B-Instruct
- Qwen2.5-VL-7B-Instruct
- Idefics3-8B-Llama3
- LLaVA-OneVision-7B
- BLIP2-OPT-2.7B

## Dataset

A fixed balanced subset of 500 images from FakeBench was used for evaluation:

- 250 real images
- 250 AI-generated images

For LoRA fine-tuning, the fixed 500-image evaluation split was excluded from training and validation. The remaining data were split into:

- 4,500 training images
- 1,000 validation images
- 500 held-out test images

The original dataset is not included in this repository.

## Prompting Strategies

### Baseline
The model directly predicts whether an image is REAL or AI-GENERATED.

### Cause-to-Effect (CtE)
The model first inspects visible image evidence and then predicts the final authenticity label.

### Effect-to-Cause (EtC)
The model first produces an authenticity prediction and then explains the visual evidence supporting that result.

## Evaluation Metrics

Detection performance is evaluated using Accuracy, Precision, Recall, F1-score, Matthews Correlation Coefficient (MCC), confusion-matrix counts, mean inference latency, and total runtime.

## Experimental Hardware

Experiments were conducted on the UTS HPC cluster using an NVIDIA RTX PRO 6000 Blackwell GPU with 96 GB VRAM.

## Main Results

| Model | Strategy | N | Accuracy | Precision | Recall | F1 | MCC | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B | Baseline | 500 | **86.80%** | 85.11% | **89.20%** | **87.11%** | **0.7368** | 211 | 39 | 27 | 223 |
| Qwen2.5-VL-3B | EtC | 500 | 75.60% | 96.38% | 53.20% | 68.56% | 0.5727 | 245 | 5 | 117 | 133 |
| Qwen2.5-VL-3B | CtE | 500 | 61.00% | 58.62% | 74.80% | 65.73% | 0.2289 | 118 | 132 | 63 | 187 |
| Qwen2.5-VL-7B | CtE | 500 | 66.20% | 67.38% | 62.80% | 65.01% | 0.3248 | 174 | 76 | 93 | 157 |
| Qwen2.5-VL-7B | EtC | 500 | 68.20% | 97.89% | 37.20% | 53.91% | 0.4639 | 248 | 2 | 157 | 93 |
| Qwen2.5-VL-7B | Baseline | 500 | 68.00% | 97.87% | 36.80% | 53.49% | 0.4607 | 248 | 2 | 158 | 92 |
| LLaVA-OneVision-7B | EtC | 500 | 80.20% | 98.71% | 61.20% | 75.56% | 0.6530 | 248 | 2 | 97 | 153 |
| LLaVA-OneVision-7B | Baseline | 500 | 69.80% | 99.01% | 40.00% | 56.98% | 0.4932 | 249 | 1 | 150 | 100 |
| LLaVA-OneVision-7B | CtE | 497 | 63.18% | 98.48% | 26.32% | 41.53% | 0.3818 | 249 | 1 | 182 | 65 |
| Idefics3-8B | CtE | 495 | 72.32% | 83.83% | 56.00% | 67.15% | 0.4756 | 218 | 27 | 110 | 140 |
| Idefics3-8B | Baseline | 500 | 71.80% | 96.58% | 45.20% | 61.58% | 0.5149 | 246 | 4 | 137 | 113 |
| Idefics3-8B | EtC | 500 | 70.20% | 96.33% | 42.00% | 58.50% | 0.4892 | 246 | 4 | 145 | 105 |
| BLIP2-OPT-2.7B | EtC | 500 | 50.00% | 50.00% | 100.00% | 66.67% | 0.0000 | 0 | 250 | 0 | 250 |
| BLIP2-OPT-2.7B | Baseline | 500 | 50.00% | 0.00% | 0.00% | 0.00% | 0.0000 | 250 | 0 | 250 | 0 |
| BLIP2-OPT-2.7B | CtE | 500 | 50.00% | 0.00% | 0.00% | 0.00% | 0.0000 | 250 | 0 | 250 | 0 |

Full numerical results are available in `github_results/final_5models_with_mcc_confusion.csv`.

## Qwen2.5-VL-3B LoRA Fine-tuning

After the benchmark, Qwen2.5-VL-3B was selected for parameter-efficient fine-tuning using LoRA.

### Fine-tuning Setup

- Base model: Qwen2.5-VL-3B-Instruct
- Training method: LoRA
- Training samples: 4,500
- Validation samples: 1,000
- Held-out test samples: 500
- Epochs: 3
- Best checkpoint: Epoch 2
- Batch size: 2
- Gradient accumulation steps: 8
- Effective batch size: 16
- Learning rate: 1e-4
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Precision: BF16
- Hardware: UTS HPC NVIDIA RTX PRO 6000 Blackwell 96GB

The fixed 500-image test set was excluded from both training and validation.

### Training Results

| Epoch | Train Loss | Validation Loss | Time (min) |
|---|---:|---:|---:|
| 1 | 0.02117 | 0.002447 | 25.20 |
| 2 | 0.001283 | **0.002103** | 36.68 |
| 3 | 0.000268 | 0.002190 | 36.47 |

### LoRA Training Curve

![Qwen2.5-VL-3B LoRA Training Curve](figures/qwen3b_lora_loss_curve.svg)

Validation loss reached its minimum at **Epoch 2**, which was selected as the `best_adapter` checkpoint. Training loss continued to decrease at Epoch 3 while validation loss rose slightly, indicating the beginning of mild overfitting.

Training history is available in `github_results/qwen3b_lora_training_history.csv`.

### Base vs LoRA Evaluation

For a controlled before/after comparison, the original Qwen2.5-VL-3B model and the LoRA-adapted model were evaluated with the **same classification-only prompt** on the same held-out 500-image test set.

| Model | N | Accuracy | Precision | Recall | F1 | MCC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B Base | 500 | 77.40% | 96.60% | 56.80% | 71.54% | 0.6014 | 245 | 5 | 108 | 142 |
| **Qwen2.5-VL-3B + LoRA** | **500** | **99.40%** | **99.60%** | **99.20%** | **99.40%** | **0.9880** | **249** | **1** | **2** | **248** |

Under the controlled classification-only setting, LoRA fine-tuning improved Accuracy by **22.0 percentage points**, Recall by **42.4 percentage points**, and F1 by approximately **27.86 percentage points**. MCC increased from **0.6014 to 0.9880**, while false negatives decreased from **108 to 2**.

> **Important:** The 86.8% Qwen2.5-VL-3B benchmark baseline used the original benchmark prompt, which also requested an explanation. The 77.4% Base result above uses the classification-only prompt used for the controlled LoRA comparison. These values are therefore not directly interchangeable.

> **Generalization note:** The 99.4% result is an in-domain result on a held-out FakeBench split. Cross-dataset and unseen-generator generalization have not yet been established.

The evaluation summary is available in `github_results/qwen3b_base_vs_lora_summary.csv`.

### Fine-tuning Code

- `src/train_qwen3b_lora.py` — LoRA training and split construction
- `src/eval_qwen3b_lora.py` — controlled Base vs LoRA evaluation
- `scripts/run_qwen3b_lora.sh` — PBS GPU training job
- `scripts/run_qwen3b_lora_eval.sh` — PBS GPU evaluation job

### Model Weights

The best LoRA adapter is published on Hugging Face:

**[NTX88666/Qwen2.5-VL-3B-FakeBench-LoRA](https://huggingface.co/NTX88666/Qwen2.5-VL-3B-FakeBench-LoRA)**

The adapter weights are intentionally not duplicated in this GitHub repository.

## Key Observations

- Qwen2.5-VL-3B achieved the strongest overall benchmark baseline result, with 86.8% accuracy and 87.1% F1 under the original benchmark prompt.
- In the controlled classification-only evaluation, LoRA fine-tuning reached 99.4% in-domain accuracy and 0.9880 MCC.
- The LoRA model reduced false negatives from 108 to 2.
- LLaVA-OneVision-7B benefited substantially from the EtC strategy.
- Larger parameter count did not consistently lead to better fake-image detection performance.
- Cross-dataset generalization remains an important next step before making claims about robustness beyond FakeBench.

## Runtime

Approximate end-to-end runtime for completed 500-image benchmark experiments:

| Model | Runtime |
|---|---:|
| Qwen2.5-VL-7B | 1h 07m |
| Idefics3-8B | 35m |
| LLaVA-OneVision-7B | 1h 02m |
| BLIP2-OPT-2.7B | 5m |
| Qwen2.5-VL-3B | Recorded in final runtime CSV |

LoRA fine-tuning took approximately **98 minutes** across three epochs on a single NVIDIA RTX PRO 6000 Blackwell 96GB GPU.

## Result Visualisations

### Accuracy Comparison

![Accuracy Comparison](figures/accuracy_comparison.png)

### F1 Score Comparison

![F1 Comparison](figures/f1_comparison.png)

### Inference Latency

![Latency Comparison](figures/latency_comparison.png)

## Confusion Matrices

### Qwen2.5-VL-3B

| Baseline | CtE | EtC |
| --- | --- | --- |
| ![](github_results/confusion_matrices/qwen25_vl_3b_baseline_cm.png) | ![](github_results/confusion_matrices/qwen25_vl_3b_cause_to_effect_cm.png) | ![](github_results/confusion_matrices/qwen25_vl_3b_effect_to_cause_cm.png) |

### Qwen2.5-VL-7B

| Baseline | CtE | EtC |
| --- | --- | --- |
| ![](github_results/confusion_matrices/qwen25_vl_7b_baseline_cm.png) | ![](github_results/confusion_matrices/qwen25_vl_7b_cause_to_effect_cm.png) | ![](github_results/confusion_matrices/qwen25_vl_7b_effect_to_cause_cm.png) |

### Idefics3-8B

| Baseline | CtE | EtC |
| --- | --- | --- |
| ![](github_results/confusion_matrices/idefics3_8b_baseline_cm.png) | ![](github_results/confusion_matrices/idefics3_8b_cause_to_effect_cm.png) | ![](github_results/confusion_matrices/idefics3_8b_effect_to_cause_cm.png) |

### LLaVA-OneVision-7B

| Baseline | CtE | EtC |
| --- | --- | --- |
| ![](github_results/confusion_matrices/llava_onevision_7b_baseline_cm.png) | ![](github_results/confusion_matrices/llava_onevision_7b_cause_to_effect_cm.png) | ![](github_results/confusion_matrices/llava_onevision_7b_effect_to_cause_cm.png) |

### BLIP2-OPT-2.7B

| Baseline | CtE | EtC |
| --- | --- | --- |
| ![](github_results/confusion_matrices/blip2_opt_27b_baseline_cm.png) | ![](github_results/confusion_matrices/blip2_opt_27b_cause_to_effect_cm.png) | ![](github_results/confusion_matrices/blip2_opt_27b_effect_to_cause_cm.png) |

# Explainable Fake Image Detection using Large Multimodal Models

Benchmarking Reasoning Strategies and LoRA Fine-tuning for AI-Generated Image Detection

This repository investigates explainable AI-generated image detection using compact and medium-scale vision-language models. The study combines multi-model benchmarking, reasoning-strategy evaluation, and parameter-efficient LoRA fine-tuning of Qwen2.5-VL-3B.

**LoRA model weights (Hugging Face):** Two versions are available. **V1** is the initial fine-tuning experiment; **V2** adds a stricter split and held-out-generator testing. Both are LoRA adapters for `Qwen/Qwen2.5-VL-3B-Instruct`, not standalone full models.

- **[LoRA V1 — Original Fine-tuning](https://huggingface.co/NTX88666/Qwen2.5-VL-3B-FakeBench-LoRA)** · 5-epoch follow-up results are documented in this repository; this HF release is the earlier V1 adapter.
- **[LoRA V2 — Leakage-Aware and OOD Evaluation](https://huggingface.co/NTX88666/Qwen2.5-VL-3B-FakeBench-LoRA-V2)** · Best checkpoint: Epoch 3; Test-ID accuracy 93.99%, held-out-generator Test-OOD accuracy 91.00%.

The V1 and V2 scores come from **different evaluation splits/protocols** and should not be compared as a direct before-and-after benchmark.

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
- Epochs: 5
- Best checkpoint: Epoch 5
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
| 1 | 0.02117 | 0.002321 | 25.09 |
| 2 | 0.002520 | 0.004123 | 35.70 |
| 3 | 0.000159 | 0.000196 | 27.84 |
| 4 | 0.000331 | 0.000248 | 22.70 |
| 5 | **0.0000237** | **0.0000235** | 22.84 |

### LoRA Training Curve

![Qwen2.5-VL-3B LoRA Training Curve](figures/qwen3b_lora_loss_curve.svg)

The extended five-epoch experiment showed continued in-domain convergence, with the lowest validation loss achieved at **Epoch 5**. This checkpoint was therefore selected as the `best_adapter`.

The very low validation loss suggests strong task adaptation on FakeBench. However, because both training and validation performance are close to saturation, further evaluation is required to rule out dataset bias, near-duplicate leakage, or excessive domain specialisation.

Training history is available in:

`github_results/qwen3b_lora_training_history.csv`

### Base vs LoRA Evaluation

For a controlled before/after comparison, the original Qwen2.5-VL-3B model and the LoRA-adapted model were evaluated with the **same classification-only prompt** on the same held-out 500-image test set.

| Model | N | Accuracy | Precision | Recall | F1 | MCC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B Base | 500 | 77.40% | 96.60% | 56.80% | 71.54% | 0.6014 | 245 | 5 | 108 | 142 |
| **Qwen2.5-VL-3B + LoRA (5 epochs)** | **500** | **99.60%** | **100.00%** | **99.20%** | **99.60%** | **0.9920** | **250** | **0** | **2** | **248** |

Under the controlled classification-only setting, the five-epoch LoRA run improved Accuracy by **22.2 percentage points**, Recall by **42.4 percentage points**, and F1 by approximately **28.06 percentage points**. MCC increased from **0.6014 to 0.9920**. False positives decreased from **5 to 0**, while false negatives decreased from **108 to 2**.

> **Important:** The 86.8% Qwen2.5-VL-3B benchmark baseline used the original benchmark prompt, which also requested an explanation. The 77.4% Base result above uses the classification-only prompt used for the controlled LoRA comparison. These values are therefore not directly interchangeable.

> **Generalization note:** The 99.6% result is an in-domain result on a held-out FakeBench split. Cross-dataset and unseen-generator generalization have not yet been established. Because performance is close to saturation, the next priority is to check exact/near-duplicate leakage and evaluate external generalisation.

The evaluation summary is available in `github_results/qwen3b_base_vs_lora_summary.csv`.

### Fine-tuning Code

- `src/train_qwen3b_lora.py` — LoRA training and split construction
- `src/eval_qwen3b_lora.py` — controlled Base vs LoRA evaluation
- `scripts/run_qwen3b_lora.sh` — PBS GPU training job
- `scripts/run_qwen3b_lora_eval.sh` — PBS GPU evaluation job

### Model Weights

The previously published LoRA adapter is available on Hugging Face:

**[NTX88666/Qwen2.5-VL-3B-FakeBench-LoRA](https://huggingface.co/NTX88666/Qwen2.5-VL-3B-FakeBench-LoRA)**

The adapter weights are intentionally not duplicated in this GitHub repository. The latest five-epoch experiment reported above was evaluated on UTS HPC; the Hugging Face repository should be refreshed separately if the five-epoch checkpoint is to replace the previously published adapter.

## LoRA V2: A Fairer Test of Fake Image Detection

### Why did we build V2?

In V1, we fine-tuned Qwen2.5-VL-3B with LoRA and achieved **99.6% accuracy** on a 500-image test set. This was encouraging, but it did not answer an important question: **did the model learn general clues about AI-generated images, or just patterns specific to FakeBench?**

We identified three possible problems:

1. **Similar images across splits:** two slightly different copies of an image could make testing too easy.
2. **Familiar AI generators:** if training and testing use the same generators, the model might mainly learn their particular styles.
3. **Dataset shortcuts:** real and fake files might differ in size, format or compression. These differences can reveal the label even without understanding the picture.

V2 is designed to test these concerns rather than simply chase a higher score.

### What changed in V2?

**1. Check similar images before splitting**

We use perceptual hashing (**dHash**) to group images that look very similar. Across **6,000 images**, the audit found **5,979 groups**, meaning **21 images were additional members of near-duplicate groups**. This is **not** the same as deleting 21 images: 5,979 is a *group count*, not a new image count. Three groups contained both real and fake labels.

We keep each group in just one split to reduce the chance that the model sees one version during training and another during testing. Perceptual hashing reduces this risk, but cannot guarantee that every near-duplicate has been found.

**2. Hold out two AI generators**

FakeBench contains images from **10 AI generators**. V2 uses eight generators in the training-side distribution and saves **DALL·E 3 and StyleGAN** for a different test. The model has not trained on fake images from these two held-out generators.

Why? It is like learning from questions written by eight teachers, then taking an exam written by two new teachers. We want to see whether the model learned something useful beyond the styles it already knows.

**3. Use two test sets**

| Split | Real | Fake | Total | Purpose |
|---|---:|---:|---:|---|
| Train | 1,944 | 1,593 | 3,537 | Fine-tune the LoRA adapter |
| Validation | 500 | 500 | 1,000 | Choose the best checkpoint |
| Test-ID | 250 | 199 | 449 | Test on **seen generators**, using unseen images |
| Test-OOD | 250 | 250 | 500 | Test on **unseen generators** |

- **ID** means *in-distribution*. Its fake images come from generators represented during training, but the specific test images were not used for training.
- **OOD** means *out-of-distribution*. Its fake images come from the two held-out generators.
- **Why is Test-ID 449 instead of 500?** We started with the original 500-image test selection. **51 images** were excluded from Test-ID because they belonged to the held-out-generator groups. This left **250 real and 199 fake** images. We kept the remaining original test images instead of filling the gap with new ones.
- **Why are Train real/fake counts unequal?** Holding out generators reduces the pool of available fake images. However, the currently published `balanced()` function in [make_clean_splits.py](src/make_clean_splits.py) should produce equal real/fake counts and an even total. **The reported 3,537 (1,944 real / 1,593 fake) does not match that code as published.** We need to check which script version produced these split files before claiming the split is exactly reproducible.

**4. Make image handling more consistent**

V2 adds [imgnorm.py](src/imgnorm.py), which can apply the same RGB conversion, resizing, cropping and JPEG re-encoding to real and fake images. The training and evaluation scripts support this preprocessing through `--norm-size` (default: 448). **The recorded V2 HPC logs show that the reported run actually used `--norm-size 224`.** This is intended to reduce easy file-format and resolution clues; it does not prove those clues were fully removed.

### Did the new LoRA model work?

Both models were evaluated under the V2 protocol:

| Model | Test-ID Accuracy | Test-OOD Accuracy | Test-ID F1 | Test-OOD F1 |
|---|---:|---:|---:|---:|
| Base Qwen2.5-VL-3B | 73.05% | 62.60% | 59.26% | 47.91% |
| **LoRA V2** | **93.99%** | **91.00%** | **92.99%** | **90.72%** |

The LoRA model improved accuracy by **20.94 percentage points on Test-ID** and **28.40 percentage points on Test-OOD**. Its OOD score is promising because the held-out generators did not provide fake training images.

**Important:** The earlier V1 score of 99.6% used a **different test split and protocol**. Do not interpret the difference between 99.6% and 93.99% as a direct measure of V2 improving or worsening the model.

Detailed files: [summary.csv](github_results/lora_v2/summary.csv), [per_generator.csv](github_results/lora_v2/per_generator.csv) and [split_report.txt](github_results/lora_v2/split_report.txt).

### LoRA V2: Four Single-Factor Ablation Experiments

We compared the recorded V2 LoRA checkpoint against four additional training configurations. Each ablation targets one setting while using the **same V2 split files** (train 3,537, validation 1,000, Test-ID 449, Test-OOD 500). The recorded V2 HPC log reported **224 × 224 image normalization**, even though the training script's default is 448; the ablation commands explicitly use `--norm-size 224`.

| Configuration | Test-ID Accuracy | Test-OOD Accuracy | OOD F1 | OOD MCC |
|---|---:|---:|---:|---:|
| **Recorded LoRA V2** | **93.99%** | **91.00%** | **0.9072** | **0.8215** |
| Fixed training JPEG quality (85; no random quality 70–95) | 92.43% | 90.00% | 0.8932 | 0.8066 |
| Center crop for training (no random crop) | 95.10% | 90.80% | 0.9038 | 0.8192 |
| LoRA rank 8 (instead of 16; alpha kept at 32) | 94.21% | 92.40% | 0.9198 | 0.8526 |
| Skip JPEG re-encoding (keep resizing and cropping) | 85.52% | 88.00% | 0.8864 | 0.7648 |

**Interpretation:** JPEG re-encoding has the largest observed effect in this set of comparisons (−3.0 percentage points on Test-OOD when removed). Random cropping and random JPEG quality have smaller observed differences. Rank 8 slightly exceeds rank 16 on this held-out-generator test, suggesting that higher adapter rank is not necessarily better in this run. These are **single-run observations**, not proof of statistically stable causal effects or external-dataset generalization. FakeBench metadata shortcuts remain a concern; the training-run settings and generated variant source files should be audited before asserting a fully controlled comparison.

**Experiment result files:**

- [Fixed JPEG quality: summary](github_results/lora_v2/ablations/ablation_no_jpeg_quality.csv) · [Per-generator](github_results/lora_v2/ablations/ablation_no_jpeg_quality_per_generator.csv)
- [No random crop: summary](github_results/lora_v2/ablations/ablation_no_random_crop.csv) · [Per-generator](github_results/lora_v2/ablations/ablation_no_random_crop_per_generator.csv)
- [LoRA rank 8: summary](github_results/lora_v2/ablations/ablation_rank8.csv) · [Per-generator](github_results/lora_v2/ablations/ablation_rank8_per_generator.csv)
- [No JPEG re-encoding: summary](github_results/lora_v2/ablations/ablation_no_jpeg_reencode.csv) · [Per-generator](github_results/lora_v2/ablations/ablation_no_jpeg_reencode_per_generator.csv)

[Four-ablation HPC batch script](scripts/run_four_ablations.sh) documents the transformations and training commands. This version is reconstructed from the earlier submitted script, with the queue walltime corrected to six hours; it is **not an independently audited snapshot** of the exact HPC file used for these finished runs. Each uploaded ZIP also includes image-level prediction CSVs; these have not yet been committed to this repository.

### Published V2 Adapter

The best V2 adapter (Epoch 3) is available at **[NTX88666/Qwen2.5-VL-3B-FakeBench-LoRA-V2 on Hugging Face](https://huggingface.co/NTX88666/Qwen2.5-VL-3B-FakeBench-LoRA-V2)**. This is a separate release from [V1](https://huggingface.co/NTX88666/Qwen2.5-VL-3B-FakeBench-LoRA), so the original adapter remains available.

### V2 Training: Three Epochs

We also checked the actual training-history CSV and the separately uploaded Epoch 1, Epoch 2, Epoch 3, best-adapter and final-adapter archives.

| Epoch | Training Loss | Validation Loss | Validation Accuracy | Time (min) |
|---|---:|---:|---:|---:|
| 1 | 0.085577 | 0.055767 | 89.10% | 18.15 |
| 2 | 0.035787 | **0.042099** | 92.20% | 17.59 |
| 3 | **0.018480** | 0.045996 | **93.10%** | 17.57 |

![LoRA V2 training and validation loss](figures/qwen3b_lora_v2_loss_curve.svg)

**How to read this:** Training loss steadily decreased. Validation loss was lowest at **Epoch 2**, then rose slightly at Epoch 3. This may suggest early overfitting, but accuracy on validation images still increased. The V2 training code selects the best checkpoint by **highest validation accuracy** (using validation loss to break ties), so **Epoch 3** is correctly selected.

The uploaded `best_adapter`, `final_adapter`, and `epoch_3` archives have **identical adapter weight SHA-256 hashes**. All five separately uploaded checkpoint archives contained the adapter weights and configuration. This confirms the packaged best and final adapters correspond to Epoch 3. It does **not** independently verify the HPC run itself or cross-dataset robustness.

- [Actual three-epoch training-history CSV](github_results/lora_v2/training_history.csv)
- [Checkpoint file audit](github_results/lora_v2/checkpoint_audit.csv)
- [V2 loss curve SVG](figures/qwen3b_lora_v2_loss_curve.svg)

**Model files:** LoRA adapter weights are approximately **148.7 MB per checkpoint uncompressed** and are intentionally **not committed to GitHub**. Keeping five large, mostly overlapping checkpoints out of the source repository avoids unnecessary bloat. Both V1 and V2 adapters are published separately on Hugging Face; large weight files are not copied into GitHub.

### What still needs to be checked?

Our metadata-only shortcut test was surprisingly strong:

| Split | Accuracy using only metadata |
|---|---:|
| Validation | 99.5% |
| Test-ID | 99.8% |
| Test-OOD | 77.6% |

This test uses properties such as image dimensions, file format and file size instead of image content. These results show **strong dataset bias remains a serious concern**. Even a high OOD result does not yet prove the model will work reliably on arbitrary online images or completely independent datasets.

Our next priorities are to resolve the train-split/code mismatch, audit exact training parameters and generated ablation variants, and evaluate on an external dataset. The four ablation comparisons reported below are not a substitute for external-data testing.

### Code and Reproducibility

The V2 workflow is:

`make_clean_splits.py` (audit and split) → `imgnorm.py` (consistent image input) → `train_qwen3b_lora_v2.py` (LoRA training) → `eval_qwen3b_lora_v2.py` (ID/OOD evaluation).

- [Split generation](src/make_clean_splits.py)
- [Image preprocessing](src/imgnorm.py)
- [LoRA V2 training](src/train_qwen3b_lora_v2.py)
- [Base vs LoRA V2 evaluation](src/eval_qwen3b_lora_v2.py)
- [HPC job script](scripts/run_lora_v2_all.sh)
- V2 HPC: `cd ~/fakebench` then `qsub scripts/run_lora_v2_all.sh`; use `qstat -u $USER` to monitor, and inspect `results/v2/summary.csv` and `results/v2/per_generator.csv`. To preserve the recorded test design, use the original saved split CSVs and `NORM_SIZE=224`; do not regenerate splits without documenting the change.

The repository includes V2 source code and summary results. **It does not currently include the full split CSVs, individual-image predictions, source images or V2 adapter weights.** Local HPC paths may need to be changed when reproducing the experiment elsewhere.

## External Cross-Dataset Generalization: Tiny-GenImage (Completed)

To test whether FakeBench-trained **LoRA V2** transfers to a genuinely different data source, we evaluated the *unchanged V2 best adapter* (no additional training) on the **validation split of [TheKernel01/Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage)**, a third-party curated subset of [GenImage](https://github.com/GenImage-Dataset/GenImage). This is distinct from the FakeBench held-out-generator Test-OOD.

### Evaluation protocol

- Sample: **1,000 images = 500 real + 250 ADM fake + 250 BigGAN fake**.
- Preprocessing: **224 × 224**, matching the recorded LoRA V2 HPC settings; evaluation used center cropping and JPEG re-encoding.
- Model: **Qwen2.5-VL-3B + original V2 LoRA best adapter**, with no tuning on the external test set.
- Metrics: Accuracy, 95% Wilson CI, Precision, fake-class Recall, F1, MCC, confusion counts, and per-generator detection rate.

### Overall detection results

| Evaluation set | N | Accuracy | F1 | MCC |
|---|---:|---:|---:|---:|
| FakeBench Test-ID | 449 | 93.99% | 92.99% | 0.8789 |
| FakeBench Test-OOD (held-out generators) | 500 | 91.00% | 90.72% | 0.8215 |
| **Tiny-GenImage external dataset** | **1,000** | **79.60%** | **77.87%** | **0.5993** |

On the external data, **Precision = 85.07%**, **Recall = 71.80%**, and the **95% accuracy CI = 76.99%–81.98%**. The confusion matrix counts are **TN 437, FP 63, FN 141, TP 359**; there were no unparseable predictions.

### By-generator generalization

| Image source | N | Correct rate |
|---|---:|---:|
| Real | 500 | **87.4%** |
| ADM generated | 250 | **44.4%** |
| BigGAN generated | 250 | **99.2%** |

**Main finding:** The model transfers to a different dataset with **79.6% overall accuracy**, but performance is highly generator-dependent. **139 of the 141 missed fake images came from ADM**, while only two came from BigGAN. This asymmetry is more informative than overall accuracy alone.

External accuracy was **11.4 percentage points below** the FakeBench Test-OOD result. Because data sources, image distributions, and generator types differ, this is a *descriptive cross-dataset contrast*, not an isolated causal effect. This experiment does **not** establish robustness to arbitrary AI generators or show the external benefit of LoRA over a base model. Image overlap, potential shortcut signals and effects of dataset curation still require independent auditing.

**Files and code:**

- [Full protocol, analysis and limitations](reports/genimage_external.md)
- [External test summary CSV](github_results/genimage_external/summary.csv) and [per-generator CSV](github_results/genimage_external/per_generator.csv)
- [Dataset sampling script](src/prepare_genimage_external.py) and [HPC evaluation job](scripts/run_genimage_external.sh)
- [Shared V2 evaluation code](src/eval_qwen3b_lora_v2.py)

## Key Observations

- Qwen2.5-VL-3B achieved the strongest overall benchmark baseline result, with 86.8% accuracy and 87.1% F1 under the original benchmark prompt.
- In the controlled classification-only evaluation, the five-epoch LoRA run reached 99.6% in-domain accuracy and 0.9920 MCC.
- The LoRA model reduced false negatives from 108 to 2.
- LLaVA-OneVision-7B benefited substantially from the EtC strategy.
- Larger parameter count did not consistently lead to better fake-image detection performance.
- External Tiny-GenImage evaluation achieved **79.6% accuracy**, with a substantial ADM (44.4%) vs BigGAN (99.2%) detection gap. External robustness remains limited and generator-dependent.

## Runtime

Approximate end-to-end runtime for completed 500-image benchmark experiments:

| Model | Runtime |
|---|---:|
| Qwen2.5-VL-7B | 1h 07m |
| Idefics3-8B | 35m |
| LLaVA-OneVision-7B | 1h 02m |
| BLIP2-OPT-2.7B | 5m |
| Qwen2.5-VL-3B | Recorded in final runtime CSV |

LoRA fine-tuning took approximately **134 minutes** across five epochs on a single NVIDIA RTX PRO 6000 Blackwell 96GB GPU.

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

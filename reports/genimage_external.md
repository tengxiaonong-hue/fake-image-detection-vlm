# External Generalization Evaluation: Tiny-GenImage

## Objective

Evaluate whether the **original LoRA V2 adapter**, fine-tuned using FakeBench, transfers to a **separate data source** without further fine-tuning. This experiment assesses cross-dataset classification performance, not the causal effect of LoRA versus an unfine-tuned base model on GenImage.

## Protocol

- Model: Qwen2.5-VL-3B-Instruct with the original LoRA V2 best adapter (no training or checkpoint selection on GenImage).
- Source: [TheKernel01/Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage), validation split; this is a curated subset, **not the full official GenImage benchmark**.
- Sample: 1,000 images: 500 real, 250 ADM-generated, 250 BigGAN-generated.
- Preprocessing: V2 evaluation function with `--norm-size 224` (RGB/EXIF handling, resize and center crop to 224×224, JPEG re-encoding at quality 85).
- Model output: binary `REAL` / `AI-GENERATED`; unknown predictions counted as errors.
- Split and generation code: [prepare_genimage_external.py](../src/prepare_genimage_external.py).
- HPC evaluation: [run_genimage_external.sh](../scripts/run_genimage_external.sh), using [eval_qwen3b_lora_v2.py](../src/eval_qwen3b_lora_v2.py).

## Results

| Test source | N | Accuracy | Precision | Fake Recall | F1 (fake class) | MCC |
|---|---:|---:|---:|---:|---:|---:|
| FakeBench Test-ID | 449 | 93.99% | 96.24% | 89.95% | 92.99% | 0.8789 |
| FakeBench held-out-generator Test-OOD | 500 | 91.00% | 93.62% | 88.00% | 90.72% | 0.8215 |
| **Tiny-GenImage external** | **1,000** | **79.60%** | **85.07%** | **71.80%** | **77.87%** | **0.5993** |

External accuracy 95% Wilson confidence interval: **76.99%–81.98%**. The external score is 11.40 percentage points lower than FakeBench Test-OOD, but the two datasets are different populations; that contrast is **descriptive, not a controlled causal estimate**.

### External confusion matrix

| Ground truth / prediction | REAL | AI-GENERATED |
|---|---:|---:|
| Real (500) | **437** (TN) | **63** (FP) |
| Fake (500) | **141** (FN) | **359** (TP) |

There were **no unparseable outputs**. The model missed 141 of 500 synthetic images; its external fake-class recall is 71.8%.

### Breakdown by generator

| Source | Number of images | Correct |
|---|---:|---:|
| Real | 500 | **87.4% (437/500)** |
| ADM fake | 250 | **44.4% (111/250)** |
| BigGAN fake | 250 | **99.2% (248/250)** |

**The main finding is strong variation by generator.** Of the 141 synthetic images missed, **139 were ADM** and only **2 were BigGAN**. Overall accuracy alone hides this asymmetry. This may reflect generator-specific properties, distribution shifts or shortcuts in the curated data; the experiment does not establish which mechanism explains the gap.

## Interpretation and limitations

The model retains **some ability to classify images from a different dataset** without additional fine-tuning, but generalization is uneven. The 79.6% external accuracy and particularly the 44.4% ADM detection rate prevent claims of reliable, generator-independent detection.

This is **one curated external subset**, with only two synthetic generators; performance may change with another selection of real images, other generator families, different resolutions/compression, or an independently curated full-scale GenImage sample. We have **not yet ruled out overlaps/near-duplicates with FakeBench** or measured metadata-only shortcuts in Tiny-GenImage. There is also no matched evaluation of the unfine-tuned base model on this external subset, so this result by itself cannot quantify LoRA's cross-dataset benefit.

## Reproduce / inspect

```bash
cd ~/fakebench
source ~/miniconda3/bin/activate
conda activate fakebench
python -m pip install datasets  # if not installed
python src/prepare_genimage_external.py  # outputs a fixed-order curated sample
qsub scripts/run_genimage_external.sh
# Once the GPU job finishes:
cat results/genimage_external/summary.csv
cat results/genimage_external/per_generator.csv
```

The preparation script refuses to overwrite an existing prepared CSV. Preserve the original prepared images and manifest for exact test reproducibility; the metadata CSV and images were **not** included in the ZIP provided for this report.

Data files committed:
- [summary.csv](../github_results/genimage_external/summary.csv)
- [per_generator.csv](../github_results/genimage_external/per_generator.csv)

The uploaded ZIP also contained `predictions_genimage_external.csv` (1,000 item-level predictions). That larger file is **not yet committed** in this update; the aggregate statistics above were checked against its predictions.

# OpenFake v2 External Generalization: 5,000-Image Evaluation

## Research question

Can the **unchanged Qwen2.5-VL-3B LoRA V2 adapter**, trained on FakeBench, detect real photographs and synthetic images from newer generators in a separate dataset?

## Protocol

- Dataset: [ComplexDataLab/OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake), **`core/test`**, independently sourced from FakeBench.
- Sampling: first distinct decoded images encountered in streaming order, until **2,500 real + 2,500 fake**, rather than the entire OpenFake benchmark. Class balancing does *not* balance generators.
- Data preparation: [`src/prepare_openfake_external.py`](../src/prepare_openfake_external.py) verifies literal `real` / `fake` labels, converts decoded images to RGB and saves lossless PNG, and tracks image hashes. Cross-dataset duplicate/near-duplicate matching against FakeBench has *not* yet been completed.
- Model: original V2 **best adapter**, without further fine-tuning or selecting checkpoints on OpenFake.
- Image preprocessing at inference: V2 **224 × 224** normalization (resize, center crop, JPEG re-encoding).
- Command: [`scripts/run_openfake_external.sh`](../scripts/run_openfake_external.sh) calls [`src/eval_qwen3b_lora_v2.py`](../src/eval_qwen3b_lora_v2.py).
- Raw input images and model weights are not included in this repository.

## Results

| Evaluation | N | Accuracy | Precision | Fake recall | F1 (fake class) | MCC |
|---|---:|---:|---:|---:|---:|---:|
| FakeBench Test-ID | 449 | 93.99% | 96.24% | 89.95% | 92.99% | 0.8789 |
| FakeBench held-out-generator Test-OOD | 500 | 91.00% | 93.62% | 88.00% | 90.72% | 0.8215 |
| Tiny-GenImage external | 1,000 | 79.60% | 85.07% | 71.80% | 77.87% | 0.5993 |
| **OpenFake external** | **5,000** | **81.60%** | **79.92%** | **84.40%** | **82.10%** | **0.6330** |

**OpenFake 95% Wilson interval for accuracy: 80.50%–82.65%.** There were **0 unknown/unparseable outputs**.

### Confusion matrix

| Actual / predicted | Real | Fake |
|---|---:|---:|
| Real (n=2,500) | **1,970 TN** | **530 FP** |
| Fake (n=2,500) | **390 FN** | **2,110 TP** |

The **real-image false-positive rate was 21.2% (530/2,500)**. For fake images, the **miss rate was 15.6% (390/2,500)**. The classifier detected 84.4% of synthetic images but frequently misclassified real photographs.

## Generator-level analysis

**For fake-image rows, correct rate = fake detection recall for that generator; for real-image rows, correct rate = real detection specificity.** Do not interpret the fake-source correct rate as an entire per-generator balanced accuracy.

| Group / generator | Images | Correct | Correct rate |
|---|---:|---:|---:|
| GPT Image 1.5 | 317 | 308 | **97.16%** |
| GPT Image 2 | 30 | 23 | **76.67%** |
| Nano Banana Pro | 12 | 9 | **75.00%** |
| Seedream v5.0 | 20 | 20 | **100.00%** |
| Midjourney 7 | 194 | 176 | **90.72%** |
| FLUX.2 Klein 9B | 474 | 289 | **60.97%** |
| Z-Image Turbo | 701 | 572 | **81.60%** |
| Illustrious | 326 | 324 | **99.39%** |
| ImageNet real | 1,681 | 1,354 | **80.55%** |
| DOCCI real | 819 | 616 | **75.21%** |

These ten groups are a **selection for readability**. Complete counts and rates for all **22** reported source groups appear in [per_generator.csv](../github_results/openfake_external/per_generator.csv).

**Key finding:** The same fixed detector performs very differently across source models. For example, it detected **308/317 GPT Image 1.5 fakes** but only **289/474 FLUX.2 Klein 9B fakes**. GPT Image 2, Nano Banana Pro and Seedream v5.0 are relatively small samples (30, 12 and 20 respectively): their point estimates must not be over-generalized. Among the real images, specificity also varies by origin: **80.5% for ImageNet** and **75.2% for DOCCI**. The observed generator and real-source gaps could arise from distribution shift, dataset artifacts, content differences and/or compression clues. They do not independently establish the cause.

The sample deliberately balanced real vs fake, but the **synthetic generators have highly unequal representation**. Thus the 81.6% aggregate depends on the chosen mix, and is not a population-level estimate of performance on all current AI image platforms.

## Interpretation and limitations

1. The fixed LoRA V2 adapter achieved **81.6%** on a different dataset containing newer generation sources. This demonstrates some cross-dataset transfer but not reliable detection for every generator.
2. The **9.4 percentage point** contrast versus FakeBench Test-OOD (91.0%) is *descriptive*: the sets differ in generator identity, image content, source selection and data processing. It does not isolate why performance declined.
3. The real-image false positives (530) and particularly weak FLUX.2 Klein 9B fake detection are priorities for future analysis. The small samples for some current models do not permit strong generator-wide claims.
4. No unfine-tuned Qwen2.5-VL-3B comparison was run on this external set, so these results **do not quantify LoRA's incremental generalization benefit**.
5. This report does **not** certify all sampled images are unrelated to FakeBench: only exact duplicates *within* the prepared OpenFake subset were filtered; cross-dataset near-duplicate checking and a metadata-only shortcut audit are outstanding.
6. The sample is a curated 5,000-image subset, not the full OpenFake benchmark. OpenFake also includes video-source-labelled groups in its image test metadata; the per-generator table reflects the source labels as supplied.

## Files and repeatability

- [Overall summary](../github_results/openfake_external/summary.csv)
- [All per-generator/source breakdowns](../github_results/openfake_external/per_generator.csv)
- [Data preparation](../src/prepare_openfake_external.py)
- [HPC inference job](../scripts/run_openfake_external.sh)
- [Model evaluator](../src/eval_qwen3b_lora_v2.py)

The uploaded experiment archive also contained **5,000 image-level prediction records** in `predictions_openfake_external.csv`. The aggregate table and group breakdown were cross-checked against those predictions. The large item-level CSV is not included here; it can be added separately if detailed audit or error inspection is required.

To reproduce on UTS HPC with a prepared OpenFake sample and original V2 adapter:

```bash
cd ~/fakebench
source ~/miniconda3/bin/activate && conda activate fakebench
python src/prepare_openfake_external.py
qsub scripts/run_openfake_external.sh
# after the GPU job completes
cat results/openfake_external/summary.csv
cat results/openfake_external/per_generator.csv
```

# LoRA v2 (无泄漏切分) 运行说明

## 1. 上传并解压
把 `fakebench_lora_v2_bundle.zip` 上传到 HPC 的 `~/fakebench/`,然后:

```bash
cd ~/fakebench
unzip -o fakebench_lora_v2_bundle.zip      # 只会新增 src/*_v2.py、src/make_clean_splits.py、scripts/*.sh
```
不会覆盖你现有的结果:新输出都在 `data_splits_v2/`、`models/qwen25vl_3b_lora_v2/`、`results/v2/`。

## 2. 先看生成器列表(登录节点上几秒钟)
```bash
source ~/miniconda3/bin/activate && conda activate fakebench
python -c "import pandas as pd; m=pd.read_csv('results/metadata.csv'); print(m[m.ground_truth=='fake'].generator.value_counts())"
```
挑 1~2 个图片较多的生成器作为"留出生成器"(只出现在 OOD 测试集,训练时完全看不到)。

## 3. 提交任务
```bash
cd ~/fakebench
qsub -v HOLDOUT="生成器A 生成器B" scripts/run_lora_v2_all.sh     # 推荐:切分 -> 训练 -> 评测一次跑完
# 不想留出生成器就:  qsub scripts/run_lora_v2_all.sh
# 只想先看切分报告:  qsub -v HOLDOUT="生成器A" scripts/run_clean_splits_only.sh
qstat -u $USER
```
可选环境变量:`EPOCHS=3`(默认)、`FORCE_SPLITS=1`(重建切分)、`HAMMING=6`(近重复阈值)。
日志是 `~/fakebench/lora_v2.o<jobid>`。

## 4. 看结果
- `results/v2/split_report.txt`:近重复数量、各 split 大小、**元数据捷径检查**(远高于 50% 说明数据本身泄漏标签)
- `results/v2/summary.csv`:base vs LoRA,在 `test_id`(见过的生成器)和 `test_ood`(没见过的生成器)上的准确率及 95% 置信区间、MCC、混淆矩阵
- `results/v2/per_generator.csv`:每个生成器的检出率
- `models/qwen25vl_3b_lora_v2/best_adapter`:按验证集准确率选出的 adapter

## 5. 注意
- `scripts/*.sh` 里的 conda 环境名 `fakebench`、队列 `large_gpuq` 沿用了你仓库里原有脚本;如果你实际用的是 `venv/`,把 `source ... conda activate ...` 两行换成你的激活命令即可。
- 评测集 `test_id` 默认就是你原来的 `evaluation_sample_500.csv`(去掉了属于留出生成器的图),所以新旧数字可直接对比。

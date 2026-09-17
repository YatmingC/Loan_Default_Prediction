#!/bin/bash
# 完整baseline测试（全部样本），快速启动baseline测试可启用sample_size参数（小规模测试，10个样本）

python benchmark_test/run_baseline_test.py \
    --data_path data/SBA/SBAcase.11.13.17.csv \
    --output_dir benchmark_test/results/SBA_baseline \
    # --sample_size 10 \
    --prompt_template benchmark_test/prompts/default_prompt.txt

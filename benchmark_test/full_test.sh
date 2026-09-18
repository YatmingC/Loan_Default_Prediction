#!/bin/bash
# 完整baseline测试（全部样本），快速启动baseline测试可启用sample_size参数（小规模测试，10个样本）

# SBA
python benchmark_test/run_benchmark.py --dataset sba \
    --data_path data/SBA/SBAcase.11.13.17.csv \
    --output_dir benchmark_test/results/sba_baseline \
    --prompt_template benchmark_test/prompts/default_prompt.txt
    # --sample_size 10

# LendingClub
# python benchmark_test/run_benchmark.py --dataset lendingclub \
#     --output_dir benchmark_test/results/lendingclub_baseline \
#     --sample_size 200

# Home Credit
# python benchmark_test/run_benchmark.py --dataset homecredit \
#     --output_dir benchmark_test/results/homecredit_baseline \
#     --sample_size 200

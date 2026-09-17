# Benchmark Test - SBA Baseline 推理测试

## 概述

`benchmark_test` 模块用于在 SBA (Small Business Administration) 贷款数据集上运行裸 LLM 推理，评估模型在 **无聚类、无 skill 演进** 条件下的信贷违约预测能力，作为 adaptive clustering 系统的 baseline 对照。

## 目录结构

```
benchmark_test/
├── __init__.py              # 模块初始化，导出 SBADataLoader 和 BaselineInferenceEngine
├── sba_data_loader.py       # SBA 数据集加载与预处理
├── baseline_inference.py    # Baseline 推理引擎（支持并发）
├── run_baseline_test.py     # 测试主程序入口
├── test_data_loading.py     # 独立数据加载验证脚本（不依赖 LLM）
└── results/                 # 推理结果输出目录（自动创建）
```

## 运行流程

### 整体流水线

```
SBA CSV 数据 → 数据加载与预处理 → 特征筛选与采样 → LLM 并发推理 → 结果收集与排序 → 指标评估 → 输出文件
```

### 详细步骤

| 步骤 | 说明 | 对应代码 |
|------|------|----------|
| 1. 加载数据 | 读取 SBA CSV，解析日期/数值列，处理缺失值 | `SBADataLoader.load_data()` → `preprocess()` |
| 2. 特征筛选 | 从 35 列中选取 18 个风险评估特征 + 2 个衍生特征 | `SBADataLoader.get_feature_columns()` |
| 3. 采样 | 按 `--sample_size` 随机采样（固定种子 42），默认全量 | `SBADataLoader.prepare_for_inference()` |
| 4. 并发推理 | 以 `--batch_size` 为并行度，多线程调用 LLM API | `BaselineInferenceEngine.infer_batch()` |
| 5. 结果评估 | 计算 Accuracy / Precision / Recall / F1 / ROC AUC | `BaselineInferenceEngine.evaluate()` |
| 6. 输出保存 | 保存推理结果（CSV + JSON）和评估指标（JSON） | `results/` 目录 |

## 使用方法

### 前置条件

- Python 3.14+
- 依赖：pandas, numpy, scikit-learn, requests, json_repair, tqdm, openpyxl
- SBA 数据集文件：`data/SBA/SBAcase.11.13.17.csv`
- LLM API 访问（OpenAI 兼容接口）

### 1. 验证数据加载（不需要 LLM）

```bash
python benchmark_test/test_data_loading.py
```

独立脚本，验证 CSV 能否正确加载、预处理和特征提取，无需配置 API。

### 2. 运行 Baseline 推理测试

```bash
# 基本用法 - 通过环境变量配置 API
export LLM_API_URL="https://your-api-endpoint/v1/chat/completions"
export LLM_API_KEY="your-api-key"
export LLM_MODEL_NAME="gpt-4"

python benchmark_test/run_baseline_test.py

# 完整参数示例
python benchmark_test/run_baseline_test.py \
    --data_path data/SBA/SBAcase.11.13.17.csv \
    --output_dir benchmark_test/results \
    --sample_size 200 \
    --batch_size 10 \
    --model_name gpt-4 \
    --api_url https://your-api-endpoint/v1/chat/completions \
    --api_key your-api-key
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | `data/SBA/SBAcase.11.13.17.csv` | SBA 数据集路径 |
| `--output_dir` | `benchmark_test/results` | 结果输出目录 |
| `--sample_size` | 全部 | 从数据集中随机采样的样本数 |
| `--batch_size` | 串行 | 并发推理并行度（即同时发出的 LLM 请求数） |
| `--prompt_template` | 内置模板 | 自定义 prompt 模板文件路径 |
| `--api_url` | 环境变量 `LLM_API_URL` | LLM API 地址（OpenAI 兼容） |
| `--api_key` | 环境变量 `LLM_API_KEY` | LLM API 密钥 |
| `--model_name` | 环境变量 `LLM_MODEL_NAME` 或 `gpt-4` | 模型名称 |

## 输入输出

### 输入

- **SBA CSV 数据集**（`SBAcase.11.13.17.csv`）：约 899,164 条贷款记录，35 列

### 选用的 18 个特征

| 特征 | 语义 | 类型 |
|------|------|------|
| Term | 贷款期限（月） | 数值 |
| NoEmp | 企业员工数 | 数值 |
| NewExist | 新企业(2) / 现有企业(1) | 分类 |
| CreateJob | 创造就业数 | 数值 |
| RetainedJob | 保留就业数 | 数值 |
| DisbursementGross | 放款总额 | 数值 |
| GrAppv | 银行批准金额 | 数值 |
| SBA_Appv | SBA 批准金额 | 数值 |
| SBA_Guarantee_Rate | SBA 担保比例（衍生自 Portion） | 数值 |
| New | 是否新业务 | 二值 |
| RealEstate | 是否有房地产担保 | 二值 |
| Recession | 贷款是否处于经济衰退期 | 二值 |
| Loan_Utilization | 贷款使用率（衍生：BalanceGross / DisbursementGross） | 数值 |
| RevLineCr | 是否循环信贷 | 文本(Y/N) |
| LowDoc | 是否低文档贷款 | 文本(Y/N) |
| NAICS | 北美行业分类代码 | 分类 |
| FranchiseCode | 特许经营代码（0/1 为无特许经营） | 分类 |
| UrbanRural | 城市(1) / 农村(2) / 未知(0) | 分类 |

### 排除的列及原因

| 排除列 | 原因 |
|--------|------|
| Selected, LoanNr_ChkDgt, Name | 标识符，无预测意义 |
| City, Zip, State, Bank, BankState | 高基数地理/机构信息 |
| ApprovalDate, ApprovalFY, DisbursementDate | 日期列，未做特征工程 |
| MIS_Status | 贷款最终状态，与 Default 直接对应（数据泄露） |
| ChgOffDate, ChgOffPrinGr | 仅违约贷款有值（数据泄露） |
| BalanceGross, Portion | 已用于衍生特征计算 |
| daysterm | 与 Term 语义重复 |
| xx | 含义不明 |

### 输出

推理完成后在 `--output_dir` 下生成以下文件：

| 文件 | 内容 |
|------|------|
| `baseline_inference_{timestamp}.csv` | 全量推理结果（原始特征 + 预测 + 推理理由） |
| `baseline_inference_{timestamp}.json` | 同上，JSON 格式 |
| `baseline_inference_intermediate_{timestamp}.json` | 中间结果（每 20 条保存一次，用于断点恢复参考） |
| `baseline_metrics_{timestamp}.json` | 评估指标（Accuracy / Precision / Recall / F1 / ROC AUC） |

### 评估指标示例

```json
{
  "accuracy": 0.7850,
  "precision": 0.6200,
  "recall": 0.5400,
  "f1_score": 0.5770,
  "roc_auc": 0.7100,
  "total_samples": 200,
  "default_count": 50,
  "predicted_default_count": 43
}
```

## 关键设计

- **并发推理**：`batch_size` 控制 `ThreadPoolExecutor` 的 `max_workers`，`batch_size=10` 表示同时发出 10 个 LLM 请求。不设置或设为 1 时退化为串行。
- **容错机制**：每个 LLM 调用最多重试 3 次（超时、HTTP 错误、JSON 解析失败均重试）；并发回调中捕获异常，单条失败不影响整体。
- **结果有序**：并发推理完成后按原始索引排序，保证输出顺序与输入一致。
- **中间保存**：每 20 条保存一次中间结果，长时间运行出错时可减少数据损失。
- **API 兼容**：使用 OpenAI Chat Completions 格式，兼容任何 OpenAI 兼容接口。

# Benchmark Test - 多数据集 Baseline 推理测试

## 概述

`benchmark_test` 模块用于在多个信贷数据集上运行裸 LLM 推理，评估模型在 **无聚类、无 skill 演进** 条件下的信贷违约预测能力，作为 adaptive clustering 系统的 baseline 对照。

当前支持的数据集：
- **SBA** — Small Business Administration 小企业贷款（~90万条，35列）
- **LendingClub** — P2P 个人贷款（~9578条，14列）
- **Home Credit** — 消费信贷违约风险（~30.7万条，122列→筛选50个特征）

## 目录结构

```
benchmark_test/
├── __init__.py                  # 模块初始化，导出所有 DataLoader 和 BaselineInferenceEngine
├── baseline_inference.py        # Baseline 推理引擎（支持并发，target_column 可配置）
├── logger.py                    # 日志工具
├── data_loader_sba.py           # SBA 数据集加载与预处理
├── data_loader_lendingclub.py   # LendingClub 数据集加载与预处理
├── data_loader_homecredit.py    # Home Credit 数据集加载与预处理
├── run_benchmark.py             # 统一测试入口（通过 --dataset 选择数据集）
├── adapt_to_clustering.py       # 离线适配器：benchmark 数据集 → main_multi_algo.py 输入目录
├── generate_sample_list.py      # 生成固定采样列表（行索引持久化到 JSON）
├── test_data_loading.py         # 独立数据加载验证脚本（不依赖 LLM，支持全部数据集）
├── prompts/
│   ├── default_prompt.txt       # SBA 默认 prompt
│   ├── lendingclub_prompt.txt   # LendingClub prompt
│   └── homecredit_prompt.txt    # Home Credit prompt
├── sample_lists/                # 固定采样列表目录（自动创建）
├── full_test.sh                 # 完整测试脚本
└── results/                     # 推理结果输出目录（自动创建）
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
# 测试全部数据集
python benchmark_test/test_data_loading.py

# 仅测试某个数据集
python benchmark_test/test_data_loading.py sba
python benchmark_test/test_data_loading.py lendingclub
python benchmark_test/test_data_loading.py homecredit
```

独立脚本，验证 CSV 能否正确加载、预处理和特征提取，无需配置 API。

### 2. 生成固定采样列表（可选，适合大数据集）

对于 Home Credit 等大数据集，可以先生成一份固定的采样索引列表，后续所有测试都基于同一批样本：

```bash
# Home Credit 抽 10%
python benchmark_test/generate_sample_list.py --dataset homecredit --sample_ratio 0.1

# 也可以按绝对数量抽样
python benchmark_test/generate_sample_list.py --dataset homecredit --sample_size 5000

# 自定义输出路径
python benchmark_test/generate_sample_list.py --dataset homecredit --sample_ratio 0.1 --output my_list.json
```

生成的 JSON 文件保存在 `benchmark_test/sample_lists/` 下，包含采样元信息和行索引列表。后续推理时通过 `--sample_list` 指定即可复用：

```bash
python benchmark_test/run_benchmark.py --dataset homecredit \
    --sample_list benchmark_test/sample_lists/homecredit_10pct.json \
    --batch_size 10
```

### 3. 运行 Baseline 推理测试

统一入口 `run_benchmark.py`，通过 `--dataset` 参数选择数据集：

#### SBA 数据集

```bash
# 基本用法 - 通过环境变量配置 API
export LLM_API_URL="https://your-api-endpoint/v1/chat/completions"
export LLM_API_KEY="your-api-key"
export LLM_MODEL_NAME="gpt-4"

python benchmark_test/run_benchmark.py --dataset sba

# 完整参数示例
python benchmark_test/run_benchmark.py --dataset sba \
    --data_path data/SBA/SBAcase.11.13.17.csv \
    --output_dir benchmark_test/results/sba_baseline \
    --sample_size 200 \
    --batch_size 10 \
    --model_name gpt-4 \
    --api_url https://your-api-endpoint/v1/chat/completions \
    --api_key your-api-key
```

#### LendingClub 数据集

```bash
# 小规模测试（10个样本）
python benchmark_test/run_benchmark.py --dataset lendingclub --sample_size 10

# 完整参数示例
python benchmark_test/run_benchmark.py --dataset lendingclub \
    --data_path data/predict_who_pays_back_loans/loan_data.csv \
    --output_dir benchmark_test/results/lendingclub_baseline \
    --sample_size 200 \
    --batch_size 10 \
    --prompt_template benchmark_test/prompts/lendingclub_prompt.txt
```

#### Home Credit 数据集

```bash
# 小规模测试（10个样本）
python benchmark_test/run_benchmark.py --dataset homecredit --sample_size 10

# 完整参数示例（注意文件名含空格，需要引号）
python benchmark_test/run_benchmark.py --dataset homecredit \
    --data_path "data/house_loan_data_analysis/loan_data (1).csv" \
    --output_dir benchmark_test/results/homecredit_baseline \
    --sample_size 200 \
    --batch_size 10 \
    --prompt_template benchmark_test/prompts/homecredit_prompt.txt
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | （必填） | 数据集名称：`sba` / `lendingclub` / `homecredit` |
| `--data_path` | 各数据集默认路径 | 数据集 CSV 文件路径 |
| `--output_dir` | 各数据集默认输出目录 | 结果输出目录 |
| `--sample_size` | 全部 | 从数据集中随机采样的样本数（与 `--sample_list` 互斥） |
| `--sample_list` | 无 | 固定采样列表文件路径（由 `generate_sample_list.py` 生成，与 `--sample_size` 互斥） |
| `--batch_size` | 串行 | 并发推理并行度（即同时发出的 LLM 请求数） |
| `--prompt_template` | 内置模板 | 自定义 prompt 模板文件路径 |
| `--api_url` | 环境变量 `LLM_API_URL` | LLM API 地址（OpenAI 兼容） |
| `--api_key` | 环境变量 `LLM_API_KEY` | LLM API 密钥 |
| `--model_name` | 环境变量 `LLM_MODEL_NAME` 或 `gpt-4` | 模型名称 |

## 输入输出

### 输入

支持三个数据集：

#### 1. SBA 数据集

- **文件**：`data/SBA/SBAcase.11.13.17.csv`，约 899,164 条贷款记录，35 列
- **目标列**：`Default`（0=正常，1=违约）

选用的 18 个特征：

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

排除的列及原因：

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

#### 2. LendingClub 数据集

- **文件**：`data/predict_who_pays_back_loans/loan_data.csv`，约 9,578 条个人贷款记录，14 列
- **目标列**：`not.fully.paid`（0=正常还款，1=未完全偿还）
- **违约率**：约 16%

全部 13 个特征（数据集本身很紧凑，无需筛选）：

| 特征 | 语义 | 类型 |
|------|------|------|
| credit.policy | 是否满足LendingClub信贷审核标准 | 二值(0/1) |
| purpose | 贷款用途 | 分类 |
| int.rate | 贷款利率 | 数值 |
| installment | 月还款额 | 数值 |
| log.annual.inc | 年收入的自然对数 | 数值 |
| dti | 债务收入比 | 数值 |
| fico | FICO信用评分 | 数值 |
| days.with.cr.line | 信用记录天数 | 数值 |
| revol.bal | 循环信贷余额 | 数值 |
| revol.util | 循环信贷使用率 (%) | 数值 |
| inq.last.6mths | 近6个月信用查询次数 | 数值 |
| delinq.2yrs | 近2年逾期次数 | 数值 |
| pub.rec | 公共不良记录数 | 数值 |

#### 3. Home Credit 数据集

- **文件**：`data/house_loan_data_analysis/loan_data (1).csv`，约 307,511 条消费信贷申请记录，122 列
- **目标列**：`TARGET`（0=正常还款，1=违约）
- **违约率**：约 8%

从 122 列中筛选 48 个原始特征（含 5 个 DAYS 列转换为可读格式）+ 2 个衍生特征（共 50 个）：

| 特征 | 语义 | 类型 |
|------|------|------|
| NAME_CONTRACT_TYPE | 合同类型 (Cash/Revolving) | 分类 |
| CODE_GENDER | 性别 | 分类 |
| FLAG_OWN_CAR | 是否拥有汽车 | 二值(Y/N) |
| FLAG_OWN_REALTY | 是否拥有房产 | 二值(Y/N) |
| CNT_CHILDREN | 子女数量 | 数值 |
| AMT_INCOME_TOTAL | 年收入总额 | 数值 |
| AMT_CREDIT | 贷款信用额度 | 数值 |
| AMT_ANNUITY | 贷款年金（每期还款额） | 数值 |
| AMT_GOODS_PRICE | 贷款对应商品价格 | 数值 |
| NAME_INCOME_TYPE | 收入来源类型 | 分类 |
| NAME_EDUCATION_TYPE | 教育程度 | 分类 |
| NAME_FAMILY_STATUS | 家庭状况 | 分类 |
| NAME_HOUSING_TYPE | 住房类型 | 分类 |
| AGE_YEARS | 申请人年龄（衍生自 DAYS_BIRTH） | 数值 |
| EMPLOYMENT_YEARS | 工作年限（衍生自 DAYS_EMPLOYED） | 数值 |
| OCCUPATION_TYPE | 职业类型 | 分类 |
| CNT_FAM_MEMBERS | 家庭成员数 | 数值 |
| REGION_RATING_CLIENT | 客户所在区域评级 (1最好, 3最差) | 分类 |
| REGION_RATING_CLIENT_W_CITY | 区域评级（含城市维度） | 分类 |
| REGION_POPULATION_RELATIVE | 所在地区相对人口密度 | 数值 |
| EXT_SOURCE_1 | 外部数据源评分1 | 数值 |
| EXT_SOURCE_2 | 外部数据源评分2 | 数值 |
| EXT_SOURCE_3 | 外部数据源评分3 | 数值 |
| OBS_30_CNT_SOCIAL_CIRCLE | 社交圈中30天可观测人数 | 数值 |
| DEF_30_CNT_SOCIAL_CIRCLE | 社交圈中30天内违约人数 | 数值 |
| OBS_60_CNT_SOCIAL_CIRCLE | 社交圈中60天可观测人数 | 数值 |
| DEF_60_CNT_SOCIAL_CIRCLE | 社交圈中60天内违约人数 | 数值 |
| AMT_REQ_CREDIT_BUREAU_MON | 过去1月征信查询次数 | 数值 |
| AMT_REQ_CREDIT_BUREAU_QRT | 过去1季度征信查询次数 | 数值 |
| AMT_REQ_CREDIT_BUREAU_YEAR | 过去1年征信查询次数 | 数值 |
| REGISTRATION_YEARS | 更换登记信息距今年数（衍生自 DAYS_REGISTRATION） | 数值 |
| ID_PUBLISH_YEARS | 身份证件更换距今年数（衍生自 DAYS_ID_PUBLISH） | 数值 |
| LAST_PHONE_CHANGE_DAYS | 最后一次更换电话距今天数（衍生自 DAYS_LAST_PHONE_CHANGE） | 数值 |
| HOUR_APPR_PROCESS_START | 申请提交时刻（0-23） | 数值 |
| WEEKDAY_APPR_PROCESS_START | 申请提交星期几 | 分类 |
| NAME_TYPE_SUITE | 申请时陪同人类型 | 分类 |
| ORGANIZATION_TYPE | 雇主单位类型 | 分类 |
| FLAG_EMP_PHONE | 是否提供工作电话 | 二值(0/1) |
| FLAG_WORK_PHONE | 是否提供座机 | 二值(0/1) |
| FLAG_CONT_MOBILE | 手机是否可联系 | 二值(0/1) |
| FLAG_PHONE | 是否有家庭电话 | 二值(0/1) |
| FLAG_EMAIL | 是否有邮箱 | 二值(0/1) |
| REG_REGION_NOT_LIVE_REGION | 注册地与居住地不在同一地区 | 二值(0/1) |
| REG_REGION_NOT_WORK_REGION | 注册地与工作地不在同一地区 | 二值(0/1) |
| LIVE_REGION_NOT_WORK_REGION | 居住地与工作地不在同一地区 | 二值(0/1) |
| REG_CITY_NOT_LIVE_CITY | 注册地与居住地不在同一城市 | 二值(0/1) |
| REG_CITY_NOT_WORK_CITY | 注册地与工作地不在同一城市 | 二值(0/1) |
| LIVE_CITY_NOT_WORK_CITY | 居住地与工作地不在同一城市 | 二值(0/1) |
| CREDIT_INCOME_RATIO | 信贷收入比（衍生：AMT_CREDIT / AMT_INCOME_TOTAL） | 数值 |
| ANNUITY_INCOME_RATIO | 年金收入比（衍生：AMT_ANNUITY / AMT_INCOME_TOTAL） | 数值 |

排除的列及原因：

| 排除列类别 | 数量 | 原因 |
|------------|------|------|
| SK_ID_CURR | 1 | 标识符，无预测意义 |
| *_AVG / *_MODE / *_MEDI 及住房相关 | 47 | 住房信息三种统计口径高度冗余，缺失率均 >50%，标准化值无实际单位 |
| FLAG_DOCUMENT_2~21 | 20 | 大部分只有单一取值（unique=1），无区分度 |
| FLAG_MOBIL | 1 | 全部为 1，零方差 |
| OWN_CAR_AGE | 1 | 缺失率 70% |
| AMT_REQ_CREDIT_BUREAU_HOUR/DAY/WEEK | 3 | unique=1~2，几乎全为 0，无区分度 |

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

## 接入自适应聚类主流程（adapt_to_clustering.py）

`baseline` 只做直推评测；若要把 benchmark 数据集喂给 **自适应聚类 + skill 自演进** 主流程
（`main_multi_algo.py`），需要一个**离线适配器**把数据集转成主流程期望的输入目录。
`adapt_to_clustering.py` 就是做这件事的，**核心聚类算法零改动**，只复用本模块的 DataLoader。

### 主流程期望的输入（硬编码，见 `src/utils.py::dataload`）

- `<dir>/sampled_test_data_add_prompt.xlsx`：数据，列顺序为
  `特征列… → Default → prompt → reasoning → answer → feedback`
- `<dir>/columns_description.xlsx`：3 列 `name, dtype, role_desc`，与数据列**逐列一一对应**
  - 特征列 → role `聚类`（dtype 由 pandas 类型推断：int/float，object 按基数 <50 记 `enum` 否则 `text`）
  - `Default` → `enum` + `目标列`（0/1 int）
  - `prompt` → `text` + `无`；`reasoning`/`answer`/`feedback` → `text` + `skill生成`

> **关键耦合**：`main_multi_algo.py::get_score` 对每条样本读取 **`reasoning`**，strip 后
> `endswith("是")`→pred=1 / `endswith("否")`→pred=0。因此适配器产出的 `reasoning`
> **必须以 是/否 结尾**，语义取自直推 baseline 的 `predicted_default`（1→是，0→否），
> 这样每簇的 `pre_compute_score` 就等于“直推 baseline 分数”。

### 用法

```bash
# 1) 先跑/复用直推 baseline，产出 baseline_inference_*.json（见上文 run_benchmark）
# 2) 适配：把数据集转成主流程输入目录
python benchmark_test/adapt_to_clustering.py --dataset sba \
    --baseline_result "benchmark_test/results/sba_baseline/baseline_inference_*.json" \
    --output_dir benchmark_adapted/sba
# 3) 跑聚类 + skill 自演进（与 mock_data 用法完全一致）
python main_multi_algo.py -i benchmark_adapted/sba -o output_sba \
    --max_iterations 3 --clustering_workspace ./clustering_workspace
```

### 复用固定采样列表（--sample_list）

适配器与 `run_benchmark.py` / `generate_sample_list.py` **共用同一套选样逻辑**，
所以同一个 sample_list 喂给 baseline 和适配器时，**取到的样本完全对齐**（行、特征、标签逐列一致），
直推结果能按 `index` 精确匹配回来：

```bash
# 生成一次固定采样列表
python benchmark_test/generate_sample_list.py --dataset homecredit --sample_ratio 0.1
# baseline 与适配器都用同一份 list（样本严格对齐）
python benchmark_test/run_benchmark.py --dataset homecredit \
    --sample_list benchmark_test/sample_lists/homecredit_10pct.json
python benchmark_test/adapt_to_clustering.py --dataset homecredit \
    --sample_list benchmark_test/sample_lists/homecredit_10pct.json \
    --baseline_result "benchmark_test/results/homecredit_baseline/baseline_inference_*.json" \
    --output_dir benchmark_adapted/homecredit
```

### 适配器命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | （必填） | `sba` / `lendingclub` / `homecredit` |
| `--data_path` | 数据集默认路径 | 覆盖默认 CSV 路径 |
| `--sample_size` | 全部 | 随机采样数（固定种子 42，与 `--sample_list` 互斥） |
| `--sample_list` | 无 | 复用 `generate_sample_list.py` 生成的固定采样列表（与 `--sample_size` 互斥） |
| `--baseline_result` | 无 | 直推 baseline 结果 JSON（**支持 glob**，多匹配取最新）；缺失时全部落占位 stub |
| `--prompt_template` | 数据集默认模板 | 自定义 prompt 模板 |
| `--output_dir` | `benchmark_adapted/<dataset>` | 输出目录 |

> **缺失 `--baseline_result` 时**：`reasoning`/`answer` 落中性占位（结尾统一 `否`，明确标注“占位待补”），
> `pre_compute_score` 恒为 0 → 无条件触发 skill 演进。补上真实 baseline 后仅需改 `--baseline_result` 重跑。
> baseline 里缺失（index 未对齐）的行也走同样的 stub 并计数告警。

## 关键设计

- **并发推理**：`batch_size` 控制 `ThreadPoolExecutor` 的 `max_workers`，`batch_size=10` 表示同时发出 10 个 LLM 请求。不设置或设为 1 时退化为串行。
- **容错机制**：每个 LLM 调用最多重试 3 次（超时、HTTP 错误、JSON 解析失败均重试）；并发回调中捕获异常，单条失败不影响整体。
- **结果有序**：并发推理完成后按原始索引排序，保证输出顺序与输入一致。
- **中间保存**：每 20 条保存一次中间结果，长时间运行出错时可减少数据损失。
- **API 兼容**：使用 OpenAI Chat Completions 格式，兼容任何 OpenAI 兼容接口。

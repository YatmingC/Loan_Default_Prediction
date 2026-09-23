"""离线适配器：把 benchmark 数据集转换成 main_multi_algo.py 期望的输入目录。

核心思想（详见 benchmark_test/ADAPT_PLAN.md）：
- 复用 benchmark_test 的 DataLoader（sba/lendingclub/homecredit），核心聚类算法零改动。
- 产出两个文件到 <output_dir>：
    - sampled_test_data_add_prompt.xlsx  （数据，含 Default/prompt/reasoning/answer/feedback）
    - columns_description.xlsx            （3 列：name, dtype, role_desc）
- 关键耦合：main_multi_algo.py::get_score 对每条样本读取 `reasoning`，strip 后
  endswith("是")→pred=1 / endswith("否")→pred=0，否则算错样本。因此适配器产出的
  `reasoning` 必须以 是/否 结尾，语义来自直推 baseline 的 predicted_default。
  （注意：mock_data 的 reasoning 实际以整句结尾、不含 是/否，故其 pre_compute_score 恒为 0；
   本适配器刻意让 reasoning 以 是/否 收尾，从而得到有意义的“直推 baseline 分数”。）

用法：
    python benchmark_test/adapt_to_clustering.py --dataset sba \
        --baseline_result "benchmark_test/results/sba_baseline/baseline_inference_*.json" \
        --output_dir benchmark_adapted/sba

    # 之后（与 mock_data 用法完全一致）：
    python main_multi_algo.py -i benchmark_adapted/sba -o output_sba \
        --max_iterations 3 --clustering_workspace ./clustering_workspace
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_test.run_benchmark import DATASET_CONFIGS, _load_dataset
from benchmark_test.logger import get_logger

logger = get_logger(__name__)

# object 列唯一值 < 该阈值 → enum，否则 → text
ENUM_CARDINALITY_THRESHOLD = 50

# 缺失 baseline 时的中性占位（结论统一为 否，明确标注待补）
STUB_REASONING = "【占位待补：未提供直推 baseline 结果】暂无直推分析，默认中性判断。结论：该笔贷款不存在违约风险，答案：否"
STUB_ANSWER = "【占位待补：未提供直推 baseline 结果】该企业不存在违约风险。答案：否"


def _resolve_baseline_path(pattern: str) -> str | None:
    """把 --baseline_result 解析成单个文件路径，支持 glob 通配。

    多个匹配时取按文件名排序的最后一个（时间戳最新）。
    """
    if not pattern:
        return None
    p = Path(pattern)
    if p.exists() and p.is_file():
        return str(p)
    matches = sorted(glob.glob(pattern))
    if not matches:
        logger.warning(f"--baseline_result 未匹配到任何文件: {pattern}，将全部使用占位 stub")
        return None
    if len(matches) > 1:
        logger.info(f"--baseline_result 匹配到 {len(matches)} 个文件，取最新: {matches[-1]}")
    return matches[-1]


def _load_baseline_index(baseline_path: str | None) -> dict:
    """读取 baseline JSON（list of records），按 record['index'] 建立索引映射。"""
    if not baseline_path:
        return {}
    with open(baseline_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if isinstance(records, dict):  # 兼容 {"results": [...]} 之类包装
        for key in ("results", "records", "data"):
            if key in records and isinstance(records[key], list):
                records = records[key]
                break
    index_map = {}
    for rec in records:
        if "index" in rec:
            index_map[rec["index"]] = rec
    logger.info(f"加载 baseline: {baseline_path}，共 {len(index_map)} 条带 index 的记录")
    return index_map


def _build_reasoning_answer(pred: int, base_reasoning: str) -> tuple[str, str]:
    """根据直推 predicted_default 构造以 是/否 结尾的 reasoning 与 answer。"""
    base = (base_reasoning or "").strip()
    if pred == 1:
        reasoning = (base + "\n" if base else "") + "结论：该笔贷款存在违约风险，答案：是"
        answer = "综合分析，该企业存在违约风险。答案：是"
    else:
        reasoning = (base + "\n" if base else "") + "结论：该笔贷款不存在违约风险，答案：否"
        answer = "综合分析，该企业不存在违约风险。答案：否"
    return reasoning, answer


def _infer_column_dtype(series: pd.Series) -> str:
    """推断特征列的 columns_description dtype 字符串（int/float/enum/text）。"""
    s = series
    if pd.api.types.is_bool_dtype(s):
        return "enum"
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_float_dtype(s):
        return "float"
    if pd.api.types.is_numeric_dtype(s):
        # 兜底：数值但非纯 int/float（如混合），按 float 处理
        return "float"
    # object / string：按基数区分 enum vs text
    nunique = s.nunique(dropna=True)
    if nunique < ENUM_CARDINALITY_THRESHOLD:
        return "enum"
    return "text"


def _format_loan_info(row: pd.Series, feature_cols: list[str]) -> str:
    """与 baseline_inference._format_loan_info 一致：`- k: v`，跳过 target 列。

    这里传入的 row 只含特征列（target 已剔除），故直接全量输出，保证同分布。
    """
    return "\n".join(f"- {k}: {row[k]}" for k in feature_cols)


def _select_samples(loader, target_column, sample_list, sample_size):
    """镜像 run_benchmark 的选样逻辑，返回 (X 特征, y 标签)，保留原始 index。"""
    if sample_list:
        with open(sample_list, "r", encoding="utf-8") as f:
            indices = json.load(f)["indices"]
        X, y = loader.prepare_for_inference(sample_size=None)
        valid = [i for i in indices if i in X.index]
        X, y = X.loc[valid], y.loc[valid]
        logger.info(f"sample_list: {sample_list}（{len(valid)}/{len(indices)} 命中）")
    else:
        # sample_size=None 表示全量；否则走 loader 内部固定 random_state=42 采样，
        # 与同 size 的 baseline 运行对齐。
        X, y = loader.prepare_for_inference(sample_size=sample_size)
        logger.info(f"sample_size={sample_size if sample_size else 'ALL'}，选中 {len(X)} 条")
    return X, y


def build_adapted_dataframe(X, y, feature_cols, prompt_template_text, baseline_index):
    """组装 sampled_test_data_add_prompt.xlsx 对应的 DataFrame。"""
    matched = 0
    stub = 0
    rows = []
    for idx, (_, row) in zip(X.index, X.iterrows()):
        loan_info = _format_loan_info(row, feature_cols)
        prompt = prompt_template_text.format(loan_info=loan_info)

        rec = baseline_index.get(idx)
        if rec is not None and "predicted_default" in rec:
            try:
                pred = int(rec.get("predicted_default", 0))
            except (ValueError, TypeError):
                pred = 0
            reasoning, answer = _build_reasoning_answer(pred, rec.get("reasoning", ""))
            matched += 1
        else:
            reasoning, answer = STUB_REASONING, STUB_ANSWER
            stub += 1

        record = {c: row[c] for c in feature_cols}
        record["Default"] = int(y.loc[idx])
        record["prompt"] = prompt
        record["reasoning"] = reasoning
        record["answer"] = answer
        record["feedback"] = ""
        rows.append(record)

    # 列顺序：特征列 → Default → prompt → reasoning → answer → feedback
    ordered_cols = list(feature_cols) + ["Default", "prompt", "reasoning", "answer", "feedback"]
    df = pd.DataFrame(rows, columns=ordered_cols)
    return df, matched, stub


def build_columns_description(df, feature_cols):
    """生成 columns_description.xlsx 对应的 DataFrame（3 列：name, dtype, role_desc）。

    行顺序与数据列一一对应：特征列(聚类) → Default(目标列) → prompt(无) →
    reasoning/answer/feedback(skill生成)。
    """
    desc_rows = []
    for col in feature_cols:
        desc_rows.append({"name": col, "dtype": _infer_column_dtype(df[col]), "role_desc": "聚类"})
    desc_rows.append({"name": "Default", "dtype": "enum", "role_desc": "目标列-样本答案"})
    desc_rows.append({"name": "prompt", "dtype": "text", "role_desc": "无"})
    desc_rows.append({"name": "reasoning", "dtype": "text", "role_desc": "skill生成-一期模型推理过程"})
    desc_rows.append({"name": "answer", "dtype": "text", "role_desc": "skill生成-一期模型回复"})
    desc_rows.append({"name": "feedback", "dtype": "text", "role_desc": "skill生成-人类对一期模型结果的反馈"})
    return pd.DataFrame(desc_rows, columns=["name", "dtype", "role_desc"])


def main():
    dataset_names = list(DATASET_CONFIGS.keys())
    parser = argparse.ArgumentParser(
        description="把 benchmark 数据集适配成 main_multi_algo.py 的输入目录"
    )
    parser.add_argument("--dataset", type=str, required=True, choices=dataset_names)
    parser.add_argument("--data_path", type=str, default=None, help="覆盖默认数据路径")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="采样数量（与 --sample_list 互斥；都不给=全量）")
    parser.add_argument("--sample_list", type=str, default=None,
                        help="固定采样列表 JSON（generate_sample_list.py 生成），与 --sample_size 互斥")
    parser.add_argument("--baseline_result", type=str, default=None,
                        help="直推 baseline 结果 JSON 路径（支持 glob）；缺失时全部落占位 stub")
    parser.add_argument("--prompt_template", type=str, default=None,
                        help="prompt 模板文本文件，默认取 DATASET_CONFIGS")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录，默认 benchmark_adapted/<dataset>")
    args = parser.parse_args()

    if args.sample_list and args.sample_size:
        parser.error("--sample_list 和 --sample_size 不能同时使用")

    cfg = DATASET_CONFIGS[args.dataset]
    output_dir = Path(args.output_dir or f"benchmark_adapted/{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_template_path = args.prompt_template or cfg["prompt_template"]
    with open(prompt_template_path, "r", encoding="utf-8") as f:
        prompt_template_text = f.read()

    logger.info("=" * 80)
    logger.info(f"适配数据集: {cfg['label']} → {output_dir}")
    logger.info("=" * 80)

    # 1. 加载数据集（内部已 preprocess）
    loader, target_column = _load_dataset(cfg, args.data_path)

    # 2. 选样
    X, y = _select_samples(loader, target_column, args.sample_list, args.sample_size)
    feature_cols = list(X.columns)
    if y is None:
        raise ValueError(f"数据集 {args.dataset} 未返回标签列（target={target_column}）")

    # 3. baseline 索引
    baseline_path = _resolve_baseline_path(args.baseline_result)
    baseline_index = _load_baseline_index(baseline_path)

    # 4. 组装数据 + 列描述
    df, matched, stub = build_adapted_dataframe(
        X, y, feature_cols, prompt_template_text, baseline_index
    )
    desc = build_columns_description(df, feature_cols)

    # 5. 写出
    data_out = output_dir / "sampled_test_data_add_prompt.xlsx"
    desc_out = output_dir / "columns_description.xlsx"
    df.to_excel(data_out, index=False)
    desc.to_excel(desc_out, index=False)

    # 6. 摘要
    pos_rate = float(df["Default"].mean()) if len(df) else 0.0
    logger.info("-" * 80)
    logger.info(f"样本数:        {len(df)}")
    logger.info(f"正样本率:      {pos_rate:.4f}")
    logger.info(f"特征列数:      {len(feature_cols)}")
    logger.info(f"baseline 匹配: {matched}")
    logger.info(f"占位 stub:     {stub}")
    logger.info(f"数据写出:      {data_out}")
    logger.info(f"列描述写出:    {desc_out}")
    logger.info("-" * 80)
    if stub and matched == 0:
        logger.warning("全部为占位 stub（reasoning 结尾均为 否）。提供 --baseline_result 后重跑即可替换为真实直推结果。")
    logger.info("适配完成。可运行：")
    logger.info(f"  python main_multi_algo.py -i {output_dir} -o output_{args.dataset} "
                f"--max_iterations 3 --clustering_workspace ./clustering_workspace")


if __name__ == "__main__":
    main()


"""多数据集 Baseline 推理测试统一入口

通过 --dataset 参数选择数据集（sba / lendingclub / homecredit），
替代原先三个独立的 run_xxx_test.py 脚本。
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_test.logger import get_logger

logger = get_logger(__name__)

DATASET_CONFIGS = {
    "sba": {
        "label": "SBA",
        "data_path": "data/SBA/SBAcase.11.13.17.csv",
        "output_dir": "benchmark_test/results/sba_baseline",
        "prompt_template": "benchmark_test/prompts/default_prompt.txt",
        "loader_module": "benchmark_test.data_loader_sba",
        "loader_class": "SBADataLoader",
        "target_column": "Default",
    },
    "lendingclub": {
        "label": "LendingClub",
        "data_path": "data/predict_who_pays_back_loans/loan_data.csv",
        "output_dir": "benchmark_test/results/lendingclub_baseline",
        "prompt_template": "benchmark_test/prompts/lendingclub_prompt.txt",
        "loader_module": "benchmark_test.data_loader_lendingclub",
        "loader_class": "LendingClubDataLoader",
        "target_column": "not.fully.paid",
    },
    "homecredit": {
        "label": "Home Credit",
        "data_path": "data/house_loan_data_analysis/loan_data (1).csv",
        "output_dir": "benchmark_test/results/homecredit_baseline",
        "prompt_template": "benchmark_test/prompts/homecredit_prompt.txt",
        "loader_module": "benchmark_test.data_loader_homecredit",
        "loader_class": "HomeCreditDataLoader",
        "target_column": "TARGET",
    },
}

def _load_dataset(config, data_path_override=None):
    """动态加载数据集对应的 DataLoader 并完成预处理"""
    import importlib
    mod = importlib.import_module(config["loader_module"])
    loader_cls = getattr(mod, config["loader_class"])
    data_path = data_path_override or config["data_path"]
    loader = loader_cls(data_path=data_path)
    loader.preprocess()
    return loader, config["target_column"]


def main():
    dataset_names = list(DATASET_CONFIGS.keys())
    parser = argparse.ArgumentParser(description="多数据集 Baseline 推理测试")
    parser.add_argument(
        "--dataset", type=str, required=True, choices=dataset_names,
        help=f"数据集名称: {', '.join(dataset_names)}",
    )
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--sample_list", type=str, default=None,
                        help="固定采样列表文件路径（由 generate_sample_list.py 生成），与 --sample_size 互斥")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--prompt_template", type=str, default=None)
    parser.add_argument("--api_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "skill"],
                        help="推理模式: baseline(直接LLM API) / skill(Claude Code加载skill)")
    parser.add_argument("--skill_path", type=str, default=None,
                        help="skill目录路径（仅 --mode skill 时生效）")
    parser.add_argument("--claude_model", type=str, default=None,
                        help="Claude模型名称（仅 --mode skill 时生效）")
    args = parser.parse_args()

    if args.sample_list and args.sample_size:
        parser.error("--sample_list 和 --sample_size 不能同时使用")

    cfg = DATASET_CONFIGS[args.dataset]
    label = cfg["label"]
    if args.output_dir:
        output_dir = args.output_dir
    elif args.mode == "skill":
        output_dir = cfg["output_dir"].replace("_baseline", "_skill")
    else:
        output_dir = cfg["output_dir"]
    prompt_template = args.prompt_template or cfg["prompt_template"]
    mode_label = "Skill (Claude Code)" if args.mode == "skill" else "Baseline"

    logger.info("=" * 80)
    logger.info(f"{label} {mode_label} Inference Test")
    logger.info("=" * 80)
    logger.info(f"Data path: {args.data_path or cfg['data_path']}")
    if args.sample_list:
        logger.info(f"Sample list: {args.sample_list}")
    else:
        logger.info(f"Sample size: {args.sample_size if args.sample_size else 'ALL'}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # Step 1: 加载数据
    logger.info(f"Step 1: Loading {label} dataset...")
    data_loader, target_column = _load_dataset(cfg, args.data_path)

    summary = data_loader.get_data_summary()
    logger.info("Data summary:")
    logger.info(f"  Total records: {summary['total_records']}")
    logger.info(f"  Default rate: {summary['default_rate']:.4f}")
    logger.info(f"  Default count: {summary['default_count']}")
    logger.info(f"  Non-default count: {summary['non_default_count']}")
    logger.info(f"  Feature count: {summary['features']}")
    logger.info("")

    # Step 2: 准备推理数据
    logger.info("Step 2: Preparing inference data...")
    if args.sample_list:
        with open(args.sample_list, "r", encoding="utf-8") as f:
            sample_meta = json.load(f)
        indices = sample_meta["indices"]
        X, y = data_loader.prepare_for_inference(sample_size=None)
        valid_indices = [i for i in indices if i in X.index]
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        logger.info(f"Loaded sample list: {args.sample_list} ({len(valid_indices)} / {len(indices)} indices matched)")
    else:
        X, y = data_loader.prepare_for_inference(sample_size=args.sample_size)
    inference_df = X.copy()
    inference_df[target_column] = y
    logger.info(f"Inference data prepared: {len(inference_df)} samples")
    logger.info("")

    # Step 3: 初始化推理引擎
    if args.mode == "skill":
        logger.info("Step 3: Initializing Skill Inference Engine (Claude Code)...")
        from benchmark_test.skill_inference import SkillInferenceEngine
        engine = SkillInferenceEngine(
            skill_path=args.skill_path,
            output_dir=output_dir,
            target_column=target_column,
            max_workers=args.batch_size or 4,
            claude_model=args.claude_model,
        )
    else:
        logger.info("Step 3: Initializing Baseline Inference Engine...")
        from benchmark_test.baseline_inference import BaselineInferenceEngine
        engine = BaselineInferenceEngine(
            prompt_template_path=prompt_template,
            output_dir=output_dir,
            api_url=args.api_url,
            api_key=args.api_key,
            model_name=args.model_name,
            target_column=target_column,
        )
    logger.info("Engine initialized")
    logger.info("")

    # Step 4: 批量推理
    logger.info(f"Step 4: Running {mode_label.lower()} inference...")
    start_time = datetime.now()
    result_df = engine.infer_batch(
        df=inference_df,
        batch_size=args.batch_size,
        save_interval=20,
    )
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"Inference completed in {duration:.2f} seconds")
    logger.info(f"Average time per sample: {duration / len(result_df):.2f} seconds")
    logger.info("")

    # Step 5: 评估性能
    logger.info(f"Step 5: Evaluating {mode_label.lower()} performance...")
    metrics = engine.evaluate(result_df)

    logger.info(f"{mode_label} Performance Metrics:")
    logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
    logger.info(f"  Recall: {metrics.get('recall', 0):.4f}")
    logger.info(f"  F1 Score: {metrics.get('f1_score', 0):.4f}")
    if "roc_auc" in metrics:
        logger.info(f"  ROC AUC: {metrics.get('roc_auc', 0):.4f}")
    logger.info(f"  Total samples: {metrics.get('total_samples', 0)}")
    logger.info(f"  Actual defaults: {metrics.get('default_count', 0)}")
    logger.info(f"  Predicted defaults: {metrics.get('predicted_default_count', 0)}")
    logger.info("")

    metrics_prefix = "skill_metrics" if args.mode == "skill" else "baseline_metrics"
    metrics_path = Path(output_dir) / f"{metrics_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    logger.info("=" * 80)
    logger.info(f"{label} {mode_label.lower()} test completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

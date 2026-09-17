"""Baseline测试主程序

在SBA数据集上运行baseline推理测试，评估裸模型性能。
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_test.sba_data_loader import SBADataLoader
from benchmark_test.baseline_inference import BaselineInferenceEngine
from benchmark_test.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="SBA数据集Baseline推理测试")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/SBA/SBAcase.11.13.17.csv",
        help="SBA数据集路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_test/results",
        help="结果输出目录"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="测试样本数量（默认全部）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="并发推理并行度（默认串行）"
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=None,
        help="自定义prompt模板路径"
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default=None,
        help="LLM API地址（或通过环境变量LLM_API_URL设置）"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="LLM API密钥（或通过环境变量LLM_API_KEY设置）"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="模型名称（或通过环境变量LLM_MODEL_NAME设置，默认gpt-4）"
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("SBA Baseline Inference Test")
    logger.info("="*80)
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Sample size: {args.sample_size if args.sample_size else 'ALL'}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")

    # Step 1: 加载数据
    logger.info("Step 1: Loading SBA dataset...")
    data_loader = SBADataLoader(data_path=args.data_path)
    data_loader.preprocess()

    # 获取数据摘要
    summary = data_loader.get_data_summary()
    logger.info(f"Data summary:")
    logger.info(f"  Total records: {summary['total_records']}")
    logger.info(f"  Default rate: {summary['default_rate']:.4f}")
    logger.info(f"  Default count: {summary['default_count']}")
    logger.info(f"  Non-default count: {summary['non_default_count']}")
    logger.info(f"  Feature count: {summary['features']}")
    logger.info("")

    # Step 2: 准备推理数据
    logger.info("Step 2: Preparing inference data...")
    X, y = data_loader.prepare_for_inference(sample_size=args.sample_size)

    # 合并特征和标签
    inference_df = X.copy()
    inference_df['Default'] = y

    logger.info(f"Inference data prepared: {len(inference_df)} samples")
    logger.info("")

    # Step 3: 初始化Baseline推理引擎
    logger.info("Step 3: Initializing Baseline Inference Engine...")
    engine = BaselineInferenceEngine(
        prompt_template_path=args.prompt_template,
        output_dir=args.output_dir,
        api_url=args.api_url,
        api_key=args.api_key,
        model_name=args.model_name
    )
    logger.info("Engine initialized")
    logger.info("")

    # Step 4: 批量推理
    logger.info("Step 4: Running baseline inference...")
    start_time = datetime.now()

    result_df = engine.infer_batch(
        df=inference_df,
        batch_size=args.batch_size,
        save_interval=20
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Inference completed in {duration:.2f} seconds")
    logger.info(f"Average time per sample: {duration/len(result_df):.2f} seconds")
    logger.info("")

    # Step 5: 评估性能
    logger.info("Step 5: Evaluating baseline performance...")
    metrics = engine.evaluate(result_df)

    logger.info("Baseline Performance Metrics:")
    logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
    logger.info(f"  Recall: {metrics.get('recall', 0):.4f}")
    logger.info(f"  F1 Score: {metrics.get('f1_score', 0):.4f}")
    if 'roc_auc' in metrics:
        logger.info(f"  ROC AUC: {metrics.get('roc_auc', 0):.4f}")
    logger.info(f"  Total samples: {metrics.get('total_samples', 0)}")
    logger.info(f"  Actual defaults: {metrics.get('default_count', 0)}")
    logger.info(f"  Predicted defaults: {metrics.get('predicted_default_count', 0)}")
    logger.info("")

    # 保存评估结果
    import json
    metrics_path = Path(args.output_dir) / f"baseline_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    logger.info("="*80)
    logger.info("Baseline test completed successfully!")
    logger.info("="*80)


if __name__ == "__main__":
    main()

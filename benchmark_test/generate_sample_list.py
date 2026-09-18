"""生成固定采样列表

对指定数据集按比例或数量采样，将行索引保存到 JSON 文件。
后续 run_benchmark.py --sample_list 可复用同一批样本。

用法：
    python benchmark_test/generate_sample_list.py --dataset homecredit --sample_ratio 0.1
    python benchmark_test/generate_sample_list.py --dataset homecredit --sample_size 5000
    python benchmark_test/generate_sample_list.py --dataset lendingclub --sample_ratio 0.5 --output my_list.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_test.run_benchmark import DATASET_CONFIGS, _load_dataset


def main():
    dataset_names = list(DATASET_CONFIGS.keys())
    parser = argparse.ArgumentParser(description="生成固定采样列表")
    parser.add_argument("--dataset", type=str, required=True, choices=dataset_names)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--sample_ratio", type=float, default=None, help="采样比例，如 0.1 表示 10%%")
    parser.add_argument("--sample_size", type=int, default=None, help="采样数量（与 sample_ratio 二选一）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（仅用于生成时）")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径（默认自动命名）")
    args = parser.parse_args()

    if args.sample_ratio is None and args.sample_size is None:
        parser.error("必须指定 --sample_ratio 或 --sample_size 其中之一")
    if args.sample_ratio is not None and args.sample_size is not None:
        parser.error("--sample_ratio 和 --sample_size 不能同时指定")

    cfg = DATASET_CONFIGS[args.dataset]
    loader, target_column = _load_dataset(cfg, args.data_path)

    total = len(loader.df)

    if args.sample_ratio is not None:
        sample_size = max(1, int(total * args.sample_ratio))
        tag = f"{int(args.sample_ratio * 100)}pct"
    else:
        sample_size = min(args.sample_size, total)
        tag = f"{sample_size}"
    sampled = loader.df.sample(n=sample_size, random_state=args.seed)
    indices = sorted(sampled.index.tolist())

    output_dir = Path("benchmark_test/sample_lists")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else output_dir / f"{args.dataset}_{tag}.json"

    payload = {
        "dataset": args.dataset,
        "data_path": str(args.data_path or cfg["data_path"]),
        "total_records": total,
        "sample_size": sample_size,
        "sample_ratio": round(sample_size / total, 6),
        "seed_used": args.seed,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "indices": indices,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Dataset:       {args.dataset} ({total} records)")
    print(f"Sample size:   {sample_size} ({sample_size / total:.2%})")
    print(f"Saved to:      {output_path}")


if __name__ == "__main__":
    main()

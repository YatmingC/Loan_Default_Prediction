"""独立测试脚本 - 验证多数据集数据加载
不依赖LLM环境，仅测试数据加载、预处理和特征提取逻辑。

用法：
    python benchmark_test/test_data_loading.py              # 测试全部数据集
    python benchmark_test/test_data_loading.py sba          # 仅测试SBA
    python benchmark_test/test_data_loading.py lendingclub   # 仅测试LendingClub
    python benchmark_test/test_data_loading.py homecredit    # 仅测试Home Credit
"""

import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
PROJECT_ROOT = BENCHMARK_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_module(name):
    """按模块名加载 data loader，绕过 benchmark_test.__init__ 的 json_repair 依赖"""
    import importlib
    return importlib.import_module(f"benchmark_test.{name}")


def test_sba():
    data_path = Path("data/SBA/SBAcase.11.13.17.csv")
    if not data_path.exists():
        print(f"[SKIP] SBA data not found: {data_path}")
        return True

    print("=" * 80)
    print("SBA Data Loading Test")
    print("=" * 80)

    mod = _load_module("data_loader_sba")
    loader = mod.SBADataLoader(data_path=str(data_path))
    loader.preprocess()

    summary = loader.get_data_summary()
    features = loader.get_feature_columns()
    X, y = loader.prepare_for_inference(sample_size=50)

    print(f"  Total records: {summary['total_records']}")
    print(f"  Default rate:  {summary['default_rate']:.4f}")
    print(f"  Features:      {len(features)} -> {features}")
    print(f"  Sample shape:  X={X.shape}, y={len(y)}")
    assert len(features) == 18, f"Expected 18 SBA features, got {len(features)}"
    assert X.shape == (50, 18), f"Unexpected sample shape: {X.shape}"
    assert y is not None and len(y) == 50

    print("[OK] SBA data loading test passed\n")
    return True


def test_lendingclub():
    data_path = Path("data/predict_who_pays_back_loans/loan_data.csv")
    if not data_path.exists():
        print(f"[SKIP] LendingClub data not found: {data_path}")
        return True

    print("=" * 80)
    print("LendingClub Data Loading Test")
    print("=" * 80)

    mod = _load_module("data_loader_lendingclub")
    loader = mod.LendingClubDataLoader(data_path=str(data_path))
    loader.preprocess()

    summary = loader.get_data_summary()
    features = loader.get_feature_columns()
    X, y = loader.prepare_for_inference(sample_size=50)

    print(f"  Total records: {summary['total_records']}")
    print(f"  Default rate:  {summary['default_rate']:.4f}")
    print(f"  Features:      {len(features)} -> {features}")
    print(f"  Sample shape:  X={X.shape}, y={len(y)}")

    assert len(features) == 13, f"Expected 13 LendingClub features, got {len(features)}"
    assert X.shape == (50, 13), f"Unexpected sample shape: {X.shape}"
    assert y is not None and len(y) == 50

    print("[OK] LendingClub data loading test passed\n")
    return True


def test_homecredit():
    data_path = Path("data/house_loan_data_analysis/loan_data (1).csv")
    if not data_path.exists():
        print(f"[SKIP] Home Credit data not found: {data_path}")
        return True

    print("=" * 80)
    print("Home Credit Data Loading Test")
    print("=" * 80)

    mod = _load_module("data_loader_homecredit")
    loader = mod.HomeCreditDataLoader(data_path=str(data_path))
    loader.preprocess()

    summary = loader.get_data_summary()
    features = loader.get_feature_columns()
    X, y = loader.prepare_for_inference(sample_size=50)

    print(f"  Total records: {summary['total_records']}")
    print(f"  Default rate:  {summary['default_rate']:.4f}")
    print(f"  Features:      {len(features)} -> {features}")
    print(f"  Sample shape:  X={X.shape}, y={len(y)}")

    assert len(features) == 50, f"Expected 50 Home Credit features, got {len(features)}"
    assert X.shape == (50, 50), f"Unexpected sample shape: {X.shape}"
    assert y is not None and len(y) == 50

    derived_cols = {"AGE_YEARS", "EMPLOYMENT_YEARS", "REGISTRATION_YEARS",
                    "ID_PUBLISH_YEARS", "LAST_PHONE_CHANGE_DAYS",
                    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"}
    missing_derived = derived_cols - set(features)
    assert not missing_derived, f"Missing derived features: {missing_derived}"

    print("[OK] Home Credit data loading test passed\n")
    return True


ALL_TESTS = {
    "sba": test_sba,
    "lendingclub": test_lendingclub,
    "homecredit": test_homecredit,
}


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(ALL_TESTS.keys())
    invalid = [t for t in targets if t not in ALL_TESTS]
    if invalid:
        print(f"Unknown dataset(s): {invalid}. Choose from: {list(ALL_TESTS.keys())}")
        sys.exit(1)

    results = {}
    for name in targets:
        try:
            results[name] = ALL_TESTS[name]()
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    print("=" * 80)
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
    print("=" * 80)

    sys.exit(0 if all(results.values()) else 1)

"""独立测试脚本 - 验证SBA数据加载
不依赖项目环境，仅测试数据加载和预处理逻辑
"""

import pandas as pd
import numpy as np
from pathlib import Path

def test_sba_data_loading():
    """测试SBA数据加载"""
    data_path = Path('data/SBA/SBAcase.11.13.17.csv')

    print("="*80)
    print("SBA Data Loading Test")
    print("="*80)

    # 1. 加载原始数据
    print("\n[Step 1] Loading raw data...")
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    print(f"[OK] Loaded {len(df)} records, {len(df.columns)} columns")
    print(f"[OK] Data shape: {df.shape}")

    # 2. 查看列名
    print("\n[Step 2] Column names:")
    print(df.columns.tolist())

    # 3. 查看前几行
    print("\n[Step 3] First 5 rows preview:")
    print(df.head())

    # 4. 检查目标变量
    print("\n[Step 4] Target variable check:")
    if 'Default' in df.columns:
        print(f"[OK] Target 'Default' exists")
        print(f"  - Default samples: {df['Default'].sum()}")
        print(f"  - Non-default samples: {(df['Default'] == 0).sum()}")
        print(f"  - Default rate: {df['Default'].mean():.4f}")
        print(f"  - Data type: {df['Default'].dtype}")
    else:
        print("[FAIL] Target 'Default' not found")
        return False

    # 5. 数据预处理测试
    print("\n[Step 5] Preprocessing test...")

    # 处理日期列
    date_columns = ['ApprovalDate', 'ChgOffDate', 'DisbursementDate']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%d%m%Y', errors='coerce')
    print(f"[OK] Date columns processed: {date_columns}")

    # 处理数值列
    numeric_columns = ['Term', 'NoEmp', 'CreateJob', 'RetainedJob',
                      'DisbursementGross', 'BalanceGross', 'ChgOffPrinGr',
                      'GrAppv', 'SBA_Appv', 'daysterm']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"[OK] Numeric columns processed")

    # 计算衍生特征
    df['Portion'] = pd.to_numeric(df['Portion'], errors='coerce').fillna(0)
    df['SBA_Guarantee_Rate'] = df['Portion']
    df['Loan_Utilization'] = np.where(
        df['DisbursementGross'] > 0,
        df['BalanceGross'] / df['DisbursementGross'],
        0
    )
    print(f"[OK] Derived features calculated")

    # 6. 特征列检查
    print("\n[Step 6] Feature columns check:")
    feature_columns = [
        'Term', 'NoEmp', 'NewExist', 'CreateJob', 'RetainedJob',
        'DisbursementGross', 'GrAppv', 'SBA_Appv', 'SBA_Guarantee_Rate',
        'New', 'RealEstate', 'Recession', 'Loan_Utilization',
        'RevLineCr', 'LowDoc',
        'NAICS', 'FranchiseCode', 'UrbanRural'
    ]

    available_features = [col for col in feature_columns if col in df.columns]
    missing_features = [col for col in feature_columns if col not in df.columns]

    print(f"[OK] Available features: {len(available_features)}/{len(feature_columns)}")
    print(f"  - Available: {available_features}")
    if missing_features:
        print(f"  - Missing: {missing_features}")

    # 7. 数据质量检查
    print("\n[Step 7] Data quality check:")
    print(f"  - Overall missing rate: {df.isnull().sum().sum() / (len(df) * len(df.columns)):.4f}")
    print(f"  - Top 5 columns with missing values:")
    missing_counts = df.isnull().sum().sort_values(ascending=False).head(5)
    for col, count in missing_counts.items():
        print(f"    * {col}: {count} ({count/len(df):.2%})")

    # 8. 统计摘要
    print("\n[Step 8] Data summary:")
    summary = {
        'total_records': len(df),
        'default_rate': df['Default'].mean(),
        'default_count': int(df['Default'].sum()),
        'non_default_count': int((df['Default'] == 0).sum()),
        'features': len(available_features),
        'missing_rate': (df.isnull().sum() / len(df)).mean(),
    }

    for key, value in summary.items():
        print(f"  - {key}: {value}")

    # 9. 采样测试
    print("\n[Step 9] Sampling test:")
    sample_sizes = [10, 50, 100]
    for size in sample_sizes:
        if size <= len(df):
            sample_df = df.sample(n=size, random_state=42)
            sample_default_rate = sample_df['Default'].mean()
            print(f"  - Sample {size} records: default_rate={sample_default_rate:.4f}")

    print("\n" + "="*80)
    print("[SUCCESS] Data loading test passed!")
    print("="*80)

    return True

if __name__ == "__main__":
    try:
        success = test_sba_data_loading()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

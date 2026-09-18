"""LendingClub数据集加载器

加载并预处理LendingClub个人贷款数据集（predict_who_pays_back_loans/loan_data.csv），
用于信贷风险评估baseline测试。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from benchmark_test.logger import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "not.fully.paid"

FEATURE_COLUMNS = [
    "credit.policy",
    "purpose",
    "int.rate",
    "installment",
    "log.annual.inc",
    "dti",
    "fico",
    "days.with.cr.line",
    "revol.bal",
    "revol.util",
    "inq.last.6mths",
    "delinq.2yrs",
    "pub.rec",
]

FEATURE_DESCRIPTIONS = {
    "credit.policy": "是否满足LendingClub信贷审核标准 (1=是, 0=否)",
    "purpose": "贷款用途",
    "int.rate": "贷款利率",
    "installment": "月还款额",
    "log.annual.inc": "年收入的自然对数",
    "dti": "债务收入比 (debt-to-income)",
    "fico": "FICO信用评分",
    "days.with.cr.line": "信用记录天数",
    "revol.bal": "循环信贷余额",
    "revol.util": "循环信贷使用率 (%)",
    "inq.last.6mths": "近6个月信用查询次数",
    "delinq.2yrs": "近2年逾期次数",
    "pub.rec": "公共不良记录数",
}


class LendingClubDataLoader:
    """LendingClub个人贷款数据集加载与预处理"""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading LendingClub data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        return self.df

    def preprocess(self) -> pd.DataFrame:
        if self.df is None:
            self.load_data()
        df = self.df.copy()

        for col in FEATURE_COLUMNS:
            if col in df.columns and col != "purpose":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(0).astype(int)

        logger.info(f"Preprocessing completed. Final shape: {df.shape}")
        self.df = df
        return df

    def get_feature_columns(self) -> List[str]:
        available = [c for c in FEATURE_COLUMNS if c in self.df.columns]
        logger.info(f"Available features: {len(available)}/{len(FEATURE_COLUMNS)}")
        return available

    def prepare_for_inference(
        self, sample_size: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if self.df is None:
            self.preprocess()

        df = self.df.copy()
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} records for inference")

        feature_cols = self.get_feature_columns()
        X = df[feature_cols].copy()
        y = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None
        return X, y

    def get_data_summary(self) -> Dict:
        if self.df is None:
            self.preprocess()
        return {
            "total_records": len(self.df),
            "default_rate": self.df[TARGET_COLUMN].mean(),
            "default_count": int(self.df[TARGET_COLUMN].sum()),
            "non_default_count": int((self.df[TARGET_COLUMN] == 0).sum()),
            "features": len(self.get_feature_columns()),
            "missing_rate": (self.df.isnull().sum() / len(self.df)).mean(),
        }

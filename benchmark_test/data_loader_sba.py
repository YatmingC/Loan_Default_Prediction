"""SBA数据集加载器

加载并预处理SBA (Small Business Administration) 贷款数据集，
用于信贷风险评估baseline测试。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from benchmark_test.logger import get_logger

logger = get_logger(__name__)


class SBADataLoader:
    """SBA数据集加载与预处理"""

    def __init__(self, data_path: str):
        """
        初始化SBA数据加载器

        Args:
            data_path: SBA CSV文件路径
        """
        self.data_path = Path(data_path)
        self.df = None
        self.feature_columns = []

    def load_data(self) -> pd.DataFrame:
        """加载SBA原始数据"""
        logger.info(f"Loading SBA data from {self.data_path}")
        self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
        logger.info(f"Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        return self.df

    def preprocess(self) -> pd.DataFrame:
        """预处理SBA数据集

        Returns:
            预处理后的DataFrame
        """
        if self.df is None:
            self.load_data()

        df = self.df.copy()

        # 基础数据清洗
        logger.info("Starting data preprocessing...")

        # 1. 处理日期列
        date_columns = ['ApprovalDate', 'ChgOffDate', 'DisbursementDate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='%d%m%Y', errors='coerce')

        # 2. 处理数值列
        numeric_columns = ['Term', 'NoEmp', 'CreateJob', 'RetainedJob',
                          'DisbursementGross', 'BalanceGross', 'ChgOffPrinGr',
                          'GrAppv', 'SBA_Appv', 'daysterm']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. 处理分类列
        df['NewExist'] = df['NewExist'].fillna(0).astype(int)
        df['New'] = df['New'].fillna(0).astype(int)
        df['RealEstate'] = df['RealEstate'].fillna(0).astype(int)
        df['Recession'] = df['Recession'].fillna(0).astype(int)

        # 4. 计算衍生特征
        df['Portion'] = pd.to_numeric(df['Portion'], errors='coerce').fillna(0)
        df['SBA_Guarantee_Rate'] = df['Portion']  # SBA担保比例
        df['Loan_Utilization'] = np.where(
            df['DisbursementGross'] > 0,
            df['BalanceGross'] / df['DisbursementGross'],
            0
        )

        # 5. 处理目标变量 Default
        if 'Default' in df.columns:
            df['Default'] = df['Default'].fillna(0).astype(int)
        else:
            logger.warning("Target column 'Default' not found!")

        # 6. 填充缺失值
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        logger.info(f"Preprocessing completed. Final shape: {df.shape}")

        self.df = df
        return df

    def get_feature_columns(self) -> List[str]:
        """获取用于风险评估的特征列

        Returns:
            特征列名列表
        """
        # 核心风险评估特征
        self.feature_columns = [
            # 贷款基本信息
            'Term',  # 贷款期限
            'NoEmp',  # 员工数量
            'NewExist',  # 新企业/现有企业
            'CreateJob',  # 创造就业数
            'RetainedJob',  # 保留就业数

            # 贷款金额相关
            'DisbursementGross',  # 放款总额
            'GrAppv',  # 批准金额
            'SBA_Appv',  # SBA批准金额
            'SBA_Guarantee_Rate',  # SBA担保比例

            # 贷款特征
            'New',  # 是否新业务
            'RealEstate',  # 是否房地产担保
            'Recession',  # 是否衰退期
            'Loan_Utilization',  # 贷款使用率
            'RevLineCr',  # 是否循环信贷
            'LowDoc',  # 是否低文档贷款

            # 企业与地域信息
            'NAICS',  # 北美行业分类代码
            'FranchiseCode',  # 特许经营代码
            'UrbanRural',  # 城市/农村
        ]

        # 确保列存在
        available_features = [col for col in self.feature_columns if col in self.df.columns]
        logger.info(f"Available features: {len(available_features)}/{len(self.feature_columns)}")

        return available_features

    def prepare_for_inference(self, sample_size: int = None) -> Tuple[pd.DataFrame, pd.Series]:
        """准备用于推理的数据

        Args:
            sample_size: 采样数量，None表示使用全部数据

        Returns:
            (特征DataFrame, 标签Series)
        """
        if self.df is None:
            self.preprocess()

        df = self.df.copy()

        # 采样
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} records for inference")

        # 获取特征列
        feature_cols = self.get_feature_columns()

        # 准备特征和标签
        X = df[feature_cols].copy()
        y = df['Default'] if 'Default' in df.columns else None

        return X, y

    def get_data_summary(self) -> Dict:
        """获取数据集摘要统计

        Returns:
            包含统计信息的字典
        """
        if self.df is None:
            self.preprocess()

        summary = {
            'total_records': len(self.df),
            'default_rate': self.df['Default'].mean() if 'Default' in self.df.columns else None,
            'default_count': self.df['Default'].sum() if 'Default' in self.df.columns else None,
            'non_default_count': (self.df['Default'] == 0).sum() if 'Default' in self.df.columns else None,
            'features': len(self.get_feature_columns()),
            'missing_rate': (self.df.isnull().sum() / len(self.df)).mean(),
        }

        logger.info(f"Data summary: {summary}")
        return summary

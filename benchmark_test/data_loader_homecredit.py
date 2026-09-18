"""Home Credit数据集加载器

加载并预处理Home Credit Default Risk数据集
（house_loan_data_analysis/loan_data (1).csv），
用于信贷风险评估baseline测试。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from benchmark_test.logger import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "TARGET"

FEATURE_COLUMNS = [
    # 合同与资产
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    # 金额
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    # 申请人背景
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS",
    # 区域
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "REGION_POPULATION_RELATIVE",
    # 外部评分
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    # 社交圈违约
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    # 征信查询
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
    # 杂项时间
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    "HOUR_APPR_PROCESS_START",
    "WEEKDAY_APPR_PROCESS_START",
    "NAME_TYPE_SUITE",
    "ORGANIZATION_TYPE",
    # 联系方式标志（FLAG_MOBIL 全为1无区分度，不纳入）
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    # 地址一致性
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY",
]

FEATURE_DESCRIPTIONS = {
    "NAME_CONTRACT_TYPE": "合同类型 (Cash loans / Revolving loans)",
    "CODE_GENDER": "性别",
    "FLAG_OWN_CAR": "是否拥有汽车 (Y/N)",
    "FLAG_OWN_REALTY": "是否拥有房产 (Y/N)",
    "CNT_CHILDREN": "子女数量",
    "AMT_INCOME_TOTAL": "年收入总额",
    "AMT_CREDIT": "贷款信用额度",
    "AMT_ANNUITY": "贷款年金（每期还款额）",
    "AMT_GOODS_PRICE": "贷款对应商品价格",
    "NAME_INCOME_TYPE": "收入来源类型",
    "NAME_EDUCATION_TYPE": "教育程度",
    "NAME_FAMILY_STATUS": "家庭状况",
    "NAME_HOUSING_TYPE": "住房类型",
    "DAYS_BIRTH": "申请人年龄（以天为负数表示，距申请日）",
    "DAYS_EMPLOYED": "工作年限（以天为负数表示，距申请日；正值365243表示无业）",
    "OCCUPATION_TYPE": "职业类型",
    "CNT_FAM_MEMBERS": "家庭成员数",
    "REGION_RATING_CLIENT": "客户所在区域评级 (1最好, 3最差)",
    "REGION_RATING_CLIENT_W_CITY": "客户所在区域评级（含城市维度，1最好, 3最差）",
    "REGION_POPULATION_RELATIVE": "所在地区相对人口密度（标准化）",
    "EXT_SOURCE_1": "外部数据源评分1（标准化）",
    "EXT_SOURCE_2": "外部数据源评分2（标准化）",
    "EXT_SOURCE_3": "外部数据源评分3（标准化）",
    "OBS_30_CNT_SOCIAL_CIRCLE": "社交圈中30天可观测人数",
    "DEF_30_CNT_SOCIAL_CIRCLE": "社交圈中30天内违约人数",
    "OBS_60_CNT_SOCIAL_CIRCLE": "社交圈中60天可观测人数",
    "DEF_60_CNT_SOCIAL_CIRCLE": "社交圈中60天内违约人数",
    "AMT_REQ_CREDIT_BUREAU_MON": "过去1月征信查询次数",
    "AMT_REQ_CREDIT_BUREAU_QRT": "过去1季度征信查询次数",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "过去1年征信查询次数",
    "DAYS_REGISTRATION": "更换登记信息距今天数（负数）",
    "DAYS_ID_PUBLISH": "身份证件更换距今天数（负数）",
    "DAYS_LAST_PHONE_CHANGE": "最后一次更换电话距今天数（负数）",
    "HOUR_APPR_PROCESS_START": "申请提交时刻（小时，0-23）",
    "WEEKDAY_APPR_PROCESS_START": "申请提交星期几",
    "NAME_TYPE_SUITE": "申请时陪同人类型",
    "ORGANIZATION_TYPE": "雇主单位类型",
    "FLAG_EMP_PHONE": "是否提供工作电话 (0/1)",
    "FLAG_WORK_PHONE": "是否提供座机 (0/1)",
    "FLAG_CONT_MOBILE": "手机是否可联系 (0/1)",
    "FLAG_PHONE": "是否有家庭电话 (0/1)",
    "FLAG_EMAIL": "是否有邮箱 (0/1)",
    "REG_REGION_NOT_LIVE_REGION": "注册地与居住地不在同一地区 (0/1)",
    "REG_REGION_NOT_WORK_REGION": "注册地与工作地不在同一地区 (0/1)",
    "LIVE_REGION_NOT_WORK_REGION": "居住地与工作地不在同一地区 (0/1)",
    "REG_CITY_NOT_LIVE_CITY": "注册地与居住地不在同一城市 (0/1)",
    "REG_CITY_NOT_WORK_CITY": "注册地与工作地不在同一城市 (0/1)",
    "LIVE_CITY_NOT_WORK_CITY": "居住地与工作地不在同一城市 (0/1)",
}


class HomeCreditDataLoader:
    """Home Credit Default Risk数据集加载与预处理"""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading Home Credit data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df)} records with {len(self.df.columns)} columns")
        return self.df

    def preprocess(self) -> pd.DataFrame:
        if self.df is None:
            self.load_data()
        df = self.df.copy()

        logger.info("Starting data preprocessing...")

        df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(0).astype(int)

        # 将DAYS_BIRTH转换为年龄（正数年），更易于LLM理解
        if "DAYS_BIRTH" in df.columns:
            df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).round(1)

        # 将DAYS_EMPLOYED转换为工作年限（正数年）；365243是无业占位符
        if "DAYS_EMPLOYED" in df.columns:
            df["EMPLOYMENT_YEARS"] = df["DAYS_EMPLOYED"].apply(
                lambda x: 0.0 if x == 365243 else round(-x / 365.25, 1)
            )

        # 将其余DAYS列转为正数年
        if "DAYS_REGISTRATION" in df.columns:
            df["REGISTRATION_YEARS"] = (-df["DAYS_REGISTRATION"] / 365.25).round(1)
        if "DAYS_ID_PUBLISH" in df.columns:
            df["ID_PUBLISH_YEARS"] = (-df["DAYS_ID_PUBLISH"] / 365.25).round(1)
        if "DAYS_LAST_PHONE_CHANGE" in df.columns:
            df["LAST_PHONE_CHANGE_DAYS"] = (-df["DAYS_LAST_PHONE_CHANGE"]).round(0)

        # 衍生特征：信贷收入比、年金收入比
        df["CREDIT_INCOME_RATIO"] = np.where(
            df["AMT_INCOME_TOTAL"] > 0,
            (df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]).round(2),
            0,
        )
        df["ANNUITY_INCOME_RATIO"] = np.where(
            df["AMT_INCOME_TOTAL"] > 0,
            (df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]).round(4),
            0,
        )

        # 填充关键数值列缺失值
        for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
                     "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
                     "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
                     "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT",
                     "AMT_REQ_CREDIT_BUREAU_YEAR", "CNT_FAM_MEMBERS",
                     "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(f"Preprocessing completed. Final shape: {df.shape}")
        self.df = df
        return df

    def get_feature_columns(self) -> List[str]:
        days_replacements = {
            "DAYS_BIRTH": "AGE_YEARS",
            "DAYS_EMPLOYED": "EMPLOYMENT_YEARS",
            "DAYS_REGISTRATION": "REGISTRATION_YEARS",
            "DAYS_ID_PUBLISH": "ID_PUBLISH_YEARS",
            "DAYS_LAST_PHONE_CHANGE": "LAST_PHONE_CHANGE_DAYS",
        }
        cols = []
        for c in FEATURE_COLUMNS:
            derived = days_replacements.get(c)
            if derived and derived in self.df.columns:
                cols.append(derived)
            elif c in self.df.columns:
                cols.append(c)
        for derived in ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"]:
            if derived in self.df.columns:
                cols.append(derived)
        logger.info(f"Available features: {len(cols)}")
        return cols

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

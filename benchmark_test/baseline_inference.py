"""Baseline推理引擎

使用LLM直接对贷款样本进行风险评估，无聚类、无skill演进，
作为adaptive clustering系统的baseline对照。
"""

import json
import json_repair
import os
import concurrent.futures
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import requests

from benchmark_test.logger import get_logger

logger = get_logger(__name__)


class BaselineInferenceEngine:
    """Baseline推理引擎 - 直接使用LLM评估风险"""

    def __init__(
        self,
        prompt_template_path: Optional[str] = None,
        output_dir: str = "./benchmark_test/results",
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        target_column: str = "Default"
    ):
        """
        初始化Baseline推理引擎

        Args:
            prompt_template_path: prompt模板文件路径
            output_dir: 结果输出目录
            api_url: LLM API地址（优先使用，其次从环境变量LLM_API_URL读取）
            api_key: LLM API密钥（优先使用，其次从环境变量LLM_API_KEY读取）
            model_name: 模型名称（优先使用，其次从环境变量LLM_MODEL_NAME读取）
        """
        self.target_column = target_column
        self.prompt_template_path = prompt_template_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 从环境变量或参数获取API配置
        self.api_url = api_url or os.getenv("LLM_API_URL")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model_name = model_name or os.getenv("LLM_MODEL_NAME", "gpt-4")

        if not self.api_url:
            raise ValueError("LLM_API_URL must be provided via parameter or environment variable")
        if not self.api_key:
            raise ValueError("LLM_API_KEY must be provided via parameter or environment variable")

        logger.info(f"Using LLM API: {self.api_url}")
        logger.info(f"Using model: {self.model_name}")

        # 加载prompt模板
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """加载prompt模板"""
        if self.prompt_template_path and Path(self.prompt_template_path).exists():
            with open(self.prompt_template_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # 使用默认模板（对齐当前算法的初始推理prompt）
            return self._get_default_prompt_template()

    def _get_default_prompt_template(self) -> str:
        """获取默认prompt模板 - 极简baseline版本"""
        return """根据以下贷款信息，判断该笔贷款是否存在违约风险。
{loan_info}
请对贷款违约风险进行分析，最终回答是否存在违约风险（0表示不违约，1表示违约），答案用JSON格式输出：
{{"predicted_default": 0或1, "reasoning": "分析理由"}}
"""

    def _format_loan_info(self, loan_record: Dict) -> str:
        """格式化贷款信息为可读文本"""
        info_lines = []
        for key, value in loan_record.items():
            if key != self.target_column:
                info_lines.append(f"- {key}: {value}")
        return "\n".join(info_lines)

    def _call_llm(self, prompt: str, max_retries: int = 3) -> Dict:
        """调用LLM API进行推理

        Args:
            prompt: 输入prompt
            max_retries: 最大重试次数

        Returns:
            解析后的JSON响应
        """
        for attempt in range(max_retries):
            try:
                # 构造API请求
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }

                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 8192
                }

                # 调用API
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=300
                )

                if response.status_code != 200:
                    logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {response.status_code} {response.text}")
                    continue

                # 解析响应
                response_data = response.json()

                # 提取LLM输出文本（兼容OpenAI格式）
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    response_text = response_data["choices"][0].get("message", {}).get("content", "")
                else:
                    response_text = response_data.get("result", response_data.get("output", ""))

                response_text = (response_text or "").strip()

                # 尝试提取JSON块
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()

                # 使用json_repair容错解析
                parsed = json_repair.loads(response_text)
                return parsed

            except requests.Timeout:
                logger.warning(f"API call timeout (attempt {attempt + 1}/{max_retries})")
            except requests.RequestException as e:
                logger.warning(f"API request error (attempt {attempt + 1}/{max_retries}): {e}")
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error (attempt {attempt + 1}/{max_retries}): {e}")
            except Exception as e:
                logger.warning(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")

        # 所有重试失败，返回默认结果
        logger.error("All LLM call attempts failed")
        return {
            "predicted_default": 0,
            "confidence": 0.5,
            "reasoning": "LLM调用失败，无法评估"
        }

    def infer_single(self, loan_record: Dict) -> Dict:
        """对单个贷款样本进行推理

        Args:
            loan_record: 贷款记录字典

        Returns:
            包含预测结果的字典
        """
        # 格式化贷款信息
        loan_info = self._format_loan_info(loan_record)

        # 构造prompt
        prompt = self.prompt_template.format(loan_info=loan_info)

        # 调用LLM
        result = self._call_llm(prompt)

        # 提取预测结果 - 支持两种格式
        if "risk_assessment" in result:
            assessment = result["risk_assessment"]
        elif "predicted_default" in result:
            assessment = result
        else:
            assessment = {}

        try:
            pred = int(assessment.get("predicted_default", 0))
        except (ValueError, TypeError):
            pred = 0

        try:
            conf = float(assessment.get("confidence", 0.5))
        except (ValueError, TypeError):
            conf = 0.5

        return {
            "predicted_default": pred,
            "confidence": conf,
            "risk_level": assessment.get("risk_level", "未知"),
            "key_risk_signals": assessment.get("key_risk_signals", []),
            "reasoning": assessment.get("reasoning", "")
        }

    def _infer_single_with_index(self, idx: int, loan_record: Dict) -> Dict:
        prediction = self.infer_single(loan_record)
        return {
            "index": idx,
            **loan_record,
            **prediction
        }

    def infer_batch(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = None,
        save_interval: int = 50
    ) -> pd.DataFrame:
        """批量推理

        Args:
            df: 包含贷款数据的DataFrame
            batch_size: 并发推理的并行度（None表示全部串行）
            save_interval: 每多少条保存一次中间结果

        Returns:
            包含预测结果的DataFrame
        """
        total = len(df)
        max_workers = batch_size if batch_size and batch_size > 1 else 1
        logger.info(f"Batch inference on {total} samples, max_workers={max_workers}")

        results = []
        results_lock = threading.Lock()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pbar = tqdm(total=total, desc="Baseline Inference")

        def on_done(future, idx):
            try:
                result_record = future.result()
            except Exception as e:
                logger.error(f"Inference failed for sample {idx}: {e}")
                result_record = {
                    "index": idx,
                    "predicted_default": 0,
                    "confidence": 0.5,
                    "reasoning": f"推理异常: {e}"
                }
            with results_lock:
                results.append(result_record)
                pbar.update(1)
                if len(results) % save_interval == 0:
                    self._save_intermediate_results(results, timestamp)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, row in df.iterrows():
                loan_record = row.to_dict()
                fut = executor.submit(self._infer_single_with_index, idx, loan_record)
                fut.add_done_callback(lambda f, i=idx: on_done(f, i))
                futures.append(fut)
            concurrent.futures.wait(futures)

        pbar.close()

        result_df = pd.DataFrame(results)
        result_df.sort_values("index", inplace=True, ignore_index=True)

        self._save_final_results(result_df, timestamp)

        return result_df

    def _save_intermediate_results(self, results: List[Dict], timestamp: str):
        """保存中间结果"""
        output_path = self.output_dir / f"baseline_inference_intermediate_{timestamp}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved intermediate results: {len(results)} records")

    def _save_final_results(self, result_df: pd.DataFrame, timestamp: str):
        """保存最终结果"""
        # 保存CSV
        csv_path = self.output_dir / f"baseline_inference_{timestamp}.csv"
        result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved final results to {csv_path}")

        # 保存JSON
        json_path = self.output_dir / f"baseline_inference_{timestamp}.json"
        result_df.to_json(json_path, orient='records', force_ascii=False, indent=2)
        logger.info(f"Saved final results to {json_path}")

    def evaluate(self, result_df: pd.DataFrame) -> Dict:
        """评估baseline性能

        Args:
            result_df: 包含真实标签和预测结果的DataFrame

        Returns:
            评估指标字典
        """
        if self.target_column not in result_df.columns or 'predicted_default' not in result_df.columns:
            logger.error(f"Missing required columns for evaluation (need '{self.target_column}' and 'predicted_default')")
            return {}

        y_true = pd.to_numeric(result_df[self.target_column], errors='coerce').fillna(0).astype(int).values
        y_pred = pd.to_numeric(result_df['predicted_default'], errors='coerce').fillna(0).astype(int).values

        # 计算评估指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "total_samples": len(result_df),
            "default_count": int(y_true.sum()),
            "predicted_default_count": int(y_pred.sum())
        }

        # 尝试计算AUC（需要confidence分数）
        if 'confidence' in result_df.columns:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, result_df['confidence'].values)
            except Exception as e:
                logger.warning(f"Failed to calculate ROC AUC: {e}")

        logger.info(f"Baseline evaluation metrics: {metrics}")
        return metrics

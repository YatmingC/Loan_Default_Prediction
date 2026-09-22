"""Skill推理引擎

通过 Claude Code CLI 加载 skill 进行贷款风险评估。
Claude 从 benchmark_test/.claude/skills/ 自动加载 skill。

使用前需要手动准备 skill 目录结构：
  benchmark_test/
    .claude/
      skills/
        post-loan-management/
          SKILL.md
          references/
          scripts/

可以从 default_skill/skills/post-loan-management/ 复制过来。
"""

import json
import json_repair
import shutil
import subprocess
import threading
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from benchmark_test.logger import get_logger

logger = get_logger(__name__)


def _find_claude_executable() -> str:
    """查找 claude 可执行文件路径"""
    path = shutil.which("claude")
    if path:
        return path
    import sys
    venv_scripts = Path(sys.executable).parent
    for name in ("claude", "claude.exe", "claude.cmd"):
        candidate = venv_scripts / name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "Cannot find 'claude' executable. "
        "Make sure Claude Code CLI is installed and available in PATH or the current virtual environment."
    )


class SkillInferenceEngine:
    """Skill推理引擎 - 通过 Claude Code agent 加载 skill 进行风险评估"""

    def __init__(
        self,
        skill_name: str = "post-loan-management",
        output_dir: str = "./benchmark_test/results",
        target_column: str = "Default",
        max_workers: int = 4,
        claude_model: Optional[str] = None,
    ):
        """
        Args:
            skill_name: skill 名称（对应 .claude/skills/ 下的子目录名）
            output_dir: 输出目录
            target_column: 目标列名
            max_workers: 并发数
            claude_model: Claude 模型名称（如 claude-sonnet-4-6）
        """
        self.benchmark_dir = Path(__file__).parent.resolve()
        self.skill_dir = self.benchmark_dir / ".claude" / "skills" / skill_name
        self.skill_name = skill_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_column = target_column
        self.max_workers = max_workers
        self.claude_model = claude_model

        self.claude_executable = _find_claude_executable()

        if not self.skill_dir.exists():
            raise FileNotFoundError(
                f"Skill directory not found: {self.skill_dir}\n"
                f"Please copy skill files to benchmark_test/.claude/skills/{skill_name}/\n"
                f"Expected structure:\n"
                f"  benchmark_test/.claude/skills/{skill_name}/SKILL.md\n"
                f"  benchmark_test/.claude/skills/{skill_name}/references/\n"
                f"  benchmark_test/.claude/skills/{skill_name}/scripts/"
            )

        logger.info(f"Claude executable: {self.claude_executable}")
        logger.info(f"Working directory: {self.benchmark_dir}")
        logger.info(f"Skill directory: {self.skill_dir}")
        logger.info(f"Max workers: {self.max_workers}")
        if self.claude_model:
            logger.info(f"Claude model: {self.claude_model}")

    def _format_loan_info(self, loan_record: Dict) -> str:
        info_lines = []
        for key, value in loan_record.items():
            if key != self.target_column:
                info_lines.append(f"- {key}: {value}")
        return "\n".join(info_lines)

    def _call_claude(self, prompt: str, max_retries: int = 3) -> Dict:
        """通过 Claude Code CLI 调用 agent 进行推理

        agent 工作目录设为 skill 目录，prompt 引导它先阅读 SKILL.md 再进行评估。
        """
        for attempt in range(max_retries):
            try:
                cmd = [self.claude_executable, "--print", "-p", prompt]
                if self.claude_model:
                    cmd.extend(["--model", self.claude_model])

                result = subprocess.run(
                    cmd,
                    cwd=str(self.benchmark_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=600,
                    encoding="utf-8",
                    errors="replace",
                )

                logger.debug(f"Claude returncode={result.returncode}")
                if result.stderr:
                    logger.debug(f"Claude stderr: {result.stderr[:500]}")

                if result.returncode != 0:
                    logger.warning(
                        f"Claude CLI failed (attempt {attempt + 1}/{max_retries}): "
                        f"returncode={result.returncode}, stderr={result.stderr[:500]}"
                    )
                    continue

                response_text = (result.stdout or "").strip()
                if not response_text:
                    logger.warning(
                        f"Empty stdout (attempt {attempt + 1}/{max_retries}), "
                        f"stderr={result.stderr[:300] if result.stderr else 'none'}"
                    )
                    continue

                # 提取 JSON 块
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    extracted = response_text[json_start:json_end].strip()
                    if extracted.startswith("{"):
                        response_text = extracted

                parsed = json_repair.loads(response_text)
                return parsed

            except subprocess.TimeoutExpired:
                logger.warning(f"Claude CLI timeout (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                logger.warning(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")

        logger.error("All Claude CLI call attempts failed")
        return {
            "predicted_default": 0,
            "confidence": 0.5,
            "reasoning": "Claude CLI调用失败，无法评估",
        }

    def _build_prompt(self, loan_info: str) -> str:
        return (
            f"请根据你的 {self.skill_name} skill 中的评估框架和标准，"
            "对以下贷款信息进行风险评估。\n\n"
            f"## 贷款信息\n{loan_info}\n\n"
            "请输出JSON格式结果（不要输出其他内容）：\n"
            '{"predicted_default": 0或1, "confidence": 0.0到1.0, "risk_level": "风险等级", '
            '"key_risk_signals": ["信号1", "信号2"], "reasoning": "分析理由"}'
        )

    def infer_single(self, loan_record: Dict) -> Dict:
        loan_info = self._format_loan_info(loan_record)
        prompt = self._build_prompt(loan_info)
        result = self._call_claude(prompt)

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
            "reasoning": assessment.get("reasoning", ""),
        }

    def _infer_single_with_index(self, idx: int, loan_record: Dict) -> Dict:
        prediction = self.infer_single(loan_record)
        return {"index": idx, **loan_record, **prediction}

    def infer_batch(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = None,
        save_interval: int = 50,
    ) -> pd.DataFrame:
        total = len(df)
        max_workers = batch_size if batch_size and batch_size > 1 else self.max_workers
        logger.info(f"Skill inference on {total} samples, max_workers={max_workers}")

        results: List[Dict] = []
        results_lock = threading.Lock()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pbar = tqdm(total=total, desc="Skill Inference")

        def on_done(future, idx):
            try:
                result_record = future.result()
            except Exception as e:
                logger.error(f"Inference failed for sample {idx}: {e}")
                result_record = {
                    "index": idx,
                    "predicted_default": 0,
                    "confidence": 0.5,
                    "reasoning": f"推理异常: {e}",
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
        output_path = self.output_dir / f"skill_inference_intermediate_{timestamp}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved intermediate results: {len(results)} records")

    def _save_final_results(self, result_df: pd.DataFrame, timestamp: str):
        csv_path = self.output_dir / f"skill_inference_{timestamp}.csv"
        result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved final results to {csv_path}")

        json_path = self.output_dir / f"skill_inference_{timestamp}.json"
        result_df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        logger.info(f"Saved final results to {json_path}")

    def evaluate(self, result_df: pd.DataFrame) -> Dict:
        if self.target_column not in result_df.columns or "predicted_default" not in result_df.columns:
            logger.error(
                f"Missing required columns for evaluation "
                f"(need '{self.target_column}' and 'predicted_default')"
            )
            return {}

        y_true = pd.to_numeric(result_df[self.target_column], errors="coerce").fillna(0).astype(int).values
        y_pred = pd.to_numeric(result_df["predicted_default"], errors="coerce").fillna(0).astype(int).values

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "total_samples": len(result_df),
            "default_count": int(y_true.sum()),
            "predicted_default_count": int(y_pred.sum()),
        }

        if "confidence" in result_df.columns:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, result_df["confidence"].values)
            except Exception as e:
                logger.warning(f"Failed to calculate ROC AUC: {e}")

        logger.info(f"Skill inference evaluation metrics: {metrics}")
        return metrics

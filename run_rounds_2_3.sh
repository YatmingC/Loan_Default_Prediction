#!/usr/bin/env bash
#
# run_rounds_2_3.sh
# -----------------
# 1) 轮询等待 Round 1 的「重新验证」跑完（产物 round_1/outputs/results.json 落盘且可解析）。
# 2) 确认完成后，依次手动跑 Round 2、Round 3（每轮 3 个 step：分析上一轮 -> 演化 skill -> 重新验证）。
#
# 刻意写成「一字排开、不循环」的形式，方便你逐 step 对照监控。
# 每个 step 开始前会在终端打印步骤名。
#
# 参数路径完全参照 windows_manual_run.md 里 Round 1 的输出结构：
#   outputs/sba_multi_round/round_{N}/{error_analysis,success_analysis,skill,outputs,logs}
#
# 用法：
#   export OPENAI_API_KEY=<your_key>
#   # 如用兼容端点： export OPENAI_BASE_URL=<endpoint>
#   bash run_rounds_2_3.sh
#
# 可用环境变量覆盖默认值：DATA / MODEL / WORKERS / POLL_INTERVAL

set -uo pipefail

# 切到脚本所在目录（= 项目根），保证相对路径稳定
cd "$(dirname "$0")"

# ----------------------------------------------------------------------------
# 可调参数（默认对齐 windows_manual_run.md）
# ----------------------------------------------------------------------------
DATA="${DATA:-data/SBA/SBAcase.11.13.17.csv}"
MODEL="${MODEL:-gpt-4o}"
WORKERS="${WORKERS:-64}"
# LLM 端点（OpenAI 兼容）。默认取环境变量 OPENAI_BASE_URL。
BASE_URL="${BASE_URL:-${OPENAI_BASE_URL:-}}"
TASK_TYPE="credit_risk"

# 轮询间隔（秒）
POLL_INTERVAL="${POLL_INTERVAL:-60}"

BASE="outputs/sba_multi_round"

# ----------------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------------
step() {
  echo
  echo "============================================================"
  echo ">>> $1"
  echo "    [$(date '+%Y-%m-%d %H:%M:%S')]"
  echo "============================================================"
  echo
}

die() {
  echo "ERROR: $1" >&2
  exit 1
}

# 校验某轮的 results.json 是否已完整落盘（存在 + 能解析 + 有 accuracy + results 非空）
# 返回 0 = 完成，非 0 = 未完成
results_ready() {
  local results_json="$1"
  [ -f "$results_json" ] || return 1
  python -c "
import json, sys
try:
    with open(r'''$results_json''', encoding='utf-8') as f:
        d = json.load(f)
except Exception:
    sys.exit(1)
if not isinstance(d, dict):
    sys.exit(1)
if 'accuracy' not in d:
    sys.exit(1)
if not d.get('results'):
    sys.exit(1)
sys.exit(0)
" 2>/dev/null
}

# 打印某轮 accuracy
print_accuracy() {
  local label="$1"
  local results_json="$2"
  python -c "import json;print('$label accuracy =', json.load(open(r'''$results_json''', encoding='utf-8'))['accuracy'])" 2>/dev/null \
    || echo "$label accuracy = <读取失败: $results_json>"
}

# 跑完一轮「分析 -> 演化 -> 重新验证」的三个 step。
# 参数：prev_round_num  cur_round_num
run_round() {
  local PREV_N="$1"
  local CUR_N="$2"

  local PREV="$BASE/round_${PREV_N}"
  local CUR="$BASE/round_${CUR_N}"

  # --------------------------------------------------------------------------
  # Round CUR - Step 1：分析上一轮（round PREV）
  # --------------------------------------------------------------------------
  step "Round ${CUR_N} - Step 1a: 错误分析（分析 round_${PREV_N} 的 _FAILED 日志）"
  python analysis/run_error_analysis.py \
    --data_path "$PREV/outputs/results.json" \
    --logs_dir  "$PREV/logs" \
    --output_dir "$CUR/error_analysis" \
    --base_url "$BASE_URL" \
    --model "$MODEL" --task_type "$TASK_TYPE" --workers "$WORKERS" \
    || die "Round ${CUR_N} Step 1a 错误分析失败"

  step "Round ${CUR_N} - Step 1b: 成功分析（分析 round_${PREV_N} 的 _SUCCEED 日志）"
  python analysis/run_success_analysis_llm.py \
    --logs_dir  "$PREV/logs" \
    --output_dir "$CUR/success_analysis" \
    --base_url "$BASE_URL" \
    --model "$MODEL" --task_type "$TASK_TYPE" --max_workers "$WORKERS" \
    || die "Round ${CUR_N} Step 1b 成功分析失败"

  step "Round ${CUR_N} - Step 1c: 解析错误分析输出为结构化 JSON"
  python analysis/parse_error_analysis_outputs.py \
    --input_dir "$CUR/error_analysis" \
    --output    "$CUR/error_analysis_parsed.json" \
    || die "Round ${CUR_N} Step 1c 解析错误分析失败"

  step "Round ${CUR_N} - Step 1d: 解析成功分析输出为结构化 JSON"
  python analysis/parse_success_analysis_outputs.py \
    --input_dir "$CUR/success_analysis" \
    --output    "$CUR/success_analysis_parsed.json" \
    || die "Round ${CUR_N} Step 1d 解析成功分析失败"

  # --------------------------------------------------------------------------
  # Round CUR - Step 2：演化 skill（起点 = 上一轮演化后的 skill）
  # --------------------------------------------------------------------------
  step "Round ${CUR_N} - Step 2a: 从 round_${PREV_N}/skill 拷贝起点 skill 到 round_${CUR_N}/skill"
  mkdir -p "$CUR/skill"
  [ -f "$PREV/skill/SKILL.md" ] || die "找不到上一轮 skill: $PREV/skill/SKILL.md"
  cp -r "$PREV/skill/." "$CUR/skill/" \
    || die "Round ${CUR_N} Step 2a 拷贝 skill 失败"

  step "Round ${CUR_N} - Step 2b: 并行演化 skill（error + success 联合）"
  python -m skill_evolver.run_parallel_combined_skill_evolution \
    --error-json   "$CUR/error_analysis_parsed.json" \
    --success-json "$CUR/success_analysis_parsed.json" \
    --skill-dir    "$CUR/skill" \
    --base-url "$BASE_URL" \
    --model "$MODEL" --task-type "$TASK_TYPE" --max-workers "$WORKERS" --batch-size 4 \
    --continue-evolving \
    --changelog    "$CUR/evolution_changelog.md" \
    || die "Round ${CUR_N} Step 2b 演化失败"

  # --------------------------------------------------------------------------
  # Round CUR - Step 3：用演化后的 skill 重新验证
  # --------------------------------------------------------------------------
  step "Round ${CUR_N} - Step 3a: 用 round_${CUR_N}/skill 重新验证（生成轨迹 + results.json）"
  python run_sba_trace2skill.py \
    --data_path "$DATA" \
    --skill_path "$CUR/skill/SKILL.md" \
    --base_url "$BASE_URL" \
    --model "$MODEL" \
    --output_dir "$CUR/outputs" \
    --log_dir   "$CUR/logs" \
    --workers "$WORKERS" \
    || die "Round ${CUR_N} Step 3a 重新验证失败"

  step "Round ${CUR_N} - Step 3b: 给 round_${CUR_N} 日志打 _SUCCEED/_FAILED 后缀（供下一轮分析）"
  python rename_sba_logs.py \
    --results  "$CUR/outputs/results.json" \
    --logs_dir "$CUR/logs" \
    || die "Round ${CUR_N} Step 3b 打后缀失败"

  step "Round ${CUR_N} - 对比准确率"
  print_accuracy "round_${PREV_N}" "$PREV/outputs/results.json"
  print_accuracy "round_${CUR_N}" "$CUR/outputs/results.json"
}

# ============================================================================
# Step 0：轮询等待 Round 1 重新验证跑完
# ============================================================================
ROUND1_RESULTS="$BASE/round_1/outputs/results.json"
ROUND1_LOGS="$BASE/round_1/logs"

step "Step 0: 轮询等待 Round 1 重新验证完成（监测 $ROUND1_RESULTS）"
echo "轮询间隔：${POLL_INTERVAL}s。results.json 仅在整轮 run_and_save 末尾一次性落盘，是可靠的完成信号。"
echo "（校验：文件存在 + JSON 可解析 + 含 accuracy + results 非空，规避半写窗口。）"
echo

while true; do
  if results_ready "$ROUND1_RESULTS"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 检测到 Round 1 已完成。"
    break
  fi
  # 顺带打印一下已有多少 .md 日志，方便判断进度
  n_md=0
  if [ -d "$ROUND1_LOGS" ]; then
    n_md=$(find "$ROUND1_LOGS" -maxdepth 1 -name '*.md' 2>/dev/null | wc -l | tr -d ' ')
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 尚未完成（当前 logs 下 .md 数量: ${n_md}）。${POLL_INTERVAL}s 后重试..."
  sleep "$POLL_INTERVAL"
done

print_accuracy "round_1" "$ROUND1_RESULTS"

# Round 1 日志打后缀（幂等）。你手动跑时可能还没来得及执行 Round 1 Step 3b，
# 这里补一次，确保 Round 2 分析能发现 _SUCCEED / _FAILED 日志。
step "Step 0b: 确保 Round 1 日志已打后缀（幂等，供 Round 2 分析）"
python rename_sba_logs.py \
  --results  "$ROUND1_RESULTS" \
  --logs_dir "$ROUND1_LOGS" \
  || die "Round 1 日志打后缀失败"

# ============================================================================
# Round 2
# ============================================================================
run_round 1 2

# ============================================================================
# Round 3
# ============================================================================
run_round 2 3

step "全部完成：Round 2、Round 3 已跑完"
print_accuracy "round_1" "$BASE/round_1/outputs/results.json"
print_accuracy "round_2" "$BASE/round_2/outputs/results.json"
print_accuracy "round_3" "$BASE/round_3/outputs/results.json"
echo
echo "如需最终 held-out 评估，参考 windows_manual_run.md 末尾（--start_idx 200 --end_idx 400，选最佳轮次的 skill）。"

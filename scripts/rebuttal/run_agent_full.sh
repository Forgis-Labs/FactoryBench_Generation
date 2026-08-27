#!/usr/bin/env bash
# Full-test-set agentic eval driven by OpenAI direct gpt-5 (the same
# underlying model as the panel's gpt-5.1-1, routed through the openai.com
# endpoint because ETH's Foundry deployment blocks sync tool-calling).
#
# Reads prompts from output/test_eval/prompts/level<N>/, writes replies to
# output/test_eval/replies/level<N>/gpt-5-agent/, in the same JSON shape as
# every other panel row so downstream aggregators and the 3-judge L4 batch
# scorer pick them up unchanged.
#
# Log tails:
#   output/kuka_eval/agent_full_L1.log
#   output/kuka_eval/agent_full_L2.log
#   output/kuka_eval/agent_full_L3.log
#   output/kuka_eval/agent_full_L4.log
set -euo pipefail

REPO=/c/Users/ymerz/OneDrive/Documents/Work/Forgis/FactoryBench_Generation
PY="/c/Users/ymerz/miniconda3/envs/factorybench/python.exe"
MODEL=gpt-5-openai
CONC="${CONC:-4}"
COST_L1="${COST_L1:-120}"
COST_L2="${COST_L2:-110}"
COST_L3="${COST_L3:-15}"
COST_L4="${COST_L4:-120}"
PROMPTS_ROOT=output/test_eval/prompts
QUESTIONS_ROOT=output/test_eval/questions
REPLIES_ROOT=output/test_eval/replies
LOG_DIR=output/kuka_eval

cd "$REPO"
mkdir -p "$LOG_DIR"

_run_level() {
  local L="$1"; local CAP="$2"
  local out="$REPLIES_ROOT/level${L}/gpt-5-agent"
  mkdir -p "$out"
  echo "[$(date +%H:%M:%S)] L${L}: firing (cost cap \$${CAP}, concurrency ${CONC})"
  nohup env PYTHONIOENCODING=utf-8 PYTHONPATH=. "$PY" -u scripts/run_agent_eval.py \
    --input "$PROMPTS_ROOT/level${L}" \
    --output-dir "$out" \
    --questions "$QUESTIONS_ROOT/level${L}" \
    --model "$MODEL" \
    --cost-limit "$CAP" \
    --concurrency "$CONC" \
    > "$LOG_DIR/agent_full_L${L}.log" 2>&1 &
  local pid=$!
  echo "  L${L} PID $pid"
}

_run_level 1 "$COST_L1"
_run_level 2 "$COST_L2"
_run_level 3 "$COST_L3"
_run_level 4 "$COST_L4"

echo
echo "All four backgrounded. Tail with:"
echo "  tail -f $LOG_DIR/agent_full_L1.log"

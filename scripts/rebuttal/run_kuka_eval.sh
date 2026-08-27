#!/usr/bin/env bash
# Rebuttal W.2: KUKA industrial-episodes eval, run as a batched job via the
# same orchestrator the paper's main eval used.
#
# `scripts.eval_test_set` handles all of:
#   - chunked provider batch APIs (Bedrock CreateModelInvocationJob,
#     Foundry batch endpoint) with automatic sync fallback for tail
#     chunks smaller than 100 items,
#   - per-cell subprocess logs under <out>/logs/level<N>_<model_slug>.err.log,
#   - concurrent (level, model) cells,
#   - a top-line run log at <out>/run.log for at-a-glance status.
#
# We just point it at the KUKA questions we pre-staged with
# `stage_kuka_from_hf.py`. Nothing else about the eval loop differs from
# the main paper run.
#
# Prereqs (inside the target env):
#   pip install openai boto3        # foundry (Azure) + bedrock (AWS)
#   .env with FOUNDRY_*, AWS_*, HF_TOKEN
#
# To run as a background job (Git-Bash on Windows):
#   bash scripts/rebuttal/run_kuka_eval.sh > output/kuka_eval/run.log 2>&1 &
#   disown
#   tail -f output/kuka_eval/run.log
#
# Env overrides:
#   PY               interpreter (default: conda env's python.exe on Windows,
#                    otherwise python / python3)
#   OUT_ROOT         output root  (default: output/kuka_eval)
#   QA_DIR           staged KUKA questions dir  (default: output/kuka_qa)
#   COST_LIMIT       USD cap per cell  (default: 40)
#   MODELS           space-separated model list (default: full 6-model panel)
#   MODEL_CONCURRENCY  parallel (level, model) subprocesses (default: 4)
#   CHUNK_SIZE       items per batch job (default: 1000, orchestrator default)
#   SYNC_THRESHOLD   tail chunks smaller than this run sync (default: 100)

set -euo pipefail

# Force UTF-8 for Python I/O so ✓/✗ status glyphs and other non-ASCII output
# from the orchestrator don't hit cp1252 encode errors on Windows.
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

# --- interpreter selection (see notes at end of file) ------------------
if [ -n "${PY:-}" ]; then
  :
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/python.exe" ]; then
  PY="${CONDA_PREFIX}/python.exe"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PY="${CONDA_PREFIX}/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/Scripts/python.exe" ]; then
  PY="${VIRTUAL_ENV}/Scripts/python.exe"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
  PY="${VIRTUAL_ENV}/bin/python"
elif command -v python.exe >/dev/null 2>&1; then
  PY=python.exe
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  PY=python3
fi
echo "using interpreter: $("$PY" -c 'import sys; print(sys.executable)')"

if ! "$PY" -c 'import openai' >/dev/null 2>&1; then
  echo "!! openai not importable from $PY — install it into this env before continuing:"
  echo "     $PY -m pip install openai boto3"
  exit 1
fi

# --- config -----------------------------------------------------------
OUT_ROOT="${OUT_ROOT:-output/kuka_eval}"
QA_DIR="${QA_DIR:-output/kuka_qa}"
COST_LIMIT="${COST_LIMIT:-40.0}"
MODEL_CONCURRENCY="${MODEL_CONCURRENCY:-4}"
CHUNK_SIZE="${CHUNK_SIZE:-1000}"
SYNC_THRESHOLD="${SYNC_THRESHOLD:-100}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-}"

DEFAULT_MODELS=(
  gpt-5.1-1
  claude-sonnet-4.6
  mistral-large-3
  deepseek-v3.2
  qwen-3-235b
  qwen-3-4b
)
if [ -n "${MODELS:-}" ]; then
  read -ra MODEL_ARR <<< "$MODELS"
else
  MODEL_ARR=("${DEFAULT_MODELS[@]}")
fi

# --- stage questions from HF if not already done ----------------------
_have_staged() {
  for l in 1 2 3 4; do
    if ! ls "$QA_DIR/level$l"/*.json >/dev/null 2>&1; then return 1; fi
  done
  return 0
}
if _have_staged; then
  echo "==> phase 0: KUKA questions already staged under $QA_DIR (skipping stage_kuka_from_hf)"
else
  echo "==> phase 0: stage KUKA questions from Forgis/FactoryBench"
  PYTHONPATH="$PWD" "$PY" -m scripts.rebuttal.stage_kuka_from_hf --out-root "$QA_DIR"
fi

# --- fire the paper orchestrator as a batched job ---------------------
mkdir -p "$OUT_ROOT"
echo "==> firing scripts.eval_test_set on local KUKA questions"
EXTRA_EVAL=()
if [ -n "$MAX_OUTPUT_TOKENS" ]; then
  EXTRA_EVAL+=(--max-output-tokens "$MAX_OUTPUT_TOKENS")
fi
PYTHONPATH="$PWD" "$PY" -m scripts.eval_test_set \
  --local-questions-root "$QA_DIR" \
  --levels 1 2 3 4 \
  --models "${MODEL_ARR[@]}" \
  --output-root "$OUT_ROOT" \
  --cost-limit "$COST_LIMIT" \
  --model-concurrency "$MODEL_CONCURRENCY" \
  --chunk-size "$CHUNK_SIZE" \
  --sync-threshold "$SYNC_THRESHOLD" \
  "${EXTRA_EVAL[@]}"

# --- aggregate --------------------------------------------------------
echo "==> aggregating under signed chance-correction"
PYTHONPATH="$PWD" "$PY" scripts/rescore_signed.py \
  --replies-root "${OUT_ROOT}/replies" \
  --questions-root "$QA_DIR" \
  --out "${OUT_ROOT}/rescored_signed.json"

echo "==> done. summary: ${OUT_ROOT}/rescored_signed.json"
echo "    per-cell logs: ${OUT_ROOT}/logs/"
echo "    top-line log:  ${OUT_ROOT}/run.log"

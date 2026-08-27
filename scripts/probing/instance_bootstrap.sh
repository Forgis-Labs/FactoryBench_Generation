#!/usr/bin/env bash
# Runs ON the GPU instance. Sets up the env and runs the probing experiment.
# Expects the bundle unpacked at $HOME/factorybench (scripts/ + output/test_eval/).
set -euo pipefail

REPO="${1:-$HOME/factorybench}"
MODEL="${2:-Qwen/Qwen3-4B}"
NPC="${3:-600}"

cd "$REPO"
echo "== python / gpu =="
python3 --version
nvidia-smi || echo "WARNING: no GPU visible; will run on CPU (slow)"

echo "== deps (bare CUDA image: bootstrap pip + torch) =="
if ! python3 -m pip --version >/dev/null 2>&1; then
    sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip
fi
python3 -m pip install -q --upgrade pip
# torch is not preinstalled on the base image; install the CUDA 12.x wheel.
python3 -c "import torch" 2>/dev/null || \
    python3 -m pip install -q torch --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install -q --upgrade "transformers>=4.51" "accelerate>=0.30" \
    "scikit-learn>=1.3" "matplotlib>=3.7" "numpy<2" "safetensors>=0.4"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

OUT="$REPO/output/probing/$(echo "$MODEL" | tr '/:' '__')"
RAND_OUT="${OUT}_randinit"
mkdir -p "$OUT" "$RAND_OUT"

echo "== run probing (real model, 3 seeds, behavioural + negated control): model=$MODEL n_per_concept=$NPC =="
# Activations are extracted once at max_length=16384 and cached; the 3 seeds
# re-split/re-fit the probes (CPU) and the behavioural generation runs once.
python3 scripts/probing/run_probing.py \
    --repo "$REPO" --model "$MODEL" --out "$OUT" \
    --n-per-concept "$NPC" --seeds 0,1,2 --mlp --behavioural \
    2>&1 | tee "$OUT/run.log"

echo "== random-init control (untrained weights, same probe pipeline; fix 10) =="
python3 scripts/probing/run_probing.py \
    --repo "$REPO" --model "$MODEL" --out "$RAND_OUT" \
    --n-per-concept "$NPC" --randomize-weights --seeds 0 \
    2>&1 | tee "$RAND_OUT/run.log"

echo "== figures + summary (merging random-init control) =="
python3 scripts/probing/plot_probes.py \
    --results "$OUT/results.json" \
    --results-random "$RAND_OUT/results.json" \
    --out "$OUT"

echo "== DONE. Results in $OUT =="
ls -la "$OUT"

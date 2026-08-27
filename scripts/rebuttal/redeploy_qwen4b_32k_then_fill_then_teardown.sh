#!/usr/bin/env bash
# Second-pass Qwen3-4B: wait for the current 16k eval to finish, then swap the
# Vertex endpoint to a 32k-context deploy so we can hit the 12 items whose
# prompts overflow 16k. Runs the missing fill, then tears everything down.
set -euo pipefail

REPO=/c/Users/ymerz/OneDrive/Documents/Work/Forgis/FactoryBench_Generation
GCLOUD="/c/Users/ymerz/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd"
PROJECT=forgisprova
REGION=us-central1
PY="/c/Users/ymerz/miniconda3/envs/factorybench/python.exe"
RUN7_LOG="$REPO/output/kuka_eval/qwen4b_run7.log"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for run7 to complete"
until grep -q "progress: 4/4 cells done" "$RUN7_LOG" 2>/dev/null; do sleep 30; done
log "run7 done"

source "$REPO/output/kuka_eval/vertex_qwen4b.env"
log "current endpoint=$ENDPOINT_ID  16k-model=$MODEL_ID"

log "-- undeploy current model --"
DEPLOYED_ID=$(MSYS_NO_PATHCONV=1 "$GCLOUD" ai endpoints describe "$ENDPOINT_ID" --project=$PROJECT --region=$REGION --format="value(deployedModels[0].id)" 2>&1 | grep -oE '^[0-9]+$' | head -1)
log "deployedModelId=$DEPLOYED_ID"
MSYS_NO_PATHCONV=1 "$GCLOUD" ai endpoints undeploy-model "$ENDPOINT_ID" --project=$PROJECT --region=$REGION --deployed-model-id="$DEPLOYED_ID" --quiet 2>&1 | tail -3

log "-- delete 16k model resource --"
MSYS_NO_PATHCONV=1 "$GCLOUD" ai models delete "$MODEL_ID" --project=$PROJECT --region=$REGION --quiet 2>&1 | tail -3

log "-- upload 32k model --"
MSYS_NO_PATHCONV=1 "$GCLOUD" ai models upload --project=$PROJECT --region=$REGION \
  --display-name=qwen3-4b-instruct-2507-32k \
  --container-image-uri=us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20250601_0916_RC01 \
  --container-command="python,-m,vllm.entrypoints.openai.api_server" \
  --container-args="--model=Qwen/Qwen3-4B-Instruct-2507,--dtype=half,--max-model-len=32768,--gpu-memory-utilization=0.93,--max-num-seqs=1,--served-model-name=qwen-3-4b,--host=0.0.0.0,--port=8080" \
  --container-ports=8080 --container-predict-route=/v1/chat/completions --container-health-route=/health 2>&1 | tail -3

MODEL32K=$(MSYS_NO_PATHCONV=1 "$GCLOUD" ai models list --project=$PROJECT --region=$REGION \
  --filter="displayName=qwen3-4b-instruct-2507-32k" --format="value(name.basename())" \
  --sort-by="~updateTime" 2>&1 | grep -oE '^[0-9]+$' | head -1)
log "uploaded 32k model=$MODEL32K"

log "-- deploy 32k on T4 (blocking) --"
MSYS_NO_PATHCONV=1 "$GCLOUD" ai endpoints deploy-model "$ENDPOINT_ID" --project=$PROJECT --region=$REGION \
  --model="$MODEL32K" --display-name=qwen3-4b-32k-t4 \
  --machine-type=n1-standard-4 --accelerator=type=nvidia-tesla-t4,count=1 \
  --min-replica-count=1 --max-replica-count=1 --traffic-split=0=100 2>&1 | tail -3
log "32k deploy done"

# update state file with new model id
sed -i "s|^MODEL_ID=.*|MODEL_ID=$MODEL32K|" "$REPO/output/kuka_eval/vertex_qwen4b.env"

log "-- run KUKA fill (existing 575 items skipped) --"
cd "$REPO"
env PY="$PY" MODELS="qwen-3-4b" MODEL_CONCURRENCY=1 bash scripts/rebuttal/run_kuka_eval.sh 2>&1 | tail -50

log "-- final per-cell counts --"
for L in level1 level2 level3 level4; do
  case $L in level1) e=98;; level2) e=346;; level3) e=24;; level4) e=119;; esac
  n=$(ls "$REPO/output/kuka_eval/replies/$L/qwen-3-4b/"*_answer.json 2>/dev/null | wc -l)
  log "  $L: $n/$e"
done

log "-- teardown all Vertex resources --"
DEPLOYED_ID2=$(MSYS_NO_PATHCONV=1 "$GCLOUD" ai endpoints describe "$ENDPOINT_ID" --project=$PROJECT --region=$REGION --format="value(deployedModels[0].id)" 2>&1 | grep -oE '^[0-9]+$' | head -1)
if [ -n "$DEPLOYED_ID2" ]; then
  MSYS_NO_PATHCONV=1 "$GCLOUD" ai endpoints undeploy-model "$ENDPOINT_ID" --project=$PROJECT --region=$REGION --deployed-model-id="$DEPLOYED_ID2" --quiet 2>&1 | tail -2
fi
MSYS_NO_PATHCONV=1 "$GCLOUD" ai models delete "$MODEL32K" --project=$PROJECT --region=$REGION --quiet 2>&1 | tail -2
MSYS_NO_PATHCONV=1 "$GCLOUD" ai endpoints delete "$ENDPOINT_ID" --project=$PROJECT --region=$REGION --quiet 2>&1 | tail -2
rm -f "$REPO/output/kuka_eval/vertex_qwen4b.env"
log "TEARDOWN COMPLETE — no lingering Vertex resources"

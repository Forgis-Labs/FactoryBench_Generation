#!/usr/bin/env bash
# Self-host Qwen3-4B-Instruct-2507 on Vertex AI as a custom-container endpoint
# on a single T4 GPU. Container: prebuilt vLLM OpenAI-compatible api_server
# from Vertex Model Garden. All resource IDs land in an .env-style file at
# output/kuka_eval/vertex_qwen4b.env so the runner + teardown share state.
#
# Cleanup: run this script with the arg "teardown" to undeploy + delete
# everything provisioned here.
#
# Cost budget: ~$0.54/hr while endpoint is up (n1-standard-4 + 1x T4). The
# deploy → eval → teardown loop takes ~45-60 min end-to-end -> ~$0.50-0.80.
#
# Prereqs: gcloud installed + `gcloud auth login` done; billing enabled on
# `forgisprova`; aiplatform.googleapis.com and compute.googleapis.com
# enabled; T4 quota >= 1 in us-central1.

set -euo pipefail

GCLOUD_BIN="${GCLOUD_BIN:-/c/Users/ymerz/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd}"
GCLOUD() { MSYS_NO_PATHCONV=1 "$GCLOUD_BIN" "$@"; }

PROJECT="${PROJECT:-forgisprova}"
REGION="${REGION:-us-central1}"
MODEL_HF_ID="${MODEL_HF_ID:-Qwen/Qwen3-4B-Instruct-2507}"
DISPLAY_NAME="${DISPLAY_NAME:-qwen3-4b-instruct-2507}"
STATE_FILE="${STATE_FILE:-output/kuka_eval/vertex_qwen4b.env}"
mkdir -p "$(dirname "$STATE_FILE")"

# vLLM image + args tuned for T4 (fp16 only, small max-model-len to fit KV
# cache in 16 GB VRAM alongside 8 GB of fp16 weights).
CONTAINER_IMAGE="us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20250601_0916_RC01"

_state_load() { [ -f "$STATE_FILE" ] && . "$STATE_FILE" || true; }
_state_save() {
  echo "MODEL_ID=${MODEL_ID:-}"           > "$STATE_FILE"
  echo "ENDPOINT_ID=${ENDPOINT_ID:-}"    >> "$STATE_FILE"
  echo "DEPLOYED_MODEL_ID=${DEPLOYED_MODEL_ID:-}" >> "$STATE_FILE"
  echo "PROJECT=$PROJECT"                >> "$STATE_FILE"
  echo "REGION=$REGION"                  >> "$STATE_FILE"
  echo "DISPLAY_NAME=$DISPLAY_NAME"      >> "$STATE_FILE"
}

log() { echo "[$(date +%H:%M:%S)] $*"; }

_state_load

# ---------------------------------------------------------------------
if [ "${1:-deploy}" = "teardown" ]; then
  log "== teardown =="
  if [ -n "${ENDPOINT_ID:-}" ]; then
    if [ -n "${DEPLOYED_MODEL_ID:-}" ]; then
      log "undeploying model $DEPLOYED_MODEL_ID from endpoint $ENDPOINT_ID"
      GCLOUD ai endpoints undeploy-model "$ENDPOINT_ID" \
        --project="$PROJECT" --region="$REGION" \
        --deployed-model-id="$DEPLOYED_MODEL_ID" --quiet 2>&1 | tail -3 || true
    fi
    log "deleting endpoint $ENDPOINT_ID"
    GCLOUD ai endpoints delete "$ENDPOINT_ID" \
      --project="$PROJECT" --region="$REGION" --quiet 2>&1 | tail -3 || true
  fi
  if [ -n "${MODEL_ID:-}" ]; then
    log "deleting model $MODEL_ID"
    GCLOUD ai models delete "$MODEL_ID" \
      --project="$PROJECT" --region="$REGION" --quiet 2>&1 | tail -3 || true
  fi
  rm -f "$STATE_FILE"
  log "teardown done"
  exit 0
fi

# ---------------------------------------------------------------------
log "== 1/4 upload model resource =="
if [ -z "${MODEL_ID:-}" ]; then
  # vLLM OpenAI-compatible server; predict route is /v1/chat/completions,
  # health on /health. Tuned args: dtype=half (T4 no bf16), tight max-model-len.
  RESP=$(GCLOUD ai models upload \
    --project="$PROJECT" \
    --region="$REGION" \
    --display-name="$DISPLAY_NAME" \
    --container-image-uri="$CONTAINER_IMAGE" \
    --container-command="python,-m,vllm.entrypoints.openai.api_server" \
    --container-args="--model=$MODEL_HF_ID,--dtype=half,--max-model-len=8192,--gpu-memory-utilization=0.90,--served-model-name=qwen-3-4b,--host=0.0.0.0,--port=8080" \
    --container-ports=8080 \
    --container-predict-route=/v1/chat/completions \
    --container-health-route=/health \
    --format="value(name)" 2>&1)
  MODEL_ID=$(echo "$RESP" | grep -oE "[0-9]{5,}" | tail -1)
  if [ -z "$MODEL_ID" ]; then
    echo "!! failed to parse MODEL_ID from upload response:"; echo "$RESP"; exit 1
  fi
  _state_save
fi
log "MODEL_ID=$MODEL_ID"

log "== 2/4 create endpoint =="
if [ -z "${ENDPOINT_ID:-}" ]; then
  RESP=$(GCLOUD ai endpoints create \
    --project="$PROJECT" --region="$REGION" \
    --display-name="${DISPLAY_NAME}-endpoint" \
    --format="value(name)" 2>&1)
  ENDPOINT_ID=$(echo "$RESP" | grep -oE "[0-9]{5,}" | tail -1)
  if [ -z "$ENDPOINT_ID" ]; then
    echo "!! failed to parse ENDPOINT_ID:"; echo "$RESP"; exit 1
  fi
  _state_save
fi
log "ENDPOINT_ID=$ENDPOINT_ID"

log "== 3/4 deploy model to endpoint (n1-standard-4 + 1x T4) =="
if [ -z "${DEPLOYED_MODEL_ID:-}" ]; then
  RESP=$(GCLOUD ai endpoints deploy-model "$ENDPOINT_ID" \
    --project="$PROJECT" --region="$REGION" \
    --model="$MODEL_ID" \
    --display-name="${DISPLAY_NAME}-deploy" \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --min-replica-count=1 --max-replica-count=1 \
    --traffic-split=0=100 2>&1)
  # deploy-model prints the deployedModelId in stderr progress logs
  DEPLOYED_MODEL_ID=$(echo "$RESP" | grep -oE '"id":\s*"[0-9]+"' | head -1 | grep -oE '[0-9]+' || true)
  if [ -z "$DEPLOYED_MODEL_ID" ]; then
    # fallback: list deployed models on the endpoint
    DEPLOYED_MODEL_ID=$(GCLOUD ai endpoints describe "$ENDPOINT_ID" \
      --project="$PROJECT" --region="$REGION" \
      --format="value(deployedModels[0].id)" 2>&1 | tr -d '\r')
  fi
  _state_save
fi
log "DEPLOYED_MODEL_ID=$DEPLOYED_MODEL_ID"

log "== 4/4 smoke test =="
TOKEN=$(GCLOUD auth print-access-token 2>&1 | tr -d '\r')
BASE="https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT}/locations/${REGION}/endpoints/${ENDPOINT_ID}"
curl -sS "${BASE}:rawPredict" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"qwen-3-4b","messages":[{"role":"user","content":"reply with the single word: pong"}],"max_tokens":8}' \
  | head -c 800; echo

log "== done. state saved to $STATE_FILE =="
log "     use \`bash $0 teardown\` to remove endpoint + model when eval is done."

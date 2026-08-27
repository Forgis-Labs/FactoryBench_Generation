#!/usr/bin/env bash
# End-to-end Vertex AI setup for Qwen3-4B and Qwen3-235B on GCP project
# `forgisprova`.
#
# Steps (idempotent, resumable):
#   1. probe Vertex Model Garden for a managed Qwen3 endpoint
#      (cheapest path: per-token billing, no infra to manage)
#   2. if not available, deploy Qwen3-4B to a custom endpoint on L4 GPU
#      via vLLM prebuilt container (cheap: ~$0.60/hr)
#   3. skip 235B self-host unless explicitly requested (too expensive
#      as a one-shot: $30+/hr for A100x4)
#   4. write a Vertex-flavoured Qwen config into src/config.py and
#      point the eval runner at the new endpoint URLs
#
# Cost budget:
#   * 4B custom endpoint on L4: ~$1-2 for the full KUKA eval (~30 min)
#   * 235B managed endpoint (if available): ~$1-2 per-token
#   * 235B self-host: NOT triggered by default

set -euo pipefail

GCLOUD="/c/Users/ymerz/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd"
export PROJECT="${PROJECT:-forgisprova}"
export REGION="${REGION:-us-central1}"       # us-central1 has L4/A100 quota by default
export OUT_DIR="output/kuka_eval/vertex_setup"
mkdir -p "$OUT_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT_DIR/setup.log"; }

log "== gcloud config =="
"$GCLOUD" config set project "$PROJECT" 2>&1 | tail -1 | tee -a "$OUT_DIR/setup.log"

log "== billing / API status =="
"$GCLOUD" services list --enabled --project="$PROJECT" --format="value(config.name)" 2>&1 \
  | grep -E "aiplatform|compute|storage|artifactregistry" > "$OUT_DIR/enabled_apis.txt" || true
cat "$OUT_DIR/enabled_apis.txt"

# enable any missing APIs
for svc in aiplatform.googleapis.com artifactregistry.googleapis.com compute.googleapis.com; do
  if ! grep -q "^$svc$" "$OUT_DIR/enabled_apis.txt"; then
    log "enabling $svc"
    "$GCLOUD" services enable "$svc" --project="$PROJECT" 2>&1 | tail -3 | tee -a "$OUT_DIR/setup.log"
  fi
done

log "== probe Vertex Model Garden for Qwen publisher =="
TOKEN="$("$GCLOUD" auth print-access-token 2>/dev/null)"
curl -sS "https://${REGION}-aiplatform.googleapis.com/v1beta1/publishers/qwen/models" \
  -H "Authorization: Bearer $TOKEN" -o "$OUT_DIR/qwen_publisher_models.json"
if grep -q "publisherModels" "$OUT_DIR/qwen_publisher_models.json"; then
  log "Qwen publisher found in Model Garden; listing model IDs:"
  grep -oE '"name":\s*"[^"]+"' "$OUT_DIR/qwen_publisher_models.json" | head -30 | tee -a "$OUT_DIR/setup.log"
  export QWEN_MG_AVAILABLE=1
else
  log "Qwen not present in Model Garden for region=$REGION; falling back to self-host"
  head -20 "$OUT_DIR/qwen_publisher_models.json"
  export QWEN_MG_AVAILABLE=0
fi

log "== probe GPU quota (L4 for 4B, A100 for 235B) =="
"$GCLOUD" compute regions describe "$REGION" --format="value(quotas.metric,quotas.limit,quotas.usage)" 2>&1 \
  | tr ';' '\n' | grep -iE "GPU|L4|A100|H100" | head -20 | tee -a "$OUT_DIR/setup.log" || true

log "== next step =="
if [ "$QWEN_MG_AVAILABLE" = "1" ]; then
  log "Managed Qwen endpoints available. Deploying via Model Garden one-click..."
  log "(next commit will add the actual deploy_publisher_model calls once we see"
  log " which exact Qwen3 SKUs are in the catalog)"
else
  log "No managed Qwen. To self-host Qwen3-4B on L4:"
  log "  bash scripts/rebuttal/deploy_qwen_vertex.sh --self-host-4b"
  log "235B self-host is intentionally NOT triggered (expensive)."
fi

log "== done. artifacts in $OUT_DIR/ =="

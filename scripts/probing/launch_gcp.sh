#!/usr/bin/env bash
# Orchestrates the FactoryBench probing run on a GCP GPU instance:
#   1. bundles the scripts + faithful prompts/questions (levels 1 & 4)
#   2. creates a Deep-Learning GPU VM (spot, auto-deletes on stop)
#   3. copies the bundle up, runs instance_bootstrap.sh
#   4. copies output/probing/ back down
#
# Prereqs (do these once, interactively, before running this script):
#   gcloud auth login <account>
#   gcloud config set project <PROJECT_ID>
#   # GPU quota must be > 0 in $ZONE (else use the CPU fallback in the README).
#
# Usage:
#   bash scripts/probing/launch_gcp.sh PROJECT_ID [ZONE] [MACHINE] [GPU] [MODEL]
set -euo pipefail

PROJECT="${1:?need PROJECT_ID}"
ZONE="${2:-us-central1-a}"
MACHINE="${3:-n1-standard-8}"
GPU="${4:-nvidia-tesla-t4}"
MODEL="${5:-Qwen/Qwen3-4B}"
NPC="${6:-600}"
VM="factorybench-probe"
REPO_LOCAL="$(cd "$(dirname "$0")/../.." && pwd)"

echo "== bundling data from $REPO_LOCAL =="
BUNDLE="$(mktemp -d)/fb_bundle.tar.gz"
tar -czf "$BUNDLE" -C "$REPO_LOCAL" \
    scripts/probing/run_probing.py \
    scripts/probing/plot_probes.py \
    scripts/probing/instance_bootstrap.sh \
    output/test_eval/prompts/level1 \
    output/test_eval/prompts/level4 \
    output/test_eval/questions/level1 \
    output/test_eval/questions/level4
echo "bundle: $(du -h "$BUNDLE" | cut -f1)"

echo "== creating spot GPU VM $VM ($MACHINE + $GPU) in $ZONE =="
gcloud compute instances create "$VM" \
    --project="$PROJECT" --zone="$ZONE" \
    --machine-type="$MACHINE" \
    --accelerator="type=$GPU,count=1" \
    --image-family="common-cu121-debian-11" --image-project="deeplearning-platform-release" \
    --maintenance-policy=TERMINATE --provisioning-model=SPOT \
    --instance-termination-action=DELETE \
    --boot-disk-size=100GB --metadata="install-nvidia-driver=True"

echo "== waiting for SSH =="
for i in $(seq 1 30); do
    if gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" \
        --command="echo up" >/dev/null 2>&1; then break; fi
    echo "  ...retry $i"; sleep 15
done

echo "== uploading bundle =="
gcloud compute scp "$BUNDLE" "$VM":~/fb_bundle.tar.gz --zone="$ZONE" --project="$PROJECT"
gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" \
    --command="mkdir -p ~/factorybench && tar -xzf ~/fb_bundle.tar.gz -C ~/factorybench && chmod +x ~/factorybench/scripts/probing/instance_bootstrap.sh"

echo "== running experiment (this streams; ~20-40 min on T4) =="
gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" \
    --command="bash ~/factorybench/scripts/probing/instance_bootstrap.sh ~/factorybench '$MODEL' '$NPC'"

echo "== pulling results back =="
mkdir -p "$REPO_LOCAL/output/probing"
gcloud compute scp --recurse \
    "$VM":~/factorybench/output/probing "$REPO_LOCAL/output/" \
    --zone="$ZONE" --project="$PROJECT"

echo "== deleting VM =="
gcloud compute instances delete "$VM" --zone="$ZONE" --project="$PROJECT" --quiet

echo "== DONE. Local results in $REPO_LOCAL/output/probing =="

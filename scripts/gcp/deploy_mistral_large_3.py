#!/usr/bin/env python3
"""Deploy / inspect / tear down Mistral Large 3 on Vertex AI.

Mistral Large 3 is the one model in the benchmark that did not survive the AWS
to GCP migration as a managed service. On Bedrock it was serverless and billed
per token. On Vertex it is not offered as MaaS at all — Model Garden ships it
only as a self-deploy vLLM container:

    image  us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/
           pytorch-vllm-serve:20251205_0916_RC01
    model  gs://vertex-model-garden-restricted-us/mistralai/
           Mistral-Large-3-675B-Instruct-2512
    shape  a3-ultragpu-8g (8x H200 141GB) or a4-highgpu-8g (8x B200)

Same weights as the Bedrock model (675B MoE, Instruct-2512), so the evaluation
stays comparable. The cost model does not: this bills per GPU-hour from the
moment the endpoint comes up until it is deleted, whether or not you send it a
single prompt. Deploying and forgetting is the expensive failure mode here, so
this script pairs every --create with a --delete and refuses to act without an
explicit --yes.

Usage:
    python scripts/gcp/deploy_mistral_large_3.py                  # plan only
    python scripts/gcp/deploy_mistral_large_3.py --create --yes   # deploy
    python scripts/gcp/deploy_mistral_large_3.py --status
    python scripts/gcp/deploy_mistral_large_3.py --delete --yes   # tear down

After --create, put the printed endpoint id in .env:
    MISTRAL_LARGE_3_VERTEX_ENDPOINT=<numeric id>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import GCP_PROJECT_DEFAULT, VERTEX_MODELS  # noqa: E402
from src.evaluation.run_foundry_eval import _vertex_access_token  # noqa: E402
from src.evaluation.test_gpt_5mini import load_dotenv_file  # noqa: E402

PUBLISHER_MODEL = "publishers/mistralai/models/mistral-large-3"
DISPLAY_NAME = "factorybench-mistral-large-3"

# The two shapes Model Garden publishes for this checkpoint. H200 first: it is
# the cheaper of the two and the one the catalog lists as the default.
MACHINE_SHAPES = {
    "h200": {"machineType": "a3-ultragpu-8g", "acceleratorType": "NVIDIA_H200_141GB", "acceleratorCount": 8},
    "b200": {"machineType": "a4-highgpu-8g", "acceleratorType": "NVIDIA_B200", "acceleratorCount": 8},
}

# Deliberately not hardcoding a dollar figure — GPU list prices move and vary
# by region and commitment. This is the order of magnitude to sanity-check
# against, and the calculator link gives the current number.
COST_WARNING = """
  COST: an 8-GPU A3-Ultra / A4 node is on the order of tens of dollars per HOUR
  in us-central1, billed from endpoint creation until deletion regardless of
  traffic. Left up for a week that is four figures. Confirm the current rate at
  https://cloud.google.com/products/calculator before running with --yes, and
  tear the endpoint down with --delete the moment the eval finishes.
"""


def _project() -> str:
    return os.environ.get("GCP_PROJECT") or GCP_PROJECT_DEFAULT


def _region() -> str:
    cfg = VERTEX_MODELS["mistral-large-3"]
    return os.environ.get(cfg["region_env"]) or cfg["region_default"]


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_vertex_access_token()}",
        "Content-Type": "application/json",
        "x-goog-user-project": _project(),
    }


def _host(region: str) -> str:
    return f"{region}-aiplatform.googleapis.com"


def cmd_status(project: str, region: str) -> int:
    """List Vertex endpoints matching our display name."""
    url = f"https://{_host(region)}/v1/projects/{project}/locations/{region}/endpoints"
    r = requests.get(url, headers=_headers(), timeout=120)
    if r.status_code >= 400:
        print(f"endpoints.list failed: {r.status_code} {r.text[:300]}")
        return 1
    endpoints = [
        e for e in r.json().get("endpoints", [])
        if DISPLAY_NAME in (e.get("displayName") or "")
    ]
    if not endpoints:
        print(f"No endpoint named {DISPLAY_NAME} in {project}/{region}. Nothing is running.")
        return 0
    for e in endpoints:
        eid = e["name"].split("/")[-1]
        deployed = e.get("deployedModels", [])
        print(f"endpoint {eid}  displayName={e.get('displayName')}")
        print(f"  createTime    : {e.get('createTime')}")
        print(f"  deployedModels: {len(deployed)}")
        for d in deployed:
            spec = (d.get("dedicatedResources") or {}).get("machineSpec", {})
            print(
                f"    - {d.get('displayName')} id={d.get('id')} "
                f"machine={spec.get('machineType')} "
                f"accel={spec.get('acceleratorType')}x{spec.get('acceleratorCount')}"
            )
        if deployed:
            print(f"  THIS IS BILLING NOW. Tear down: --delete --yes")
        print(f"  .env: MISTRAL_LARGE_3_VERTEX_ENDPOINT={eid}")
    return 0


def cmd_create(project: str, region: str, shape: str, apply: bool) -> int:
    machine = MACHINE_SHAPES[shape]
    body = {
        "publisherModelName": PUBLISHER_MODEL,
        "destination": f"projects/{project}/locations/{region}",
        "modelConfig": {"acceptEula": True},
        "endpointConfig": {"endpointDisplayName": DISPLAY_NAME},
        "deployConfig": {
            "dedicatedResources": {
                "machineSpec": machine,
                "minReplicaCount": 1,
                "maxReplicaCount": 1,
            }
        },
    }
    url = f"https://{_host(region)}/v1beta1/projects/{project}/locations/{region}:deploy"

    print(f"POST {url}")
    print(json.dumps(body, indent=2))
    print(COST_WARNING)
    if not apply:
        print("Plan only. Nothing was deployed. Re-run with --create --yes to execute.")
        return 0

    r = requests.post(url, headers=_headers(), json=body, timeout=600)
    if r.status_code >= 400:
        print(f"deploy failed: {r.status_code} {r.text[:800]}")
        if "quota" in r.text.lower():
            print(
                "\nThis reads like a quota denial. 8xH200 needs "
                "`custom_model_serving_a3_ultra_gpus` (or the B200 equivalent) "
                "in the target region. Request it in IAM & Admin -> Quotas."
            )
        return 1
    op = r.json()
    print(f"\nDeploy operation started: {op.get('name')}")
    print("This takes 30-60 minutes for a 675B checkpoint.")
    print("Poll with: python scripts/gcp/deploy_mistral_large_3.py --status")
    return 0


def cmd_delete(project: str, region: str, apply: bool) -> int:
    """Undeploy models then delete the endpoint. Both are required — deleting
    an endpoint with a model still attached fails, and an undeployed-but-alive
    endpoint costs nothing but leaves confusing state behind."""
    url = f"https://{_host(region)}/v1/projects/{project}/locations/{region}/endpoints"
    r = requests.get(url, headers=_headers(), timeout=120)
    if r.status_code >= 400:
        print(f"endpoints.list failed: {r.status_code} {r.text[:300]}")
        return 1
    endpoints = [
        e for e in r.json().get("endpoints", [])
        if DISPLAY_NAME in (e.get("displayName") or "")
    ]
    if not endpoints:
        print("Nothing to delete.")
        return 0

    for e in endpoints:
        name = e["name"]
        for d in e.get("deployedModels", []):
            print(f"undeploy {d.get('id')} from {name}")
            if apply:
                resp = requests.post(
                    f"https://{_host(region)}/v1/{name}:undeployModel",
                    headers=_headers(), json={"deployedModelId": d["id"]}, timeout=600,
                )
                print(f"  -> {resp.status_code}")
                if resp.status_code < 400:
                    # Endpoint deletion 400s while an undeploy is still in
                    # flight, so give the operation a moment to settle.
                    time.sleep(10)
        print(f"delete endpoint {name}")
        if apply:
            resp = requests.delete(
                f"https://{_host(region)}/v1/{name}", headers=_headers(), timeout=600
            )
            print(f"  -> {resp.status_code} {resp.text[:200]}")

    if not apply:
        print("\nPlan only. Nothing was deleted. Re-run with --delete --yes.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--create", action="store_true", help="Deploy the endpoint.")
    action.add_argument("--status", action="store_true", help="Show what is running.")
    action.add_argument("--delete", action="store_true", help="Undeploy and delete the endpoint.")
    parser.add_argument("--shape", choices=list(MACHINE_SHAPES), default="h200")
    parser.add_argument("--yes", action="store_true", help="Actually execute. Without it, plan only.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    if args.env_file.exists():
        load_dotenv_file(args.env_file)

    project, region = _project(), _region()
    print(f"project={project} region={region}\n")

    if args.status:
        return cmd_status(project, region)
    if args.delete:
        return cmd_delete(project, region, args.yes)
    if args.create:
        return cmd_create(project, region, args.shape, args.yes)

    parser.print_help()
    print(COST_WARNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

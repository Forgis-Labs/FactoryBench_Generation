"""Shared configuration for FactoryBench pipeline and evaluation.

Single source of truth for the model catalog (provider routing, endpoints,
batch support) and pipeline-wide defaults. Adding a new model is one dict edit
here and every pipeline/evaluation script picks it up automatically.

Three providers are supported:

  * ``foundry``   — Microsoft Azure AI Foundry. Used for OpenAI-style models
                    (GPT-5.x) that are not available on AWS.
  * ``bedrock``   — AWS Bedrock. Native managed inference for Anthropic
                    (CRIS profile in EU), Mistral, and DeepSeek (fully-managed
                    serverless). Native batch via ``CreateModelInvocationJob``
                    (S3-in / S3-out, ~50% cheaper than on-demand).
  * ``sagemaker`` — AWS SageMaker Async Inference. Used for self-hosted models
                    (e.g. Qwen via JumpStart). Per-request S3 input + S3 output
                    via ``InvokeEndpointAsync`` against a deployed endpoint with
                    scale-to-zero.

Each AWS model declares its own region because availability differs per model
(Mistral Large 3 has no EU region; DeepSeek V3.1 is in eu-west-2 / eu-north-1
only; Sonnet 4.6 in EU uses the CRIS inference profile from eu-central-1). See
``src/evaluation/aws-setup.md`` for the verified id / region table.

Model ids and regions are resolved at runtime from environment variables — the
catalog ships without baking in values that change per AWS account.
"""
from __future__ import annotations

from typing import Any, Dict, List


MODELS: Dict[str, Dict[str, Any]] = {
    # --- Azure Foundry (kept for OpenAI proxy; AWS does not host GPT-5.x) ---
    "gpt-5.1-1": {
        "provider": "foundry",
        "endpoint_env": "CHAT_ENDPOINT",
        "endpoint_default": "https://student-research-lab-resource.services.ai.azure.com/openai/v1",
        "api_style": "openai",
        "supports_batch": True,
    },

    # --- AWS Bedrock (managed; native batch via S3) -----------------------
    # Verified ids/regions per src/evaluation/aws-setup.md (Apr 2026).
    "claude-sonnet-4.6": {
        "provider": "bedrock",
        "model_id_env": "CLAUDE_SONNET_46_MODEL_ID",        # eu.anthropic.claude-sonnet-4-6 (CRIS)
        "region_env": "CLAUDE_SONNET_46_REGION",            # eu-central-1
        "api_style": "anthropic",
        "supports_batch": True,
    },
    "mistral-large-3": {
        "provider": "bedrock",
        "model_id_env": "MISTRAL_LARGE_3_MODEL_ID",         # mistral.mistral-large-3-675b-instruct
        "region_env": "MISTRAL_LARGE_3_REGION",             # us-west-2 (no EU region yet)
        "api_style": "mistral",
        "supports_batch": True,
    },
    "deepseek-v3.2": {
        "provider": "bedrock",
        "model_id_env": "DEEPSEEK_V32_MODEL_ID",            # deepseek.v3.2
        "region_env": "DEEPSEEK_V32_REGION",                # eu-west-2 or eu-north-1
        "api_style": "deepseek",
        "supports_batch": True,
    },

    # --- AWS SageMaker (Async Inference; scale-to-zero JumpStart endpoint) -
    "qwen-3.5-4b": {
        "provider": "sagemaker",
        "endpoint_env": "QWEN_SAGEMAKER_ENDPOINT",          # endpoint name from the deploy
        "region_env": "QWEN_SAGEMAKER_REGION",              # eu-central-1
        # JumpStart Qwen typically deploys with a HuggingFace TGI container;
        # request body is {"inputs": "...", "parameters": {...}}.
        "api_style": "tgi",
        "supports_async": True,
    },
}

# Per-provider views derived from the unified MODELS table.
FOUNDRY_MODELS: Dict[str, Dict[str, Any]] = {
    name: cfg for name, cfg in MODELS.items() if cfg.get("provider") == "foundry"
}
BEDROCK_MODELS: Dict[str, Dict[str, Any]] = {
    name: cfg for name, cfg in MODELS.items() if cfg.get("provider") == "bedrock"
}
SAGEMAKER_MODELS: Dict[str, Dict[str, Any]] = {
    name: cfg for name, cfg in MODELS.items() if cfg.get("provider") == "sagemaker"
}

MODEL_NAMES: List[str] = list(MODELS.keys())
FOUNDRY_MODEL_NAMES: List[str] = list(FOUNDRY_MODELS.keys())
BEDROCK_MODEL_NAMES: List[str] = list(BEDROCK_MODELS.keys())
SAGEMAKER_MODEL_NAMES: List[str] = list(SAGEMAKER_MODELS.keys())


def get_provider(model_name: str) -> "str | None":
    cfg = MODELS.get(model_name)
    return cfg.get("provider") if cfg else None


def get_model_config(model_name: str) -> "Dict[str, Any] | None":
    return MODELS.get(model_name)


DEFAULT_JUDGE_MODEL: str = "gpt-5.1-1"

# AWS defaults (can be overridden per-call or via env)
AWS_REGION_DEFAULT: str = "eu-central-1"  # Frankfurt

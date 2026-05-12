"""Shared configuration for FactoryBench pipeline and evaluation.

Single source of truth for the model catalog (provider routing, endpoints,
batch support) and pipeline-wide defaults. Adding a new model is one dict edit
here and every pipeline/evaluation script picks it up automatically.

Three providers are supported:

  * ``foundry``   — OpenAI-compatible HTTP endpoints (Azure AI Foundry for
                    GPT-5.x; OpenRouter for open models like qwen3-4b). The
                    foundry path also handles per-model overrides for the
                    base URL, API key env var, and the upstream model id sent
                    to ``client.chat.completions.create``.
  * ``bedrock``   — AWS Bedrock. Native managed inference for Anthropic
                    (CRIS profile in EU), Mistral, and DeepSeek (fully-managed
                    serverless). Native batch via ``CreateModelInvocationJob``
                    (S3-in / S3-out, ~50% cheaper than on-demand).
  * ``sagemaker`` — AWS SageMaker Async Inference. Currently unused (the
                    JumpStart Qwen deploy was flaky); kept for the option of
                    re-enabling self-hosted endpoints later.

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
        # Azure /v1/batches requires a deployment whose SKU is `globalbatch`
        # or `datazonebatch`. The default `gpt-5.1-1` deployment is
        # `GlobalStandard` (sync only) and rejects batch with HTTP 400. Set
        # GPT_5_1_BATCH_DEPLOYMENT to a separate batch-capable deployment
        # name; absent the env var, batch falls back to concurrent sync.
        "batch_deployment_env": "GPT_5_1_BATCH_DEPLOYMENT",
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

    # --- Together AI (sync-only; OpenAI-compatible) ----------------------
    # Routed through the foundry path because Together exposes the same
    # /chat/completions API surface as Azure / OpenAI. Tried OpenRouter and
    # smaller Together variants first — none host qwen3 below 235B
    # serverless. Qwen/Qwen3-235B-A22B-Instruct-2507-tput is a 235B MoE
    # with 22B active per token; the `-tput` suffix is Together's reliable
    # serverless / throughput-tier marker.
    "qwen-3-235b": {
        "provider": "foundry",
        "endpoint_env": "TOGETHER_BASE_URL",
        "endpoint_default": "https://api.together.xyz/v1",
        "api_style": "openai",
        "api_key_env": "TOGETHER_API_KEY",
        "model_id_env": "TOGETHER_QWEN_MODEL",
        "model_id_default": "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        "supports_batch": False,
    },
    "qwen-3-4b": {
        "provider": "foundry",
        "endpoint_env": "TOGETHER_BASE_URL",
        "endpoint_default": "https://api.together.xyz/v1",
        "api_style": "openai",
        "api_key_env": "TOGETHER_API_KEY",
        "model_id_default": "ymerzouki001_2159/Qwen/Qwen3-4B-Instruct-2507-9bb0515d",
        "supports_batch": False,
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


def get_upstream_model_id(model_name: str) -> str:
    """Upstream model id to send to the API.

    Lets the FactoryBench-side name diverge from the provider's model id —
    e.g. ``qwen-3-4b`` (FactoryBench) -> ``qwen/qwen3-4b`` (OpenRouter). When
    ``model_id_env`` is set and populated it wins; otherwise falls back to
    ``model_id_default``; otherwise the FactoryBench name itself.
    """
    import os
    cfg = MODELS.get(model_name) or {}
    env_key = cfg.get("model_id_env")
    if env_key:
        override = os.getenv(env_key)
        if override:
            return override
    return cfg.get("model_id_default") or model_name


def get_api_key_env(model_name: str) -> "str | None":
    """Env var name to consult for this model's API key, if any.

    When a foundry-style model declares ``api_key_env``, that env var is
    checked before falling back to the default Azure/OpenAI keys. Lets
    multiple OpenAI-compatible providers (Azure, OpenRouter, ...) coexist
    without sharing one key.
    """
    cfg = MODELS.get(model_name) or {}
    return cfg.get("api_key_env")


def get_batch_deployment(model_name: str) -> str:
    """Deployment name to send to Azure /v1/batches for ``model_name``.

    Returns the value of ``batch_deployment_env`` (when set on a foundry model
    AND the env var is populated), otherwise the model name itself. Lets a
    sync-only ``GlobalStandard`` deployment coexist with a separate
    ``globalbatch`` deployment used only by batch jobs.
    """
    import os
    cfg = MODELS.get(model_name) or {}
    env_key = cfg.get("batch_deployment_env")
    if env_key:
        override = os.getenv(env_key)
        if override:
            return override
    return model_name


DEFAULT_JUDGE_MODEL: str = "gpt-5.1-1"

# AWS defaults (can be overridden per-call or via env)
AWS_REGION_DEFAULT: str = "eu-central-1"  # Frankfurt

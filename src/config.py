"""Shared configuration for FactoryBench pipeline and evaluation.

Single source of truth for the Microsoft Foundry model catalog (names +
endpoint routing) and pipeline-wide defaults. Add a new Foundry model here
and every pipeline/evaluation script picks it up automatically.
"""
from __future__ import annotations

from typing import Any, Dict, List


FOUNDRY_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-5.1": {
        "endpoint_env": "CHAT_ENDPOINT",
        "endpoint_default": "https://student-research-lab-resource.services.ai.azure.com/openai/v1",
        "api_style": "openai",
        "supports_batch": True,
    },
    "claude-haiku-4-5": {
        "endpoint_env": "REASONING_ENDPOINT",
        "endpoint_default": "https://student-research-lab-resource.services.ai.azure.com/anthropic/v1",
        "api_style": "anthropic",
        "supports_batch": True,
    },
    "DeepSeek-V3.1": {
        "endpoint_env": "PROJECT_ENDPOINT",
        "endpoint_default": "https://student-research-lab-resource.services.ai.azure.com/models",
        "api_style": "deepseek",
        "requires_api_version": True,
        "api_version_env": "PROJECT_API_VERSION",
        "api_version_default": "2024-05-01-preview",
        "supports_batch": False,
    },
    "Mistral-Large-3": {
        "endpoint_env": "PROJECT_ENDPOINT",
        "endpoint_default": "https://student-research-lab-resource.services.ai.azure.com/models",
        "api_style": "mistral",
        "requires_api_version": True,
        "api_version_env": "PROJECT_API_VERSION",
        "api_version_default": "2024-05-01-preview",
        # Azure Foundry may not actually support batch for Mistral; if the
        # batch submission 404s the built-in fallback runs concurrent sync.
        "supports_batch": True,
    },
}

FOUNDRY_MODEL_NAMES: List[str] = list(FOUNDRY_MODELS.keys())

DEFAULT_JUDGE_MODEL: str = "gpt-5.1"

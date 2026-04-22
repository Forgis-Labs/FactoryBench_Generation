"""Shared configuration for FactoryBench pipeline and evaluation.

Single source of truth for the Microsoft Foundry model catalog (names +
endpoint routing) and pipeline-wide defaults. Add a new Foundry model here
and every pipeline/evaluation script picks it up automatically.
"""
from __future__ import annotations

from typing import Any, Dict, List


FOUNDRY_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-5-mini": {
        "endpoint_env": "CHAT_ENDPOINT",
        "endpoint_default": "https://student-research-lab-resource.services.ai.azure.com/openai/v1",
        "api_style": "openai",
    },
    "claude-haiku-4-5": {
        "endpoint_env": "REASONING_ENDPOINT",
        "endpoint_default": "https://student-research-lab-resource.services.ai.azure.com/anthropic/v1",
        "api_style": "anthropic",
    },
    "DeepSeek-V3.1": {
        "endpoint_env": "PROJECT_ENDPOINT",
        "endpoint_default": "https://student-research-lab-resource.services.ai.azure.com/openai/v1",
        "api_style": "deepseek",
    },
    "Mistral-Large-3": {
        "endpoint_env": "PROJECT_ENDPOINT",
        "endpoint_default": "https://student-research-lab-resource.services.ai.azure.com/openai/v1",
        "api_style": "mistral",
    },
}

FOUNDRY_MODEL_NAMES: List[str] = list(FOUNDRY_MODELS.keys())

DEFAULT_JUDGE_MODEL: str = "gpt-5-mini"

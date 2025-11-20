import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RUN_DIR = Path(os.getenv("FACTORYBENCH_RUN_DIR", "runs")).resolve()
RUN_DIR.mkdir(parents=True, exist_ok=True)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Azure OpenAI Configuration
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Dataset Registry
DATASETS = {
    "telemetry_literacy": [
        {
            "id": "local_basic",
            "name": "Basic Statistics (10 samples)",
            "source": "local",
            "fixture_path": "datasets/basic_statistics.json",
            "split": "train",
        },
        {
            "id": "local_step_functions",
            "name": "Step Functions (15 samples)",
            "source": "local",
            "fixture_path": "datasets/step_functions.json",
            "split": "train",
        },
        {
            "id": "local_patterns",
            "name": "Pattern Recognition (12 samples)",
            "source": "local",
            "fixture_path": "datasets/pattern_recognition.json",
            "split": "train",
        },
        {
            "id": "hf_factoryset",
            "name": "FS-TL v0.1 (50k samples)",
            "source": "hf",
            "hf_slug": "Forgis/FactorySet",
            "split": "train",
        },
    ]
}

# Model Registry
MODELS = [
    {"id": "mock", "name": "Mock Adapter", "provider": "local"},
    # Current generation (2024)
    {"id": "azure:gpt-4o", "name": "GPT-4o (Azure)", "provider": "azure"},
    {"id": "azure:gpt-4o-mini", "name": "GPT-4o Mini (Azure)", "provider": "azure"},
    # Next generation (2025)
    {"id": "azure:gpt-5", "name": "GPT-5 Global (SOTA)", "provider": "azure"},
    {"id": "azure:o3-2025-04-16", "name": "o3 2025-04-16 (SOTA Affordable)", "provider": "azure"},
    {"id": "azure:o4-mini-2025-04-16", "name": "o4-mini 2025-04-16 (Efficient RAG)", "provider": "azure"},
    {"id": "azure:gpt-5-nano", "name": "GPT-5-nano (Super Inexpensive)", "provider": "azure"},
]

# Azure OpenAI pricing (USD per 1K tokens) - updated Nov 2024
# Source: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
# Prices converted from per 1M tokens to per 1K tokens (divide by 1000)
AZURE_PRICING = {
    # Current generation (2024)
    # gpt-4o-2024-11-20
    "azure:gpt-4o": {"input_per_1k": 0.0025, "output_per_1k": 0.01},
    # gpt-4o-mini-2024-07-18
    "azure:gpt-4o-mini": {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
    # Next generation (2025) - Global pricing per 1M tokens converted to per 1K
    # GPT-5: $1.25/$10.00 per 1M → $0.00125/$0.01 per 1K
    "azure:gpt-5": {"input_per_1k": 0.00125, "output_per_1k": 0.01},
    # o3 2025-04-16: $2.00/$8.00 per 1M → $0.002/$0.008 per 1K
    "azure:o3-2025-04-16": {"input_per_1k": 0.002, "output_per_1k": 0.008},
    # o4-mini 2025-04-16: $1.10/$4.40 per 1M → $0.0011/$0.0044 per 1K
    "azure:o4-mini-2025-04-16": {"input_per_1k": 0.0011, "output_per_1k": 0.0044},
    # GPT-5-nano: $0.05/$0.40 per 1M → $0.00005/$0.0004 per 1K
    "azure:gpt-5-nano": {"input_per_1k": 0.00005, "output_per_1k": 0.0004},
}

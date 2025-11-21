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
            "name": "FactorySet v0.1 (50k)",
            "source": "hf",
            "hf_slug": "Forgis/FactorySet",
            "split": "train",
        },
    ]
}

# Model Registry
MODELS = [
    {"id": "mock", "name": "Mock Adapter", "provider": "local"},
    {"id": "azure:gpt-4o", "name": "GPT-4o", "provider": "azure"},
    {"id": "azure:gpt-4o-mini", "name": "GPT-4o Mini", "provider": "azure"},
    {"id": "azure:o1", "name": "o1 (Reasoning)", "provider": "azure"},
    {"id": "azure:o1-mini", "name": "o1-mini (Efficient)", "provider": "azure"},
]

# Azure OpenAI pricing (USD per 1K tokens) - updated Nov 2024
# Source: https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
# Prices converted from per 1M tokens to per 1K tokens (divide by 1000)
AZURE_PRICING = {
    # Current generation (2024)
    "azure:gpt-4o": {"input_per_1k": 0.0025, "output_per_1k": 0.01},
    "azure:gpt-4o-mini": {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
    # o1 series (reasoning models)
    "azure:o1": {"input_per_1k": 0.015, "output_per_1k": 0.06},
    "azure:o1-mini": {"input_per_1k": 0.003, "output_per_1k": 0.012},
}

# Cost Limits (USD)
MAX_COST_PER_RUN = 1.0  # Maximum spend per benchmark run
MAX_COST_PER_DAY = 20.0  # Maximum total spend per day

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
    # Stage 1: Telemetry Literacy (Rung 1 - Associational)
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
    ],

    # Stage 2: Root Cause Analysis (Rung 2 - Interventional)
    "root_cause_analysis": [
        {
            "id": "causrca_coolant",
            "name": "CausRCA Coolant (30 scenarios)",
            "source": "adapted",
            "fixture_path": "datasets/causrca/coolant.json",
            "split": "test",
            "n_variables": 11,
            "n_edges": 11,
        },
        {
            "id": "causrca_hydraulic",
            "name": "CausRCA Hydraulic (41 scenarios)",
            "source": "adapted",
            "fixture_path": "datasets/causrca/hydraulic.json",
            "split": "test",
            "n_variables": 9,
            "n_edges": 9,
        },
        {
            "id": "causrca_probe",
            "name": "CausRCA Probe (34 scenarios)",
            "source": "adapted",
            "fixture_path": "datasets/causrca/probe.json",
            "split": "test",
            "n_variables": 15,
            "n_edges": 15,
        },
        {
            "id": "synthetic_tier1",
            "name": "Synthetic Apprentice (1000 scenarios)",
            "source": "synthetic",
            "tier": "apprentice",
            "split": "train",
        },
        {
            "id": "synthetic_tier2",
            "name": "Synthetic Technician (1000 scenarios)",
            "source": "synthetic",
            "tier": "technician",
            "split": "train",
        },
        {
            "id": "synthetic_tier3",
            "name": "Synthetic Expert (500 scenarios)",
            "source": "synthetic",
            "tier": "expert",
            "split": "test",
        },
    ],

    # Stage 3: Guided Remediation (Rung 4 - Applied)
    "guided_remediation": [
        {
            "id": "manuals_ur",
            "name": "UR Robot Manuals",
            "source": "local",
            "fixture_path": "datasets/manuals/ur_sections.json",
            "n_sections": 100,
        },
        {
            "id": "manuals_abb",
            "name": "ABB Robot Manuals",
            "source": "local",
            "fixture_path": "datasets/manuals/abb_sections.json",
            "n_sections": 80,
        },
        {
            "id": "procedures_synthetic",
            "name": "Synthetic Procedures",
            "source": "synthetic",
            "fixture_path": "datasets/manuals/synthetic_procedures.json",
            "n_procedures": 500,
        },
    ],
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

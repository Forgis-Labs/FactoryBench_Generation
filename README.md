# FactoryBench

**A Q&A benchmark for machine understanding** — evaluating whether AI can reason about industrial machines, not just detect anomalies.

📦 **Dataset:** [huggingface.co/datasets/FactoryBench/FactoryBench](https://huggingface.co/datasets/FactoryBench/FactoryBench)
📜 **License:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

---

## The Idea

When a robot stops at 3 AM, operators face a cascade of questions:

- *"What is happening right now?"* — State
- *"What if I do this now?"* — Intervention
- *"What if I had done that in the past?"* — Counterfactual
- *"How do we recover?"* — Decision making

**FactoryBench tests all four levels** on collaborative robots (UR5e), going deeper on one machine family than existing benchmarks go across many. The four levels mirror Pearl's causal hierarchy and probe distinct reasoning capabilities, from raw sensor interpretation up to closed-loop troubleshooting.

---

## The 4-Level Q&A Framework

| Level | Task | Example question | Ground-truth source | Pearl rung |
|-------|------|------------------|---------------------|------------|
| **1 — State** | Interpret what the machine is doing now (mode, anomaly, signal pattern). | *"What is the current of joint 3 right now?"* | Sensor data (time-series window) | Rung 1 |
| **2 — Intervention** | Reason about what an action **now** would change. | *"If force in joint 3 increases to X now, what happens?"* | Simulation (present state injection) | Rung 2 |
| **3 — Counterfactual** | Reason about what a different past would have produced. | *"If force had increased to X at t=20 ms, what would have happened?"* | Simulation (past state replay) | Rung 3 |
| **4 — Decision Making** | Generate a remediation/optimization plan from the trace. | *"Robot stopped with error C203A. What to do?"* | Manual + sensor fusion | Composite |

Each level builds on the previous; failure at level *N* implies failure at level *N+1*.

### Answer formats supported per level

| Answer format | L1 State | L2 Intervention | L3 Counterfactual | L4 Decision |
| --- | :---: | :---: | :---: | :---: |
| Multiple choice (single / multi) | ✓ | ✓ | ✓ | ✓ |
| Scalar / numerical (with tolerance) | ✓ | ✓ | ✓ | — |
| Tensor (fixed-shape vector) | ✓ | ✓ | ✓ | — |
| Ranking (Kendall-τ scored) | ✓ | ✓ | ✓ | ✓ |
| Free-form (LLM-judge scored) | ✓ | ✓ | ✓ | ✓ |

---

## Overview

Modern industrial systems generate large volumes of multivariate time-series data from sensors, actuators, and controllers. Traditional time-series models excel at narrow tasks but lack explicit reasoning, while general LLMs show strong reasoning but struggle with dense numerical telemetry.

FactoryBench bridges that gap by offering:

- **Scalable framework.** Structured question templates spanning state, intervention, counterfactual, and decision-making reasoning.
- **FactoryWave dataset.** A dense multivariate sensor stream from UR5e-class collaborative robots under varied operational conditions and systematically injected anomalies.
- **SCE causal schema.** A unified timeline schema mapping **S**etpoint, **C**ontext, and **E**ffort/feedback channels, making causal relationships explicit.
- **A scoring cascade.** Deterministic parsers for structured formats (numerical with tolerance, ranking via Kendall-τ, multi-bit T/F, tensor with per-channel margin) and an LLM-judge fallback for free-form items.

---

## Repository Structure

```text
FactoryBench/
├── factorybench/             # Python package
│   ├── adapters/             # LLM adapters (Azure OpenAI, etc.)
│   ├── api/                  # FastAPI backend for question serving
│   ├── data/                 # Dataset loaders
│   ├── eval/                 # Evaluation pipeline runner
│   ├── metrics/              # Scoring utilities
│   ├── cli.py                # `factorybench` CLI entry point
│   ├── config.py             # Configuration / dataclasses
│   ├── stages.py             # Pipeline stage definitions
│   └── state.py              # Pipeline state model
├── src/
│   ├── evaluation/           # End-to-end eval drivers for HF/Bedrock/Foundry
│   ├── scoring/              # Cascade scorer (parsers + LLM-judge)
│   ├── question_generation/  # Q&A template + generator code
│   └── pipeline/             # Data pipeline + setup docs
├── scripts/                  # Plotting + ad-hoc analysis scripts
├── archive/                  # Deprecated causal-framework prototypes
├── factorybench_croissant.json   # Croissant dataset descriptor
├── environment.yaml          # Conda environment spec
├── pyproject.toml            # Package metadata + deps
├── requirements.txt          # Minimal pinned install list
└── README.md                 # You are here
```

---

## Quick Start

### 1. Install

```bash
# pip (recommended)
git clone https://github.com/Forgis-Labs/FactoryBench_Generation.git
cd FactoryBench_Generation
pip install -e .

# or conda
conda env create -f environment.yaml
conda activate factorybench
```

### 2. Configure model access

Create a `.env` at the repo root with the credentials you'll use. Required env vars depend on which provider you target — see `src/pipeline/README.md` for the full list. Minimal example:

```bash
# Azure Foundry (Anthropic / OpenAI via Foundry endpoint)
AZURE_FOUNDRY_ENDPOINT="<your-endpoint>"
AZURE_FOUNDRY_API_KEY="<your-key>"

# Or AWS Bedrock (Claude / Llama / DeepSeek)
AWS_REGION="us-east-2"
# IAM credentials picked up from the standard AWS chain

# Or HuggingFace inference / Together / Modal (optional)
HF_TOKEN="<token>"
TOGETHER_API_KEY="<key>"

# LLM-as-judge (only needed for free-form L4 scoring)
JUDGE_MODEL="gpt-5-mini"
```

### 3. Pull the dataset

The full Q&A test set is on HuggingFace:

```python
from datasets import load_dataset

ds = load_dataset("FactoryBench/FactoryBench", split="test")
print(ds[0])
# {'id': ..., 'level': 1, 'question': ..., 'options': {...},
#  'answer': ..., 'context': {'time_series': [...], ...}}
```

The Croissant metadata for the dataset lives at `factorybench_croissant.json` in this repo.

### 4. Run evaluation

End-to-end driver for a single model on the test set:

```bash
python -m src.evaluation.run_foundry_eval \
    --model gpt-5-mini \
    --level 1,2,3,4 \
    --output-dir runs/gpt5mini-fbench
```

Then score the resulting predictions with the cascade:

```bash
python -m src.scoring.score_traces \
    --predictions runs/gpt5mini-fbench \
    --judge-model gpt-5-mini   # only used for L4 free-form
```

Outputs: per-prediction JSONL with `score`, `provenance`, and `score_reason`, plus an aggregated `summary.json`.

### 5. (Optional) Run the question-serving API

For interactive use or for building a custom evaluation harness:

```bash
uvicorn factorybench.api.app:app --reload --port 5173
# OpenAPI docs at http://localhost:5173/docs
```

---

## Scoring

FactoryBench answers come in five formats, each scored on `[0, 1]`:

| Format | Parser | Score definition |
| --- | --- | --- |
| `multiple_choice_single_select` | strict + lenient cascade | 1 if extracted letter matches GT, else 0 |
| `multiple_choice_multi_select` (T/F string) | per-bit | fraction of positions matching GT |
| `ranking` (permutation string) | Kendall-τ | `(τ + 1) / 2`, so 1 = exact match, 0.5 ≈ random, 0 = fully reversed |
| `numerical` | tolerance band | 1 within margin, 0.5 within 2× margin, 0 otherwise; margin from `acceptance_bounds` or default |
| `tensor` (fixed-length vector) | per-channel margin | same three-level piecewise as numerical, applied per channel and averaged |
| `free_form` | LLM judge | rubric-graded 0 / 0.5 / 1 by a configurable judge model |

Raw scores can be chance-corrected with `src/evaluation/chance_correct.py` for cross-format comparability.

---

## Dataset Citation

If you use FactoryBench in your research, please cite the dataset descriptor:

```bibtex
@misc{merzouki2026factorybench,
  title        = {FactoryBench: A Q\&A Benchmark for Machine Understanding},
  author       = {Merzouki, Yanis and collaborators},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/FactoryBench/FactoryBench}},
  note         = {Dataset and evaluation framework}
}
```

A formal preprint is in preparation; this bib entry will be updated once a DOI / arXiv ID is available.

---

## License

The codebase and dataset are released under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/), matching the license declared in `factorybench_croissant.json`. You may share and adapt the material for non-commercial purposes with appropriate attribution.

---

## Contributing

Issues and pull requests are welcome. For substantial changes (new question templates, new answer formats, new datasets), please open an issue first to discuss the scope.

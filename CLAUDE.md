# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FactoryBench is a benchmark for evaluating LLM agents on machine understanding over industrial time-series data. It consists of:
- A **question generation framework** (structured Q&A tasks at 4+ reasoning levels)
- A **FactoryWave dataset** (robotic sensor telemetry in ICO schema)
- An **evaluation pipeline** (LLM adapters, scoring, run tracking)
- A **FastAPI backend** + **Remix frontend** for running and visualizing benchmarks

## Setup

```bash
# Python environment (conda or venv, Python >=3.11)
conda env create -f environment.yaml
conda activate factorybench
# or
pip install -e ".[dev]"

# Copy and fill in environment variables
cp .env.example .env
```

Required `.env` variables for real model runs:
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`
- `HF_API_TOKEN` (only if FactorySet HuggingFace dataset is private)
- `FACTORYBENCH_RUN_DIR` (defaults to `runs/`)

## Common Commands

### Backend (FastAPI)
```bash
uvicorn factorybench.api.app:app --reload
```

### CLI benchmark runs
```bash
# Run Stage 1 (telemetry literacy) with mock model
python -m factorybench.cli run-stage1 --dataset-id local_basic --limit 10 --model mock

# Run with Azure OpenAI
python -m factorybench.cli run-stage1 --dataset-id local_basic --model azure:gpt-4o

# Verify plumbing
python -m factorybench.cli components:test
```

### Question generation (from `src/`)
```bash
# Generate Level 1 questions from a normalized episode
python -m src.questions.cli level1 \
  --input datasets/normalized_episodes/dummy/ABB.json \
  --output datasets/questions/dummy/sample_questions.json \
  --n 100 --seed 42

# Full config with thresholds
python -m src.questions.cli level1 \
  --input datasets/normalized_episodes/dummy/ABB.json \
  --output datasets/questions/dummy/questions_configured.json \
  --n 100 --eps-1 0.05 --eps-2 15 --delta-1 500 --eps-3 1.0 --seed 42
```

### Dataset installation & normalization
```bash
# Install AURSAD dataset
python -m src.data.data_installation.install_aursad --max-timestamps 100000

# Install CNC dataset
python -m src.data.data_installation.install_cnc --setup
python -m src.data.data_installation.install_cnc

# Normalize to JSON episodes
python -m src.data.data_normalization.mapped_dataset_normalizer \
    --dataset aursad --input datasets/open_datasets/aursad --output datasets/normalized_episodes

# CWRU bearing dataset converter
python -m factorybench.data.cwru_converter \
    --input-dir datasets/open_datasets/CWRU/cwru \
    --output-dir datasets/open_datasets/CWRU/converted --format both

# UR3e normalizer
python -m factorybench.data.ur3e_normalizer \
    --input datasets/open_datasets/ur3+cobotops/dataset_02052023.xlsx \
    --output datasets/open_datasets/ur3+cobotops/normalized --episode-id ur3e_episode_001
```

### Frontend (Remix)
```bash
cd frontend
npm install
npm run dev        # Dev server
npm run build      # Production build
npm run typecheck  # TypeScript check
```

### Linting & type checking
```bash
ruff check .
mypy factorybench/
```

### Tests
```bash
pytest
pytest tests/test_level1.py -v
pytest tests/test_level1.py::test_q1_position_check -v
```

## Architecture

### Two parallel code trees

**`factorybench/`** — installable Python package (benchmark runner + API):
- `config.py` — dataset registry, model registry, Azure pricing, cost limits
- `stages.py` — `Stage` enum (telemetry_literacy, root_cause_analysis, guided_remediation) + aliases
- `state.py` — thread-safe `RunStateManager` singleton for tracking active runs and daily costs
- `cli.py` — Click CLI entry point
- `api/app.py` — FastAPI app; runs benchmarks in background tasks, writes incremental JSON to `RUN_DIR`
- `adapters/` — `ModelAdapter` ABC; implementations: `MockAdapter`, `AzureOpenAIAdapter`
- `eval/runner.py` — `run_telemetry_literacy()`: drives sample loop, calls adapter, scores, writes `runs/<run_id>.json` after each sample
- `metrics/telemetry_literacy.py` — parsing and scoring (mean/min/max absolute error, ok_rate, performance)
- `data/loader_tl.py` — dataset loader for Stage 1 (local JSON or HuggingFace)
- `viz/charts.py` — chart generation from completed runs

**`src/`** — research/data pipeline (question generation + dataset tools):
- `src/questions/level1/` — Level 1 (State) question generator: 6 question types (position, friction, acceleration, force, jerk, torque) computed deterministically from normalized episodes
- `src/questions/level{2-5}/` — Higher-level question generators (Intervention, Counterfactual, Decision Making)
- `src/data/data_installation/` — dataset downloaders (AURSAD, CNC)
- `src/data/data_normalization/` — normalizers mapping raw datasets to ICO schema JSON

### Data flow
```
Raw sensor data (AURSAD, CNC, UR3e, CWRU, FactoryWave)
    → src/data normalizers
    → datasets/normalized_episodes/**/*.json  (ICO schema: timestamp_ms + signal columns)
    → src/questions/level{N}/  (deterministic Q&A generation)
    → datasets/questions/**/*.json
    → factorybench/api or CLI  (LLM inference via adapter)
    → runs/<run_id>.json  (scored results)
    → factorybench/viz/  (charts)
    → frontend/  (Remix dashboard)
```

### ICO Schema
Normalized episodes are arrays of dicts with `timestamp_ms` and signal columns:
- **Intent:** `setpoint_pos_*`, `setpoint_speed_*`, gripper commands
- **Context:** `joint_temp_*`, voltage, safety signals
- **Outcome:** `feedback_pos_*`, `feedback_speed_*`, `effort_current_*`, `vibration_*`, `true_force_*`/`est_contact_force_*`, `true_torque_*`/`est_contact_torque_*`

### Run files
`RUN_DIR` (default `runs/`) stores one JSON file per benchmark run (`<run_id>.json`). The runner writes after every sample for incremental progress. The API serves these files directly. Daily cost limits are computed by scanning all run files for the current date.

### Question levels
- **Level 1 (State):** 6 deterministic types from sensor data — position change, friction degradation, end-effector acceleration, external force, joint jerk, torque magnitude
- **Levels 2–4:** Intervention, Counterfactual, Decision Making (see `src/questions/level{2-4}/`)
- Questions follow the format: `{id, question: {text, level, type}, context, params, answer, reasoning, provenance}`

### Adding a new model adapter
Implement `ModelAdapter` ABC from `factorybench/adapters/base.py`: one method `generate(prompt: str) -> dict` returning `{text: str, usage: {prompt_tokens, completion_tokens, total_tokens}}`. Register in `MODELS` in `config.py`.

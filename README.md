# FactoryBench

Lean benchmark for evaluating AI model performance on industrial troubleshooting — from telemetry literacy to guided remediation — with a minimal Python backend and a clean Remix frontend.

## Quick Links

- **Leaderboard**: [localhost:3000/leaderboard](http://localhost:3000/leaderboard) (after `npm run dev`)
- **Methodology**: [localhost:3000/readme](http://localhost:3000/readme) (detailed metrics & dataset structure)
- **Analysis**: [localhost:3000/analysis](http://localhost:3000/analysis) (charts & visualizations)
- **API Docs**: `http://localhost:5173/docs` (FastAPI Swagger UI)

## Overview

FactoryBench evaluates AI model performance on industrial troubleshooting tasks through a three-stage progression:

1. **Telemetry Literacy** (Current): Basic time series comprehension and statistical analysis
2. **Root Cause Analysis** (Planned): Diagnostic reasoning and fault correlation
3. **Guided Remediation** (Planned): Complete troubleshooting workflow with repair instructions

### Current Status: Stage 1 - Telemetry Literacy ✅

The benchmark currently evaluates models on:
- Statistical measures (mean, min, max) from univariate time series
- Pattern recognition in temporal data
- Step function detection and analysis

**Available Datasets**:
- Local: `basic_statistics` (10 samples), `step_functions` (15 samples), `pattern_recognition` (12 samples)
- HuggingFace: `Forgis/FactorySet` (50k+ samples, gated)

**Supported Models**:
- Mock adapter (testing)
- Azure OpenAI: gpt-4o, gpt-4o-mini, gpt-5, o3-2025, o4-mini-2025, gpt-5-nano

**Key Features**:
- Cost tracking per run (token usage & USD cost)
- Sortable leaderboard with filtering by model/dataset/stage
- Interactive charts with explanations
- Run creation UI with progress tracking
- Detailed run artifacts (JSON) with per-sample metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Remix Frontend (Port 3000)              │
│  Routes: /, /leaderboard, /run, /runs/:id, /analysis        │
│  Features: Leaderboard, Run Creation, Charts, Filters       │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Backend (Port 5173)                    │
│  Endpoints: /runs, /runs/:id, /charts/:type, /metadata      │
│  Adapters: Mock, Azure OpenAI                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
┌─────────────────────┐         ┌──────────────────────┐
│  Local JSON Files   │         │  HuggingFace Hub     │
│  runs/*.json        │         │  Forgis/FactorySet   │
│  datasets/*.json    │         │  (Gated)             │
└─────────────────────┘         └──────────────────────┘
```

### Tech Stack

**Backend**:
- FastAPI + Uvicorn (Python 3.11+)
- Pydantic for data validation
- Click for CLI
- HuggingFace Datasets for remote data loading
- Matplotlib + Seaborn for chart generation
- Azure OpenAI SDK

**Frontend**:
- Remix (React-based meta-framework)
- TypeScript
- CSS variables (Forgis brand colors)
- No heavy dependencies (lean & fast)

**Storage**:
- Local JSON artifacts in `runs/` directory
- Chart PNGs cached in `charts/` directory
- Upgradable to PostgreSQL/S3 without API changes

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- npm or pnpm
- (Optional) Azure OpenAI API access

### 1. Install Python Backend

Using `uv` (recommended):
```powershell
pipx install uv
uv venv
uv pip install -e .
```

Or using standard venv:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### 2. Configure Environment (Optional)

For Azure OpenAI models:
```powershell
Copy-Item .env.example .env -ErrorAction SilentlyContinue

# Edit .env and add:
# AZURE_OPENAI_API_KEY=your-key-here
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
# AZURE_OPENAI_API_VERSION=2024-02-15-preview
# HF_API_TOKEN=your-hf-token  # For gated datasets
```

### 3. Start the Backend API

```powershell
# With uv
uv run uvicorn factorybench.api.app:app --reload --port 5173

# Or without uv
.\.venv\Scripts\Activate.ps1
uvicorn factorybench.api.app:app --reload --port 5173
```

API will be available at `http://localhost:5173`

### 4. Run a Benchmark (CLI)

```powershell
# Mock adapter (no API key needed)
uv run python -m factorybench.cli run-stage1 `
  --model mock `
  --dataset-id local_basic `
  --fixture-path datasets/basic_statistics.json `
  --limit 5

# Azure OpenAI (requires API key)
uv run python -m factorybench.cli run-stage1 `
  --model "azure:gpt-4o-mini" `
  --dataset-id local_basic `
  --fixture-path datasets/basic_statistics.json `
  --limit 10

# HuggingFace dataset (requires HF_API_TOKEN for gated datasets)
uv run python -m factorybench.cli run-stage1 `
  --dataset-source hf `
  --hf-slug Forgis/FactorySet `
  --hf-split train `
  --limit 50 `
  --model "azure:gpt-4o"
```

### 5. Start the Frontend

```powershell
cd frontend
npm install
npm run dev
```

Frontend will be available at `http://localhost:3000`

**Available Routes**:
- `/` - Overview & quick links
- `/leaderboard` - Sortable table of all runs with filters
- `/run` - Create new benchmark runs
- `/runs/:id` - Detailed view of a specific run
- `/analysis` - Interactive charts & visualizations
- `/readme` - Benchmark methodology & metrics (in-app docs)

## API Endpoints

### GET /runs
List all benchmark runs with optional filters.

**Query Parameters**:
- `model` (multiple): Filter by model ID
- `dataset` (multiple): Filter by dataset ID
- `stage`: Filter by benchmark stage

**Response**:
```json
{
  "items": [
    {
      "run_id": "tl-20251119T201643",
      "stage": "telemetry_literacy",
      "model": "azure:gpt-4o-mini",
      "status": "completed",
      "dataset": {
        "dataset_id": "local_basic",
        "source": "local",
        "fixture_path": "datasets/basic_statistics.json"
      },
      "aggregate": {
        "samples": 10.0,
        "ok_rate": 1.0,
        "mean_abs_err_mean": 0.0,
        "min_abs_err_mean": 0.0,
        "max_abs_err_mean": 0.0,
        "cost_per_sample": 0.000123,
        "cost_total": 0.00123,
        "tokens_input": 1500,
        "tokens_output": 450
      }
    }
  ]
}
```

### GET /runs/:id
Retrieve detailed run artifact including per-sample results.

### POST /runs
Create a new benchmark run.

**Request Body**:
```json
{
  "stage": "telemetry_literacy",
  "model": "azure:gpt-4o-mini",
  "dataset_id": "local_basic",
  "dataset_source": "local",
  "fixture_path": "datasets/basic_statistics.json",
  "limit": 10
}
```

### GET /charts/:type
Generate analysis charts (PNG images).

**Chart Types**: `error_distribution`, `convergence`, `ok_rate`, `model_comparison`, `error_breakdown`, `cost_quality`

**Query Parameters**:
- `model` (multiple): Filter by model
- `dataset` (multiple): Filter by dataset
- `regenerate`: Force regeneration (bypass cache)

### GET /metadata/models
List available models.

### GET /metadata/datasets
List available datasets for a stage.

## Dataset Structure

### Telemetry Literacy (Stage 1)

Each sample contains:
```json
{
  "id": "basic_001",
  "timestamps": [0.0, 1.0, 2.0, 3.0, 4.0],
  "values": [10.0, 20.0, 30.0, 40.0, 50.0],
  "domain": "synthetic",
  "subtype": "linear_increase",
  "statistics": {
    "mean": 30.0,
    "std": 14.142135,
    "min": 10.0,
    "max": 50.0
  }
}
```

**Evaluation**: Models are prompted to compute `mean`, `min`, `max`. Scores are calculated as absolute errors against ground truth.

See `/readme` route in the UI for full methodology.

## Evaluation Metrics

### Stage 1: Telemetry Literacy

**Per-Sample Metrics**:
- `mean_abs_err`: Absolute error for mean prediction
- `min_abs_err`: Absolute error for min prediction
- `max_abs_err`: Absolute error for max prediction
- `ok`: Boolean indicating if all three metrics were successfully extracted

**Aggregate Metrics**:
- `mean_abs_err_mean`: Average of mean errors across samples
- `min_abs_err_mean`: Average of min errors across samples
- `max_abs_err_mean`: Average of max errors across samples
- `ok_rate`: Percentage of samples with successful predictions
- `samples`: Total number of samples evaluated

**Cost Metrics**:
- `tokens_input`: Total input tokens consumed
- `tokens_output`: Total output tokens consumed
- `cost_input`: USD cost for input tokens
- `cost_output`: USD cost for output tokens
- `cost_total`: Total USD cost
- `cost_per_sample`: Average cost per sample

### Stage 2: Root Cause Analysis (Planned)

TBD - Will focus on diagnostic reasoning and fault correlation.

### Stage 3: Guided Remediation (Planned)

TBD - Will evaluate complete troubleshooting workflows with repair instructions.

## Development Roadmap

### ✅ Completed (Phase 1-2)
- [x] Stage 1 backend evaluation engine
- [x] FastAPI REST API with run management
- [x] CLI for benchmark execution
- [x] Mock & Azure OpenAI adapters
- [x] Remix frontend with leaderboard
- [x] Run detail views with JSON artifacts
- [x] Cost tracking & token usage
- [x] Interactive charts with filtering
- [x] Sortable leaderboard columns
- [x] Run creation UI with progress tracking
- [x] HuggingFace dataset integration

### 🚧 In Progress (Phase 3)
- [ ] Expanded Stage 1 datasets (industrial patterns)
- [ ] WandB integration for experiment tracking
- [ ] Additional chart types (latency, token efficiency)

### 📋 Planned (Phase 4-5)
- [ ] Stage 2: Root Cause Analysis specification
- [ ] Stage 2: Dataset curation & evaluation engine
- [ ] PostgreSQL/S3 backend (optional upgrade)
- [ ] Public leaderboard deployment
- [ ] API authentication & rate limiting

### 🔮 Future (Phase 6+)
- [ ] Stage 3: FactorySet curation (real industrial scenarios)
- [ ] Stage 3: Multi-modal evaluation (text + images + PDFs)
- [ ] Human-in-the-loop evaluation interface
- [ ] Model fine-tuning benchmarks
- [ ] Cross-model ensemble evaluation

## File Structure

```
FactoryBench/
├── factorybench/           # Python package
│   ├── adapters/          # Model adapters (mock, azure_openai)
│   ├── api/               # FastAPI app & chart generation
│   ├── data/              # Data loaders (local, HF)
│   ├── eval/              # Evaluation runner
│   ├── metrics/           # Scoring functions
│   ├── viz/               # Visualization (charts)
│   ├── cli.py             # Click CLI
│   ├── config.py          # Dataset/model registry, env config
│   └── stages.py          # Stage definitions
├── frontend/              # Remix app
│   ├── app/
│   │   ├── routes/        # Pages (_index, leaderboard, run, runs.$id, analysis, readme)
│   │   ├── styles/        # Global CSS
│   │   └── root.tsx       # Layout with nav
│   ├── public/            # Static assets
│   └── package.json
├── datasets/              # Local dataset fixtures
│   ├── basic_statistics.json
│   ├── step_functions.json
│   └── pattern_recognition.json
├── runs/                  # Run artifacts (JSON)
├── charts/                # Generated chart PNGs (cached)
├── pyproject.toml         # Python dependencies
├── README.md              # This file
└── .env.example           # Environment variable template
```

## Legacy & Outdated Code

⚠️ **Alert**: The following code/files are legacy or outdated:

1. **README.md references to OpenAI adapter**: Only Azure OpenAI is currently supported. Direct OpenAI adapter was removed.
2. **Duplicate "Quick Start" sections**: README has redundant setup instructions.
3. **FactorySet repository location**: References to `Xelerit-Robotics/FactorySet` are outdated; migrating to `Forgis` org.
4. **WandB placeholders**: WandB integration is mentioned but not implemented.
5. **Stage 2 & 3 datasets**: Placeholder references exist but no actual implementation.
6. **Future Azure model versions**: Model IDs like `gpt-5`, `o3-2025`, `o4-mini-2025`, `gpt-5-nano` are speculative and may not exist.

## Contributing

Contributions are welcome! Key areas:

1. **Dataset Curation**: Real-world industrial scenarios for Stage 2 & 3
2. **Model Adapters**: Support for Anthropic, Google, local models
3. **Metrics**: Enhanced evaluation metrics for Stages 2 & 3
4. **Frontend**: UI/UX improvements, chart types, export features
5. **Documentation**: Tutorials, case studies, best practices

## License

TBD

---

**Maintained by**: Forgis  
**Status**: Stage 1 MVP - Telemetry Literacy  
**Last Updated**: November 2025

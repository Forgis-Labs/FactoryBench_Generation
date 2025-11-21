# FactoryBench

Comprehensive benchmark for evaluating AI model performance on industrial troubleshooting tasks — from telemetry literacy to guided remediation — featuring real-time progress tracking, cost controls, and model comparison analytics.

## Quick Links

- **Leaderboard**: [localhost:3000/leaderboard](http://localhost:3000/leaderboard) - Sortable table with all runs
- **Analysis**: [localhost:3000/analysis](http://localhost:3000/analysis) - Model comparison charts
- **Create Run**: [localhost:3000/run](http://localhost:3000/run) - Start new benchmarks with real-time progress
- **Methodology**: [localhost:3000/readme](http://localhost:3000/readme) - In-app documentation
- **API Docs**: `http://localhost:5173/docs` - FastAPI Swagger UI

## Overview

FactoryBench evaluates AI models on industrial troubleshooting through a three-stage progression:

1. **Telemetry Literacy** ✅ (Current): Statistical analysis of time series data
2. **Root Cause Analysis** 📋 (Planned): Diagnostic reasoning and fault correlation
3. **Guided Remediation** 📋 (Planned): Complete troubleshooting workflows

### Stage 1: Telemetry Literacy (Live)

**What We Measure**:
- Statistical comprehension (mean, min, max) from univariate time series
- Pattern recognition in temporal data
- Step function detection and change point analysis

**Available Datasets**:
- `local_basic` - 10 samples, basic statistics
- `local_step_functions` - 15 samples, change detection
- `local_patterns` - 12 samples, pattern recognition
- `hf_factoryset` - 50,000+ samples, industrial telemetry (via HuggingFace)

**Supported Models**:
- `mock` - Testing adapter (no API required)
- `azure:gpt-4o` - GPT-4 Omni
- `azure:gpt-4o-mini` - GPT-4 Omni Mini (cost-effective)
- `azure:o1` - O1 reasoning model
- `azure:o1-mini` - O1 Mini

## Key Features

### 🎯 Core Functionality
- **Cost Safeguards**: $1/run and $20/day limits with pre-flight checks
- **Real-time Progress**: 2-second polling with sample-by-sample updates
- **Graceful Cancellation**: Stop button saves partial results
- **Performance Metric**: Composite score = avg(mean_err, min_err, max_err)

### 📊 Analytics & Visualization
- **Model Performance Bar**: Sorted comparison with sample counts
- **Cost vs Performance Scatter**: Efficiency analysis with sample size indicators
- **Model Metrics Heatmap**: Traffic-light coloring (red=high error, green=low)
- **Smart Run Selection**: Most samples processed, then most recent

### 🎨 User Experience
- **Default Filters**: Pre-select gpt-4o/gpt-4o-mini + FactorySet dataset
- **Connected Filtering**: Shared state across leaderboard & analysis
- **Loading Stages**: Dataset loading → Processing samples
- **Auto-refresh**: Detail pages reload on run completion
- **Responsive Tables**: 13 columns with sortable headers

### 💰 Cost Management
- **Per-run Limits**: Enforce $1 maximum per execution
- **Daily Limits**: Track $20 cumulative spending across all runs
- **Persistent Tracking**: File-based daily cost survives restarts
- **Color-coded Warnings**: Visual alerts at 80% threshold
- **Token Breakdown**: Separate input/output token counts and costs

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│               Remix Frontend (localhost:3000)                │
│  Routes: /, /leaderboard, /run, /runs/:id, /analysis         │
│  Features: Real-time progress, filters, charts, stop button  │
└────────────────────────┬─────────────────────────────────────┘
                         │ HTTP + 2s polling
                         ▼
┌──────────────────────────────────────────────────────────────┐
│            FastAPI Backend (localhost:5173)                  │
│  /runs - List/create with background tasks                  │
│  /runs/:id - Detail view with artifacts                     │
│  /runs/:id/progress - Real-time progress tracking           │
│  /runs/:id/stop - Graceful cancellation                     │
│  /charts/:type - Model comparison charts (PNG)              │
│  /metadata/* - Models, datasets, cost limits                │
└────────────────────────┬─────────────────────────────────────┘
                         │
         ┌───────────────┴────────────────┐
         ▼                                ▼
┌──────────────────┐            ┌─────────────────────┐
│  Local Storage   │            │  HuggingFace Hub    │
│  runs/*.json     │            │  Forgis/FactorySet  │
│  (50+ runs)      │            │  (50k+ samples)     │
└──────────────────┘            └─────────────────────┘
```

### Tech Stack

**Backend** (Python 3.11+):
- FastAPI + BackgroundTasks for async execution
- RunStateManager for thread-safe progress tracking
- HuggingFace Datasets (streaming disabled for reliability)
- Matplotlib with Forgis brand colors (traffic-light heatmaps)
- Azure OpenAI SDK with token counting

**Frontend** (TypeScript + React):
- Remix v2 with loader-based data fetching
- 2-second polling for progress updates
- Multi-select dropdowns with outside-click detection
- CSS variables for consistent theming
- No heavy dependencies (lean bundle)

**State Management**:
- In-memory: RunStateManager (progress, stop flags)
- Persistent: JSON files in `runs/` directory
- Daily cost: Aggregated from run files (survives restarts)

## Quick Start

### Prerequisites
- Python 3.11+ (3.12 recommended)
- Node.js 18+ and npm
- Azure OpenAI API credentials (or use mock adapter)
- HuggingFace token for gated datasets (optional)

### 1. Install Python Dependencies

Using `uv` (recommended - fast):
```powershell
pipx install uv
uv venv
uv pip install -e .
```

Or standard venv:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

### 2. Configure Environment

Create `.env` from template:
```powershell
Copy-Item .env.example .env

# Edit .env with your credentials:
# AZURE_OPENAI_API_KEY=your-key-here
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
# AZURE_OPENAI_API_VERSION=2024-02-15-preview
# HF_API_TOKEN=your-hf-token  # For Forgis/FactorySet
```

### 3. Start Backend API

```powershell
uvicorn factorybench.api.app:app --reload --port 5173
```

API available at `http://localhost:5173` (Swagger docs at `/docs`)

### 4. Start Frontend

```powershell
cd frontend
npm install
npm run dev
```

Frontend available at `http://localhost:3000`

### 5. Run Your First Benchmark

**Option A: Web UI (Recommended)**
1. Navigate to http://localhost:3000/run
2. Select model: `azure:gpt-4o-mini`
3. Select dataset: `hf_factoryset`
4. Set limit: `10` samples
5. Click "Start Run"
6. Watch real-time progress with cost tracking
7. Use Stop button if needed

**Option B: CLI**
```powershell
# Mock adapter (no API key)
python -m factorybench.cli run-stage1 --model mock --dataset-id local_basic --limit 5

# Azure OpenAI with HuggingFace dataset
python -m factorybench.cli run-stage1 --model "azure:gpt-4o-mini" --dataset-source hf --hf-slug Forgis/FactorySet --limit 50
```

## Evaluation Metrics

### Performance Metric (Primary)
**`performance = (mean_abs_err_mean + min_abs_err_mean + max_abs_err_mean) / 3`**

Lower is better. This composite metric provides a single score for model comparison.

### Detailed Metrics

**Per-Sample** (computed after each prediction):
- `mean_abs_err` - |predicted_mean - true_mean|
- `min_abs_err` - |predicted_min - true_min|
- `max_abs_err` - |predicted_max - true_max|
- `ok` - Boolean: all three metrics successfully extracted

**Aggregate** (averaged across all samples):
- `mean_abs_err_mean` - Average error for mean predictions
- `min_abs_err_mean` - Average error for min predictions
- `max_abs_err_mean` - Average error for max predictions
- `ok_rate` - Percentage with successful predictions (0.0-1.0)
- `samples` - Total samples evaluated

**Cost** (summed across run):
- `prompt_tokens_total` - Total input tokens
- `completion_tokens_total` - Total output tokens
- `cost_total` - Total USD spent
- `cost_per_sample` - Average USD per sample

## File Structure

```
FactoryBench/
├── factorybench/              # Python package
│   ├── adapters/             # Model adapters (mock, azure_openai)
│   ├── api/                  # FastAPI app (app.py, charts.py)
│   ├── data/                 # Data loaders (local JSON, HuggingFace)
│   ├── eval/                 # Evaluation engine (runner.py)
│   ├── metrics/              # Scoring functions (telemetry_literacy.py)
│   ├── viz/                  # Charts (model comparison focus)
│   ├── cli.py                # Click CLI
│   ├── config.py             # Cost limits, model/dataset registry
│   ├── stages.py             # Stage definitions
│   └── state.py              # RunStateManager (thread-safe)
├── frontend/                 # Remix app
│   ├── app/routes/           # Pages (leaderboard, run, analysis, etc.)
│   ├── app/styles/           # Global CSS (Forgis brand)
│   └── package.json
├── datasets/                 # Local JSON fixtures
├── runs/                     # Run artifacts (JSON, 50+ files)
├── charts/                   # Generated PNG cache
├── pyproject.toml            # Python dependencies
└── README.md                 # This file
```

## Development Roadmap

### ✅ Completed (Phase 1-3)
- Cost safeguards ($1/run, $20/day)
- Real-time progress tracking
- Performance metric
- Model comparison charts
- Default filters
- HuggingFace integration
- Traffic-light heatmaps
- Graceful cancellation

### 🚧 In Progress (Phase 4)
- Additional industrial patterns
- Multi-variate time series
- Anomaly detection scenarios

### 📋 Planned
- **Phase 5**: Stage 2 - Root Cause Analysis
- **Phase 6**: Production deployment (PostgreSQL, S3, auth)
- **Phase 7**: Stage 3 - Guided Remediation

## Contributing

Contributions welcome! Focus areas:
1. Dataset curation (real industrial time series)
2. Model adapters (Anthropic, Google, local models)
3. Evaluation metrics (Stage 2 & 3)
4. UI/UX improvements
5. Documentation

## License

TBD - Contact Forgis for licensing information.

---

**Maintained by**: Forgis  
**Status**: Stage 1 Production-Ready  
**Version**: 0.2.0 (Nov 2025)  
**Support**: [GitHub Issues](https://github.com/Xelerit-Robotics/FactoryBench/issues)

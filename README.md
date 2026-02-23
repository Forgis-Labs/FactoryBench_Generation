# FactoryBench

**Q&A benchmark for machine understanding** — evaluating whether AI can reason about industrial machines, not just detect anomalies.

**Target:** NeurIPS 2026 Datasets and Benchmarks Track

---

## The Idea

When a robot stops at 3 AM, operators face a cascade of questions:

- _"Is this normal?"_ (State)
- _"Is something degrading?"_ (Anomaly)
- _"Why did this happen?"_ (Root Cause)
- _"What if we had maintained it last week?"_ (Counterfactual)
- _"How do we recover?"_ (Procedure)

**FactoryBench tests all five levels** on collaborative robots (UR5e), going deeper on one machine family than existing benchmarks go across many.

---

## 5-Level Q&A Framework

| Level | Task                 | Example Question                                    | Ground Truth Source               |
| ----- | -------------------- | --------------------------------------------------- | --------------------------------- |
| **1** | State Identification | "Is joint 3 moving?"                                | Sensor data (deterministic)       |
| **2** | Anomaly Detection    | "Is friction increasing?"                           | Pattern detection (deterministic) |
| **3** | Root Cause Analysis  | "Why did cycle time increase 15%?"                  | Simulation + physics              |
| **4** | Counterfactual       | "When will thermal throttling occur at 25% faster?" | Digital twin                      |
| **5** | Procedure + Prior    | "Robot stopped. What happened and how to recover?"  | Manual + sensor fusion            |

**Key insight:** Each level builds on previous. Failure at Level N implies failure at Level N+1.

---

## Repository Structure

```
FactoryBench/
├── docs/
│   └── FactoryBench_NeurIPS_Paper_Draft.md    # Current paper (start here)
├── factorybench/                               # Python package
│   ├── adapters/          # LLM adapters (Azure OpenAI)
│   ├── api/               # FastAPI backend
│   ├── data/              # Data loaders
│   ├── eval/              # Evaluation runner
│   ├── metrics/           # Scoring (to extend for Q&A)
│   └── viz/               # Charts
├── frontend/              # Remix web UI
├── datasets/              # Local test fixtures
├── runs/                  # Benchmark results
└── archive/               # Deprecated causal framework code
```

---

## Ground Truth Generation (The Hard Problem)

Scaling Q&A requires reliable answers **without human labeling every sample**. Here's the strategy by level:

### Levels 1-2: Deterministic from Sensors

```yaml
question: "What is the mean current of joint 3 over the last hour?"
answer: 2.47 # Computed directly from RTDE data
provenance: sensor
confidence: 1.0
```

### Level 3: Physics + Simulation

```yaml
question: "Why did joint 3 current increase 12% without velocity change?"
answer: "Increasing friction in gearbox, likely due to insufficient lubrication"
provenance: simulation
evidence:
  - "Inverse dynamics: current increase without velocity change = resistive load"
  - "Thermal model: 8°C temperature rise correlates with friction"
confidence: 0.95
```

**Tools:**

- **NVIDIA Isaac Sim** — UR5e model with fault injection, ground-truth sensor export
- **Inverse dynamics** — τ = M(q)q̈ + C(q,q̇)q̇ + g(q) + τ_friction
- **Thermal models** — Predict overheating from motor currents and duty cycles

### Level 4: Digital Twin Counterfactuals

```yaml
question: "If we run 25% faster, when will thermal throttling occur?"
answer: "After 47 minutes of continuous operation"
provenance: simulation
method: "Isaac Sim with accelerated thermal model"
confidence: 0.90
```

### Level 5: Manual + LLM Consensus

```yaml
question: "Robot stopped with error C203A. What happened and how to recover?"
answer: "Joint 3 protective stop due to force limit exceeded. Recovery: 1) Clear obstruction..."
provenance: consensus
sources:
  - "UR5e Manual Section 4.3.2"
  - "3/3 LLM agreement (GPT-4o, Gemini-2.5, Claude-3.5)"
confidence: 0.85
```

**RAG Pipeline:**

1. Chunk UR5e manual by semantic sections
2. Embed with multimodal model (diagrams matter)
3. Retrieve relevant sections for question
4. Generate answer with citation tracking
5. Validate via multi-LLM consensus

---

## Scaling Strategy

| Phase         | Q&A Pairs | Method                 | Notes                               |
| ------------- | --------- | ---------------------- | ----------------------------------- |
| **Seed**      | 1,000     | Expert annotation      | You + domain experts on FactoryCell |
| **Bootstrap** | 10,000    | Template + simulation  | Level 1-2 scale easily              |
| **Scale**     | 50,000    | Consensus + validation | Level 3-4 require physics           |
| **Full**      | 200,000   | Pipeline automation    | Level 5 smaller quantities OK       |

**Key:** Levels 1-2 are cheap to scale (deterministic). Levels 3-5 need validation but smaller N is acceptable for benchmark.

---

## Key Resources

### Digital Twin

- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim) — UR robot models included
- [Universal_Robots_Isaac_Driver](https://github.com/UniversalRobots/Universal_Robots_Isaac_Driver)
- [URSim](https://www.universal-robots.com/download/software-e-series/simulator-non-linux/) — Official UR offline simulator

### Technical Documentation

- [UR5e User Manual](https://s3-eu-west-1.amazonaws.com/ur-support-site/40971/UR5e_User_Manual_en_Global.pdf) (400+ pages)
- [UR5e Technical Specs](https://www.universal-robots.com/media/1807465/ur5e_e-series_datasheets_web.pdf)

### Related Benchmarks

- **TSAQA** ([arXiv:2601.23204](https://arxiv.org/abs/2601.23204)) — 210k samples, 13 domains, broad but shallow
- **PHM-Bench** ([arXiv:2508.02490](https://arxiv.org/abs/2508.02490)) — Prognostics evaluation framework
- **OpenEQA** — Embodied Q&A, not industrial

### LLM Evaluation

- **LLM-Match** — Open-ended scoring with 0.91 human correlation ([OpenEQA protocol](https://arxiv.org/abs/2312.06648))
- **ReConcile** ([arXiv:2309.13007](https://arxiv.org/abs/2309.13007)) — Multi-LLM consensus voting

---

## Quick Start

### 1. Setup

```powershell
uv venv && uv pip install -e .
cp .env.example .env  # Add Azure OpenAI keys
```

### 2. Run Existing Stage 1 (Telemetry Literacy)

```powershell
# Backend
uvicorn factorybench.api.app:app --reload --port 5173

# Frontend (separate terminal)
cd frontend && npm install && npm run dev
```

### 3. Read the Paper Draft

```
docs/FactoryBench_NeurIPS_Paper_Draft.md
```

Section 4.2 (Pipeline Overview) is marked for expansion.

---

## Dataset Installation and Normalization

### 1. Install AURSAD dataset (HDF5 → CSV)

Download and convert the AURSAD dataset from Zenodo. By default it exports the full dataset; use `--max-timestamps` to limit:

```powershell
# Export full dataset
python -m src.data.data_installation.install_aursad

# Export first 100,000 timestamps
python -m src.data.data_installation.install_aursad --max-timestamps 100000
```

**Output:** One CSV per experiment in the same folder:

```
datasets/open_datasets/aursad/
  experiment_1.csv
  experiment_2.csv
  ...
```

Each experiment is grouped by `sample_nr` from the HDF5 dataset.

### 2. Install CNC Mill dataset (Kaggle)

**First-time setup (interactive):**

```powershell
# Sets up Kaggle credentials interactively
python -m src.data.data_installation.install_cnc --setup

# Then download and extract
python -m src.data.data_installation.install_cnc
```

**Subsequent runs (skip if already downloaded):**

```powershell
# Smart: skips download if already extracted
python -m src.data.data_installation.install_cnc

# Force re-download
python -m src.data.data_installation.install_cnc --force
```

**Features:**

- Defaults to `datasets/open_datasets/cnc/`
- Skips re-download if dataset already extracted
- Interactive Kaggle setup on first run (no manual file editing)
- Adds `timestamp_ms` column (0, 100, 200, ...) to all experiment CSVs
- Renames `experiment_0i.csv` → `experiment_i.csv`
- Environment variable support for CI/CD: `KAGGLE_USERNAME` and `KAGGLE_KEY`

**Setup instructions for new team members:**

Just run the command—the script guides you through setup interactively:

```powershell
python -m src.data.data_installation.install_cnc --setup
```

The script will:

1. Prompt for your Kaggle username and API key
2. Create `~/.kaggle/` and save credentials securely
3. You're done—future runs just work!

(No manual file editing needed.)

### 3. Normalize CSV to UR3e JSON schema

Normalize all `experiment_*.csv` files in a folder. Each CSV becomes a JSON file with the same name:

```powershell
# Normalize all AURSAD experiments
python -m src.data.data_normalization.mapped_dataset_normalizer \
  --dataset aursad \
  --input datasets/open_datasets/aursad \
  --output datasets/normalized_episodes

# Normalize all CNC experiments
python -m src.data.data_normalization.mapped_dataset_normalizer \
  --dataset cnc \
  --input datasets/open_datasets/cnc \
  --output datasets/normalized_episodes
```

### How normalization works

- **Directory input**: Reads all `experiment_*.csv` files and outputs matching `experiment_*.json` files.
- **Single file input**: Normalizes the file to a JSON file with the same base name.
- **With `--episode-column`**: Streams chunks and groups by unique values in the column, writes separate JSON per episode group.
- Mapping files live in `datasets/mappings_of_features/<dataset>.json`—define source→target column mappings.
- Output JSON matches UR3e schema: `[{col1: val, col2: val, ...}, ...]`.
- Use `--no-metadata` to skip `_metadata.json`.
- Use `-v` for verbose logging.

### Supported datasets

- **aursad** (raw: Zenodo HDF5, mapping: `datasets/mappings_of_features/aursad.json`)
- **cnc** (raw: Kaggle, mapping: `datasets/mappings_of_features/cnc.json`)

---

## Coral's Focus Areas

### Priority 1: Q&A Pair Generation

- [ ] Define question templates per level
- [ ] Implement deterministic answer generation (Levels 1-2)
- [ ] Setup Isaac Sim for UR5e fault scenarios (Levels 3-4)
- [ ] Build RAG pipeline over UR5e manual (Level 5)

### Priority 2: Evaluation Protocol

- [ ] Implement LLM-Match scoring
- [ ] Calibrate against human judgments
- [ ] Design physical validation experiments

### Priority 3: Scaling Beyond UR5e

- [ ] Identify 2-3 additional machine families
- [ ] Test template generalization
- [ ] Document scalability findings

---

## Architecture Notes

The existing codebase has a working **Stage 1 (Telemetry Literacy)** pipeline:

- FastAPI backend with cost controls ($1/run, $20/day)
- Real-time progress tracking
- Azure OpenAI integration
- Remix frontend with leaderboard

**To extend for Q&A:** The `factorybench/eval/` and `factorybench/metrics/` modules need adapting from numeric error metrics to LLM-Match scoring.

---

**Maintainer:** Jonas Petersen
**Researcher:** Coral
**Organization:** Forgis AG
**Version:** 0.3.0 (Feb 2026)

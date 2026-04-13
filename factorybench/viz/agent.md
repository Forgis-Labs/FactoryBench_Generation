# Agent Instructions: FactoryBench Figure Generation with PaperBanana

You are assisting with generating publication-quality academic figures for the **FactoryBench** paper (NeurIPS 2026). Figures are produced using [PaperBanana](https://github.com/llmsresearch/paperbanana), a multi-agent framework that turns text descriptions into methodology diagrams via a plan-generate-critique loop.

---

## Project Context

FactoryBench is a Q&A benchmark that evaluates LLM agents on causal machine understanding over industrial time-series data from collaborative robots (UR-series). It defines a four-level hierarchy: L1 State, L2 Intervention, L3 Counterfactual, L4 Decision Making.

Figures must communicate this hierarchy, the Q&A generation pipeline, the SCE causal schema, and benchmark results clearly to an academic audience.

---

## Directory Layout

```
factorybench/viz/
├── agent.md                  # This file
├── README.md                 # Setup, install, and usage guide
├── prompts/                  # Prompt files for figure generation (.txt only)
│   ├── factorybench_method.txt       # Methodology context (bullet-point format)
│   ├── factorybench_method_paper_level.txt  # Methodology context (paper-level prose)
│   └── factorybench_neurips.txt      # Detailed NeurIPS-style prompt (freeform)
├── paperbanana/              # PaperBanana library (cloned fork)
│   ├── paperbanana/          # Core source: agents, core, providers, prompts
│   ├── configs/              # YAML configs (pipeline, provider settings)
│   ├── examples/             # Example scripts and sample inputs
│   └── data/                 # Reference sets and style guidelines
├── charts.py                 # Matplotlib chart generators (bar, scatter, heatmap)
├── generate_plots.py         # Script to generate statistical plots from results
└── run_*/                    # Output directories from previous runs
```

---

## How PaperBanana Works

PaperBanana runs a multi-agent pipeline in two phases:

### Phase 1 — Planning (sequential)
1. **Retriever**: selects relevant reference diagrams from a curated set of 13 academic examples for few-shot learning.
2. **Planner**: generates a detailed textual description of the figure using in-context learning from retrieved examples.
3. **Stylist**: refines the description for NeurIPS-quality aesthetics (soft pastels, clean sans-serif, flat design).

### Phase 2 — Iterative Refinement (loop)
4. **Visualizer**: renders the description into an image using FLUX.2-pro (via Azure AI Foundry).
5. **Critic**: evaluates the image on faithfulness, conciseness, readability, and aesthetics. If the critic finds issues, it produces a revised description and the loop repeats.

Optional **Phase 0 — Input Optimization** (`--optimize` flag): enriches raw context and sharpens captions before planning.

### Key Parameters
- `--iterations N`: fixed number of refinement cycles (default 3).
- `--auto`: loop until the critic is satisfied or `--max-iterations` is reached (default cap: 30).
- `--optimize`: enable Phase 0 input preprocessing.
- `--continue` / `--continue-run <run_id>`: resume a previous run with additional iterations.
- `--feedback "..."`: inject guidance into the critique loop when continuing a run.

---

## Prompt Format

All prompts are plain-text `.txt` files stored in `prompts/`. Each file is passed directly as the `--input` argument to the PaperBanana CLI. The text serves as the methodology description that the planner agent uses to understand what to draw.

Two writing styles are supported:

- **Bullet-point format** (`factorybench_method.txt`): structured bullets with clear hierarchy. Generally easier for the VLM to parse and preferred for most figures.
- **Dense prose** (`factorybench_method_paper_level.txt`): paper-level paragraph text. Works when you want the planner to receive the same framing as the paper.
- **Detailed freeform** (`factorybench_neurips.txt`): exhaustive zone-by-zone description with explicit layout, colors (hex codes), element positions, and style rules. Best for complex multi-zone figures where precision matters.

When writing a new prompt, use **bullets with explicit structure** unless the figure is simple enough that prose suffices.

---

## Writing Effective Prompts

### Rules

1. **Be explicit about layout.** State flow direction, element count, alignment, and connections. "Five horizontal boxes connected by rightward arrows" beats "show the pipeline".

2. **Spell out every text string verbatim.** Image models hallucinate labels when given room for interpretation. Write out every label, subtitle, and callout exactly. Include a line like: "ALL text must match exactly what is specified. Do not add, remove, or rephrase any text."

3. **Use hex codes for colors.** Named colors ("blue") are ambiguous. Use exact hex values and assign them semantically (e.g., `#00796B` for teal/healthy, `#B71C1C` for red/fault). Keep the palette to 5-8 colors.

4. **Use structured sections over walls of text.** Break the diagram into logical sections with clear headings (e.g., `ZONE 1`, `ROW 2`, `CALLOUT`) rather than a single long paragraph. Compare `factorybench_neurips.txt` (structured zones) vs `factorybench_method_paper_level.txt` (dense prose) — structured prompts produce more reliable results.

5. **Anchor the academic context.** Start the prompt with a sentence describing what the figure communicates in the paper, not just what it looks like. This guides the planner agent.

6. **Constrain the style explicitly.** Always include lines stating:
   - No gradients, no 3D effects
   - Target venue and quality level: "NeurIPS 2026 camera-ready"
   - Font family: `Inter` or `Helvetica Neue`
   - Aspect ratio: `16:9`, `16:7`, etc.

7. **Include concrete domain examples.** The model cannot infer domain-specific content like joint currents, error codes, or sensor values. Write out realistic example data.

8. **Iterate via the critique loop rather than rewriting.** If results are off, tighten the most ambiguous part:
   - Text is wrong → add `do_not_hallucinate_text` and spell out strings
   - Layout is messy → add explicit position/alignment instructions
   - Colors are off → replace named colors with hex codes

### FactoryBench-Specific Color Palette

| Color              | Hex       | Usage                                 |
|--------------------|-----------|---------------------------------------|
| Dark navy          | `#122128` | Borders, arrows, headers              |
| Forgis orange      | `#FF762B` | Accents, SCE-Schema, intervention     |
| Blue               | `#3a7bd5` | Sensor-only, Level 1                  |
| Green              | `#2ecc71` | Manual+Sensor, healthy/OK             |
| Red                | `#FF4D00` | Fault states, Level 4                 |
| Panel background   | `#f0f4f8` | Card/panel fills                      |
| Warm amber         | `#FFF3CD` | Question boxes                        |
| Teal               | `#00796B` | L1 State level                        |
| Amber              | `#F57F17` | L2 Intervention level                 |
| Purple             | `#6A1B9A` | L3 Counterfactual level               |
| Deep red           | `#B71C1C` | L4 Decision Making level              |

### Level Color Mapping (for evaluation hierarchy figures)

| Level | Background | Border    |
|-------|------------|-----------|
| L1    | `#E0F2F1`  | `#00796B` |
| L2    | `#FFF8E1`  | `#F57F17` |
| L3    | `#F3E5F5`  | `#6A1B9A` |
| L4    | `#FFEBEE`  | `#B71C1C` |

---

## Running a Generation

### Prerequisites

- Python 3.10+ with the virtual environment activated (`.venv`)
- PaperBanana installed: `pip install -e "factorybench\viz\paperbanana[openai]"`
- `.env` configured with Azure OpenAI + FLUX.2-pro credentials (see `README.md`)

### Basic Generation

```bash
cd factorybench/viz/paperbanana

# Basic generation (3 iterations)
python -m paperbanana.cli generate \
  --input ../prompts/factorybench_method.txt \
  --caption "Overview of the FactoryBench evaluation hierarchy"

# With input optimization and auto-refinement
python -m paperbanana.cli generate \
  --input ../prompts/factorybench_method.txt \
  --caption "Overview of the FactoryBench evaluation hierarchy" \
  --optimize --auto --max-iterations 10

# Detailed prompt with more iterations
python -m paperbanana.cli generate \
  --input ../prompts/factorybench_neurips.txt \
  --caption "Illustration of the FactoryBench tasks" \
  --iterations 5
```

### Continuing a Previous Run

```bash
python -m paperbanana.cli generate \
  --continue-run run_20260324_015207_97cf87 \
  --iterations 3 \
  --feedback "Make the arrows thicker and fix the label on the L3 box"
```

### Output

Generated images are saved to `outputs/run_<id>/` or `figures/`. Each run produces:
- `diagram_iter_N.png` — image from each refinement iteration
- `run_input.json` — original input
- `planning.json` — retrieved examples and descriptions
- `metadata.json` — run configuration and timing

---

## Creating a New Figure Prompt

1. **Decide the figure's purpose.** What does it communicate in the paper? Write one sentence.

2. **Choose a writing style.** Use existing prompts as templates:
   - `factorybench_method.txt` — bullet-point methodology context
   - `factorybench_method_paper_level.txt` — dense prose methodology context
   - `factorybench_neurips.txt` — detailed zone-by-zone layout with hex colors and style rules

3. **Write the `.txt` prompt** following the rules above. For complex figures, use the zone-by-zone style of `factorybench_neurips.txt`. For simpler figures, bullet-point context is sufficient.

4. **Save to `prompts/`** with a descriptive name: `fig<N>_<short_description>.txt`.

5. **Run and iterate.** Start with 3 iterations, inspect the output, then either continue the run with feedback or tighten the prompt.

---

## Statistical Plots (Matplotlib)

For data-driven charts (bar charts, heatmaps, scatter plots), use the existing Python utilities instead of PaperBanana:

- `charts.py` — functions: `create_model_performance_bar_chart`, `create_cost_vs_performance_scatter`, `create_model_metrics_heatmap`, `generate_all_charts`
- `generate_plots.py` — CLI script to render all charts from evaluation results

PaperBanana's `plot` command can also generate matplotlib-based plots via VLM code generation:

```bash
python -m paperbanana.cli plot \
  --data results.csv \
  --intent "Grouped bar chart comparing model accuracy across L1-L4"
```

---

## Environment Variables

These must be set in `.env` at the project root:

| Variable                    | Purpose                         |
|-----------------------------|---------------------------------|
| `VLM_PROVIDER`              | VLM backend (`openai`)          |
| `OPENAI_API_KEY`            | Azure OpenAI API key            |
| `OPENAI_BASE_URL`           | Azure OpenAI endpoint           |
| `OPENAI_VLM_MODEL`          | Azure OpenAI model name         |
| `AZURE_OPENAI_API_VERSION`  | API version (`2024-02-01`)      |
| `IMAGE_PROVIDER`            | Image backend (`openai_imagen`) |
| `FLUX_API_KEY`              | Azure FLUX.2-pro API key        |
| `FLUX_BASE_URL`             | Azure FLUX.2-pro endpoint       |
| `OPENAI_IMAGE_MODEL`        | FLUX model name                 |

---

## Troubleshooting

| Problem                        | Fix                                                                 |
|--------------------------------|---------------------------------------------------------------------|
| Garbled/nonsensical text       | Add `do_not_hallucinate_text` rule; spell out every label verbatim  |
| Wrong layout or flow direction | Add explicit layout field with direction, count, and alignment       |
| Colors don't match             | Replace named colors with hex codes in `color_palette`              |
| Too many iterations, no improvement | Tighten the most ambiguous part of the prompt, don't rewrite all |
| API timeout or rate limit      | PaperBanana retries up to 8 times with exponential backoff          |
| Missing `.env` keys            | Run `python -m paperbanana.cli setup` or copy from `.env.example`   |

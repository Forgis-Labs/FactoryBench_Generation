# FactoryBench Visualization & Figure Generation

This directory contains the tools and scripts used to generate the figures for the FactoryBench academic paper. The pipeline is decoupled into specialized scripts for statistical plots and conceptual diagrams.

## 📂 Directory Structure

- `generate_plots.py`: Native Matplotlib script for generating Figure 3 (Model Comparison) and Figure 4 (The Causal Gap).
- `run_paperbanana.py`: PaperBanana orchestration script for generating Fig 1, 2, 5, and 6.
- `prompts/`: JSON configuration files for each PaperBanana figure.
- `charts.py`: Core Matplotlib implementations and brand styles.
- `paperbanana/`: Local integration of the PaperBanana framework.
- `figures/`: (Generated) Final PNG outputs.

## 🚀 Getting Started

### 1. Prerequisites

```bash
pip install matplotlib numpy pandas openai structlog tenacity pydantic-settings python-dotenv pillow
```

### 2. Configuration (`.env`)
```env
AZURE_OPENAI_ENDPOINT="your-endpoint"
AZURE_OPENAI_KEY="your-key"
AZURE_OPENAI_DEPLOYMENT="your-gpt-4o-deployment"
AZURE_OPENAI_IMAGE_DEPLOYMENT="your-dalle-3-deployment"
```
### 3. Usage

#### Generate Statistical Plots (Matplotlib)
```bash
python .\factorybench\viz\generate_plots.py
```

#### Generate Conceptual Diagrams (PaperBanana)
- PaperBanana (PaperPlot) website: https://paperbanana.org/ https://paperplot.org/
- GitHub repo: https://github.com/dwzhu-pku/PaperBanana
- Rsearch GitHub repo: https://github.com/llmsresearch/paperbanana

To generate all diagrams:
```bash
python .\factorybench\viz\run_paperbanana.py
```
To generate a specific figure (e.g., fig1_qa_pipeline):
```bash
python .\factorybench\viz\run_paperbanana.py fig1_qa_pipeline
```

## 📝 Customizing Prompts

You can define diagrams using a simple or structured JSON schema in the `prompts/` directory.

### Simple Schema
```json
{
    "name": "fig2_overview",
    "caption": "Your communication intent here",
    "context": "The technical background here",
    "diagram_type": "methodology"
}
```

### Structured Schema (Recommended)
For precise control over layouts, components, and exact text labels, use the structured schema:

```json
{
    "name": "fig1_qa_pipeline",
    "caption": "QA Generation Pipeline",
    "layout": "horizontal left-to-right, five boxes connected by arrows",
    "boxes": [
        {
          "position": 1,
          "label": "Step 1",
          "icon": "database icon",
          "subtitle": "Technical detail",
          "color": "light blue"
        }
    ],
    "callout_above": {
        "position": "above box 1",
        "exact_text": "Sample Data: {key: value}"
    },
    "style": {
        "font": "Inter",
        "arrows": "thick blue",
        "quality": "NeurIPS publication style"
    }
}
```
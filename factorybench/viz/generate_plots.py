"""
Script to generate FactoryBench paper statistical plots using Matplotlib.
"""

from pathlib import Path
from charts import create_reasoning_level_comparison_chart, create_causal_gap_chart

# Define Output directory
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

def generate_matplotlib_plots():
    """Generate Figure 3 and 4 using native Matplotlib charts."""
    print("--- Generating Fig 3 & 4 via Matplotlib ---")
    
    # Mock data for Figure 3
    fig3_data = {
        "GPT-4o": {"Level 1": 0.85, "Level 2": 0.72, "Level 3": 0.58, "Level 4": 0.45, "Level 5": 0.38},
        "Claude 3.5 Sonnet": {"Level 1": 0.88, "Level 2": 0.75, "Level 3": 0.62, "Level 4": 0.48, "Level 5": 0.42},
        "Gemini 1.5 Pro": {"Level 1": 0.82, "Level 2": 0.68, "Level 3": 0.55, "Level 4": 0.42, "Level 5": 0.35},
        "Llama 3.1 70B": {"Level 1": 0.78, "Level 2": 0.62, "Level 3": 0.48, "Level 4": 0.35, "Level 5": 0.28},
        "Mistral Large": {"Level 1": 0.75, "Level 2": 0.58, "Level 3": 0.45, "Level 4": 0.32, "Level 5": 0.25},
        "Qwen2.5 72B": {"Level 1": 0.72, "Level 2": 0.55, "Level 3": 0.42, "Level 4": 0.30, "Level 5": 0.22},
    }
    create_reasoning_level_comparison_chart(fig3_data, OUTPUT_DIR / "fig3_comparison.png")
    
    # Mock data for Figure 4
    fig4_data = {
        "Rung 1": (1.0, 0.85),
        "Rung 2": (0.95, 0.70),
        "Rung 3": (0.90, 0.39),
        "Rung 4": (0.85, 0.30),
    }
    create_causal_gap_chart(fig4_data, OUTPUT_DIR / "fig4_causal_gap.png")
    print("Saved Matplotlib charts to figures/")

if __name__ == "__main__":
    generate_matplotlib_plots()

"""
Model comparison charts for FactoryBench analysis.
Focus: Compare different models on the same dataset using performance metrics.
Uses matplotlib with Forgis brand colors.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# Forgis brand colors
COLORS = {
    "fire": "#FF4D00",
    "tiger": "#FF762B",
    "flicker": "#DC4B07",
    "platinum": "#CCD3D6",
    "steel": "#707B84",
    "white": "#FFFFFF",
    "gunmetal": "#122128",
}

# Scientific plotting style
STYLE_CONFIG = {
    "figure.facecolor": COLORS["gunmetal"],
    "axes.facecolor": "#0e1a22",
    "axes.edgecolor": COLORS["steel"],
    "axes.labelcolor": COLORS["platinum"],
    "axes.grid": True,
    "grid.color": COLORS["steel"],
    "grid.alpha": 0.2,
    "grid.linewidth": 0.5,
    "text.color": COLORS["white"],
    "xtick.color": COLORS["platinum"],
    "ytick.color": COLORS["platinum"],
    "legend.facecolor": "#0e1a22",
    "legend.edgecolor": COLORS["steel"],
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Arial", "Helvetica"],
    "font.size": 12,  # Increased from 10
}


def apply_style():
    """Apply Forgis brand style to matplotlib."""
    plt.rcParams.update(STYLE_CONFIG)


def _select_best_run(runs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Select the best run from a group: most samples processed, then most recent.
    
    Args:
        runs: List of run dictionaries
        
    Returns:
        Selected run or None if empty
    """
    if not runs:
        return None
    
    # Sort by samples (desc) then by started_at (desc for most recent)
    sorted_runs = sorted(
        runs,
        key=lambda r: (
            r.get("aggregate", {}).get("samples", 0),
            r.get("started_at", "")
        ),
        reverse=True
    )
    return sorted_runs[0]


def _group_runs_by_model_dataset(runs: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """
    Group runs by (model, dataset) pairs.
    
    Args:
        runs: List of run dictionaries
        
    Returns:
        Dict mapping (model, dataset) to list of runs
    """
    groups = {}
    for run in runs:
        model = run.get("model", "unknown")
        dataset_id = run.get("dataset", {}).get("dataset_id", "unknown")
        key = (model, dataset_id)
        if key not in groups:
            groups[key] = []
        groups[key].append(run)
    return groups


def create_model_performance_bar_chart(runs: List[Dict[str, Any]], output_path: Path) -> Figure:
    """
    Create sorted bar chart comparing model performance.
    Lower performance score is better.
    
    Args:
        runs: List of run dictionaries
        output_path: Path to save the chart
        
    Returns:
        Matplotlib figure
    """
    apply_style()
    
    if not runs:
        return _create_empty_chart("No data available")
    
    # Group by (model, dataset) and select best run
    groups = _group_runs_by_model_dataset(runs)
    data = []
    for (model, dataset), group_runs in groups.items():
        best_run = _select_best_run(group_runs)
        if best_run:
            perf = best_run.get("aggregate", {}).get("performance", 0)
            samples = int(best_run.get("aggregate", {}).get("samples", 0))
            data.append({
                "model": model,
                "dataset": dataset,
                "performance": perf,
                "samples": samples
            })
    
    if not data:
        return _create_empty_chart("No valid data")
    
    # Sort by performance (ascending - lower is better)
    data.sort(key=lambda x: x["performance"])
    
    labels = [f"{d['model']}\n({d['dataset']})" for d in data]
    performances = [d["performance"] for d in data]
    samples = [d["samples"] for d in data]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Color gradient from best (fire) to worst (steel)
    colors = [COLORS["fire"] if i == 0 else COLORS["tiger"] if i == 1 else COLORS["flicker"] if i == 2 else COLORS["steel"] for i in range(len(data))]
    
    bars = ax.barh(labels, performances, color=colors, alpha=0.9)
    
    # Add sample count annotations
    for i, (bar, samp) in enumerate(zip(bars, samples)):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"n={samp}",
                va="center", fontsize=11, color=COLORS["platinum"])
    
    ax.set_xlabel("Performance (Lower is Better)", fontweight="bold", fontsize=14)
    ax.set_ylabel("Model (Dataset)", fontweight="bold", fontsize=14)
    ax.set_title("Model Performance Comparison", fontsize=16, fontweight="bold", pad=15)
    ax.grid(True, axis="x", alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=COLORS["gunmetal"], edgecolor="none")
    
    return fig


def create_cost_vs_performance_scatter(runs: List[Dict[str, Any]], output_path: Path) -> Figure:
    """
    Create scatter plot showing cost per sample vs performance tradeoff.
    Each point is the best run for a (model, dataset) combination.
    
    Args:
        runs: List of run dictionaries
        output_path: Path to save the chart
        
    Returns:
        Matplotlib figure
    """
    apply_style()
    
    if not runs:
        return _create_empty_chart("No data available")
    
    # Group by (model, dataset) and select best run
    groups = _group_runs_by_model_dataset(runs)
    data = []
    for (model, dataset), group_runs in groups.items():
        best_run = _select_best_run(group_runs)
        if best_run:
            perf = best_run.get("aggregate", {}).get("performance", 0)
            cost = best_run.get("aggregate", {}).get("cost_per_sample", 0)
            samples = int(best_run.get("aggregate", {}).get("samples", 0))
            data.append({
                "model": model,
                "dataset": dataset,
                "performance": perf,
                "cost": cost,
                "samples": samples
            })
    
    if not data:
        return _create_empty_chart("No valid data")
    
    costs = [d["cost"] for d in data]
    performances = [d["performance"] for d in data]
    samples = [d["samples"] for d in data]
    labels = [f"{d['model']} ({d['dataset']})" for d in data]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Size points by sample count
    sizes = [50 + s/10 for s in samples]
    scatter = ax.scatter(costs, performances, s=sizes, alpha=0.7,
                        c=range(len(data)), cmap="autumn",
                        edgecolors=COLORS["white"], linewidth=1.5)
    
    # Add labels
    for i, label in enumerate(labels):
        ax.annotate(label, (costs[i], performances[i]),
                   xytext=(5, 5), textcoords="offset points",
                   fontsize=10, color=COLORS["platinum"])
    
    ax.set_xlabel("Cost per Sample ($)", fontweight="bold", fontsize=14)
    ax.set_ylabel("Performance (Lower is Better)", fontweight="bold", fontsize=14)
    ax.set_title("Cost vs Performance Tradeoff", fontsize=16, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3)
    
    # Add Pareto frontier hint
    ax.text(0.02, 0.06, "← Ideal: Low cost, low error",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            color=COLORS["fire"], style="italic")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=COLORS["gunmetal"], edgecolor="none")
    
    return fig


def create_model_metrics_heatmap(runs: List[Dict[str, Any]], output_path: Path) -> Figure:
    """
    Create heatmap showing all error metrics for each model-dataset combination.
    
    Args:
        runs: List of run dictionaries
        output_path: Path to save the chart
        
    Returns:
        Matplotlib figure
    """
    apply_style()
    
    if not runs:
        return _create_empty_chart("No data available")
    
    # Group by (model, dataset) and select best run
    groups = _group_runs_by_model_dataset(runs)
    data = []
    for (model, dataset), group_runs in groups.items():
        best_run = _select_best_run(group_runs)
        if best_run:
            agg = best_run.get("aggregate", {})
            data.append({
                "label": f"{model}\n({dataset})",
                "mean_err": agg.get("mean_abs_err_mean", 0),
                "min_err": agg.get("min_abs_err_mean", 0),
                "max_err": agg.get("max_abs_err_mean", 0),
                "performance": agg.get("performance", 0),
                "ok_rate": agg.get("ok_rate", 0) * 100
            })
    
    if not data:
        return _create_empty_chart("No valid data")
    
    # Sort by performance
    data.sort(key=lambda x: x["performance"])
    
    labels = [d["label"] for d in data]
    metrics = ["Mean Error", "Min Error", "Max Error", "Performance"]
    # Separate error metrics from OK rate (different magnitude)
    error_values = np.array([
        [d["mean_err"] for d in data],
        [d["min_err"] for d in data],
        [d["max_err"] for d in data],
        [d["performance"] for d in data]
    ])
    ok_rates = [d["ok_rate"] for d in data]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Use Red-Yellow-Green colormap (inverted so green=low error=good)
    im = ax.imshow(error_values, cmap="RdYlGn_r", aspect="auto", interpolation="nearest")
    
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(metrics, fontsize=12)
    
    # Add value annotations for error metrics
    for i in range(len(metrics)):
        for j in range(len(labels)):
            text = ax.text(j, i, f"{error_values[i, j]:.3f}",
                          ha="center", va="center", color=COLORS["gunmetal"],
                          fontsize=9, fontweight="bold")
    
    ax.set_title("Model Metrics Heatmap", fontsize=16, fontweight="bold", pad=15)
    
    # Colorbar for error metrics
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Error (Red=High, Green=Low)", rotation=270, labelpad=15, color=COLORS["platinum"], fontsize=12)
    cbar.ax.tick_params(colors=COLORS["platinum"], labelsize=11)
    
    # Add OK rate as text below the chart
    ok_text = "OK Rate: " + ", ".join([f"{labels[i].replace(chr(10), ' ')}: {ok_rates[i]:.1f}%" for i in range(len(labels))])
    ax.text(0.5, -0.15, ok_text, ha="center", va="top", transform=ax.transAxes,
            fontsize=10, color=COLORS["platinum"], wrap=True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=COLORS["gunmetal"], edgecolor="none")
    
    return fig


def _create_empty_chart(message: str) -> Figure:
    """Create a placeholder chart for missing data."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.text(0.5, 0.5, message, ha="center", va="center",
            fontsize=16, color=COLORS["steel"])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    return fig


def generate_all_charts(runs_dir: Path, output_dir: Path, model_filters: list = None, dataset_filters: list = None):
    """
    Generate all charts for model comparison analysis.
    
    Args:
        runs_dir: Directory containing run JSON files
        output_dir: Directory to save chart images
        model_filters: Optional list of model IDs to filter runs
        dataset_filters: Optional list of dataset sources to filter runs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all runs
    runs = []
    all_runs_count = 0
    for run_file in runs_dir.glob("*.json"):
        all_runs_count += 1
        with open(run_file) as f:
            run_data = json.load(f)
            
            # Apply filters
            if model_filters and run_data.get("model") not in model_filters:
                continue
            if dataset_filters:
                dataset_id = run_data.get("dataset", {}).get("dataset_id")
                if dataset_id not in dataset_filters:
                    continue
                    
            runs.append(run_data)
    
    if not runs:
        print(f"No runs found matching filters (total runs: {all_runs_count})")
        print(f"  model_filters: {model_filters}")
        print(f"  dataset_filters: {dataset_filters}")
        # Create empty/placeholder charts
        _create_empty_chart("No data matches current filters").savefig(output_dir / "model_performance.png", dpi=300, facecolor=COLORS["gunmetal"])
        _create_empty_chart("No data matches current filters").savefig(output_dir / "cost_vs_performance.png", dpi=300, facecolor=COLORS["gunmetal"])
        _create_empty_chart("No data matches current filters").savefig(output_dir / "metrics_heatmap.png", dpi=300, facecolor=COLORS["gunmetal"])
        return
    
    filter_desc = []
    if model_filters:
        filter_desc.append(f"models={','.join(model_filters)}")
    if dataset_filters:
        filter_desc.append(f"datasets={','.join(dataset_filters)}")
    filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""
    
    print(f"Generating charts for {len(runs)}/{all_runs_count} runs{filter_str}...")
    
    create_model_performance_bar_chart(runs, output_dir / "model_performance.png")
    create_cost_vs_performance_scatter(runs, output_dir / "cost_vs_performance.png")
    create_model_metrics_heatmap(runs, output_dir / "metrics_heatmap.png")
    
    print(f"Charts saved to {output_dir}")


if __name__ == "__main__":
    from factorybench.config import RUN_DIR
    
    runs_dir = RUN_DIR
    output_dir = Path("charts")
    
    generate_all_charts(runs_dir, output_dir)

"""
Paper-quality chart generation for FactoryBench analysis.
Uses matplotlib with Forgis brand colors for scientific visualization.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    "font.size": 10,
}


def apply_style():
    """Apply Forgis brand style to matplotlib."""
    plt.rcParams.update(STYLE_CONFIG)


def create_error_distribution_chart(runs: List[Dict[str, Any]], output_path: Path) -> Figure:
    """
    Create error distribution bar chart for mean/min/max absolute errors.
    
    Args:
        runs: List of run dictionaries with aggregate metrics
        output_path: Path to save the chart
        
    Returns:
        Matplotlib figure
    """
    apply_style()
    
    if not runs:
        return _create_empty_chart("No data available")
    
    # Extract error metrics
    mean_errors = [r.get("aggregate", {}).get("mean_abs_err_mean", 0) for r in runs]
    # Correct metric keys: aggregator outputs min_abs_err_mean / max_abs_err_mean
    min_errors = [r.get("aggregate", {}).get("min_abs_err_mean", 0) for r in runs]
    max_errors = [r.get("aggregate", {}).get("max_abs_err_mean", 0) for r in runs]
    labels = [r.get("run_id", f"Run {i+1}")[:8] for i, r in enumerate(runs)]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    x = np.arange(len(runs))
    width = 0.25
    
    ax.bar(x - width, mean_errors, width, label="Mean Error", color=COLORS["fire"], alpha=0.9)
    ax.bar(x, min_errors, width, label="Min Error", color=COLORS["tiger"], alpha=0.9)
    ax.bar(x + width, max_errors, width, label="Max Error", color=COLORS["flicker"], alpha=0.9)
    
    ax.set_xlabel("Run ID", fontweight="bold", fontsize=11)
    ax.set_ylabel("Absolute Error", fontweight="bold", fontsize=11)
    ax.set_title("Error Distribution by Metric Type", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(framealpha=0.9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=COLORS["gunmetal"], edgecolor="none")
    
    return fig


def create_convergence_chart(runs: List[Dict[str, Any]], output_path: Path) -> Figure:
    """
    Create model convergence line chart showing improvement over time.
    
    Args:
        runs: List of run dictionaries sorted by time
        output_path: Path to save the chart
        
    Returns:
        Matplotlib figure
    """
    apply_style()
    
    if not runs:
        return _create_empty_chart("No data available")
    
    # Sort by timestamp
    sorted_runs = sorted(runs, key=lambda r: r.get("started_at", ""))
    errors = [r.get("aggregate", {}).get("mean_abs_err_mean", 0) for r in sorted_runs]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    x = np.arange(len(errors))
    ax.plot(x, errors, marker="o", linewidth=2, markersize=6, 
            color=COLORS["tiger"], markerfacecolor=COLORS["fire"], 
            markeredgecolor=COLORS["white"], markeredgewidth=1)
    
    # Add trend line
    if len(errors) > 1:
        z = np.polyfit(x, errors, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "--", color=COLORS["platinum"], alpha=0.5, linewidth=1.5, label="Trend")
    
    ax.set_xlabel("Run Number (Chronological)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Mean Absolute Error", fontweight="bold", fontsize=11)
    ax.set_title("Model Convergence Over Time", fontsize=13, fontweight="bold", pad=15)
    ax.legend(framealpha=0.9, loc="upper right")
    ax.grid(True, alpha=0.3)
    
    # Add improvement percentage
    if len(errors) > 1 and errors[0] > 0:
        improvement = ((errors[0] - errors[-1]) / errors[0]) * 100
        ax.text(0.02, 0.98, f"Improvement: {improvement:.1f}%", 
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor=COLORS["gunmetal"], alpha=0.8, edgecolor=COLORS["fire"]))
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=COLORS["gunmetal"], edgecolor="none")
    
    return fig


def create_ok_rate_chart(runs: List[Dict[str, Any]], output_path: Path) -> Figure:
    """
    Create OK rate over time chart showing percentage of correct predictions.
    
    Args:
        runs: List of run dictionaries sorted by time
        output_path: Path to save the chart
        
    Returns:
        Matplotlib figure
    """
    apply_style()
    
    if not runs:
        return _create_empty_chart("No data available")
    
    sorted_runs = sorted(runs, key=lambda r: r.get("started_at", ""))
    ok_rates = [(r.get("aggregate", {}).get("ok_rate", 0) * 100) for r in sorted_runs]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    x = np.arange(len(ok_rates))
    ax.plot(x, ok_rates, marker="s", linewidth=2, markersize=6,
            color=COLORS["flicker"], markerfacecolor=COLORS["tiger"],
            markeredgecolor=COLORS["white"], markeredgewidth=1)
    
    ax.axhline(y=100, color=COLORS["platinum"], linestyle="--", alpha=0.3, linewidth=1)
    ax.fill_between(x, ok_rates, alpha=0.2, color=COLORS["fire"])
    
    ax.set_xlabel("Run Number (Chronological)", fontweight="bold", fontsize=11)
    ax.set_ylabel("OK Rate (%)", fontweight="bold", fontsize=11)
    ax.set_title("OK Rate Over Time", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.grid(True, alpha=0.3)
    
    # Add latest value
    if ok_rates:
        ax.text(0.98, 0.98, f"Latest: {ok_rates[-1]:.1f}%",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor=COLORS["gunmetal"], alpha=0.8, edgecolor=COLORS["fire"]))
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=COLORS["gunmetal"], edgecolor="none")
    
    return fig


def create_model_comparison_chart(runs: List[Dict[str, Any]], output_path: Path) -> Figure:
    """
    Create model comparison chart showing performance of different models.
    
    Args:
        runs: List of run dictionaries
        output_path: Path to save the chart
        
    Returns:
        Matplotlib figure
    """
    apply_style()
    
    if not runs:
        return _create_empty_chart("No data available")
    
    # Group by model
    by_model = {}
    for r in runs:
        model = r.get("model", "unknown")
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(r.get("aggregate", {}).get("mean_abs_err_mean", 0))
    
    models = list(by_model.keys())
    avg_errors = [np.mean(by_model[m]) for m in models]
    std_errors = [np.std(by_model[m]) for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    colors = [COLORS["fire"], COLORS["tiger"], COLORS["flicker"], COLORS["steel"]]
    bars = ax.barh(models, avg_errors, xerr=std_errors, 
                   color=colors[:len(models)], alpha=0.9, capsize=5)
    
    ax.set_xlabel("Mean Absolute Error (avg ± std)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Model", fontweight="bold", fontsize=11)
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, axis="x", alpha=0.3)
    
    # Add value labels
    for i, (bar, avg, std) in enumerate(zip(bars, avg_errors, std_errors)):
        ax.text(bar.get_width() + std + 0.01, bar.get_y() + bar.get_height()/2,
                f"{avg:.3f} ± {std:.3f}",
                va="center", fontsize=9, color=COLORS["platinum"])
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=COLORS["gunmetal"], edgecolor="none")
    
    return fig


def create_error_breakdown_chart(runs: List[Dict[str, Any]], output_path: Path) -> Figure:
    """
    Create stacked bar chart showing error breakdown by metric (mean/min/max).
    
    Args:
        runs: List of run dictionaries
        output_path: Path to save the chart
        
    Returns:
        Matplotlib figure
    """
    apply_style()
    
    if not runs:
        return _create_empty_chart("No data available")
    
    mean_errors = [r.get("aggregate", {}).get("mean_abs_err_mean", 0) for r in runs]
    min_errors = [r.get("aggregate", {}).get("min_abs_err_mean", 0) for r in runs]
    max_errors = [r.get("aggregate", {}).get("max_abs_err_mean", 0) for r in runs]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    metrics = ["Mean Error", "Min Error", "Max Error"]
    values = [np.mean(mean_errors), np.mean(min_errors), np.mean(max_errors)]
    colors_list = [COLORS["fire"], COLORS["tiger"], COLORS["flicker"]]
    
    wedges, texts, autotexts = ax.pie(values, labels=metrics, colors=colors_list,
                                        autopct="%1.1f%%", startangle=90,
                                        textprops={"color": COLORS["white"], "fontsize": 11},
                                        wedgeprops={"edgecolor": COLORS["gunmetal"], "linewidth": 2})
    
    for autotext in autotexts:
        autotext.set_color(COLORS["white"])
        autotext.set_fontweight("bold")
    
    ax.set_title("Average Error Breakdown by Metric", fontsize=13, fontweight="bold", pad=15)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=COLORS["gunmetal"], edgecolor="none")
    
    return fig


def create_cost_quality_chart(runs: List[Dict[str, Any]], output_path: Path) -> Figure:
    """
    Create scatter plot showing performance vs cost tradeoff.
    
    Args:
        runs: List of run dictionaries
        output_path: Path to save the chart
        
    Returns:
        Matplotlib figure
    """
    apply_style()
    
    if not runs:
        return _create_empty_chart("No data available")
    
    # Mock cost data (in production, this would come from run metadata)
    errors = [r.get("aggregate", {}).get("mean_abs_err_mean", 0) for r in runs]
    costs = [np.random.uniform(0.01, 0.5) for _ in runs]  # Placeholder
    models = [r.get("model", "unknown") for r in runs]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    scatter = ax.scatter(costs, errors, s=100, alpha=0.7, c=range(len(runs)),
                        cmap="autumn", edgecolors=COLORS["white"], linewidth=1.5)
    
    # Add model labels
    for i, model in enumerate(models):
        ax.annotate(model[:10], (costs[i], errors[i]), 
                   xytext=(5, 5), textcoords="offset points",
                   fontsize=8, color=COLORS["platinum"])
    
    ax.set_xlabel("Cost per Sample ($)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Mean Absolute Error", fontweight="bold", fontsize=11)
    ax.set_title("Performance vs Cost Tradeoff", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3)
    
    # Add Pareto frontier indicator
    ax.text(0.02, 0.98, "← Lower cost, better performance",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            color=COLORS["fire"], style="italic")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor=COLORS["gunmetal"], edgecolor="none")
    
    return fig


def _create_empty_chart(message: str) -> Figure:
    """Create a placeholder chart for missing data."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.text(0.5, 0.5, message, ha="center", va="center",
            fontsize=14, color=COLORS["steel"])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    return fig


def generate_all_charts(runs_dir: Path, output_dir: Path, model_filters: list = None, dataset_filters: list = None):
    """
    Generate all charts for all runs in the runs directory.
    
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
        _create_empty_chart("No data matches current filters").savefig(output_dir / "error_distribution.png", dpi=300, facecolor=COLORS["gunmetal"])
        _create_empty_chart("No data matches current filters").savefig(output_dir / "convergence.png", dpi=300, facecolor=COLORS["gunmetal"])
        _create_empty_chart("No data matches current filters").savefig(output_dir / "ok_rate.png", dpi=300, facecolor=COLORS["gunmetal"])
        _create_empty_chart("No data matches current filters").savefig(output_dir / "model_comparison.png", dpi=300, facecolor=COLORS["gunmetal"])
        _create_empty_chart("No data matches current filters").savefig(output_dir / "error_breakdown.png", dpi=300, facecolor=COLORS["gunmetal"])
        _create_empty_chart("No data matches current filters").savefig(output_dir / "cost_quality.png", dpi=300, facecolor=COLORS["gunmetal"])
        return
    
    filter_desc = []
    if model_filters:
        filter_desc.append(f"models={','.join(model_filters)}")
    if dataset_filters:
        filter_desc.append(f"datasets={','.join(dataset_filters)}")
    filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""
    
    print(f"Generating charts for {len(runs)}/{all_runs_count} runs{filter_str}...")
    
    create_error_distribution_chart(runs, output_dir / "error_distribution.png")
    create_convergence_chart(runs, output_dir / "convergence.png")
    create_ok_rate_chart(runs, output_dir / "ok_rate.png")
    create_model_comparison_chart(runs, output_dir / "model_comparison.png")
    create_error_breakdown_chart(runs, output_dir / "error_breakdown.png")
    create_cost_quality_chart(runs, output_dir / "cost_quality.png")
    
    print(f"Charts saved to {output_dir}")


if __name__ == "__main__":
    from factorybench.config import RUN_DIR
    
    runs_dir = RUN_DIR
    output_dir = Path("charts")
    
    generate_all_charts(runs_dir, output_dir)

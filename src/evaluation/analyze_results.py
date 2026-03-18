import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Analyze results and generate heatmaps.")
    parser.add_argument("--input", type=Path, required=True, help="Directory containing LLM replies")
    parser.add_argument("--questions", type=Path, required=True, help="Directory containing generated questions")
    parser.add_argument("--figures-dir", type=Path, required=True, help="Directory to save figures")
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load questions to map stem -> template_type
    question_map = {}
    for q_path in args.questions.rglob("*.json"):
        try:
            q_data = load_json(q_path)
            question_map[q_path.stem] = q_data.get("template_type", "unknown")
        except Exception:
            pass

    # 2. Load replies
    results = []
    for r_path in args.input.rglob("*_answer.json"):
        try:
            data = load_json(r_path)
            results.append(data)
        except Exception:
            continue

    if not results:
        print(f"No valid result files found in {args.input}")
        return

    # 3. Aggregate data
    # model -> template_type -> list of scores
    model_template_scores = defaultdict(lambda: defaultdict(list))
    for res in results:
        model = res.get("model", "unknown")
        score = res.get("score")
        if score is None:
            continue # Skipped or error
            
        prompt_file = Path(res.get("prompt_file", ""))
        template_type = question_map.get(prompt_file.stem, "unknown")
                
        model_template_scores[model][template_type].append(float(score))

    # 4. Create heatmap for each model
    for model, template_scores in model_template_scores.items():
        templates = sorted(template_scores.keys())
        # We'll create a 1D heatmap (bar) for the model across templates since we only have 1 level (Level 1)
        # But we can format it nicely as a 2D matrix: 1 x len(templates)
        
        acc_values = []
        labels = []
        for t in templates:
            scores = template_scores[t]
            acc = sum(scores) / len(scores) if scores else 0
            acc_values.append(acc * 100)
            labels.append(f"{t}\n(n={len(scores)})")
            
        acc_matrix = np.array([acc_values])
        
        plt.figure(figsize=(max(8, len(templates) * 1.5), 3))
        ax = sns.heatmap(
            acc_matrix, 
            annot=True, 
            fmt=".1f", 
            cmap="YlGnBu", 
            cbar=True, 
            vmin=0, 
            vmax=100,
            cbar_kws={'label': 'Accuracy (%)'}
        )
        
        ax.set_yticklabels([model], rotation=0, fontsize=10, fontweight='bold')
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        plt.title(f"Performance Heatmap - {model}", pad=20)
        plt.tight_layout()
        
        safe_model_name = model.replace("/", "_").replace(":", "_")
        out_path = args.figures_dir / f"heatmap_{safe_model_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Heatmap saved to {out_path}")

if __name__ == "__main__":
    main()

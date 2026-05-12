"""Unified FactoryBench pipeline runner for Levels 1-3.

Stages (runnable subsets via ``--stages``):
    generate  -> src.question_generation.level{N}.level{N}
    fetch     -> pull QA JSONs from a HF dataset repo
                 (mutually exclusive with generate)
    prompts   -> src.question_generation.build_prompts_from_questions
    eval      -> src.evaluation.run_foundry_eval (per model)

Post-run analysis and figure generation are performed manually via
``scripts/evaluate_opik_results.ipynb``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from src.config import (
    DEFAULT_JUDGE_MODEL,
    FOUNDRY_MODEL_NAMES,
    MODEL_NAMES,
    get_provider,
)

load_dotenv()


LEVEL_CONFIGS = {
    1: {
        "module": "src.question_generation.level1.level1",
        "output_flag": "--output",
        "supports_test_mode": False,
        "supports_questions_per_template": False,
        "supports_dataset_repo": False,
    },
    2: {
        "module": "src.question_generation.level2.level2",
        "output_flag": "--output",
        "supports_test_mode": False,
        "supports_questions_per_template": False,
        "supports_dataset_repo": False,
    },
    3: {
        "module": "src.question_generation.level3.level3",
        "output_flag": "--output",
        "supports_test_mode": False,
        "supports_questions_per_template": False,
        "supports_dataset_repo": False,
    },
    4: {
        "module": "src.question_generation.level4.level4",
        "output_flag": "--output",
        "supports_test_mode": False,
        "supports_questions_per_template": False,
        "supports_dataset_repo": False,
    },
}

ALL_STAGES = ("generate", "fetch", "prompts", "eval")
DEFAULT_STAGES = ("generate", "prompts", "eval")
DEFAULT_QA_REPO = "FactoryBench/FactoryBench"


def _model_slug(model: str) -> str:
    return model.replace(".", "_").replace("/", "_")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def stage_generate(level: int, args: argparse.Namespace, q_dir: Path) -> None:
    cfg = LEVEL_CONFIGS[level]
    print("=" * 60)
    print(f"[L{level}] Generating questions -> {q_dir}")
    print("=" * 60)
    cmd = [
        sys.executable, "-m", cfg["module"],
        "-n", str(args.num_questions),
        cfg["output_flag"], str(q_dir),
    ]
    if cfg.get("supports_dataset_repo", True):
        cmd.extend(["--dataset-repo", args.dataset_repo])
    if args.questions_per_template is not None and cfg["supports_questions_per_template"]:
        cmd.extend(["--questions-per-template", str(args.questions_per_template)])
    if args.test_mode and cfg["supports_test_mode"]:
        cmd.append("--test-mode")
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    _run(cmd)


def stage_fetch(level: int, args: argparse.Namespace, q_dir: Path) -> None:
    """Download `<folder>/level_<N>/<split>.jsonl` from HF and explode into per-question JSONs.

    The downstream stages (`build_prompts_from_questions`, `run_*_eval`) discover
    questions via `rglob("*.json")` and index them by filename stem, so we
    materialise one JSON per line using the payload's `id` field as the stem.
    """
    remote_path = f"{args.hf_dataset_folder}/level_{level}/{args.split}.jsonl"
    print("=" * 60)
    print(f"[L{level}] Fetching '{args.split}' split from {args.hf_qa_repo}:{remote_path} -> {q_dir}")
    print("=" * 60)

    try:
        local_jsonl = Path(hf_hub_download(
            repo_id=args.hf_qa_repo,
            repo_type="dataset",
            filename=remote_path,
        ))
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not fetch {args.hf_qa_repo}:{remote_path} ({exc}). "
            f"Check --hf-qa-repo, --hf-dataset-folder, --split, and that level {level} exists."
        ) from exc

    q_dir.mkdir(parents=True, exist_ok=True)
    # Wipe stale JSONs so a re-fetch with a different split doesn't mix items.
    for stale in q_dir.glob("*.json"):
        stale.unlink()

    count = 0
    with local_jsonl.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            qid = payload.get("id") or f"level{level}_{idx:06d}"
            out_path = q_dir / f"{qid}.json"
            with out_path.open("w", encoding="utf-8") as out:
                json.dump(payload, out, indent=2, ensure_ascii=False)
            count += 1
    print(f"[L{level}] Fetched {count} item(s) into {q_dir}")


def stage_prompts(level: int, q_dir: Path, p_dir: Path, kg_path: str) -> None:
    print("=" * 60)
    print(f"[L{level}] Building prompts -> {p_dir}")
    print("=" * 60)
    _run([
        sys.executable, "-m", "src.question_generation.build_prompts_from_questions",
        "--input", str(q_dir),
        "--output", str(p_dir),
        "--machines", kg_path,
    ])


def stage_eval(
    level: int,
    model: str,
    args: argparse.Namespace,
    q_dir: Path,
    p_dir: Path,
    r_dir: Path,
) -> None:
    """Dispatch evaluation to the right backend based on provider.

    Foundry models -> src.evaluation.run_foundry_eval (Azure)
    Bedrock + SageMaker models -> src.evaluation.run_aws_eval (AWS)
    """
    eval_level_tag = args.eval_level or f"level_{level}"
    provider = get_provider(model)
    if provider in ("bedrock", "sagemaker"):
        eval_module = "src.evaluation.run_aws_eval"
    else:
        eval_module = "src.evaluation.run_foundry_eval"

    print("-" * 60)
    print(f"[L{level} | {model} | {provider or 'unknown'}] Running evaluation -> {r_dir}")
    print("-" * 60)
    cmd = [
        sys.executable, "-m", eval_module,
        "--input", str(p_dir),
        "--output-dir", str(r_dir),
        "--questions", str(q_dir),
        "--model", model,
    ]
    cmd.extend(["--eval-level", eval_level_tag])
    if args.overwrite:
        cmd.append("--overwrite")
    if args.judge_model:
        cmd.extend(["--judge-model", args.judge_model])
    if getattr(args, "no_judge", False):
        cmd.append("--no-judge")
    if args.max_output_tokens is not None:
        cmd.extend(["--max-output-tokens", str(args.max_output_tokens)])
    # --cost-limit applies to both providers (different semantics: Foundry
    # tracks live cost mid-run, AWS pre-flight-estimates batch and tracks
    # live in the sync fallback). --concurrency is foundry-only.
    if args.cost_limit is not None:
        cmd.extend(["--cost-limit", str(args.cost_limit)])
    if eval_module.endswith("run_foundry_eval") and args.concurrency is not None:
        cmd.extend(["--concurrency", str(args.concurrency)])
    if args.no_batch:
        cmd.append("--no-batch")
    if args.strict_batch:
        cmd.append("--strict-batch")
    if args.poll_interval is not None:
        cmd.extend(["--poll-interval", str(args.poll_interval)])
    summary_path = r_dir / "_summary.json"
    cmd.extend(["--summary-file", str(summary_path)])
    _run(cmd)
    return summary_path


def run_level(
    level: int,
    models: list[str],
    stages: set[str],
    args: argparse.Namespace,
    kg_path: Optional[str],
    totals: Dict[str, int],
) -> None:
    header = f" LEVEL {level} "
    print("#" * 60)
    print(f"#{header.center(58)}#")
    print("#" * 60)

    q_dir = Path(args.questions_dir.format(level=level))
    p_dir = Path(args.prompts_dir.format(level=level))

    if "generate" in stages:
        stage_generate(level, args, q_dir)
    if "fetch" in stages:
        stage_fetch(level, args, q_dir)
    if "prompts" in stages:
        if kg_path is None:
            print(f"  L{level}: skipping 'prompts' stage (no Knowledge Graph available).")
        else:
            stage_prompts(level, q_dir, p_dir, kg_path)

    if "eval" not in stages:
        return

    def _run_one(model: str) -> Optional[Path]:
        slug = _model_slug(model)
        r_dir = Path(args.replies_dir.format(level=level, slug=slug))
        banner = f" L{level} x {model} "
        print("*" * 60)
        print(f"*{banner.center(58)}*")
        print("*" * 60)
        return stage_eval(level, model, args, q_dir, p_dir, r_dir)

    model_conc = max(1, args.model_concurrency)
    if model_conc <= 1 or len(models) <= 1:
        summary_paths = [_run_one(m) for m in models]
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"Running {len(models)} models in parallel (model-concurrency={model_conc})")
        with ThreadPoolExecutor(max_workers=model_conc) as ex:
            future_map = {ex.submit(_run_one, m): m for m in models}
            summary_paths = []
            for fut in as_completed(future_map):
                try:
                    summary_paths.append(fut.result())
                except Exception as exc:
                    print(f"WARN: eval for {future_map[fut]} raised: {exc}")
                    summary_paths.append(None)

    for summary_path in summary_paths:
        if summary_path and summary_path.exists():
            try:
                with open(summary_path) as _f:
                    s = json.load(_f)
                totals["completed"] += int(s.get("completed", 0))
                totals["failed"] += int(s.get("failed", 0))
                totals["skipped"] += int(s.get("skipped", 0))
            except Exception as exc:
                print(f"WARN: failed to read {summary_path}: {exc}")


def _parse_csv(value: str, label: str) -> list[str]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError(f"--{label} must not be empty")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Unified FactoryBench pipeline runner. Generates Q&A, builds prompts, "
            "and evaluates Foundry models for levels 1-4. "
            "Use --stages to restrict which steps run (e.g. --stages generate,prompts "
            "to skip inference). "
            "Post-run figures are produced via scripts/evaluate_opik_results.ipynb."
        )
    )
    parser.add_argument("-n", "--num-questions", type=int, default=100,
                        help="Number of questions to generate per level (default: 100)")
    parser.add_argument("-t", "--questions-per-template", type=int, default=None,
                        help="Generate exactly X questions per template (overrides -n)")
    parser.add_argument("--dataset-repo", type=str, default="FactoryBench/FactoryNet_Dataset",
                        help="Source dataset repo id (default: FactoryBench/FactoryNet_Dataset)")
    parser.add_argument("--kg-repo", type=str, default="FactoryBench/FactoryBench",
                        help="Knowledge graph repo id (default: FactoryBench/FactoryBench)")
    parser.add_argument("--test-mode", action="store_true",
                        help="Run generation in test mode (faster)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--levels", type=str, default="1,2,3,4",
                        help="Comma-separated levels to run (default: 1,2,3,4)")
    parser.add_argument("--stages", type=str, default=",".join(DEFAULT_STAGES),
                        help=(
                            f"Comma-separated stages to run. "
                            f"Choices: {','.join(ALL_STAGES)} "
                            f"(default: {','.join(DEFAULT_STAGES)}). "
                            f"Use 'fetch' instead of 'generate' to pull QA pairs "
                            f"from a HF dataset repo."
                        ))
    parser.add_argument("--hf-qa-repo", type=str, default=DEFAULT_QA_REPO,
                        help=f"HF dataset repo id for the 'fetch' stage "
                             f"(default: {DEFAULT_QA_REPO})")
    parser.add_argument("--hf-dataset-folder", type=str, default=None,
                        help="Top-level folder inside --hf-qa-repo to fetch from "
                             "(e.g. factorybench_qa). Required when 'fetch' stage is active.")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "train", "validation"],
                        help="Which JSONL split to fetch from level_<N>/<split>.jsonl "
                             "(default: test). Only applies to the 'fetch' stage.")
    parser.add_argument("--models", type=str, default=",".join(MODEL_NAMES),
                        help=(
                            f"Comma-separated models to evaluate (Foundry, Bedrock, "
                            f"or SageMaker; provider is resolved automatically from "
                            f"src/config.py). Default: {','.join(MODEL_NAMES)}"
                        ))
    parser.add_argument("--no-judge", action="store_true",
                        help="Disable LLM-as-judge across all eval calls. Free-form items get score=None.")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
                        help=(
                            f"LLM-as-judge model for free-form scoring "
                            f"(default: {DEFAULT_JUDGE_MODEL})"
                        ))
    parser.add_argument("--eval-level", type=str, default=None,
                        help="Evaluation level tag for Opik tracing (default: level_<N> per level)")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="Max output tokens forwarded to eval")
    parser.add_argument("--cost-limit", type=float, default=None,
                        help="Max USD spend per model/level eval run")
    parser.add_argument("--questions-dir", type=str,
                        default="output/questions/level{level}",
                        help="Questions dir template with {level} placeholder "
                             "(default: output/questions/level{level})")
    parser.add_argument("--prompts-dir", type=str,
                        default="output/prompts/level{level}",
                        help="Prompts dir template with {level} placeholder "
                             "(default: output/prompts/level{level})")
    parser.add_argument("--replies-dir", type=str,
                        default="output/replies/level{level}/{slug}",
                        help="Replies dir template with {level} and {slug} placeholders "
                             "(default: output/replies/level{level}/{slug})")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing per-prompt replies (default: skip already-computed)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Number of concurrent API calls per model (default: 8)")
    parser.add_argument("--model-concurrency", type=int, default=4,
                        help="Number of models to run in parallel per level (default: 4)")
    parser.add_argument("--no-batch", action="store_true",
                        help="Disable provider batch APIs; force concurrent sync for all models")
    parser.add_argument("--strict-batch", action="store_true",
                        help="If batch submission fails, error out instead of falling back to "
                             "concurrent sync. Protects against silent cost doubling when batch "
                             "is misconfigured (e.g. missing GPT_5_1_BATCH_DEPLOYMENT). Forwarded "
                             "to both run_foundry_eval and run_aws_eval.")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Batch polling interval in seconds (default: 30)")

    args = parser.parse_args()

    try:
        levels_to_run = [int(x) for x in _parse_csv(args.levels, "levels")]
    except ValueError:
        parser.error(f"--levels must be a comma-separated list of integers, got: {args.levels!r}")

    invalid_levels = [lv for lv in levels_to_run if lv not in LEVEL_CONFIGS]
    if invalid_levels:
        parser.error(
            f"Unsupported level(s): {invalid_levels}. "
            f"Supported: {sorted(LEVEL_CONFIGS.keys())}"
        )

    stages = set(_parse_csv(args.stages, "stages"))
    invalid_stages = stages - set(ALL_STAGES)
    if invalid_stages:
        parser.error(
            f"Unsupported stage(s): {sorted(invalid_stages)}. Supported: {list(ALL_STAGES)}"
        )
    if {"generate", "fetch"} <= stages:
        parser.error("--stages cannot contain both 'generate' and 'fetch' (they are alternatives).")
    if "fetch" in stages and not args.hf_dataset_folder:
        parser.error("--hf-dataset-folder is required when 'fetch' stage is active.")

    models_to_run = _parse_csv(args.models, "models")
    eval_active = "eval" in stages
    unknown_models = [m for m in models_to_run if m not in MODEL_NAMES]
    if eval_active and unknown_models:
        parser.error(
            f"Unknown model(s): {unknown_models}. Supported: {MODEL_NAMES}"
        )

    kg_path: Optional[str] = None
    if "prompts" in stages:
        print("=" * 60)
        print("0. Downloading Knowledge Graph from Hugging Face...")
        print("=" * 60)
        try:
            kg_path = hf_hub_download(
                repo_id=args.kg_repo, repo_type="dataset", filename="knowledge_graph/knowledge_graph.json"
            )
            print(f"Knowledge Graph downloaded to: {kg_path}\n")
        except Exception as exc:
            print(f"WARNING: Knowledge Graph not available on HF ({exc}); 'prompts' stage will be skipped.\n")

    ordered_stages = sorted(stages, key=ALL_STAGES.index)
    print(
        f"Plan: levels={levels_to_run} | stages={ordered_stages} "
        f"| models={models_to_run if eval_active else '(n/a)'}\n"
    )

    totals: Dict[str, int] = {"completed": 0, "failed": 0, "skipped": 0}
    for level in levels_to_run:
        run_level(level, models_to_run, stages, args, kg_path, totals)

    print("=" * 60)
    print("Pipeline Completed Successfully!")
    print(f"  Levels : {levels_to_run}")
    print(f"  Stages : {ordered_stages}")
    if eval_active:
        print(f"  Models : {models_to_run}")
        print(
            f"  Aggregate totals across all models and levels: "
            f"Completed={totals['completed']}, "
            f"Failed={totals['failed']}, "
            f"Skipped={totals['skipped']}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()

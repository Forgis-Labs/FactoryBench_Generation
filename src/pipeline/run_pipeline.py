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
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download

from src.config import DEFAULT_JUDGE_MODEL, FOUNDRY_MODEL_NAMES

load_dotenv()


LEVEL_CONFIGS = {
    1: {
        "module": "src.question_generation.level1.level1",
        "output_flag": "--output-dir",
        "supports_test_mode": True,
        "supports_questions_per_template": True,
    },
    2: {
        "module": "src.question_generation.level2.level2",
        "output_flag": "--output",
        "supports_test_mode": True,
        "supports_questions_per_template": True,
    },
    3: {
        "module": "src.question_generation.level3.level3",
        "output_flag": "--output",
        "supports_test_mode": True,
        "supports_questions_per_template": True,
    },
}

ALL_STAGES = ("generate", "fetch", "prompts", "eval")
DEFAULT_STAGES = ("generate", "prompts", "eval")
DEFAULT_QA_REPO = "Forgis/FactoryBench_QA_pairs"


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
        "--dataset-repo", args.dataset_repo,
        cfg["output_flag"], str(q_dir),
    ]
    if args.questions_per_template is not None and cfg["supports_questions_per_template"]:
        cmd.extend(["--questions-per-template", str(args.questions_per_template)])
    if args.test_mode and cfg["supports_test_mode"]:
        cmd.append("--test-mode")
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    _run(cmd)


def stage_fetch(level: int, args: argparse.Namespace, q_dir: Path) -> None:
    remote_dir = f"{args.hf_dataset_folder}/level_{level}"
    print("=" * 60)
    print(f"[L{level}] Fetching QA pairs from {args.hf_qa_repo}:{remote_dir}/ -> {q_dir}")
    print("=" * 60)

    snapshot_root = Path(snapshot_download(
        repo_id=args.hf_qa_repo,
        repo_type="dataset",
        allow_patterns=f"{remote_dir}/*.json",
    ))
    src_dir = snapshot_root / remote_dir
    if not src_dir.exists() or not any(src_dir.iterdir()):
        raise FileNotFoundError(
            f"No QA JSONs found at {args.hf_qa_repo}:{remote_dir}/. "
            f"Check --hf-dataset-folder and the level exists on the HF repo."
        )

    q_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for json_file in sorted(src_dir.glob("*.json")):
        shutil.copy2(json_file, q_dir / json_file.name)
        copied += 1
    print(f"[L{level}] Fetched {copied} QA pair(s) into {q_dir}")


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
    eval_level_tag = args.eval_level or f"level_{level}"
    print("-" * 60)
    print(f"[L{level} | {model}] Running evaluation -> {r_dir}")
    print("-" * 60)
    cmd = [
        sys.executable, "-m", "src.evaluation.run_foundry_eval",
        "--input", str(p_dir),
        "--output-dir", str(r_dir),
        "--questions", str(q_dir),
        "--model", model,
        "--eval-level", eval_level_tag,
        "--overwrite",
    ]
    if args.judge_model:
        cmd.extend(["--judge-model", args.judge_model])
    if args.max_output_tokens is not None:
        cmd.extend(["--max-output-tokens", str(args.max_output_tokens)])
    if args.cost_limit is not None:
        cmd.extend(["--cost-limit", str(args.cost_limit)])
    _run(cmd)


def run_level(
    level: int,
    models: list[str],
    stages: set[str],
    args: argparse.Namespace,
    kg_path: Optional[str],
) -> None:
    header = f" LEVEL {level} "
    print("#" * 60)
    print(f"#{header.center(58)}#")
    print("#" * 60)

    base_name = f"level{level}_pipeline"
    q_dir = Path(f"datasets/questions/{base_name}")
    p_dir = Path(f"datasets/prompts/{base_name}")

    if "generate" in stages:
        stage_generate(level, args, q_dir)
    if "fetch" in stages:
        stage_fetch(level, args, q_dir)
    if "prompts" in stages:
        assert kg_path is not None, "prompts stage requires KG download"
        stage_prompts(level, q_dir, p_dir, kg_path)

    if "eval" not in stages:
        return

    for model in models:
        slug = _model_slug(model)
        r_dir = Path(f"datasets/replies/{base_name}/{slug}")

        banner = f" L{level} x {model} "
        print("*" * 60)
        print(f"*{banner.center(58)}*")
        print("*" * 60)

        stage_eval(level, model, args, q_dir, p_dir, r_dir)


def _parse_csv(value: str, label: str) -> list[str]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError(f"--{label} must not be empty")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Unified FactoryBench pipeline runner. Generates Q&A, builds prompts, "
            "and evaluates Foundry models for levels 1-3. "
            "Use --stages to restrict which steps run (e.g. --stages generate,prompts "
            "to skip inference). Level 4 is not yet implemented. "
            "Post-run figures are produced via scripts/evaluate_opik_results.ipynb."
        )
    )
    parser.add_argument("-n", "--num-questions", type=int, default=100,
                        help="Number of questions to generate per level (default: 100)")
    parser.add_argument("-t", "--questions-per-template", type=int, default=None,
                        help="Generate exactly X questions per template (overrides -n)")
    parser.add_argument("--dataset-repo", type=str, default="Forgis/FactoryNet_Dataset",
                        help="Source dataset repo id (default: Forgis/FactoryNet_Dataset)")
    parser.add_argument("--kg-repo", type=str, default="Forgis/FactoryBench-KnowledgeGraph",
                        help="Knowledge graph repo id (default: Forgis/FactoryBench-KnowledgeGraph)")
    parser.add_argument("--test-mode", action="store_true",
                        help="Run generation in test mode (faster)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--levels", type=str, default="1,2,3",
                        help="Comma-separated levels to run (default: 1,2,3)")
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
                             "(e.g. factorynet_qa_260). Required when 'fetch' stage is active.")
    parser.add_argument("--models", type=str, default=",".join(FOUNDRY_MODEL_NAMES),
                        help=(
                            f"Comma-separated Foundry models to evaluate. "
                            f"Default: {','.join(FOUNDRY_MODEL_NAMES)}"
                        ))
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
    unknown_models = [m for m in models_to_run if m not in FOUNDRY_MODEL_NAMES]
    if eval_active and unknown_models:
        parser.error(
            f"Unknown model(s): {unknown_models}. Supported: {FOUNDRY_MODEL_NAMES}"
        )

    kg_path: Optional[str] = None
    if "prompts" in stages:
        print("=" * 60)
        print("0. Downloading Knowledge Graph from Hugging Face...")
        print("=" * 60)
        kg_path = hf_hub_download(
            repo_id=args.kg_repo, repo_type="dataset", filename="machines.json"
        )
        print(f"Knowledge Graph downloaded to: {kg_path}\n")

    ordered_stages = sorted(stages, key=ALL_STAGES.index)
    print(
        f"Plan: levels={levels_to_run} | stages={ordered_stages} "
        f"| models={models_to_run if eval_active else '(n/a)'}\n"
    )

    for level in levels_to_run:
        run_level(level, models_to_run, stages, args, kg_path)

    print("=" * 60)
    print("Pipeline Completed Successfully!")
    print(f"  Levels : {levels_to_run}")
    print(f"  Stages : {ordered_stages}")
    if eval_active:
        print(f"  Models : {models_to_run}")
    print("=" * 60)


if __name__ == "__main__":
    main()

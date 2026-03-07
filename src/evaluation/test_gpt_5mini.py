#!/usr/bin/env python3
"""
Generate answers for prompt files using direct OpenAI Responses API calls.

This script:
1. Loads pre-generated prompt JSON files from a folder
2. Calls the model per prompt (Responses endpoint)
3. Saves one output JSON per prompt with the generated answer
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from openai import AzureOpenAI, OpenAI
except ImportError:
    AzureOpenAI = None
    OpenAI = None

logger = logging.getLogger(__name__)


def load_dotenv_file(env_file: Path) -> None:
    if not env_file.exists() or not env_file.is_file():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_api_key(cli_api_key: Optional[str]) -> str:
    if cli_api_key:
        return cli_api_key

    key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "Missing API key. Provide --api-key or set OPENAI_API_KEY/AZURE_OPENAI_API_KEY in .env"
        )
    return key


def create_client_and_model(
    cli_api_key: Optional[str],
    cli_model: Optional[str],
) -> tuple[Any, str, str]:
    """
    Returns (client, model, provider), where provider is 'openai' or 'azure'.
    """

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21"
    # Use AZURE_OPENAI_CHAT_DEPLOYMENT as the primary model/deployment variable for Azure
    azure_model = (
        os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_MODEL")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    )

    if azure_endpoint and (azure_api_key or cli_api_key):
        key = cli_api_key or azure_api_key
        client = AzureOpenAI(
            api_key=key,
            api_version=azure_api_version,
            azure_endpoint=azure_endpoint,
        )
        model = cli_model or azure_model or "gpt-5.1"
        return client, model, "azure"

    key = resolve_api_key(cli_api_key)
    client = OpenAI(api_key=key)
    model = cli_model or os.getenv("OPENAI_MODEL") or "gpt-5.1"
    return client, model, "openai"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_ground_truth_index(questions_dir: Optional[Path]) -> Dict[str, Any]:
    """Build a mapping of filename stem -> answer from a questions directory."""
    if questions_dir is None or not questions_dir.is_dir():
        return {}
    index: Dict[str, Any] = {}
    for path in questions_dir.rglob("*.json"):
        try:
            payload = load_json(path)
        except Exception:
            continue
        if isinstance(payload, dict) and "answer" in payload:
            index[path.stem] = payload["answer"]
    logger.info(f"Loaded {len(index)} ground truth entries from {questions_dir}")
    return index


def load_prompt_entries(
    input_dir: Path,
    total_prompts: Optional[int] = None,
    batch_number: int = 0,
    batch_size: int = 1000,
) -> list[Tuple[Path, str, int, str]]:
    """
    Load pre-generated prompts and return selected slice entries.

    total_prompts:
      Total number of prompts to process across all batches. If None, processes all prompts.
    batch_number:
      Which batch this is (0-indexed), where each batch is batch_size prompts.

    Returns:
      List of tuples (prompt_path, prompt_text, global_index, custom_id).
    """
    prompt_files = sorted(input_dir.rglob("*.json"))
    if not prompt_files:
        logger.warning(f"No prompt JSON files found in {input_dir}")
        return []

    prompt_entries: list[tuple[Path, str]] = []
    for prompt_path in prompt_files:
        try:
            payload = load_json(prompt_path)
        except Exception:
            logger.warning(f"Skipping unreadable JSON: {prompt_path}")
            continue

        if not isinstance(payload, dict):
            logger.warning(f"Skipping non-object JSON: {prompt_path}")
            continue

        prompt_text = payload.get("prompt")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            logger.warning(f"Skipping JSON without 'prompt' text: {prompt_path}")
            continue

        prompt_entries.append((prompt_path, prompt_text.strip()))

    if not prompt_entries:
        logger.warning(f"No valid prompt entries found in {input_dir}")
        return []

    batch_start = batch_number * batch_size
    batch_end = (batch_number + 1) * batch_size

    if total_prompts is not None:
        prompt_entries = prompt_entries[:total_prompts]

    logger.info(f"Loaded {len(prompt_entries)} prompt entries from {input_dir}")

    selected_entries = prompt_entries[batch_start:batch_end]

    selected: list[Tuple[Path, str, int, str]] = []
    for prompt_idx, (prompt_path, prompt_text) in enumerate(selected_entries, start=batch_start):
        custom_id = f"{prompt_path.stem}_{prompt_idx}"
        selected.append((prompt_path, prompt_text, prompt_idx, custom_id))

    logger.info(f"Selected {len(selected)} prompts for processing")
    return selected


def _to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return {}


def _extract_output_text_from_responses_body(body: Dict[str, Any]) -> str:
    """
    Responses API format: body["output"] is a list of items.
    We concatenate any output_text segments found inside message items.
    """
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: list[str] = []
    for item in body.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    return "".join(chunks)


def run_direct_requests(
    entries: list[Tuple[Path, str, int, str]],
    client: Any,
    output_dir: Path,
    model: str,
    max_output_tokens: int,
    overwrite: bool,
    ground_truth_index: Dict[str, Any],
) -> tuple[int, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    failed = 0
    skipped = 0

    for prompt_path, prompt_text, prompt_idx, custom_id in entries:
        out_path = output_dir / f"{custom_id}_answer.json"
        fail_path = output_dir / f"{custom_id}_failed.json"

        if not overwrite and (out_path.exists() or fail_path.exists()):
            skipped += 1
            logger.info(f"- Skipping existing result: {custom_id}")
            continue


        # Load full Q&A object for scoring (not just answer)
        try:
            qa_payload = load_json(prompt_path)
        except Exception:
            qa_payload = {}

        answer_format = qa_payload.get("answer_format", {})
        q_type = answer_format.get("type") or qa_payload.get("template_type")
        acceptance_bounds = qa_payload.get("acceptance_bounds")

        try:
            response = client.responses.create(
                model=model,
                input=prompt_text,
                max_output_tokens=max_output_tokens,
            )
            body = _to_dict(response)
            answer = _extract_output_text_from_responses_body(body)

            # --- Scoring logic ---
            score = None
            gt = ground_truth
            pred = answer
            try:
                if q_type in ("numerical", "tensor"):
                    # Numerical: float, Tensor: underscore-separated floats
                    if q_type == "numerical":
                        try:
                            gt_val = float(gt)
                            pred_val = float(pred)
                        except Exception:
                            score = 0
                        else:
                            if acceptance_bounds:
                                margin = acceptance_bounds.get("margin", 0)
                                mean = acceptance_bounds.get("mean", 0)
                                score = int(abs(pred_val - gt_val) <= margin)
                            else:
                                score = int(abs(pred_val - gt_val) < 1e-4)
                    elif q_type == "tensor":
                        try:
                            gt_vals = [float(x) for x in str(gt).split("_")]
                            pred_vals = [float(x) for x in str(pred).split("_")]
                        except Exception:
                            score = 0
                        else:
                            if acceptance_bounds and "margin" in acceptance_bounds:
                                margins = acceptance_bounds["margin"]
                                if len(gt_vals) == len(pred_vals) == len(margins):
                                    n = len(gt_vals)
                                    n_correct = sum(abs(p - g) <= m for p, g, m in zip(pred_vals, gt_vals, margins))
                                    score = n_correct / n
                                else:
                                    score = 0.0
                            else:
                                score = float(gt_vals == pred_vals)
                elif q_type == "multiple_choice_multi_select":
                    # Multi-select MCQ: string of T/F, e.g., TFFT
                    gt_str = str(gt).strip().upper()
                    pred_str = str(pred).strip().upper()
                    if len(gt_str) == len(pred_str) and set(gt_str) <= {"T", "F"} and set(pred_str) <= {"T", "F"}:
                        n = len(gt_str)
                        n_correct = sum(g == p for g, p in zip(gt_str, pred_str))
                        if n_correct == n:
                            score = 1.0
                        elif n_correct >= n - 1:
                            score = 0.5
                        else:
                            score = 0.0
                    else:
                        score = 0.0
                elif q_type == "ranking":
                    # Ranking: permutation of A-D, e.g., DCAB
                    gt_str = str(gt).strip().upper()
                    pred_str = str(pred).strip().upper()
                    score = float(gt_str == pred_str)
                else:
                    # Fallback: exact match
                    score = float(str(gt) == str(pred))
            except Exception:
                score = None

            save_json(out_path, {
                "custom_id": custom_id,
                "prompt_index": prompt_idx,
                "prompt_file": str(prompt_path),
                "prompt": prompt_text,
                "answer": answer,
                "ground_truth": gt,
                "score": score,
                "model": body.get("model"),
                "usage": body.get("usage"),
            })
            completed += 1
            logger.info(f"✓ [{completed + failed + skipped}/{len(entries)}] Saved answer: {custom_id}")
        except Exception as exc:
            failed += 1
            save_json(fail_path, {
                "custom_id": custom_id,
                "prompt_index": prompt_idx,
                "prompt_file": str(prompt_path),
                "error": str(exc),
            })
            logger.warning(f"✗ [{completed + failed + skipped}/{len(entries)}] Failed: {custom_id} ({exc})")

    return completed, failed, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate answers via direct OpenAI Responses API calls")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/prompts/level3"),
        help="Directory containing pre-generated prompt JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/replies/level3"),
        help="Directory to save answer outputs",
    )
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key (overrides .env)")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file (default: ./.env)",
    )
    parser.add_argument("--model", type=str, default=None, help="Model name (default from env or gpt-5.1)")
    parser.add_argument("--max-output-tokens", type=int, default=2000, help="Max output tokens")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing result files")
    parser.add_argument(
        "--total-prompts",
        type=int,
        default=None,
        help="Total number of prompts to process across batches",
    )
    parser.add_argument(
        "--batch-number",
        type=int,
        default=0,
        help="Which batch slice to process (0-indexed)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="How many prompts per batch slice",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="Directory containing Q&A pair JSON files (for ground truth lookup)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    load_dotenv_file(args.env_file)
    client, model, provider = create_client_and_model(args.api_key, args.model)
    logger.info(f"Using provider={provider}, model={model}")

    if args.questions is None:
        logger.error("--questions not provided; all prompts will be skipped. Pass the Q&A directory to enable ground truth lookup.")
        return

    ground_truth_index = build_ground_truth_index(args.questions)

    entries = load_prompt_entries(
        input_dir=args.input,
        total_prompts=args.total_prompts,
        batch_number=args.batch_number,
        batch_size=args.batch_size,
    )

    if not entries:
        logger.error("No requests to process")
        return

    if args.total_prompts is not None:
        total_batches = (args.total_prompts + args.batch_size - 1) // args.batch_size
        logger.info(f"Processing batch {args.batch_number}/{total_batches - 1}")
        logger.info(f"This slice contains {len(entries)} prompts")

    completed, failed, skipped = run_direct_requests(
        entries=entries,
        client=client,
        output_dir=args.output_dir,
        model=model,
        max_output_tokens=args.max_output_tokens,
        overwrite=args.overwrite,
        ground_truth_index=ground_truth_index,
    )
    logger.info(
        f"Done. Completed={completed}, Failed={failed}, Skipped={skipped}, OutputDir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
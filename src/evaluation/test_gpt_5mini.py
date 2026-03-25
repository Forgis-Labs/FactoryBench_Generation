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


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a robotics sensor-data Q&A benchmark.
Your task is to score a model's free-form answer against a reference answer.

Scoring criteria (0-10):
  10 – Answer is semantically equivalent to the reference: correct direction of change, correct signal, correct magnitude range, correct time window.
   7 – Answer captures the main trend and signal correctly but is imprecise on magnitude or timing.
   4 – Answer mentions the right signal but the described behaviour is partially incorrect or vague.
   1 – Answer is on-topic but mostly incorrect or misleading.
   0 – Answer is completely wrong, irrelevant, or refuses to answer.

Respond ONLY with a JSON object in this exact format (no extra text):
{"score": <integer 0-10>, "reason": "<one sentence justification>"}"""


def llm_judge_score(
    client: Any,
    model: str,
    question: str,
    prediction: str,
    reference: str,
    max_tokens: int = 256,
) -> tuple[float, str]:
    """
    Ask the LLM to score `prediction` against `reference` for the given `question`.
    Returns (normalised_score 0.0–1.0, justification_string).
    Falls back to (0.0, error_message) on any failure.
    """
    user_msg = (
        f"Question: {question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"Model answer: {prediction}\n\n"
        "Please score the model answer (0-10) and provide a one-sentence justification."
    )
    try:
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                       {"role": "user",   "content": user_msg}],
                max_output_tokens=max_tokens,
            )
            raw = _extract_output_text_from_responses_body(_to_dict(response))
        except Exception:
            # Fallback to Chat Completions
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens=max_tokens,
            )
            raw = _to_dict(response).get("choices", [{}])[0].get("message", {}).get("content", "")

        # Parse the JSON response from the judge
        raw = raw.strip()
        # Sometimes the model wraps in markdown code fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        raw_score = float(parsed["score"])
        reason = str(parsed.get("reason", ""))
        normalised = max(0.0, min(1.0, raw_score / 10.0))
        return normalised, reason
    except Exception as e:
        return 0.0, f"Judge call failed: {e}"


def run_direct_requests(
    entries: list[Tuple[Path, str, int, str]],
    client: Any,
    output_dir: Path,
    model: str,
    max_output_tokens: int,
    overwrite: bool,
    ground_truth_index: Dict[str, Any],
    eval_level: str,
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


        # Token budget guard – skip prompts that are too large for the model
        estimated_tokens = len(prompt_text) // 4
        if estimated_tokens > 900_000:
            logger.warning(
                f"- Skipping {custom_id}: estimated {estimated_tokens} tokens exceeds budget"
            )
            skipped += 1
            continue

        # Load full Q&A object for scoring (not just answer)
        try:
            qa_payload = load_json(prompt_path)
        except Exception:
            qa_payload = {}

        answer_format: Dict[str, Any] = qa_payload.get("answer_format") or {}
        q_type = str(answer_format.get("type") or qa_payload.get("template_type") or qa_payload.get("type") or "unknown")
        acceptance_bounds = qa_payload.get("acceptance_bounds")

        # Derive eval level from the Q&A JSON if not provided via CLI
        level_val = qa_payload.get("level")
        effective_eval_level = eval_level or (f"level_{level_val}" if level_val is not None else None)

        try:
            try:
                response = client.responses.create(
                    model=model,
                    input=prompt_text,
                    max_output_tokens=max_output_tokens,
                )
                body = _to_dict(response)
                answer = _extract_output_text_from_responses_body(body)
            except Exception as e:
                if "404" in str(e) or "not found" in str(e).lower() or not hasattr(client, "responses"):
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt_text}],
                        max_tokens=max_output_tokens,
                    )
                    body = _to_dict(response)
                    answer = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    raise e

            # --- Scoring logic ---
            score = None
            llm_judge_result: tuple[float, str] | None = None
            gt = ground_truth_index.get(prompt_path.stem)
            pred = answer
            evaluation_method = str(answer_format.get("type") or "rule_based")
            try:
                if evaluation_method == "free_form":
                    # LLM-as-a-judge: call the model to score the free-form answer
                    question_text = qa_payload.get("question", "")
                    ref_answer = str(gt) if gt is not None else ""
                    judge_score, judge_reason = llm_judge_score(
                        client=client,
                        model=model,
                        question=question_text,
                        prediction=str(pred) if pred is not None else "",
                        reference=ref_answer,
                    )
                    score = judge_score
                    llm_judge_result = (judge_score, judge_reason)
                elif evaluation_method in ("numerical", "tensor"):
                    # Numerical: float, Tensor: underscore-separated floats
                    if evaluation_method == "numerical":
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
                    elif evaluation_method == "tensor":
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
                elif evaluation_method == "multiple_choice_multi_select":
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
                elif evaluation_method == "ranking":
                    # Ranking: permutation of A-D, e.g., DCAB
                    gt_str = str(gt).strip().upper()
                    pred_str = str(pred).strip().upper()
                    score = float(gt_str == pred_str)
                else:
                    if evaluation_method == "llm_judge":
                        # Already handled above; this branch won't be reached
                        pass
                    else:
                        # Fallback: exact match
                        score = float(str(gt) == str(pred))
            except Exception:
                score = None

            # --- Opik Tracing ---
            try:
                if os.getenv("OPIK_API_KEY"):
                    import opik
                    client_opik = opik.Opik(
                        project_name=os.getenv("OPIK_PROJECT_NAME", "FactoryBench"),
                        workspace=os.getenv("OPIK_WORKSPACE", "forgis")
                    )
                    
                    usage_raw = body.get("usage", {}) or {}
                    # Normalize across Chat Completions (prompt_tokens) and Responses API (input_tokens)
                    prompt_tokens: int = int(
                        usage_raw.get("prompt_tokens")
                        or usage_raw.get("input_tokens")
                        or 0
                    )
                    completion_tokens: int = int(
                        usage_raw.get("completion_tokens")
                        or usage_raw.get("output_tokens")
                        or 0
                    )
                    total_tokens: int = int(usage_raw.get("total_tokens") or (prompt_tokens + completion_tokens))
                    
                    # Estimate cost based on model name
                    model_name = (body.get("model") or model).lower()
                    if "mini" in model_name:
                        price_in, price_out = 0.15, 0.60
                    else:
                        # Default to GPT-4 pricing from the interactive notebook
                        price_in, price_out = 5.0, 15.0
                    
                    est_cost = (prompt_tokens / 1_000_000 * price_in) + (completion_tokens / 1_000_000 * price_out)

                    # print qa_payload for debugging to see what tags it contains
                    print(f"-------> QA Payload for {custom_id}: {qa_payload}")

                    category_tag = evaluation_method
                    metadata_payload = qa_payload.get("metadata") or {}
                    dataset_tag = metadata_payload.get("dataset")
                    qa_pair_id = metadata_payload.get("qa_pair_id")
                    
                    # Filter out None/empty values so Opik receives only real tags
                    opik_tags = [str(t) for t in [effective_eval_level, category_tag, dataset_tag] if t]

                    type_tag = metadata_payload.get("type")
                    model_tag = model_name

                    if type_tag and type_tag != "unknown":
                        opik_tags.append(str(type_tag))

                    if evaluation_method and evaluation_method != "unknown":
                        opik_tags.append(f"eval_{evaluation_method}")
                    
                    if model_tag and model_tag != "unknown":
                        opik_tags.append(model_tag)

                    if dataset_tag and dataset_tag != "unknown":
                        opik_tags.append(dataset_tag)

                    opik_metadata = {
                        "model": model_tag,
                        "evaluation_method": evaluation_method,
                        "q_type": evaluation_method,
                        "qa_pair_id": qa_pair_id,
                        "dataset": dataset_tag,
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                            "total_estimated_cost": f"${est_cost}"
                        },
                    }
                    if isinstance(metadata_payload, dict):
                        opik_metadata.update(metadata_payload)



                    trace = client_opik.trace(
                        name="factorybench_eval",
                        input={"custom_id": custom_id, "prompt": prompt_text, "ground_truth": gt, "q_type": q_type},
                        output={"answer": answer, "raw_body": body},
                        tags=opik_tags,
                        usage={
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                        },
                        metadata=opik_metadata,
                        total_estimated_cost=est_cost,
                        model=body.get("model") or model_name
                    )
                    if score is not None:
                        pred_str_repr = str(pred).strip() if pred is not None else "N/A"
                        gt_str_repr   = str(gt).strip()   if gt   is not None else "N/A"
                        # For llm_judge questions, use the judge's reason as the primary accuracy reason
                        accuracy_reason = judge_reason if (llm_judge_result and judge_reason) else (
                            f"Predicted: '{pred_str_repr}' | "
                            f"Ground truth: '{gt_str_repr}'"
                        )
                        trace.log_feedback_score(
                            name="accuracy",
                            value=float(score),
                            reason=accuracy_reason,
                        )

                        # --- Exact match ---
                        exact_match = float(pred_str_repr == gt_str_repr)
                        trace.log_feedback_score(
                            name="exact_match",
                            value=exact_match,
                            reason=accuracy_reason,
                        )

                        # --- F1 Score (type-aware) ---
                        try:
                            f1: float | None = None
                            if q_type in ("numerical", "ranking"):
                                # Binary for these types → F1 == accuracy
                                f1 = float(score)
                            elif q_type == "tensor":
                                # Element-wise F1: TP / (TP + 0.5*(FP+FN)) — same as accuracy here
                                # since each element is a numeric match, reuse score
                                f1 = float(score)
                            elif q_type == "multiple_choice_multi_select":
                                # Per-position binary F1 over T/F labels
                                gt_labels  = [1 if c == "T" else 0 for c in gt_str_repr.upper()  if c in ("T","F")]
                                pred_labels = [1 if c == "T" else 0 for c in pred_str_repr.upper() if c in ("T","F")]
                                if gt_labels and len(gt_labels) == len(pred_labels):
                                    tp = sum(g == 1 and p == 1 for g, p in zip(gt_labels, pred_labels))
                                    fp = sum(g == 0 and p == 1 for g, p in zip(gt_labels, pred_labels))
                                    fn = sum(g == 1 and p == 0 for g, p in zip(gt_labels, pred_labels))
                                    denom = 2 * tp + fp + fn
                                    f1 = (2 * tp / denom) if denom > 0 else 0.0
                                else:
                                    f1 = 0.0
                            else:
                                # Token-overlap F1 (for free-text / single-choice questions)
                                gt_tokens   = set(gt_str_repr.lower().split())
                                pred_tokens = set(pred_str_repr.lower().split())
                                if gt_tokens or pred_tokens:
                                    tp = len(gt_tokens & pred_tokens)
                                    fp = len(pred_tokens - gt_tokens)
                                    fn = len(gt_tokens - pred_tokens)
                                    denom = 2 * tp + fp + fn
                                    f1 = (2 * tp / denom) if denom > 0 else 0.0
                                else:
                                    f1 = 1.0  # both empty → perfect match

                            if f1 is not None:
                                f1_reason = (
                                    f"Token-overlap F1 for type '{q_type}' | "
                                    f"Predicted: '{pred_str_repr}' | "
                                    f"Ground truth: '{gt_str_repr}'"
                                )
                                trace.log_feedback_score(
                                    name="f1_score",
                                    value=f1,
                                    reason=f1_reason,
                                )
                        except Exception as f1_err:
                            logger.debug(f"F1 score computation failed: {f1_err}")

                        # --- LLM-as-a-judge score ---
                        if llm_judge_result is not None:
                            judge_value, judge_reason = llm_judge_result
                            trace.log_feedback_score(
                                name="llm_judge",
                                value=judge_value,
                                reason=judge_reason or accuracy_reason,
                            )

            except Exception as e:
                logger.warning(f"Opik logging failed: {e}")


            save_json(out_path, {
                "custom_id": custom_id,
                "prompt_index": prompt_idx,
                "prompt_file": str(prompt_path),
                "prompt": prompt_text,
                "answer": answer,
                "ground_truth": gt,
                "score": score,
                "llm_judge_score": llm_judge_result[0] if llm_judge_result else None,
                "llm_judge_reason": llm_judge_result[1] if llm_judge_result else None,
                "evaluation_method": evaluation_method,
                "model": body.get("model"),
                "usage": body.get("usage"),
                "estimated_cost": est_cost if 'est_cost' in locals() else None,
                "raw_api_response": body,
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
    parser.add_argument(
        "--eval-level",
        type=str,
        default=None,
        help="Evaluation level tag for Opik (e.g. level_1)",
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
        eval_level=args.eval_level,
    )
    logger.info(
        f"Done. Completed={completed}, Failed={failed}, Skipped={skipped}, OutputDir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
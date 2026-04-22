#!/usr/bin/env python3
"""Multi-endpoint evaluation for Microsoft Foundry models.

Reuses scoring, prompt loading, and ground-truth helpers from
``src.evaluation.test_gpt_5mini`` so evaluation output shape is identical.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from src.config import DEFAULT_JUDGE_MODEL, FOUNDRY_MODELS
from src.evaluation.test_gpt_5mini import (
    JUDGE_SYSTEM_PROMPT,
    _estimate_cost,
    _extract_output_text_from_responses_body,
    _to_dict,
    build_ground_truth_index,
    load_dotenv_file,
    load_json,
    load_prompt_entries,
    parse_llm_answer,
    save_json,
)

logger = logging.getLogger(__name__)


def resolve_api_key() -> str:
    key = (
        os.getenv("AZURE_API_KEY")
        or os.getenv("AZURE_OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not key:
        raise ValueError(
            "Missing API key. Set AZURE_API_KEY (preferred) or AZURE_OPENAI_API_KEY in .env"
        )
    return key


def resolve_endpoint(model: str) -> Tuple[str, str]:
    """Return (endpoint_url, api_style) for the given model."""
    cfg = FOUNDRY_MODELS.get(model)
    if cfg is None:
        raise ValueError(
            f"Unknown Foundry model: {model}. Supported: {list(FOUNDRY_MODELS.keys())}"
        )
    endpoint = (os.getenv(cfg["endpoint_env"]) or cfg["endpoint_default"]).rstrip("/").rstrip('"')
    return endpoint, cfg["api_style"]


def _openai_client(base_url: str, api_key: str) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Install with: pip install openai")
    return OpenAI(api_key=api_key, base_url=base_url)


def call_openai_style(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    api_style: str = "openai",
) -> Tuple[str, Dict[str, Any]]:
    """Call an OpenAI-compatible endpoint (gpt-5-mini, DeepSeek, Mistral)."""
    client = _openai_client(base_url, api_key)

    # Mistral on Azure doesn't support /responses and uses `max_tokens` on chat.completions.
    if api_style != "mistral":
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=max_tokens,
            )
            body = _to_dict(response)
            answer = _extract_output_text_from_responses_body(body)
            if answer:
                return answer, body
        except Exception as exc:
            logger.debug(f"responses.create failed for {model}: {exc}; falling back to chat.completions")

    completion_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if api_style == "mistral":
        completion_kwargs["max_tokens"] = max_tokens
    else:
        completion_kwargs["max_completion_tokens"] = max_tokens

    response = client.chat.completions.create(**completion_kwargs)
    body = _to_dict(response)
    answer = body.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    return answer, body


def call_anthropic_style(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    """Call Anthropic-native /messages endpoint (claude-haiku-4-5)."""
    r = requests.post(
        f"{base_url}/messages",
        headers={
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=300,
    )
    r.raise_for_status()
    body = r.json()

    answer_parts: list[str] = []
    for block in body.get("content", []) or []:
        if isinstance(block, dict) and block.get("type") == "text":
            answer_parts.append(block.get("text", ""))
    answer = "".join(answer_parts)

    usage = body.get("usage", {}) or {}
    body["usage"] = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }
    body.setdefault("model", model)
    return answer, body


def call_model(model: str, prompt: str, max_tokens: int) -> Tuple[str, Dict[str, Any]]:
    api_key = resolve_api_key()
    endpoint, api_style = resolve_endpoint(model)

    if api_style == "anthropic":
        return call_anthropic_style(endpoint, api_key, model, prompt, max_tokens)
    return call_openai_style(endpoint, api_key, model, prompt, max_tokens, api_style=api_style)


def foundry_llm_judge(
    question: str,
    prediction: str,
    reference: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    max_tokens: int = 256,
) -> Tuple[float, str]:
    """Score a free-form prediction with gpt-5-mini as judge (cheap + consistent)."""
    user_msg = (
        f"Question: {question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"Model answer: {prediction}\n\n"
        "Please score the model answer (0-10) and provide a one-sentence justification."
    )
    full_prompt = f"{JUDGE_SYSTEM_PROMPT}\n\n{user_msg}"
    try:
        raw, _body = call_model(judge_model, full_prompt, max_tokens)
        raw = (raw or "").strip()
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


def score_prediction(
    answer_format: str,
    prediction: Any,
    ground_truth: Any,
    acceptance_bounds: Optional[Dict[str, Any]],
    question_text: str,
    judge_model: str,
) -> Tuple[Optional[float], Optional[Tuple[float, str]]]:
    """Return (score, judge_result_or_none). Mirrors test_gpt_5mini scoring branches."""
    gt = ground_truth
    pred = prediction
    judge_result: Optional[Tuple[float, str]] = None
    score: Optional[float] = None

    try:
        if answer_format == "free_form":
            ref_answer = str(gt) if gt is not None else ""
            judge_score, judge_reason = foundry_llm_judge(
                question=question_text,
                prediction=str(pred) if pred is not None else "",
                reference=ref_answer,
                judge_model=judge_model,
            )
            score = judge_score
            judge_result = (judge_score, judge_reason)

        elif answer_format == "numerical":
            try:
                gt_val = round(float(gt), 4)
                pred_val = round(float(pred), 4)
            except Exception:
                score = 0.0
            else:
                if acceptance_bounds:
                    margin = acceptance_bounds.get("margin", 0)
                    score = float(abs(pred_val - gt_val) <= margin)
                else:
                    score = float(abs(pred_val - gt_val) < 1e-4)

        elif answer_format == "tensor":
            try:
                gt_vals = [float(x) for x in str(gt).split("_")]
                pred_vals = [float(x) for x in str(pred).split("_")]
            except Exception:
                score = 0.0
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

        elif answer_format == "multiple_choice_multi_select":
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

        elif answer_format == "multiple_choice_single_select":
            gt_str = str(gt).strip().upper()
            pred_letter = parse_llm_answer(str(pred).strip())
            score = 0.0 if pred_letter is None else float(gt_str == pred_letter)

        elif answer_format == "ranking":
            gt_str = str(gt).strip().upper()
            match = re.search(r"\b([A-D]{4})\b", str(pred).strip().upper())
            pred_str = match.group(1) if match else ""
            score = float(gt_str == pred_str)

        else:
            score = float(str(gt) == str(pred))
    except Exception:
        score = None

    return score, judge_result


def log_to_opik(
    qa_payload: Dict[str, Any],
    prompt_text: str,
    pred: Any,
    gt: Any,
    body: Dict[str, Any],
    model_name: str,
    eval_level: str,
    answer_format: str,
    question_type: str,
    score: Optional[float],
    judge_result: Optional[Tuple[float, str]],
    est_cost: float,
    usage_raw: Dict[str, Any],
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    if not os.getenv("OPIK_API_KEY"):
        return
    try:
        import opik
        client_opik = opik.Opik(
            project_name=os.getenv("OPIK_PROJECT_NAME", "FactoryBench"),
            workspace=os.getenv("OPIK_WORKSPACE", "forgis"),
        )
        total_tokens = int(usage_raw.get("total_tokens") or (prompt_tokens + completion_tokens))
        metadata_payload = qa_payload.get("metadata") or {}
        dataset_tag = metadata_payload.get("dataset")
        episode_tag = metadata_payload.get("episode")
        qa_pair_id = metadata_payload.get("qa_pair_id")

        opik_tags = [str(t) for t in [eval_level, question_type, dataset_tag, answer_format] if t]
        if answer_format and answer_format != "unknown":
            opik_tags.append(f"eval_{answer_format}")
        if model_name and model_name != "unknown":
            opik_tags.append(model_name)
        if qa_pair_id:
            opik_tags.append(f"template_{qa_pair_id}")

        opik_metadata = {
            "model": model_name,
            "answer_format": answer_format,
            "question_type": question_type,
            "qa_pair_id": qa_pair_id,
            "dataset": dataset_tag,
            "episode": episode_tag,
            "time_window": metadata_payload.get("time_window"),
            "correct_answer": gt,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "total_estimated_cost": f"${est_cost}",
            },
            "full_telemetry_context": {"text": prompt_text},
        }

        trace = client_opik.trace(
            name=f"factorybench_{eval_level}_{qa_pair_id}",
            input={"question": qa_payload.get("question"), "ground_truth": gt, "prediction": pred},
            output={"answer": pred, "raw_body": body},
            tags=opik_tags,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            metadata=opik_metadata,
            total_estimated_cost=est_cost,
            model=body.get("model") or model_name,
        )

        if score is not None:
            pred_str_repr = str(pred).strip() if pred is not None else "N/A"
            gt_str_repr = str(gt).strip() if gt is not None else "N/A"
            accuracy_reason = (
                judge_result[1] if (judge_result and judge_result[1])
                else f"Predicted: '{pred_str_repr}' | Ground truth: '{gt_str_repr}'"
            )
            trace.log_feedback_score(name="accuracy", value=float(score), reason=accuracy_reason)
            if judge_result is not None:
                trace.log_feedback_score(
                    name="llm_judge", value=judge_result[0], reason=judge_result[1] or accuracy_reason,
                )
    except Exception as e:
        logger.warning(f"Opik logging failed: {e}")


def run_foundry_eval(
    entries: list[Tuple[Path, str, int, str]],
    model: str,
    output_dir: Path,
    max_output_tokens: int,
    overwrite: bool,
    ground_truth_index: Dict[str, Any],
    eval_level: str,
    judge_model: str,
    cost_limit: float,
) -> Tuple[int, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = failed = skipped = 0
    total_cost = 0.0

    for prompt_path, prompt_text, prompt_idx, custom_id in entries:
        out_path = output_dir / f"{custom_id}_answer.json"
        fail_path = output_dir / f"{custom_id}_failed.json"

        if not overwrite and (out_path.exists() or fail_path.exists()):
            skipped += 1
            logger.info(f"- Skipping existing result: {custom_id}")
            continue

        if len(prompt_text) // 4 > 900_000:
            logger.warning(f"- Skipping {custom_id}: prompt too large")
            skipped += 1
            continue

        try:
            qa_payload = load_json(prompt_path)
        except Exception:
            qa_payload = {}

        answer_format = str(qa_payload.get("answer_format") or qa_payload.get("answer_format").get("type") or "unknown")
        question_type = str(qa_payload.get("type") or "unknown")
        acceptance_bounds = qa_payload.get("acceptance_bounds")
        level_val = qa_payload.get("level")
        effective_eval_level = eval_level or (f"level_{level_val}" if level_val is not None else None)

        try:
            answer, body = call_model(model, prompt_text, max_output_tokens)

            usage_raw = body.get("usage", {}) or {}
            prompt_tokens = int(usage_raw.get("prompt_tokens") or usage_raw.get("input_tokens") or 0)
            completion_tokens = int(usage_raw.get("completion_tokens") or usage_raw.get("output_tokens") or 0)
            model_name = (body.get("model") or model).lower()
            est_cost = _estimate_cost(model_name, prompt_tokens, completion_tokens)
            total_cost += est_cost

            gt = ground_truth_index.get(prompt_path.stem)
            score, judge_result = score_prediction(
                answer_format=answer_format,
                prediction=answer,
                ground_truth=gt,
                acceptance_bounds=acceptance_bounds,
                question_text=qa_payload.get("question", ""),
                judge_model=judge_model,
            )

            log_to_opik(
                qa_payload=qa_payload,
                prompt_text=prompt_text,
                pred=answer,
                gt=gt,
                body=body,
                model_name=model_name,
                eval_level=effective_eval_level or "unknown",
                answer_format=answer_format,
                question_type=question_type,
                score=score,
                judge_result=judge_result,
                est_cost=est_cost,
                usage_raw=usage_raw,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            save_json(out_path, {
                "custom_id": custom_id,
                "prompt_index": prompt_idx,
                "prompt_file": str(prompt_path),
                "prompt": prompt_text,
                "answer": answer,
                "ground_truth": gt,
                "score": score,
                "llm_judge_score": judge_result[0] if judge_result else None,
                "llm_judge_reason": judge_result[1] if judge_result else None,
                "answer_format": answer_format,
                "question_type": question_type,
                "model": body.get("model") or model,
                "usage": body.get("usage"),
                "estimated_cost": est_cost,
                "raw_api_response": body,
            })
            completed += 1
            logger.info(
                f"✓ [{completed + failed + skipped}/{len(entries)}] {custom_id}"
                f" | cost: ${est_cost:.4f} | total: ${total_cost:.4f}"
            )

            if total_cost >= cost_limit:
                logger.warning(f"Cost limit ${cost_limit:.2f} reached (${total_cost:.4f}). Stopping.")
                return completed, failed, skipped

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
    parser = argparse.ArgumentParser(description="Foundry multi-model evaluation")
    parser.add_argument("--input", type=Path, required=True, help="Directory containing prompt JSON files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for replies")
    parser.add_argument("--questions", type=Path, required=True, help="Directory with Q&A ground truth JSONs")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(FOUNDRY_MODELS.keys()),
                        help="Foundry model to evaluate")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
                        help=f"Model used for free-form scoring (default: {DEFAULT_JUDGE_MODEL})")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--max-output-tokens", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--total-prompts", type=int, default=None)
    parser.add_argument("--batch-number", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--eval-level", type=str, default=None)
    parser.add_argument("--cost-limit", type=float, default=20.0)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    load_dotenv_file(args.env_file)

    endpoint, api_style = resolve_endpoint(args.model)
    logger.info(f"Model={args.model} | api_style={api_style} | endpoint={endpoint}")

    ground_truth_index = build_ground_truth_index(args.questions)
    entries = load_prompt_entries(
        input_dir=args.input,
        total_prompts=args.total_prompts,
        batch_number=args.batch_number,
        batch_size=args.batch_size,
    )
    if not entries:
        logger.error("No prompts to process")
        return

    completed, failed, skipped = run_foundry_eval(
        entries=entries,
        model=args.model,
        output_dir=args.output_dir,
        max_output_tokens=args.max_output_tokens,
        overwrite=args.overwrite,
        ground_truth_index=ground_truth_index,
        eval_level=args.eval_level,
        judge_model=args.judge_model,
        cost_limit=args.cost_limit,
    )
    logger.info(
        f"Done. Completed={completed}, Failed={failed}, Skipped={skipped}, OutputDir={args.output_dir}"
    )


if __name__ == "__main__":
    main()

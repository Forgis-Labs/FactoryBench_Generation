"""Shared Opik tracing for FactoryBench evaluators.

Both ``run_foundry_eval`` and ``run_direct_eval`` log per-question traces with
the same schema (name, tags, metadata, usage, feedback scores). Keeping the
construction in one place avoids the two evaluators drifting apart.

The function is a no-op when ``OPIK_API_KEY`` is not set, and any client error
is swallowed and logged as a warning so a transient Opik outage never breaks
an evaluation run.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def log_eval_trace_to_opik(
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
    """Send one evaluation trace to Opik with the FactoryBench schema."""
    if not os.getenv("OPIK_API_KEY"):
        return
    try:
        import opik
        client_opik = opik.Opik(
            project_name=os.getenv("OPIK_PROJECT_NAME", "FactoryBench"),
            workspace=os.getenv("OPIK_WORKSPACE", "default"),
        )

        total_tokens = int(
            usage_raw.get("total_tokens") or (prompt_tokens + completion_tokens)
        )

        metadata_payload = qa_payload.get("metadata") or {}
        dataset_tag = metadata_payload.get("dataset")
        episode_tag = metadata_payload.get("episode")
        qa_pair_id = metadata_payload.get("qa_pair_id")
        type_tag = metadata_payload.get("type")

        # Tags: filter empties; "unknown" is treated as missing.
        opik_tags = [
            str(t)
            for t in [eval_level, question_type, dataset_tag, answer_format]
            if t
        ]
        if type_tag and type_tag != "unknown":
            opik_tags.append(str(type_tag))
        if answer_format and answer_format != "unknown":
            opik_tags.append(f"eval_{answer_format}")
        if model_name and model_name != "unknown":
            opik_tags.append(model_name)
        if dataset_tag and dataset_tag != "unknown":
            opik_tags.append(str(dataset_tag))
        if qa_pair_id and qa_pair_id != "unknown":
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
            judge_reason = judge_result[1] if judge_result else None
            accuracy_reason = (
                judge_reason
                if judge_reason
                else f"Predicted: '{pred_str_repr}' | Ground truth: '{gt_str_repr}'"
            )
            trace.log_feedback_score(
                name="accuracy",
                value=float(score),
                reason=accuracy_reason,
            )
            if judge_result is not None:
                trace.log_feedback_score(
                    name="llm_judge",
                    value=judge_result[0],
                    reason=judge_result[1] or accuracy_reason,
                )
    except Exception as e:
        logger.warning(f"Opik logging failed: {e}")

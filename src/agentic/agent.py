"""ReAct-style agent loop with OpenAI function-calling.

One call = one question. The agent is instantiated with the four tool
objects (already bound to this question's time series and the shared
RAG index), plus a chat client from the same Foundry runner path used
by the zero-shot panel — so auth and cost accounting inherit for free.

Termination conditions (whichever fires first):
  * the model returns a final assistant message with no tool_calls,
  * the tool-call budget is exhausted (default 8 calls),
  * an unrecoverable error is raised (returned to the caller as an
    error payload).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


AGENT_SYSTEM_PROMPT = """You are an industrial-telemetry reasoning agent evaluating on FactoryBench.

FIRST, THE DEFAULT: most questions here are answered best by reading the series
in front of you and thinking carefully, with no tool at all. Answering directly
is a first-class outcome, not a fallback. If no tool is clearly better than you
at the specific subtask, ignore everything below the tool list and answer the
question exactly as you would if this were a plain question with no tools
available. Calling a tool you did not need costs you the attention the question
needs, and that shows up as a wrong answer.

You have access to four tools:

  * `signal_stats` — per-channel statistics (min, max, mean, std, p05, p95, derivative).
  * `describe_dynamics` — structural read of the window: regime change points, per-channel
    drift, setpoint-vs-feedback tracking error, and strong cross-channel correlations.
  * `forecast` — Chronos-Bolt (200M-param) forecaster for future value prediction.
  * `run_python` — sandboxed Python (numpy/scipy) with the item's time series bound as `ts` (dict of channel → np.ndarray). Great for windowed searches, derivatives, pattern matching.
  * `retrieve_manual` — RAG over vendor PDFs (UR3e, KUKA KR6/KR10, voraus AI) for machine-specific concepts and protocols.

HOW TO DECIDE WHICH TOOL (map from question shape to tool):

  QUESTION SHAPE                                        → TOOL
  --------------------------------------------------------------------------
  "expected value of <signal> at T+N ms/steps?"         → forecast, THEN VERIFY
     The forecaster is a 200M general-purpose model. It is not an oracle and
     on this benchmark it is often no better than your own reading of the
     trend. Treat `predicted_value_at_horizon` as ONE estimate, not as the
     answer.
     The tool answers with four things: its point estimate, its 10-90 band, a
     `linear_trend_estimate` computed from the same channel, and a
     `reliability` verdict. READ `reliability` BEFORE ANSWERING.
       * CONSISTENT   -> the point estimate is safe to use.
       * LOW CONFIDENCE -> the band is wider than the channel's own range.
         The forecaster does not know. Answer from your own reading of the
         trend, or from `linear_trend_estimate`, not from the point estimate.
       * DISAGREEMENT -> the two estimates are far apart. Decide which is
         better supported by the series you can see, and say which you used.
     Emitting `predicted_value_at_horizon` unchanged when the verdict is not
     CONSISTENT is the single most common way to get this question wrong.

  "at which timestamp should the window begin?" /       → run_python
  "when does <event> begin?" (segment localization)        Use ts['<channel>'] and numpy.
     Typical recipe:
        v = np.abs(np.diff(ts['feedback_speed_0']))   # activity signal
        # find window of length W where activity best matches description
        onset_scores = np.array([v[i:i+W].sum() for i in range(len(v)-W)])
        answer = int(np.argmax(onset_scores))
     Return the integer timestamp answer.
     This tool is also the right one for any arithmetic over a long window:
     ratios of means, drift between segments, peak-to-final decay. Prefer it
     over doing the arithmetic in your head.

  Root cause, remediation, repair procedure, or any     → retrieve_manual (ALWAYS)
  question naming a machine-specific parameter             then answer
     THIS OVERRIDES THE DEFAULT ABOVE. Retrieval is the one case where the tool
     knows something you do not, so always call it here even though direct
     answering is the default elsewhere.
     Every Level-4 item is one of these. Do not answer a troubleshooting or
     optimization question from memory: query the manuals with the machine
     name plus the observed symptom plus the specific noun (for example
     "UR3e payload mass installation setting" or "KUKA KR10 collision
     detection reaction"), and ground the procedure you return in what comes
     back. A remediation naming the right parameter and the right corrective
     direction scores; a generic tuning checklist does not.

  "what changed / when did it change / which channel is  → describe_dynamics
  anomalous / how does this differ from normal?"
     One call returns change points, drift, tracking error and correlations for the
     whole window. Prefer it over reading hundreds of rows by eye, and over
     signal_stats when the question is about behaviour rather than a single number.

  Ranking, multi-select T/F, phase reading, or any      → DIRECT (no tool)
  comparison you can make by reading the series
     No tool beats careful reading here, and reaching for one costs you the
     attention the question needs. Consider each candidate against the series
     before committing to an ordering or a T/F string. Reason internally: the
     reply itself must still be the bare answer token and nothing else, exactly
     as you would answer with no tools at all.

RULES:
  1. Use a tool only when it is genuinely better than you at that subtask.
     The sandbox is better than you at arithmetic over hundreds of rows, and
     the manuals know vendor procedures you do not. You are usually better
     than the forecaster at reading a trend that is visible in the window.
     A wrong tool call is worse than no tool call, and never guess arguments.
     If you are unsure whether a tool would help, do not call it.
  2. Max 8 tool calls per question. Do not loop; if a tool errors, try a
     different approach or answer directly. Spending a call to check a
     surprising tool result is worth it; spending one to re-ask a tool
     that already answered is not.
  3. Final answer format is EXACT: a single letter, T/F string, ranking
     permutation, single number, or free-form protocol. No preamble, no
     units, no explanation unless the question is free-form. When the
     question says "Answer only with an integer or decimal number, nothing
     else" — do exactly that."""


class Agent:
    def __init__(
        self,
        client,
        model: str,
        tools: Sequence[Any],
        max_tool_calls: int = 6,
        temperature: float = 0.0,
    ):
        self.client = client
        self.model = model
        self.tools_by_name = {t.NAME: t for t in tools}
        self.tool_specs = [t.spec() for t in tools]
        self.max_tool_calls = max_tool_calls
        self.temperature = temperature

    def answer(self, prompt: str) -> Dict[str, Any]:
        """Run the ReAct loop on one prompt. Returns dict with keys:
        `answer`, `trace` (list of tool-call summaries), `usage`, `error`.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        trace: List[Dict[str, Any]] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        calls_used = 0
        last_error: Optional[str] = None

        while calls_used <= self.max_tool_calls:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_specs,
                    tool_choice="auto",
                )
            except Exception as exc:
                last_error = f"chat call failed: {type(exc).__name__}: {exc}"
                logger.warning(last_error)
                break

            u = getattr(resp, "usage", None)
            if u is not None:
                for k in total_usage:
                    total_usage[k] += int(getattr(u, k, 0) or 0)

            choice = resp.choices[0]
            msg = choice.message
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in (msg.tool_calls or [])
                ] or None,
            })

            if not msg.tool_calls:
                # final answer
                return {
                    "answer": (msg.content or "").strip(),
                    "trace": trace,
                    "usage": total_usage,
                    "n_tool_calls": calls_used,
                    "error": None,
                }

            for tc in msg.tool_calls:
                calls_used += 1
                name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    args = {}
                tool = self.tools_by_name.get(name)
                if tool is None:
                    tool_result: Any = {"error": f"unknown tool {name!r}"}
                else:
                    try:
                        tool_result = tool(**args)
                    except Exception as exc:
                        tool_result = {"error": f"{type(exc).__name__}: {exc}"}
                trace.append({"tool": name, "args": args, "result_repr": _truncate(repr(tool_result))})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": _serialise(tool_result),
                })
            # loop back for the model's next turn

        return {
            "answer": "",
            "trace": trace,
            "usage": total_usage,
            "n_tool_calls": calls_used,
            "error": last_error or f"tool-call budget exhausted ({self.max_tool_calls})",
        }


def _serialise(x: Any, cap: int = 8000) -> str:
    try:
        s = json.dumps(x, default=lambda o: str(o))
    except Exception:
        s = repr(x)
    return s if len(s) <= cap else s[:cap] + f"...[truncated {len(s) - cap} chars]"


def _truncate(s: str, cap: int = 400) -> str:
    return s if len(s) <= cap else s[:cap] + "…"


class BedrockAgent:
    """Same ReAct pipeline as `Agent`, but driven by AWS Bedrock Converse
    (Anthropic-native tool_use) instead of the OpenAI chat.completions API.

    Interface is deliberately identical: ``answer(prompt) -> {"answer",
    "trace", "usage", "n_tool_calls", "error"}``. Tool specs are the same
    OpenAI-flavoured dicts produced by every tool's ``spec()`` method —
    we translate the ``function`` / ``parameters`` fields into Bedrock's
    ``toolSpec`` / ``inputSchema`` on the fly, so the tools themselves
    don't have to know which backend is running them.
    """

    def __init__(
        self,
        client,        # boto3 bedrock-runtime client
        model: str,    # bedrock model id, e.g. "eu.anthropic.claude-sonnet-4-6"
        tools: Sequence[Any],
        max_tool_calls: int = 6,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
    ):
        self.client = client
        self.model = model
        self.tools_by_name = {t.NAME: t for t in tools}
        self._oai_specs = [t.spec() for t in tools]
        self.tool_config = {
            "tools": [_oai_to_bedrock_tool(s) for s in self._oai_specs],
        }
        self.max_tool_calls = max_tool_calls
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def answer(self, prompt: str) -> Dict[str, Any]:
        system_blocks = [{"text": AGENT_SYSTEM_PROMPT}]
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": [{"text": prompt}]},
        ]
        trace: List[Dict[str, Any]] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        calls_used = 0
        last_error: Optional[str] = None

        while calls_used <= self.max_tool_calls:
            try:
                resp = self.client.converse(
                    modelId=self.model,
                    messages=messages,
                    system=system_blocks,
                    toolConfig=self.tool_config,
                    inferenceConfig={
                        "temperature": self.temperature,
                        "maxTokens":  self.max_output_tokens,
                    },
                )
            except Exception as exc:
                last_error = f"converse call failed: {type(exc).__name__}: {exc}"
                logger.warning(last_error)
                break

            u = resp.get("usage") or {}
            total_usage["prompt_tokens"]     += int(u.get("inputTokens", 0) or 0)
            total_usage["completion_tokens"] += int(u.get("outputTokens", 0) or 0)
            total_usage["total_tokens"]      += int(u.get("totalTokens", 0) or 0)

            out_msg = (resp.get("output") or {}).get("message") or {}
            content_blocks: List[Dict[str, Any]] = out_msg.get("content") or []
            stop_reason = resp.get("stopReason") or ""

            # Echo the assistant turn back into the conversation.
            messages.append({"role": "assistant", "content": content_blocks})

            tool_uses = [b for b in content_blocks if "toolUse" in b]
            if not tool_uses or stop_reason not in ("tool_use", "toolUse"):
                # Final answer — concatenate all text blocks in order.
                text = "".join(b.get("text") or "" for b in content_blocks if "text" in b).strip()
                return {
                    "answer": text,
                    "trace": trace,
                    "usage": total_usage,
                    "n_tool_calls": calls_used,
                    "error": None if text else "empty assistant response",
                }

            tool_result_blocks: List[Dict[str, Any]] = []
            for b in tool_uses:
                tu = b["toolUse"]
                name = tu.get("name")
                tuid = tu.get("toolUseId")
                args = tu.get("input") or {}
                calls_used += 1
                tool = self.tools_by_name.get(name)
                if tool is None:
                    tool_result: Any = {"error": f"unknown tool {name!r}"}
                else:
                    try:
                        tool_result = tool(**args)
                    except Exception as exc:
                        tool_result = {"error": f"{type(exc).__name__}: {exc}"}
                trace.append({"tool": name, "args": args, "result_repr": _truncate(repr(tool_result))})
                tool_result_blocks.append({
                    "toolResult": {
                        "toolUseId": tuid,
                        "content":   [{"text": _serialise(tool_result)}],
                    }
                })

            messages.append({"role": "user", "content": tool_result_blocks})

        return {
            "answer": "",
            "trace": trace,
            "usage": total_usage,
            "n_tool_calls": calls_used,
            "error": last_error or f"tool-call budget exhausted ({self.max_tool_calls})",
        }


def _oai_to_bedrock_tool(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Translate an OpenAI function spec into Bedrock Converse toolSpec."""
    fn = spec.get("function") or spec
    return {
        "toolSpec": {
            "name":        fn["name"],
            "description": fn.get("description") or "",
            "inputSchema": {"json": fn.get("parameters") or {"type": "object", "properties": {}}},
        }
    }

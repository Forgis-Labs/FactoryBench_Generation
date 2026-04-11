"""NeurIPS figure generator using Azure AI Foundry (MAI-Image-2).

Generates publication-quality diagrams from text prompts stored in
scripts/prompts/, with optional AI-driven prompt refinement and cost tracking.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


SCRIPT_DIR = Path(__file__).resolve().parent
COLOR_SCHEMA_PATH = SCRIPT_DIR / "color_schema.json"
DEFAULT_PROMPT_PATH = SCRIPT_DIR / "prompts" / "factorybench_neurips.txt"


API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")

MAI_BASE = os.getenv(
    "IMAGE_ENDPOINT",
).rstrip("/").rstrip('"')

CHAT_BASE = os.getenv(
    "CHAT_ENDPOINT",
).rstrip("/").rstrip('"')

ANTHROPIC_BASE = os.getenv(
    "REASONING_ENDPOINT",
).rstrip("/").rstrip('"')

IMAGE_MODEL = os.getenv("IMAGE_MODEL", "MAI-Image-2")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5-mini")
REASONING_MODEL = os.getenv("REASONING_MODEL", "claude-opus-4-6")

# OpenAI client for the chat model
chat_client = OpenAI(api_key=API_KEY, base_url=CHAT_BASE)

COST_TABLE = {
    "MAI-Image-2":      {"per_image": 0.020},
    "gpt-5-mini":       {"input_per_1k": 0.0004, "output_per_1k": 0.0016},
    "claude-opus-4-6":  {"input_per_1k": 0.015,  "output_per_1k": 0.075},
}



def load_color_schema() -> dict:
    """Load the Forgis color schema from JSON."""
    with open(COLOR_SCHEMA_PATH) as f:
        return json.load(f)


def format_color_block(colors: dict) -> str:
    """Format the color schema as a text block to inject into prompts."""
    lines = [
        "",
        "═══════════════════════════════════════════════════════",
        "FORGIS BRAND COLOR PALETTE (use these colors)",
        "═══════════════════════════════════════════════════════",
    ]
    for section, mapping in colors.items():
        lines.append(f"\n{section.upper()}:")
        if isinstance(mapping, dict) and all(isinstance(v, str) for v in mapping.values()):
            for name, hex_val in mapping.items():
                lines.append(f"  - {name}: {hex_val}")
        elif isinstance(mapping, dict):
            for name, obj in mapping.items():
                if isinstance(obj, dict):
                    parts = ", ".join(f"{k}={v}" for k, v in obj.items())
                    lines.append(f"  - {name}: {parts}")
                else:
                    lines.append(f"  - {name}: {obj}")
    return "\n".join(lines)


def load_prompt(path: Path) -> str:
    """Read a prompt text file."""
    return path.read_text(encoding="utf-8").strip()


def inject_colors_into_prompt(prompt: str, colors: dict) -> str:
    """Append the Forgis color palette to the end of a prompt."""
    color_block = format_color_block(colors)
    return f"{prompt}\n\n{color_block}"


# Cost tracker
class CostTracker:
    """Accumulates estimated API costs across all calls."""

    def __init__(self):
        self.entries: list[dict] = []

    def add_image(self, model: str = "MAI-Image-2"):
        cost = COST_TABLE.get(model, {}).get("per_image", 0.0)
        self.entries.append({"type": "image", "model": model, "cost": cost})

    def add_chat(self, model: str, input_tokens: int, output_tokens: int):
        table = COST_TABLE.get(model, {})
        cost = (
            (input_tokens / 1000) * table.get("input_per_1k", 0)
            + (output_tokens / 1000) * table.get("output_per_1k", 0)
        )
        self.entries.append({
            "type": "chat",
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
        })

    @property
    def total(self) -> float:
        return sum(e["cost"] for e in self.entries)

    def summary(self) -> str:
        lines = ["\n--- Cost Summary ---"]
        for i, e in enumerate(self.entries, 1):
            if e["type"] == "image":
                lines.append(f"  [{i}] {e['model']} image generation: ${e['cost']:.4f}")
            else:
                lines.append(
                    f"  [{i}] {e['model']} chat "
                    f"({e['input_tokens']} in / {e['output_tokens']} out): "
                    f"${e['cost']:.4f}"
                )
        lines.append(f"  TOTAL estimated cost: ${self.total:.4f}")
        return "\n".join(lines)


# Image generation (MAI-Image-2)
MAX_RETRIES = 3
RETRY_BACKOFF = [10, 30, 60]  # seconds to wait between retries


def generate_image(
    prompt: str,
    width: int = 1366,
    height: int = 768,
    tracker: CostTracker | None = None,
) -> bytes | None:
    """Call MAI-Image-2 to generate an image with retry logic.

    Retries up to MAX_RETRIES times on 499/timeout errors with exponential
    backoff, since MAI-Image-2 can be slow for complex prompts.
    """
    print(f"[image] Generating via {IMAGE_MODEL} ({width}x{height}) ...")

    for attempt in range(1, MAX_RETRIES + 1):
        timeout = 180 + (attempt - 1) * 60  # 180s, 240s, 300s
        try:
            r = requests.post(
                f"{MAI_BASE}/images/generations",
                headers={"api-key": API_KEY, "Content-Type": "application/json"},
                json={
                    "model": IMAGE_MODEL,
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                },
                timeout=timeout,
            )
        except requests.exceptions.Timeout:
            print(f"[image] Attempt {attempt}/{MAX_RETRIES}: timed out after {timeout}s")
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt - 1]
                print(f"[image] Retrying in {wait}s ...")
                time.sleep(wait)
                continue
            print("[image] All retries exhausted (timeout).")
            if tracker:
                tracker.add_image(IMAGE_MODEL)
            return None

        if r.status_code == 499 or r.status_code == 429:
            print(f"[image] Attempt {attempt}/{MAX_RETRIES}: HTTP {r.status_code}")
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt - 1]
                print(f"[image] Retrying in {wait}s ...")
                time.sleep(wait)
                continue
            print("[image] All retries exhausted.")
            if tracker:
                tracker.add_image(IMAGE_MODEL)
            return None

        if tracker:
            tracker.add_image(IMAGE_MODEL)

        if not r.ok:
            print(f"[image] ERROR {r.status_code}: {r.text[:500]}")
            return None

        item = r.json()["data"][0]
        if "b64_json" in item:
            return base64.b64decode(item["b64_json"])
        if "url" in item:
            print("[image] Downloading from URL ...")
            img_r = requests.get(item["url"], timeout=120)
            if img_r.ok:
                return img_r.content
            print(f"[image] Download failed: {img_r.status_code}")
        return None

    return None

# AI agents for prompt refinement

CRITIQUE_SYSTEM = """\
You are an expert academic figure reviewer for NeurIPS 2026.
You will receive the text prompt that was used to generate a diagram.
Evaluate the prompt for:
1. Clarity of layout instructions (flow direction, alignment, zones)
2. Completeness of text labels (every label must be spelled out verbatim)
3. Color specification (hex codes, not named colors)
4. Academic quality (clean, flat, no gradients, no 3D, NO dark backgrounds)
5. Domain accuracy for an industrial robotics benchmark paper

Output a structured critique with:
- STRENGTHS: what the prompt does well (2-3 bullet points)
- WEAKNESSES: what needs improvement (2-5 bullet points)
- REVISED_PROMPT: the full improved prompt text, ready to use for image generation

Important: the REVISED_PROMPT must be a complete, self-contained prompt.
Do NOT include any preamble or explanation inside REVISED_PROMPT — only the
prompt text itself. Keep the Forgis brand color palette section intact.
"""


def refine_with_gpt(prompt: str, tracker: CostTracker) -> tuple[str, str]:
    """Use GPT-5-mini to critique and improve the prompt.
    Returns (critique_text, revised_prompt).
    """
    print(f"[refine] Critiquing with {CHAT_MODEL} ...")
    resp = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": CRITIQUE_SYSTEM},
            {"role": "user", "content": f"Here is the current image-generation prompt:\n\n{prompt}"},
        ],
        max_completion_tokens=4096,
    )
    msg = resp.choices[0].message.content or ""
    usage = resp.usage
    if usage:
        tracker.add_chat(CHAT_MODEL, usage.prompt_tokens, usage.completion_tokens)

    return _parse_critique(msg, prompt)


def refine_with_claude(prompt: str, tracker: CostTracker) -> tuple[str, str]:
    """Use Claude Opus via the Anthropic endpoint to critique and improve the prompt.
    Returns (critique_text, revised_prompt).
    """
    print(f"[refine] Critiquing with {REASONING_MODEL} ...")
    r = requests.post(
        f"{ANTHROPIC_BASE}/messages",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": REASONING_MODEL,
            "max_tokens": 4096,
            "system": CRITIQUE_SYSTEM,
            "messages": [
                {"role": "user", "content": f"Here is the current image-generation prompt:\n\n{prompt}"},
            ],
        },
        timeout=120,
    )
    if not r.ok:
        print(f"[refine] ERROR {r.status_code}: {r.text[:500]}")
        return "Error: could not get critique", prompt

    body = r.json()
    msg = body.get("content", [{}])[0].get("text", "")
    usage = body.get("usage", {})
    tracker.add_chat(
        REASONING_MODEL,
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )
    return _parse_critique(msg, prompt)


TEXT_CHECK_SYSTEM = """\
You are a meticulous proofreader specializing in AI-generated diagrams.
You will receive an image of a diagram along with the original prompt used to
generate it.

Your job:
1. Read EVERY piece of text visible in the image — titles, labels, captions,
   axis names, legend entries, annotations, watermarks, everything.
2. List each text element you found in the image (exactly as it appears).
3. Compare each one against the intended text in the prompt.
4. Identify ALL errors: misspellings, garbled words, missing words, extra
   characters, wrong capitalisation, truncated labels, or nonsensical text.

Output format:
EXTRACTED_TEXT:
- "<text as it appears in image>" -> INTENDED: "<correct text from prompt>" | STATUS: OK / ERROR

ERRORS_FOUND: <number>

If ERRORS_FOUND > 0, output:
REVISED_PROMPT: <the full prompt with added emphasis on correct spelling>

In the REVISED_PROMPT:
- Keep the entire original prompt intact.
- After EVERY text label, add the instruction: (spell exactly as written).
- At the very top, add a bold instruction: "CRITICAL: Every text label must be
  spelled EXACTLY as specified. Double-check every letter."
- If specific words were garbled, add explicit notes like:
  "The word 'FactoryBench' must appear exactly — not 'FctoryBnch' or similar."

If ERRORS_FOUND == 0, do NOT output REVISED_PROMPT.
"""


def check_and_fix_text(
    image_bytes: bytes,
    prompt: str,
    width: int,
    height: int,
    tracker: CostTracker,
    agent: str = "gpt",
) -> tuple[str, bytes | None]:
    """Send the generated image to a vision model to check all text.

    Returns (report, corrected_image_bytes_or_None).
    If all text is correct, corrected_image_bytes is None.
    """
    img_b64 = base64.b64encode(image_bytes).decode()

    if agent == "claude":
        return _check_text_claude(img_b64, prompt, width, height, tracker)
    return _check_text_gpt(img_b64, prompt, width, height, tracker)


def _check_text_gpt(
    img_b64: str,
    prompt: str,
    width: int,
    height: int,
    tracker: CostTracker,
) -> tuple[str, bytes | None]:
    """Use GPT-5-mini vision to check text in the image."""
    print(f"[text-fix] Checking text with {CHAT_MODEL} (vision) ...")
    resp = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": TEXT_CHECK_SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Here is the original prompt:\n\n{prompt}\n\n"
                            "And here is the generated image. "
                            "Please check ALL text in the image for errors."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}",
                        },
                    },
                ],
            },
        ],
        max_completion_tokens=4096,
    )
    msg = resp.choices[0].message.content or ""
    usage = resp.usage
    if usage:
        tracker.add_chat(CHAT_MODEL, usage.prompt_tokens, usage.completion_tokens)

    return _process_text_check(msg, width, height, tracker)


def _check_text_claude(
    img_b64: str,
    prompt: str,
    width: int,
    height: int,
    tracker: CostTracker,
) -> tuple[str, bytes | None]:
    """Use Claude Opus vision to check text in the image."""
    print(f"[text-fix] Checking text with {REASONING_MODEL} (vision) ...")
    r = requests.post(
        f"{ANTHROPIC_BASE}/messages",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": REASONING_MODEL,
            "max_tokens": 4096,
            "system": TEXT_CHECK_SYSTEM,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Here is the original prompt:\n\n{prompt}\n\n"
                                "And here is the generated image. "
                                "Please check ALL text in the image for errors."
                            ),
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        },
                    ],
                },
            ],
        },
        timeout=120,
    )
    if not r.ok:
        print(f"[text-fix] ERROR {r.status_code}: {r.text[:500]}")
        return "Error: could not check text", None

    body = r.json()
    msg = body.get("content", [{}])[0].get("text", "")
    usage = body.get("usage", {})
    tracker.add_chat(
        REASONING_MODEL,
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
    )
    return _process_text_check(msg, width, height, tracker)


def _process_text_check(
    response: str,
    width: int,
    height: int,
    tracker: CostTracker,
) -> tuple[str, bytes | None]:
    """Parse the text-check response and regenerate if errors found."""
    # Check how many errors were found
    errors_found = 0
    for line in response.splitlines():
        if line.strip().startswith("ERRORS_FOUND:"):
            try:
                errors_found = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
            break

    if errors_found == 0:
        print("[text-fix] All text looks correct!")
        return response, None

    print(f"[text-fix] Found {errors_found} text error(s). Regenerating ...")

    # Extract the revised prompt
    marker = "REVISED_PROMPT:"
    idx = response.find(marker)
    if idx == -1:
        print("[text-fix] No REVISED_PROMPT in response, cannot fix.")
        return response, None

    revised = response[idx + len(marker):].strip()
    if revised.startswith("```"):
        first_nl = revised.index("\n") if "\n" in revised else 3
        revised = revised[first_nl + 1:]
    if revised.endswith("```"):
        revised = revised[:-3].strip()

    # Regenerate with corrected prompt
    corrected = generate_image(revised, width, height, tracker)
    return response, corrected


def _parse_critique(response: str, fallback_prompt: str) -> tuple[str, str]:
    """Extract the critique summary and revised prompt from the agent response."""
    revised = fallback_prompt
    marker = "REVISED_PROMPT:"
    idx = response.find(marker)
    if idx != -1:
        revised = response[idx + len(marker):].strip()
        # Strip markdown code fences if the model wrapped it
        if revised.startswith("```"):
            first_nl = revised.index("\n") if "\n" in revised else 3
            revised = revised[first_nl + 1:]
        if revised.endswith("```"):
            revised = revised[:-3].strip()
        critique = response[:idx].strip()
    else:
        critique = response

    return critique, revised


# Main pipeline

def run(args: argparse.Namespace) -> None:
    tracker = CostTracker()
    colors = load_color_schema()
    prompt_path = Path(args.prompt)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load and prepare the initial prompt
    raw_prompt = load_prompt(prompt_path)
    prompt = inject_colors_into_prompt(raw_prompt, colors)
    print(f"[init] Prompt loaded from {prompt_path} ({len(prompt)} chars)")
    print(f"[init] Output directory: {outdir}")
    print(f"[init] Refine: {args.refine}  Agent: {args.agent}  Iterations: {args.iterations}")
    print("---")

    iterations = args.iterations if args.refine else 1
    refine_fn = refine_with_claude if args.agent == "claude" else refine_with_gpt

    for i in range(1, iterations + 1):
        print(f"\n{'='*60}")
        print(f" Iteration {i}/{iterations}")
        print(f"{'='*60}")

        t0 = time.time()
        image_bytes = generate_image(prompt, args.width, args.height, tracker)
        elapsed = time.time() - t0
        print(f"[image] Generation took {elapsed:.1f}s")

        if image_bytes:
            img_path = outdir / f"diagram_iter_{i}.png"
            img_path.write_bytes(image_bytes)
            print(f"[image] Saved -> {img_path}")
        else:
            print("[image] No image returned, skipping save.")

        prompt_out = outdir / f"prompt_iter_{i}.txt"
        prompt_out.write_text(prompt, encoding="utf-8")

        # Refine prompt for next iteration (if enabled)
        if args.refine and i < iterations:
            critique, revised_prompt = refine_fn(prompt, tracker)

            critique_path = outdir / f"critique_iter_{i}.txt"
            critique_path.write_text(critique, encoding="utf-8")
            print(f"[refine] Critique saved -> {critique_path}")

            # Show a preview of the critique
            preview = critique[:300].replace("\n", " ")
            print(f"[refine] Preview: {preview}...")

            prompt = revised_prompt
            print(f"[refine] Prompt updated ({len(prompt)} chars)")

    # Text correction step (post-generation)
    if args.fix_text and image_bytes:
        print(f"\n{'='*60}")
        print(" Text Correction Step")
        print(f"{'='*60}")

        report, corrected_bytes = check_and_fix_text(
            image_bytes, prompt, args.width, args.height, tracker, args.agent,
        )

        report_path = outdir / "text_check_report.txt"
        report_path.write_text(report, encoding="utf-8")
        print(f"[text-fix] Report saved -> {report_path}")

        if corrected_bytes:
            corrected_path = outdir / "diagram_text_fixed.png"
            corrected_bytes_obj = corrected_bytes
            corrected_path.write_bytes(corrected_bytes_obj)
            print(f"[text-fix] Corrected image saved -> {corrected_path}")
            image_bytes = corrected_bytes
        else:
            print("[text-fix] No correction needed or regeneration failed.")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "prompt_file": str(prompt_path),
        "agent": args.agent if args.refine else None,
        "refine": args.refine,
        "iterations": iterations,
        "image_model": IMAGE_MODEL,
        "chat_model": CHAT_MODEL if args.agent == "gpt" else REASONING_MODEL,
        "width": args.width,
        "height": args.height,
        "color_schema": str(COLOR_SCHEMA_PATH),
        "cost": {
            "total_usd": round(tracker.total, 6),
            "entries": tracker.entries,
        },
    }
    meta_path = outdir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\n[done] Metadata saved -> {meta_path}")
    print(tracker.summary())



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate NeurIPS-quality figures via MAI-Image-2 with optional AI refinement.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--prompt", default=str(DEFAULT_PROMPT_PATH),
        help="Path to the prompt .txt file (default: prompts/factorybench_neurips.txt)",
    )
    p.add_argument(
        "--outdir", default=str(SCRIPT_DIR / "outputs" / datetime.now().strftime("run_%Y%m%d_%H%M%S")),
        help="Output directory for images and metadata",
    )
    p.add_argument(
        "--width", type=int, default=1366,
        help="Image width in pixels (default: 1366)",
    )
    p.add_argument(
        "--height", type=int, default=768,
        help="Image height in pixels (default: 768)",
    )
    p.add_argument(
        "--refine", action="store_true",
        help="Enable AI-driven iterative prompt refinement",
    )
    p.add_argument(
        "--agent", choices=["gpt", "claude"], default="gpt",
        help="Which model to use for critique/refinement (default: gpt)",
    )
    p.add_argument(
        "--iterations", type=int, default=3,
        help="Number of generate-refine cycles when --refine is set (default: 3)",
    )
    p.add_argument(
        "--fix-text", action="store_true",
        help="After final generation, use AI vision to check all text in the image and regenerate if errors are found",
    )
    return p.parse_args(argv)


def main():
    args = parse_args()
    if not API_KEY:
        print("ERROR: AZURE_OPENAI_API_KEY not set. Check your .env file.")
        sys.exit(1)
    run(args)


if __name__ == "__main__":
    main()

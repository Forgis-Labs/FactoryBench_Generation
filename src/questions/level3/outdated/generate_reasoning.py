#!/usr/bin/env python3
"""
Generate reasoning traces for Level 3 prompts using OpenAI Batch API (GPT-4.1).

This script:
1. Dynamically generates prompts using generate_prompt from generate_promtps.py
2. Uses the system prompt from system_prompt.json
3. Creates batch requests for OpenAI GPT-4.1 (Responses endpoint)
4. Submits batch and polls for completion
5. Stores reasoning responses in datasets/questions/level3/reasoning
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

from generate_promtps import (
    generate_prompt,
    is_inactive_subseries,
    INACTIVE_CONSTANT_THRESHOLD,
    MAX_RESAMPLE_ATTEMPTS,
    load_json,
    load_phrases,
    load_root_causes,
    load_anomalies,
    load_machines,
    sample_subseries,
)

logger = logging.getLogger(__name__)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def create_batch_request(
    prompt_data: Dict[str, Any],
    system_prompt: str,
    custom_id: str,
) -> Dict[str, Any]:
    """Create a single batch request line for the Batch API (Responses endpoint)."""

    prompt_obj = {
        'question': prompt_data.get('question'),
        'root_cause': prompt_data.get('root_cause'),
        'machine': prompt_data.get('machine'),
        'notes': prompt_data.get('notes'),
        'time_series_format': prompt_data.get('time_series_format'),
        'time_series': prompt_data.get('time_series'),
    }
    
    combined_prompt = f"{system_prompt}\n\n{json.dumps(prompt_obj, ensure_ascii=False)}"

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": "gpt-4.1",
            "input": combined_prompt,
            "max_output_tokens": 2000,
            "temperature": 0.7,
        },
    }


def create_batch_file(
    input_dir: Path,
    phrases_path: Path,
    root_causes_path: Path,
    anomalies_path: Path,
    machines_path: Path,
    system_prompt_path: Path,
    output_jsonl: Path,
    min_len: int = 32,
    max_len: int = 64,
    samples_per_episode: int = 1,
    seed: Optional[int] = None,
    total_prompts: Optional[int] = None,
    batch_number: int = 0,
    batch_size: int = 1000,
) -> int:
    """
    Dynamically generate prompts and create a JSONL batch file.

    total_prompts:
      Total number of prompts to generate across all batches. If None, generates all prompts.
    batch_number:
      Which batch this is (0-indexed), where each batch is batch_size prompts.
    """
    if seed is not None:
        random.seed(seed)

    system_prompt_data = load_json(system_prompt_path)
    system_prompt = system_prompt_data.get("prompt", "")

    phrases = load_phrases(phrases_path)
    root_causes = load_root_causes(root_causes_path)
    anomalies = load_anomalies(anomalies_path)
    machines = load_machines(machines_path)

    episode_files = sorted(input_dir.glob("*.json"))
    if not episode_files:
        logger.warning(f"No episode JSON files found in {input_dir}")
        return 0

    batch_start = batch_number * batch_size
    batch_end = (batch_number + 1) * batch_size

    requests: list[dict[str, Any]] = []
    global_prompt_index = 0

    # Determine total prompts to generate
    if total_prompts is None:
        total_prompts = batch_end  # Generate enough for this batch

    # Pre-load all episodes with metadata
    episodes: list[tuple[Path, list, dict]] = []
    for episode_path in episode_files:
        rows = load_json(episode_path)
        if not isinstance(rows, list):
            logger.warning(f"Skipping non-list episode: {episode_path}")
            continue

        metadata_path = episode_path.parent / f"{episode_path.stem}_metadata.json"
        metadata = load_json(metadata_path) if metadata_path.exists() else {}

        episodes.append((episode_path, rows, metadata))

    if not episodes:
        logger.warning(f"No valid episodes found in {input_dir}")
        return 0

    logger.info(f"Generating {total_prompts} prompts from {len(episodes)} episodes")

    # Generate prompts by randomly sampling episodes until we reach total_prompts
    prompt_idx = 0
    while prompt_idx < total_prompts:
        current_episode = random.choice(episodes)
        excluded_ids: set[str] = set()
        attempts = 0

        while True:
            episode_path, rows, metadata = current_episode
            episode_stem = episode_path.stem
            exp_id = episode_stem.replace("experiment_", "") if episode_stem.startswith("experiment_") else episode_stem

            attempts += 1
            subseries = sample_subseries(rows, min_len, max_len)

            if not is_inactive_subseries(subseries, metadata, machines, INACTIVE_CONSTANT_THRESHOLD):
                break

            if attempts >= MAX_RESAMPLE_ATTEMPTS:
                excluded_ids.add(exp_id)
                candidates = [ep for ep in episodes if (ep[0].stem.replace("experiment_", "") if ep[0].stem.startswith("experiment_") else ep[0].stem) not in excluded_ids]
                if not candidates:
                    logger.warning(
                        f"All episodes inactive for prompt {prompt_idx}; keeping last inactive sample"
                    )
                    break
                current_episode = random.choice(candidates)
                attempts = 0

        # Only emit prompts that fall in this batch slice
        in_this_batch = (prompt_idx >= batch_start) and (prompt_idx < batch_end)

        if in_this_batch:
            question = random.choice(phrases)

            prompt_data = generate_prompt(
                subseries=subseries,
                metadata=metadata,
                question_text=question,
                root_causes=root_causes,
                anomalies=anomalies,
                machines=machines,
            )

            custom_id = f"experiment_{exp_id}_prompt_{prompt_idx}"
            requests.append(create_batch_request(prompt_data, system_prompt, custom_id))

        prompt_idx += 1

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    logger.info(f"Created batch file with {len(requests)} requests: {output_jsonl}")
    return len(requests)


def submit_batch(batch_file: Path, api_key: str) -> str:
    client = OpenAI(api_key=api_key)

    logger.info(f"Uploading batch file: {batch_file}")
    with batch_file.open("rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    logger.info(f"File uploaded: {file_obj.id}")

    logger.info("Creating batch...")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/responses",
        completion_window="24h",  # Batch completion window is 24h
    )

    logger.info(f"Batch created: {batch.id}")
    return batch.id


def poll_batch_status(
    batch_id: str,
    api_key: str,
    poll_interval: int = 60,
) -> Any:
    client = OpenAI(api_key=api_key)

    while True:
        batch = client.batches.retrieve(batch_id)
        status = getattr(batch, "status", None)
        logger.info(f"Batch status: {status}")

        if status == "completed":
            logger.info("Batch completed successfully!")
            return batch
        if status in {"failed", "expired", "cancelled"}:
            raise RuntimeError(f"Batch {batch_id} ended with status: {status}")

        time.sleep(poll_interval)


def _download_file_bytes(client: OpenAI, file_id: str) -> bytes:
    """
    New SDK: files.content(file_id) returns a binary response wrapper.
    Use .read() to get bytes.
    """
    content = client.files.content(file_id)
    return content.read()


def _extract_output_text_from_responses_body(body: dict[str, Any]) -> str:
    """
    Responses API batch output body format: body["output"] is a list of items.
    We concatenate any output_text segments found inside message items.
    """
    chunks: list[str] = []
    for item in body.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    return "".join(chunks)


def download_results(batch: Any, api_key: str, output_dir: Path) -> None:
    client = OpenAI(api_key=api_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file_id = getattr(batch, "output_file_id", None)
    error_file_id = getattr(batch, "error_file_id", None)

    if not output_file_id and not error_file_id:
        raise RuntimeError("Batch has neither output_file_id nor error_file_id")

    if output_file_id:
        logger.info(f"Downloading results from output file: {output_file_id}")
        file_bytes = _download_file_bytes(client, output_file_id)
        for line in file_bytes.decode("utf-8").splitlines():
            if not line.strip():
                continue

            result = json.loads(line)
            custom_id = result.get("custom_id")
            response = result.get("response", {}) or {}
            status_code = response.get("status_code")

            if status_code == 200:
                body = response.get("body", {}) or {}
                reasoning = _extract_output_text_from_responses_body(body)

                save_json(output_dir / f"{custom_id}_reasoning.json", {
                    "custom_id": custom_id,
                    "reasoning": reasoning,
                    "model": body.get("model"),
                    "usage": body.get("usage"),
                })
                logger.info(f"✓ Saved reasoning: {custom_id}")
            else:
                save_json(output_dir / f"{custom_id}_failed.json", {
                    "custom_id": custom_id,
                    "status_code": status_code,
                    "response": response,
                })
                logger.warning(f"Request {custom_id} failed with status_code={status_code}")

    # Also download the error file if present, since it contains details for failed lines.
    if error_file_id:
        logger.info(f"Downloading error details from error file: {error_file_id}")
        err_bytes = _download_file_bytes(client, error_file_id)
        save_json(output_dir / "batch_error_file_raw.json", {
            "error_file_id": error_file_id,
            "jsonl": err_bytes.decode("utf-8", errors="replace"),
        })
        logger.info("✓ Saved batch error file snapshot")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reasoning using OpenAI Batch API (Responses)")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/normalized_episodes/aursad"),
        help="Directory of normalized episode JSON files",
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=Path(__file__).parent / "system_prompt.json",
        help="Path to system prompt JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/questions/level3/reasoning"),
        help="Directory to save reasoning outputs",
    )
    parser.add_argument(
        "--batch-file",
        type=Path,
        default=Path("data/questions/level3/batch_requests.jsonl"),
        help="Path to save batch requests JSONL file",
    )
    parser.add_argument("--min-len", type=int, default=32, help="Minimum subseries length")
    parser.add_argument("--max-len", type=int, default=64, help="Maximum subseries length")
    parser.add_argument("--samples-per-episode", type=int, default=1, help="Questions to sample per episode")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--api-key", type=str, required=True, help="OpenAI API key")

    parser.add_argument("--create-only", action="store_true", help="Only create batch file, do not submit")
    parser.add_argument("--batch-id", type=str, help="Existing batch ID to poll/download")

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
        "--poll-interval",
        type=int,
        default=60,
        help="Polling interval in seconds",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    repo_root = Path(__file__).resolve().parents[3]
    phrases_path = Path(__file__).with_name("phrases_level3.json")
    root_causes_path = repo_root / "data" / "rca" / "root_causes.json"
    anomalies_path = repo_root / "data" / "rca" / "anomalies.json"
    machines_path = repo_root / "data" / "machines" / "machines.json"

    if args.batch_id:
        logger.info(f"Polling existing batch: {args.batch_id}")
        batch = poll_batch_status(args.batch_id, args.api_key, args.poll_interval)
        download_results(batch, args.api_key, args.output_dir)
        return

    num_requests = create_batch_file(
        input_dir=args.input,
        phrases_path=phrases_path,
        root_causes_path=root_causes_path,
        anomalies_path=anomalies_path,
        machines_path=machines_path,
        system_prompt_path=args.system_prompt,
        output_jsonl=args.batch_file,
        min_len=args.min_len,
        max_len=args.max_len,
        samples_per_episode=args.samples_per_episode,
        seed=args.seed,
        total_prompts=args.total_prompts,
        batch_number=args.batch_number,
        batch_size=args.batch_size,
    )

    if num_requests == 0:
        logger.error("No requests to process")
        return

    if args.total_prompts is not None:
        total_batches = (args.total_prompts + args.batch_size - 1) // args.batch_size
        logger.info(f"Processing batch {args.batch_number}/{total_batches - 1}")
        logger.info(f"This batch contains {num_requests} prompts")

    if args.create_only:
        logger.info(f"Batch file created: {args.batch_file}")
        logger.info("Submit it by running without --create-only")
        return

    # Save batch file to batch files directory instead of submitting to API
    batch_files_dir = Path("data/questions/level3/batch_files")
    batch_files_dir.mkdir(parents=True, exist_ok=True)
    
    if args.total_prompts is not None:
        batch_filename = f"batch_{args.batch_number}_of_{(args.total_prompts + args.batch_size - 1) // args.batch_size - 1}.jsonl"
    else:
        batch_filename = f"batch_{args.batch_number}.jsonl"
    
    saved_batch_path = batch_files_dir / batch_filename
    
    import shutil
    shutil.copy(args.batch_file, saved_batch_path)
    logger.info(f"✓ Batch file saved: {saved_batch_path}")
    logger.info(f"  Contains {num_requests} prompts")
    
    # Comment out API calls - uncomment to actually submit to OpenAI
    # batch_id = submit_batch(args.batch_file, args.api_key)
    # logger.info(f"Batch submitted: {batch_id}")
    #
    # batch = poll_batch_status(batch_id, args.api_key, args.poll_interval)
    # download_results(batch, args.api_key, args.output_dir)
    #
    # logger.info("✓ Reasoning generation complete!")


if __name__ == "__main__":
    main()
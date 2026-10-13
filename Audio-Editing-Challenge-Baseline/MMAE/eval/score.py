from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .core.client import QwenOmniClient, DEFAULT_MODEL
from .core.data import (
    DEFAULT_METADATA_PATH,
    build_audio_map,
    load_dataset,
    select_items,
)
from .core.report import (
    append_jsonl,
    build_per_sample,
    build_taxonomy_report,
    write_json,
    write_jsonl,
)
from .core.rubric import extract_referenced_sources, parse_rubrics, score_one_rubric

MISSING_PREDICTION = "missing_prediction"


async def _score_sample(
    client: QwenOmniClient,
    sample_index: int,
    item: dict[str, Any],
    retries: int,
    max_attempts: int,
    timeout: float,
) -> list[dict[str, Any]]:
    rubrics = parse_rubrics(item)
    sample_id = item["id"]
    has_prediction = item["_has_prediction"]
    audio_map = build_audio_map(item)

    rows: list[dict[str, Any]] = []
    for rubric in rubrics:
        base = {
            "sample_index": sample_index,
            "sample_id": sample_id,
            "complexity": item["complexity"],
            "modality": item["modality"],
            "operations": item["operations"],
            "has_prediction": has_prediction,
        }
        if not has_prediction:
            base.update(
                {
                    "rubric_id": rubric.rubric_id,
                    "category": rubric.category,
                    "question": rubric.question,
                    "right_choice": rubric.right_choice,
                    "scores": [],
                    "score": 0,
                    "attempts": 0,
                    "attempts_detail": [],
                    "errors": [MISSING_PREDICTION],
                }
            )
            rows.append(base)
            continue

        # Only pass audio sources actually referenced in the rubric text
        referenced = extract_referenced_sources(rubric)
        rubric_audio_paths = [audio_map[s] for s in referenced if s in audio_map]
        rubric_audio_labels = [f"<audio {s}>" for s in referenced if s in audio_map]
        base["audio_map"] = {
            label: path for label, path in zip(rubric_audio_labels, rubric_audio_paths)
        }

        result = await score_one_rubric(
            client,
            sample_id,
            rubric_audio_paths,
            rubric_audio_labels,
            rubric,
            retries=retries,
            max_attempts=max_attempts,
            timeout=timeout,
        )
        base.update(result)
        rows.append(base)
    return rows


async def run(args: argparse.Namespace) -> None:
    audio_root = Path(args.audio_root) if args.audio_root else None
    items = load_dataset(
        Path(args.metadata),
        Path(args.predictions) if args.predictions else None,
        audio_root=audio_root,
    )
    selected = select_items(items, limit=args.limit, index=args.index)
    n_pred = sum(1 for _, it in selected if it["_has_prediction"])
    print(
        f"loaded {len(selected)} samples ({n_pred} with predictions, "
        f"{len(selected) - n_pred} missing)"
    )

    base_urls = [u.strip() for u in args.base_urls.split(",") if u.strip()]
    client = QwenOmniClient(
        base_urls=base_urls, model=args.model, timeout=args.timeout,
        api_key=os.environ.get(args.api_key_env),
        audio_format=args.audio_format, stream=args.stream,
    )
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    output_dir = Path(args.output_dir)
    # Record the judge configuration without credentials; API model aliases alone
    # do not prove that the provider serves the official checkpoint.
    write_json(output_dir / "judge_config.json", {
        "model": args.model,
        "reference_checkpoint": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "audio_format": args.audio_format,
        "stream": args.stream,
        "temperature": 0.7,
        "max_tokens": 4096,
        "retries": args.retries,
        "max_attempts": args.max_attempts,
    })
    results_path = output_dir / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("", encoding="utf-8")

    all_rows: list[dict[str, Any]] = []
    total_rubrics = sum(len(parse_rubrics(it)) for _, it in selected)
    progress = tqdm(total=total_rubrics, desc="Scoring", unit="rubric")

    async def _bounded(idx: int, item: dict[str, Any]) -> list[dict[str, Any]]:
        async with semaphore:
            return await _score_sample(
                client, idx, item,
                retries=args.retries,
                max_attempts=args.max_attempts,
                timeout=args.timeout,
            )

    tasks = [
        asyncio.create_task(_bounded(idx, item))
        for idx, item in selected
    ]
    for task in asyncio.as_completed(tasks):
        sample_rows = await task
        for row in sample_rows:
            all_rows.append(row)
            append_jsonl(results_path, row)
            progress.update(1)
    progress.close()

    all_rows.sort(key=lambda r: (r["sample_index"], r["rubric_id"]))
    write_jsonl(results_path, all_rows)

    per_sample = build_per_sample(all_rows)
    write_json(output_dir / "per_sample.json", per_sample)

    taxonomy = build_taxonomy_report(per_sample)
    write_json(output_dir / "taxonomy.json", taxonomy)

    overall = taxonomy["overall"]
    print(
        f"\nDone: {len(all_rows)} rubrics across {overall['samples']} samples\n"
        f"  Instruction Following: {overall['instruction_following']:.4f}\n"
        f"  Consistency:           {overall['consistency']:.4f}\n"
        f"  Exact Match:           {overall['exact_match']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MMAE Rubric Evaluation - score audio editing outputs with Qwen-Omni."
    )
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--predictions", required=True,
                        help="Path to the model predictions JSON file.")
    parser.add_argument("--audio_root", default=None)
    parser.add_argument("--output_dir", default="outputs/scores")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--retries", type=int, default=3,
                        help="Number of valid scores to collect per rubric.")
    parser.add_argument("--max_attempts", type=int, default=10,
                        help="Max total attempts (including failures) per rubric.")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--model", default=os.environ.get("MMAE_JUDGE_MODEL", DEFAULT_MODEL),
                        help="API model ID serving Qwen3-Omni-30B-A3B-Instruct.")
    parser.add_argument("--base_urls", "--base-url", dest="base_urls",
                        default=os.environ.get("MMAE_JUDGE_BASE_URL"),
                        help="API base URL including /v1, or comma-separated endpoint URLs.")
    parser.add_argument("--api-key-env", default="MMAE_JUDGE_API_KEY",
                        help="Environment variable containing the API key (never pass the key itself).")
    parser.add_argument("--audio-format", choices=["audio_url", "input_audio"],
                        default="audio_url", help="Audio input wire format required by the API.")
    parser.add_argument("--stream", action="store_true", help="Read streaming chat completion responses.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()
    if not args.base_urls or not any(url.strip() for url in args.base_urls.split(",")):
        parser.error("set --base-url/--base_urls or MMAE_JUDGE_BASE_URL")
    return args


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()

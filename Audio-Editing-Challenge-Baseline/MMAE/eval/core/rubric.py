from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass
from typing import Any

from .client import QwenOmniClient, parse_response

REF_SOURCE_PATTERN = re.compile(
    r"<audio\s+(?P<source>output|input\d+)", re.IGNORECASE
)


@dataclass(frozen=True)
class RubricQuestion:
    rubric_id: str
    category: str
    question: str
    right_choice: str
    wrong_choices: list[str]  # wrong_choices[:-1] (shuffled portion)
    fixed_last: str           # wrong_choices[-1] (always last, typically "None of the above")


def parse_rubrics(item: dict[str, Any]) -> list[RubricQuestion]:
    rubrics: list[RubricQuestion] = []
    for idx, raw in enumerate(item["rubrics"], start=1):
        category = raw["category"]
        if category not in ("Instruction Following", "Consistency"):
            continue
        question = raw["question"]
        right_choice = raw["right_choice"]
        all_wrong = raw["wrong_choices"]
        wrong_choices = all_wrong[:-1]
        fixed_last = all_wrong[-1]
        rubrics.append(
            RubricQuestion(
                rubric_id=f"R{idx}",
                category=category,
                question=question,
                right_choice=right_choice,
                wrong_choices=list(wrong_choices),
                fixed_last=fixed_last,
            )
        )
    return rubrics


def extract_referenced_sources(rubric: RubricQuestion) -> list[str]:
    all_text = (
        rubric.question + " " + rubric.right_choice + " "
        + " ".join(rubric.wrong_choices) + " " + rubric.fixed_last
    )
    sources: set[str] = set()
    for match in REF_SOURCE_PATTERN.finditer(all_text):
        sources.add(match.group("source").lower())
    return sorted(sources)


def _build_choices(
    rubric: RubricQuestion, sample_id: str, attempt: int
) -> tuple[list[str], str]:
    shuffled = [rubric.right_choice] + list(rubric.wrong_choices)
    seed = f"{sample_id}:{rubric.rubric_id}:{attempt}"
    random.Random(seed).shuffle(shuffled)
    shuffled.append(rubric.fixed_last)
    right_index = shuffled.index(rubric.right_choice)
    right_letter = chr(ord("A") + right_index)
    return shuffled, right_letter


async def score_one_rubric(
    client: QwenOmniClient,
    sample_id: str,
    audio_paths: list[str],
    audio_labels: list[str],
    rubric: RubricQuestion,
    retries: int = 3,
    max_attempts: int = 10,
    timeout: float = 300.0,
) -> dict[str, Any]:
    scores: list[int] = []
    attempts_detail: list[dict[str, Any]] = []
    errors: list[str] = []
    attempts = 0

    while len(scores) < retries and attempts < max_attempts:
        attempts += 1
        choices, right_letter = _build_choices(rubric, sample_id, attempts)
        valid_letters = {chr(ord("A") + i) for i in range(len(choices))}
        formatted_choices = [f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)]
        try:
            raw = await asyncio.wait_for(
                client.answer(
                    question=rubric.question,
                    choices=choices,
                    audio_paths=audio_paths,
                    audio_labels=audio_labels,
                ),
                timeout=timeout,
            )
            parsed = parse_response(raw)
            choice = (parsed["choice"] or "").strip().upper() if "choice" in parsed else ""
            if len(choice) == 1 and choice in valid_letters:
                score = 1 if choice == right_letter else 0
                scores.append(score)
                attempts_detail.append(
                    {
                        "choices": formatted_choices,
                        "right_letter": right_letter,
                        "model_choice": choice,
                        "score": score,
                        "raw_response": raw,
                    }
                )
            else:
                errors.append(f"invalid_choice:{choice!r}")
                attempts_detail.append(
                    {
                        "choices": formatted_choices,
                        "right_letter": right_letter,
                        "model_choice": choice,
                        "score": None,
                        "raw_response": raw,
                    }
                )
        except Exception as exc:
            errors.append(str(exc).strip() or type(exc).__name__)

    # Majority vote: pass if correct >= 2 times, else 0
    n_correct = sum(scores)
    score = 1 if n_correct >= 2 else 0
    return {
        "rubric_id": rubric.rubric_id,
        "category": rubric.category,
        "question": rubric.question,
        "right_choice": rubric.right_choice,
        "scores": scores,
        "score": score,
        "attempts": attempts,
        "attempts_detail": attempts_detail,
        "errors": errors,
    }

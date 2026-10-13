from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_per_sample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["sample_id"]].append(row)

    output: list[dict[str, Any]] = []
    for sample_id, sample_rows in sorted(grouped.items()):
        head = sample_rows[0]
        if_scores = [r["score"] for r in sample_rows if r["category"] == "Instruction Following"]
        c_scores = [r["score"] for r in sample_rows if r["category"] == "Consistency"]
        all_perfect = all(r["score"] == 1 for r in sample_rows)
        output.append(
            {
                "sample_id": sample_id,
                "complexity": head["complexity"],
                "modality": head["modality"],
                "operations": head["operations"],
                "has_prediction": head["has_prediction"],
                "instruction_following": _mean(if_scores),
                "consistency": _mean(c_scores),
                "exact_match": 1 if all_perfect else 0,
            }
        )
    return output


def _aggregate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if_vals = [s["instruction_following"] for s in samples if s["instruction_following"] is not None]
    c_vals = [s["consistency"] for s in samples if s["consistency"] is not None]
    em_vals = [float(s["exact_match"]) for s in samples]
    return {
        "samples": len(samples),
        "instruction_following": _mean(if_vals),
        "consistency": _mean(c_vals),
        "exact_match": _mean(em_vals),
    }


def build_taxonomy_report(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    by_modality: dict[str, list] = defaultdict(list)
    by_complexity: dict[str, list] = defaultdict(list)
    by_cross: dict[tuple[str, str], list] = defaultdict(list)
    op_local: dict[str, list] = defaultdict(list)
    op_global: dict[str, list] = defaultdict(list)

    for sample in per_sample:
        complexity = sample["complexity"]
        modality = sample["modality"]
        by_modality[modality].append(sample)
        by_complexity[complexity].append(sample)
        by_cross[(complexity, modality)].append(sample)
        if complexity == "single":
            for op in sample["operations"]:
                if op["granularity"] == "local":
                    op_local[op["operation"]].append(sample)
                elif op["granularity"] == "global":
                    op_global[op["operation"]].append(sample)

    return {
        "overall": _aggregate_samples(per_sample),
        "by_modality": {
            k: _aggregate_samples(v) for k, v in sorted(by_modality.items())
        },
        "by_complexity": {
            k: _aggregate_samples(v) for k, v in sorted(by_complexity.items())
        },
        "by_complexity_modality": {
            f"{c} {m}": _aggregate_samples(v)
            for (c, m), v in sorted(by_cross.items())
        },
        "single_operations": {
            "local": {op: _aggregate_samples(v) for op, v in sorted(op_local.items())},
            "global": {op: _aggregate_samples(v) for op, v in sorted(op_global.items())},
        },
    }

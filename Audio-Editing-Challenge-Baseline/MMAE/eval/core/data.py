from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_METADATA_PATH = Path("MMAE-meta.json")
DEFAULT_PREDICTIONS_PATH = None
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SampleAudio:
    inputs: list[str] = field(default_factory=list)
    output: str | None = None


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _resolve_audio_path(rel_path: str, audio_roots: list[Path]) -> str | None:
    if not rel_path:
        return None
    candidate = Path(rel_path)
    if candidate.is_absolute():
        return str(candidate) if candidate.exists() else None
    for root in audio_roots:
        resolved = root / candidate
        if resolved.exists():
            return str(resolved)
    return None


def _build_sample_audio(
    messages: list[dict[str, Any]], audio_roots: list[Path], sample_id: str
) -> SampleAudio:
    audio = SampleAudio()
    seen_inputs: set[str] = set()
    last_assistant_audio: str | None = None
    for message in messages:
        role = message["role"]
        for content in message["content"]:
            if content["type"] != "audio":
                continue
            url = content["audio_url"]
            resolved = _resolve_audio_path(url, audio_roots)
            if role == "user":
                if url in seen_inputs:
                    continue
                seen_inputs.add(url)
                if resolved is None:
                    print(
                        f"[ERROR] sample {sample_id}: user audio not found: {url}\n"
                        f"  searched roots: {[str(r) for r in audio_roots]}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                audio.inputs.append(resolved)
            elif role == "assistant":
                if resolved is not None:
                    last_assistant_audio = resolved
    audio.output = last_assistant_audio
    return audio


def load_dataset(
    metadata_path: Path = DEFAULT_METADATA_PATH,
    predictions_path: Path | None = DEFAULT_PREDICTIONS_PATH,
    audio_root: Path | None = None,
) -> list[dict[str, Any]]:
    metadata = _load_json_list(metadata_path)
    predictions_by_id: dict[str, dict[str, Any]] = {}
    if predictions_path is not None and Path(predictions_path).exists():
        predictions = _load_json_list(Path(predictions_path))
        predictions_by_id = {item["id"]: item for item in predictions}

    roots: list[Path] = []
    if audio_root is not None:
        roots.append(Path(audio_root).resolve())
    if predictions_path is not None:
        roots.append(Path(predictions_path).resolve().parent)
    roots.append(REPO_ROOT)
    roots.append(Path.cwd())
    deduped: list[Path] = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)

    items: list[dict[str, Any]] = []
    for raw in metadata:
        item = copy.deepcopy(raw)
        sample_id = item["id"]
        prediction = predictions_by_id.get(sample_id)
        messages_for_audio = prediction["messages"] if prediction else item["messages"]
        audio = _build_sample_audio(messages_for_audio, deduped, sample_id)
        item["_audio"] = audio
        item["_has_prediction"] = audio.output is not None
        items.append(item)
    return items


def build_audio_map(item: dict[str, Any]) -> dict[str, str]:
    audio: SampleAudio = item["_audio"]
    audio_map: dict[str, str] = {}
    if audio.output:
        audio_map["output"] = audio.output
    for idx, path in enumerate(audio.inputs, start=1):
        audio_map[f"input{idx}"] = path
    return audio_map


def select_items(
    items: list[dict[str, Any]],
    limit: int | None = None,
    index: int | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(items))
    if index is not None:
        return [(index, items[index])]
    if limit is not None:
        return indexed[:limit]
    return indexed

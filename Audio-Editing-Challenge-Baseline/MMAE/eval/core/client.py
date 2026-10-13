from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from .prompt import SYSTEM_PROMPT, USER_PREFIX

DEFAULT_BASE_URLS: tuple[str, ...] = ()
DEFAULT_MODEL = "Qwen3Omni-Instruct"


def _audio_data_uri(path: str) -> str:
    file_path = Path(path)
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = "audio/wav"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def parse_response(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            # Model may return a bare single letter
            letter = text.strip().upper()
            if len(letter) == 1 and letter.isalpha():
                return {"choice": letter, "reason": ""}
            return {"choice": None, "reason": text}
        return json.loads(match.group(0))


class QwenOmniClient:
    def __init__(
        self,
        base_urls: tuple[str, ...] | list[str] = DEFAULT_BASE_URLS,
        model: str = DEFAULT_MODEL,
        timeout: float = 300.0,
        api_key: str | None = None,
        audio_format: str = "audio_url",
        stream: bool = False,
    ):
        self.base_urls = [url.strip().rstrip("/") for url in base_urls if url.strip()]
        if not self.base_urls or any(urlsplit(url).scheme not in {"http", "https"}
                                     or not urlsplit(url).netloc for url in self.base_urls):
            raise ValueError("Provide at least one HTTP(S) API base URL, including its /v1 path")
        if audio_format not in {"audio_url", "input_audio"}:
            raise ValueError("audio_format must be audio_url or input_audio")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.audio_format = audio_format
        self.stream = stream
        self.model = model
        self.timeout = timeout
        self._idx = 0
        self._idx_lock = asyncio.Lock()

    async def _next_url(self) -> str:
        async with self._idx_lock:
            url = self.base_urls[self._idx % len(self.base_urls)]
            self._idx += 1
            return url

    async def answer(
        self,
        question: str,
        choices: list[str],
        audio_paths: list[str],
        audio_labels: list[str],
    ) -> str:
        if len(audio_paths) != len(audio_labels):
            raise ValueError("Each audio path must have a corresponding label")
        choices_text = "\n".join(
            f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(choices)
        )
        user_text = f"{USER_PREFIX}\n\nQuestion:\n{question}\n\nChoices:\n{choices_text}"

        content: list[dict[str, Any]] = []
        for path, label in zip(audio_paths, audio_labels):
            content.append({"type": "text", "text": f"{label}: "})
            data_uri = _audio_data_uri(path)
            if self.audio_format == "audio_url":
                content.append({"type": "audio_url", "audio_url": {"url": data_uri}})
            else:
                audio_type = Path(path).suffix.lower().lstrip(".")
                if audio_type not in {"wav", "mp3"}:
                    raise ValueError("input_audio mode requires WAV or MP3 files")
                content.append({"type": "input_audio", "input_audio": {
                    "data": data_uri.split(",", 1)[1], "format": audio_type,
                }})
        content.append({"type": "text", "text": user_text})

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        if self.audio_format == "input_audio":
            payload["modalities"] = ["text"]
        if self.stream:
            payload["stream"] = True

        url = await self._next_url()
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=self.timeout, write=60.0, pool=10.0),
            trust_env=False,
        ) as http:
            if self.stream:
                chunks = []
                async with http.stream("POST", f"{url}/chat/completions",
                                       json=payload, headers=self.headers) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        event = line[5:].strip()
                        if event == "[DONE]":
                            break
                        if not event:
                            continue
                        body = json.loads(event)
                        if "error" in body:
                            raise RuntimeError("Judge API returned an error in the response stream")
                        choices = body.get("choices", [])
                        if choices:
                            text = choices[0].get("delta", {}).get("content")
                            if text:
                                chunks.append(text)
                return "".join(chunks)
            resp = await http.post(f"{url}/chat/completions", json=payload,
                                   headers=self.headers)
            resp.raise_for_status()
            body = resp.json()
        return body["choices"][0]["message"]["content"] or ""

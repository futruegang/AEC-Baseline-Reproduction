"""Offline API contract checks; never sends audio to a real service."""
import asyncio
import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from .core.client import QwenOmniClient
from .score import parse_args

class JudgeAPIContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.audio = Path(self.temp.name) / "sample.wav"
        self.audio.write_bytes(b"RIFF-test-audio-content")

    def request(self, client, handler):
        original = httpx.AsyncClient
        transport = httpx.MockTransport(handler)
        def factory(**kwargs):
            return original(transport=transport, **kwargs)
        with patch("eval.core.client.httpx.AsyncClient", side_effect=factory):
            return asyncio.run(client.answer("Compare the clips", ["yes", "no"],
                              [str(self.audio)] * 2, ["input1", "output"]))

    def test_authenticated_multiple_audio_and_legacy_settings(self):
        def handler(request):
            self.assertEqual(str(request.url), "https://judge.example/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            body = json.loads(request.content)
            self.assertEqual(body["temperature"], 0.7)
            self.assertEqual(body["max_tokens"], 4096)
            audios = [c for c in body["messages"][1]["content"] if c["type"] == "audio_url"]
            self.assertEqual(len(audios), 2)
            self.assertNotIn(str(self.audio), request.content.decode())
            self.assertEqual(base64.b64decode(audios[0]["audio_url"]["url"].split(",")[1]), self.audio.read_bytes())
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"choice":"A"}'}}]})
        self.assertEqual(self.request(QwenOmniClient(["https://judge.example/v1/"], api_key="test-key"), handler), '{"choice":"A"}')

    def test_input_audio_text_only_output(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body["modalities"], ["text"])
            audios = [c["input_audio"] for c in body["messages"][1]["content"] if c["type"] == "input_audio"]
            self.assertEqual(len(audios), 2)
            self.assertEqual(audios[0]["format"], "wav")
            self.assertEqual(base64.b64decode(audios[0]["data"]), self.audio.read_bytes())
            self.assertNotIn("authorization", request.headers)
            return httpx.Response(200, json={"choices": [{"message": {"content": "A"}}]})
        self.assertEqual(self.request(QwenOmniClient(["https://judge.example/v1"], audio_format="input_audio"), handler), "A")

    def test_streamed_response_and_usage_event(self):
        text = '\n'.join(['data: {"choices":[{"delta":{"content":"{\\"choice\\":"}}]}',
                          'data: {"choices":[{"delta":{"content":"\\"A\\"}"}}]}',
                          'data: {"choices":[],"usage":{}}', 'data: [DONE]', ''])
        def handler(request):
            self.assertTrue(json.loads(request.content)["stream"])
            return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
        self.assertEqual(self.request(QwenOmniClient(["https://judge.example/v1"], stream=True), handler), '{"choice":"A"}')

    def test_api_errors_are_not_scored_as_answers(self):
        with self.assertRaises(httpx.HTTPStatusError):
            self.request(QwenOmniClient(["https://judge.example/v1"]), lambda request: httpx.Response(401))
        with self.assertRaises(RuntimeError):
            self.request(QwenOmniClient(["https://judge.example/v1"], stream=True),
                         lambda request: httpx.Response(200, text='data: {"error":{"message":"failed"}}\n'))

    def test_environment_config_and_cli_override(self):
        with patch.dict(os.environ, {"MMAE_JUDGE_BASE_URL": "https://judge.example/v1", "MMAE_JUDGE_MODEL": "exact-model-id"}):
            with patch("sys.argv", ["score", "--predictions", "predictions.json"]):
                args = parse_args()
                self.assertEqual(args.base_urls, "https://judge.example/v1")
                self.assertEqual(args.model, "exact-model-id")
                self.assertEqual(args.api_key_env, "MMAE_JUDGE_API_KEY")
            with patch("sys.argv", ["score", "--predictions", "predictions.json", "--base-url", "https://override.example/v1"]):
                self.assertEqual(parse_args().base_urls, "https://override.example/v1")

    def test_empty_endpoints_rejected(self):
        for urls in [[], [" "], ["not-a-url"]]:
            with self.assertRaises(ValueError):
                QwenOmniClient(urls)

if __name__ == "__main__":
    unittest.main()

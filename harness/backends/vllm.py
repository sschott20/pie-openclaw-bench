"""vLLM backend driver using OpenAI-compatible API with prefix caching."""

from __future__ import annotations

import time

import aiohttp

from harness.backends.base import Backend
from harness.models import ExperimentConfig, ModularRequest, StreamingResponse


class VLLMBackend(Backend):
    """vLLM with --enable-prefix-caching as the industry-standard baseline."""

    def __init__(self):
        self._session: aiohttp.ClientSession | None = None
        self._base_url: str = ""
        self._model: str = ""

    async def setup(self, config: ExperimentConfig) -> None:
        self._base_url = config.vllm_url
        self._model = config.model
        self._session = aiohttp.ClientSession()

    async def send_request(self, request: ModularRequest) -> StreamingResponse:
        assert self._session is not None

        # Flatten modules into a single prompt
        prompt = request.flat_prompt()

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": request.max_response_tokens,
            "temperature": 0.0,
            "stream": True,
        }

        response = StreamingResponse()
        url = f"{self._base_url}/v1/chat/completions"

        t_start = time.monotonic()
        first_token_received = False

        async with self._session.post(url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.content:
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break

                import json

                chunk = json.loads(data)
                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if not first_token_received:
                        response.ttft_ms = (time.monotonic() - t_start) * 1000
                        first_token_received = True
                    response.tokens.append(content)

        response.total_latency_ms = (time.monotonic() - t_start) * 1000
        return response

    async def reset_state(self) -> None:
        """Reset vLLM prefix cache via its reset endpoint if available."""
        if self._session and self._base_url:
            try:
                async with self._session.post(
                    f"{self._base_url}/reset_prefix_cache"
                ) as _:
                    pass  # Best-effort; not all vLLM versions support this
            except Exception:
                pass

    async def teardown(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def get_server_metrics(self) -> dict:
        """Fetch vLLM metrics endpoint."""
        if not self._session or not self._base_url:
            return {}
        try:
            async with self._session.get(
                f"{self._base_url}/metrics"
            ) as resp:
                text = await resp.text()
                return {"raw_metrics": text}
        except Exception:
            return {}

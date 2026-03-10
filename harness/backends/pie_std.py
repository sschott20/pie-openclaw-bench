"""PIE std/chat backend driver — baseline without application-level caching."""

from __future__ import annotations

import json
import time

from harness.backends.base import Backend
from harness.backends.pie_common import PieConnection, Event
from harness.models import ExperimentConfig, ModularRequest, StreamingResponse


# Path to the built-in std/chat inferlet WASM (provided by PIE)
STD_CHAT_INFERLET = "std/chat"


class PIEStdBackend(Backend):
    """PIE with built-in std/chat inferlet. No application-level caching."""

    def __init__(self):
        self._conn: PieConnection | None = None
        self._model: str = ""

    async def setup(self, config: ExperimentConfig) -> None:
        self._model = config.model
        self._conn = PieConnection(config.pie_server_uri)
        await self._conn.connect()
        await self._conn.launch_inferlet(
            STD_CHAT_INFERLET,
            arguments=["--model", self._model],
        )

    async def send_request(self, request: ModularRequest) -> StreamingResponse:
        assert self._conn is not None

        # Flatten modules into single prompt (no module awareness)
        prompt = request.flat_prompt()

        # Send as chat message
        msg = json.dumps({
            "prompt": prompt,
            "max_tokens": request.max_response_tokens,
            "temperature": 0.0,
        })
        await self._conn.send(msg)

        response = StreamingResponse()
        t_start = time.monotonic()
        first_token_received = False

        # Receive streamed tokens
        async for event, data in self._conn.recv_stream():
            if event == Event.MESSAGE:
                if not first_token_received:
                    response.ttft_ms = (time.monotonic() - t_start) * 1000
                    first_token_received = True
                response.tokens.append(data)
            elif event == Event.COMPLETED:
                break
            elif event in (Event.EXCEPTION, Event.SERVER_ERROR):
                raise RuntimeError(f"PIE std error: {data}")

        response.total_latency_ms = (time.monotonic() - t_start) * 1000
        return response

    async def reset_state(self) -> None:
        """PIE std has no application-level cache to reset."""
        pass

    async def teardown(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def get_server_metrics(self) -> dict:
        return {}

"""PIE modular-cache backend driver — our contribution with per-module KV caching."""

from __future__ import annotations

import json
import time
from pathlib import Path

from harness.backends.base import Backend
from harness.backends.pie_common import PieConnection, Event
from harness.models import ExperimentConfig, ModularRequest, StreamingResponse

# Relative path to our custom inferlet WASM binary
INFERLET_WASM = Path(__file__).parent.parent.parent / "inferlet" / "target" / "wasm32-wasip2" / "release" / "modular_kv_cache.wasm"
INFERLET_MANIFEST = Path(__file__).parent.parent.parent / "inferlet" / "Pie.toml"


class PIECacheBackend(Backend):
    """PIE with our custom modular KV cache inferlet."""

    def __init__(self):
        self._conn: PieConnection | None = None
        self._model: str = ""

    async def setup(self, config: ExperimentConfig) -> None:
        self._model = config.model
        self._conn = PieConnection(config.pie_server_uri)
        await self._conn.connect()
        await self._conn.launch_inferlet(
            "modular-kv-cache",
            wasm_path=str(INFERLET_WASM),
            manifest_path=str(INFERLET_MANIFEST),
            arguments=["--model", self._model],
            detached=True,
        )

    async def send_request(self, request: ModularRequest) -> StreamingResponse:
        assert self._conn is not None

        # Send modular request as JSON (inferlet understands module boundaries)
        msg = json.dumps({
            "program_id": request.program_id,
            "turn_index": request.turn_index,
            "modules": [
                {
                    "name": m.name,
                    "content": m.content,
                    "hash": m.content_hash,
                }
                for m in request.modules
            ],
            "max_tokens": request.max_response_tokens,
        })
        await self._conn.send(msg)

        response = StreamingResponse()
        t_start = time.monotonic()
        first_token_received = False

        # Receive streamed tokens and completion metrics
        async for event, data in self._conn.recv_stream():
            if event == Event.MESSAGE:
                if data.startswith("__DONE__"):
                    # Parse completion metrics
                    metrics_json = data[8:]  # strip "__DONE__" prefix
                    metrics = json.loads(metrics_json)
                    response.cache_hits = metrics.get("cache_hits", 0)
                    response.cache_misses = metrics.get("cache_misses", 0)
                    response.tokens_saved = metrics.get("tokens_saved", 0)
                    response.tokens_computed = metrics.get("tokens_computed", 0)
                    break
                elif data.startswith("__ERROR__"):
                    raise RuntimeError(f"Inferlet error: {data[9:]}")
                else:
                    # Regular token
                    if not first_token_received:
                        response.ttft_ms = (time.monotonic() - t_start) * 1000
                        first_token_received = True
                    response.tokens.append(data)
            elif event == Event.COMPLETED:
                break
            elif event in (Event.EXCEPTION, Event.SERVER_ERROR):
                raise RuntimeError(f"PIE cache error: {data}")

        response.total_latency_ms = (time.monotonic() - t_start) * 1000
        return response

    async def reset_state(self) -> None:
        """Shutdown and relaunch inferlet to clear KV cache."""
        if self._conn:
            await self._conn.send("__SHUTDOWN__")
            await self._conn.terminate()

    async def teardown(self) -> None:
        if self._conn:
            # Signal inferlet to shut down gracefully
            try:
                await self._conn.send("__SHUTDOWN__")
            except Exception:
                pass
            await self._conn.close()
            self._conn = None

    async def get_server_metrics(self) -> dict:
        return {}

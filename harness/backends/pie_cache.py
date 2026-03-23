"""PIE modular-cache backend — one-instance-per-request with PIE store caching.

Each send_request() launches a fresh inferlet instance that:
1. Imports cached KV pages for the longest matching module prefix
2. Fills remaining modules + user prompt (no flush)
3. Generates via decode_step (all pending tokens in one forward pass)
4. Returns response + cache metrics, then exits

Cache is built lazily via separate cache_build instances (one per module layer).
Each cache_build instance does exactly one flush() on a fresh context (which is
reliable), then exports KV pages to PIE's persistent store.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from harness.backends.base import Backend
from harness.models import ExperimentConfig, ModularRequest, StreamingResponse

INFERLET_WASM = str(
    Path(__file__).parent.parent.parent
    / "inferlet"
    / "target"
    / "wasm32-wasip2"
    / "release"
    / "modular_kv_cache.wasm"
)
INFERLET_MANIFEST = str(
    Path(__file__).parent.parent.parent / "inferlet" / "Pie.toml"
)
INFERLET_NAME = "modular-kv-cache@0.8.0"

STARTUP_DRAIN_TIMEOUT = 2  # seconds to drain startup messages
RECV_TIMEOUT = 120  # seconds for generate/cache-build responses


def cache_key_for_prefix(hashes: list[str]) -> str:
    """Store key for a module hash prefix."""
    return "kv_" + "_".join(hashes)


class PIECacheBackend(Backend):
    """PIE with per-module modular KV cache, one instance per request."""

    def __init__(self):
        self._client = None
        self._installed = False
        # Track which cache keys have been built in this session
        self._cache_built: set[str] = set()

    # ── Lifecycle ────────────────────────────────────────────────

    async def setup(self, config: ExperimentConfig) -> None:
        from pie_client import PieClient

        self._client = PieClient(config.pie_server_uri)
        await self._client.__aenter__()
        await self._client.authenticate("benchmark")
        await self._client.install_program(INFERLET_WASM, INFERLET_MANIFEST)
        self._installed = True

        # Warm up PIE server forward-pass pipeline
        await self._warmup()

    async def teardown(self) -> None:
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None

    async def reset_state(self) -> None:
        """Clear cache tracking. PIE store is reset by server restart."""
        self._cache_built.clear()

    async def get_server_metrics(self) -> dict:
        return {}

    # ── Instance management ──────────────────────────────────────

    async def _launch(self) -> object:
        """Launch a fresh inferlet instance and drain startup messages."""
        inst = await self._client.launch_instance(INFERLET_NAME, arguments=[])
        try:
            while True:
                await asyncio.wait_for(inst.recv(), timeout=STARTUP_DRAIN_TIMEOUT)
        except asyncio.TimeoutError:
            pass
        return inst

    async def _recv_until(self, inst, sentinel: str) -> list[str]:
        """Receive messages from instance until one contains sentinel."""
        messages = []
        while True:
            event, data = await asyncio.wait_for(inst.recv(), timeout=RECV_TIMEOUT)
            ename = event.name if hasattr(event, "name") else str(event)
            text = data if isinstance(data, str) else data.decode() if data else ""
            if ename == "Message":
                messages.append(text)
                if sentinel in text:
                    return messages
            elif ename in ("Exception", "ServerError", "Aborted"):
                raise RuntimeError(f"PIE cache instance error: {text[:300]}")
        return messages

    # ── Warm-up ──────────────────────────────────────────────────

    async def _warmup(self) -> None:
        """Warm up PIE server with a quick forward pass."""
        inst = await self._launch()
        await inst.send(json.dumps({"mode": "warmup"}))
        await self._recv_until(inst, "__WARMUP_DONE__")
        await inst.terminate()

    # ── Cache building ───────────────────────────────────────────

    async def _build_cache_layer(
        self,
        import_key: str | None,
        module_content: str,
        export_key: str,
        is_first_module: bool,
    ) -> None:
        """Launch one cache_build instance to add a module layer."""
        inst = await self._launch()
        await inst.send(json.dumps({
            "mode": "cache_build",
            "import_key": import_key,
            "module_content": module_content,
            "export_key": export_key,
            "is_first_module": is_first_module,
        }))
        await self._recv_until(inst, "__CACHE_BUILT__")
        # Allow time for host to process the export command before cleanup
        await asyncio.sleep(0.5)
        await inst.terminate()
        self._cache_built.add(export_key)

    async def _ensure_cache(self, modules: list) -> None:
        """Build all missing per-module prefix caches for this module list."""
        hashes = [m.content_hash for m in modules]
        for i in range(len(modules)):
            key = cache_key_for_prefix(hashes[: i + 1])
            if key in self._cache_built:
                continue
            import_key = cache_key_for_prefix(hashes[:i]) if i > 0 else None
            await self._build_cache_layer(
                import_key=import_key,
                module_content=modules[i].content,
                export_key=key,
                is_first_module=(i == 0),
            )

    def _find_cache_hits(self, modules: list) -> tuple[int, str | None]:
        """Find the deepest cached prefix matching this module list."""
        hashes = [m.content_hash for m in modules]
        for depth in range(len(hashes), 0, -1):
            key = cache_key_for_prefix(hashes[:depth])
            if key in self._cache_built:
                return depth, key
        return 0, None

    # ── Request serving ──────────────────────────────────────────

    async def send_request(self, request: ModularRequest) -> StreamingResponse:
        assert self._client is not None

        # Build any missing cache layers (not counted in request timing)
        await self._ensure_cache(request.modules)

        cache_hits, import_key = self._find_cache_hits(request.modules)
        remaining = request.modules[cache_hits:]

        # Launch a fresh generate instance
        inst = await self._launch()

        msg = json.dumps({
            "mode": "generate",
            "import_key": import_key,
            "cache_hits": cache_hits,
            "modules": [{"content": m.content} for m in remaining],
            "first_module_is_system": (cache_hits == 0),
            "user_prompt": "Continue.",
            "max_tokens": request.max_response_tokens,
            "program_id": request.program_id,
            "turn_index": request.turn_index,
            "total_modules": len(request.modules),
        })

        response = StreamingResponse()
        t_start = time.monotonic()
        first_token_received = False

        await inst.send(msg)

        # Stream tokens until __DONE__ metrics
        while True:
            event, data = await asyncio.wait_for(
                inst.recv(), timeout=RECV_TIMEOUT
            )
            ename = event.name if hasattr(event, "name") else str(event)
            text = data if isinstance(data, str) else data.decode() if data else ""

            if ename == "Message":
                if text.startswith("__DONE__"):
                    metrics = json.loads(text[8:])
                    response.cache_hits = metrics.get("cache_hits", 0)
                    response.cache_misses = metrics.get("cache_misses", 0)
                    response.tokens_saved = metrics.get("tokens_saved", 0)
                    response.tokens_computed = metrics.get("tokens_computed", 0)
                    break
                elif text.startswith("__ERROR__"):
                    raise RuntimeError(f"Inferlet error: {text[9:]}")
                else:
                    if not first_token_received:
                        response.ttft_ms = (time.monotonic() - t_start) * 1000
                        first_token_received = True
                    response.tokens.append(text)
            elif ename in ("Exception", "ServerError", "Aborted"):
                raise RuntimeError(f"PIE cache generate error: {text[:300]}")
            elif ename == "Completed":
                break

        response.total_latency_ms = (time.monotonic() - t_start) * 1000

        await inst.terminate()
        return response

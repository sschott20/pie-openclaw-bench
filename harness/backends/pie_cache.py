"""PIE modular-cache backend — persistent inferlet with per-module KV caching.

A single long-running inferlet instance handles all cache-build and generate
requests. All KV state stays within one instance using intra-instance
export/import — no cross-instance persistence needed.

Cache is built lazily: before each request, any missing module-prefix KV layers
are built via cache_build messages. Each cache_build uses decode_step() (not
flush()) to commit tokens to KV, so it works reliably across unlimited calls.
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
    """PIE with per-module modular KV cache via a persistent inferlet."""

    def __init__(self):
        self._client = None
        self._installed = False
        self._instance = None  # persistent inferlet instance
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

        # Launch the persistent inferlet instance
        self._instance = await self._launch_persistent()

        # Warm up PIE server forward-pass pipeline
        await self._warmup()

    async def teardown(self) -> None:
        if self._instance:
            try:
                await self._instance.send(json.dumps({"mode": "shutdown"}))
                await asyncio.wait_for(
                    self._recv_until("__SHUTDOWN__"), timeout=5
                )
            except Exception:
                pass
            try:
                await self._instance.terminate()
            except Exception:
                pass
            self._instance = None
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None

    async def reset_state(self) -> None:
        """Reset cache tracking and clear in-memory KV cache."""
        self._cache_built.clear()
        if not self._instance:
            return
        try:
            await self._instance.send(json.dumps({"mode": "clear_cache"}))
            await self._recv_until("__CACHE_CLEARED__")
        except Exception:
            pass

    async def get_server_metrics(self) -> dict:
        return {}

    # ── Instance management ──────────────────────────────────────

    async def _launch_persistent(self) -> object:
        """Launch the persistent inferlet instance and drain startup messages."""
        inst = await self._client.launch_instance(INFERLET_NAME, arguments=[])
        try:
            while True:
                await asyncio.wait_for(inst.recv(), timeout=STARTUP_DRAIN_TIMEOUT)
        except asyncio.TimeoutError:
            pass
        return inst

    async def _recv_until(self, sentinel: str) -> list[str]:
        """Receive messages from persistent instance until one contains sentinel."""
        messages = []
        while True:
            event, data = await asyncio.wait_for(
                self._instance.recv(), timeout=RECV_TIMEOUT
            )
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
        await self._instance.send(json.dumps({"mode": "warmup"}))
        await self._recv_until("__WARMUP_DONE__")

    # ── Cache building ───────────────────────────────────────────

    async def _build_cache_layer(
        self,
        import_key: str | None,
        module_content: str,
        export_key: str,
        is_first_module: bool,
    ) -> None:
        """Send a cache_build message to the persistent instance."""
        await self._instance.send(json.dumps({
            "mode": "cache_build",
            "import_key": import_key,
            "module_content": module_content,
            "export_key": export_key,
            "is_first_module": is_first_module,
        }))
        await self._recv_until("__CACHE_BUILT__")
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
        assert self._instance is not None

        # Build any missing cache layers (not counted in request timing)
        await self._ensure_cache(request.modules)

        cache_hits, import_key = self._find_cache_hits(request.modules)
        remaining = request.modules[cache_hits:]

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

        await self._instance.send(msg)

        # Stream tokens until __DONE__ metrics
        while True:
            event, data = await asyncio.wait_for(
                self._instance.recv(), timeout=RECV_TIMEOUT
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

        return response

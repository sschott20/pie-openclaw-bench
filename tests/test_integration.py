"""Integration test: PIE modular KV cache — one-instance-per-request.

Each request launches a fresh inferlet instance. Cache is built via separate
cache_build instances (one per module layer), then generation instances import
cached KV and generate.

Run with: python tests/test_integration.py

Requires PIE server to be running:
  cd /home/gc635/Documents/pie/pie && uv run pie serve --no-auth -m
"""

from __future__ import annotations

import asyncio
import json
import time

INFERLET_WASM = "/home/gc635/Documents/pie_openclaw/inferlet/target/wasm32-wasip2/release/modular_kv_cache.wasm"
INFERLET_MANIFEST = "/home/gc635/Documents/pie_openclaw/inferlet/Pie.toml"
INFERLET_NAME = "modular-kv-cache@0.8.0"

PIE_SERVER_URI = "ws://127.0.0.1:8080"
RECV_TIMEOUT = 60


def cache_key_for_prefix(hashes: list[str]) -> str:
    """Generate a store key for a module hash prefix."""
    return "kv_" + "_".join(hashes)


async def drain_startup(instance):
    """Drain startup messages from a fresh instance."""
    try:
        while True:
            await asyncio.wait_for(instance.recv(), timeout=2)
    except asyncio.TimeoutError:
        pass


async def recv_until(instance, sentinel: str, timeout: float = RECV_TIMEOUT) -> list[str]:
    """Receive messages until one contains sentinel. Returns all messages."""
    messages = []
    while True:
        event, data = await asyncio.wait_for(instance.recv(), timeout=timeout)
        ename = event.name if hasattr(event, "name") else str(event)
        text = data if isinstance(data, str) else data.decode() if data else ""

        if ename == "Message":
            messages.append(text)
            if sentinel in text:
                return messages
        elif ename in ("Stdout", "Stderr"):
            print(f"  [{ename}] {text.rstrip()}")
        elif ename in ("Exception", "ServerError", "Aborted", "Completed"):
            raise RuntimeError(f"[{ename}] {text[:300]}")

    return messages


async def recv_response(instance) -> tuple[str, dict | None]:
    """Receive generated tokens + metrics from a generate instance."""
    tokens = []
    metrics = None

    while True:
        event, data = await asyncio.wait_for(instance.recv(), timeout=RECV_TIMEOUT)
        ename = event.name if hasattr(event, "name") else str(event)
        text = data if isinstance(data, str) else data.decode() if data else ""

        if ename == "Message":
            if text.startswith("__DONE__"):
                metrics = json.loads(text[8:])
                break
            elif text.startswith("__ERROR__"):
                print(f"  ERROR: {text}")
                break
            else:
                tokens.append(text)
        elif ename in ("Stdout", "Stderr"):
            print(f"  [{ename}] {text.rstrip()}")
        elif ename in ("Exception", "ServerError", "Aborted", "Completed"):
            print(f"  [{ename}] {text[:300]}")
            break

    return "".join(tokens), metrics


async def run_test():
    from pie_client import PieClient

    print("=" * 60)
    print("Integration Test: Modular KV Cache (one-shot architecture)")
    print("=" * 60)

    async with PieClient(PIE_SERVER_URI) as client:
        await client.authenticate("benchmark")

        print("\nInstalling inferlet...")
        await client.install_program(INFERLET_WASM, INFERLET_MANIFEST)
        print("  Installed.")

        # ── Warm-up ──────────────────────────────────────────────
        print("\nWarming up PIE server...")
        inst = await client.launch_instance(INFERLET_NAME, arguments=[])
        await drain_startup(inst)
        await inst.send(json.dumps({"mode": "warmup"}))
        msgs = await recv_until(inst, "__WARMUP_DONE__")
        for m in msgs:
            if not m.startswith("__"):
                print(f"  [warmup] {m.rstrip()}")
        await inst.terminate()
        print("  Server warmed up.")

        # ── Define test modules ──────────────────────────────────
        mod_core = {"content": "You are a helpful assistant. Be concise.", "hash": "hash_core_v1"}
        mod_user1 = {"content": "What is 2+2? Answer in one word.", "hash": "hash_user_v1"}
        mod_user2 = {"content": "What is 3+3? Answer in one word.", "hash": "hash_user_v2"}
        mod_skill = {"content": "You are now in code review mode.", "hash": "hash_skill_review"}
        mod_user3 = {"content": "Review this: print('hello')", "hash": "hash_user_v3"}

        # ── Build cache for module prefixes ──────────────────────
        # Build cache for [core] prefix
        print("\n" + "-" * 60)
        print("Building cache: [core]")
        print("-" * 60)
        inst = await client.launch_instance(INFERLET_NAME, arguments=[])
        await drain_startup(inst)
        await inst.send(json.dumps({
            "mode": "cache_build",
            "import_key": None,
            "module_content": mod_core["content"],
            "export_key": cache_key_for_prefix([mod_core["hash"]]),
            "is_first_module": True,
        }))
        msgs = await recv_until(inst, "__CACHE_BUILT__")
        for m in msgs:
            print(f"  {m.rstrip()}")
        await inst.terminate()
        print("  Cache built for [core]")

        # ── Test 1: Generate with no cache (cold start baseline) ─
        print("\n" + "=" * 60)
        print("Test 1: Cold start — no cache import (2 modules)")
        print("=" * 60)

        inst = await client.launch_instance(INFERLET_NAME, arguments=[])
        await drain_startup(inst)
        t = time.monotonic()
        await inst.send(json.dumps({
            "mode": "generate",
            "import_key": None,
            "cache_hits": 0,
            "modules": [
                {"content": mod_core["content"]},
                {"content": mod_user1["content"]},
            ],
            "first_module_is_system": True,
            "user_prompt": "Continue.",
            "max_tokens": 20,
            "program_id": "test_p1",
            "turn_index": 0,
            "total_modules": 2,
        }))
        text, metrics = await recv_response(inst)
        elapsed = (time.monotonic() - t) * 1000
        await inst.terminate()

        print(f"  Response: {text[:100]}")
        print(f"  Elapsed: {elapsed:.0f}ms")
        if metrics:
            print(f"  Cache: {metrics['cache_hits']} hits, {metrics['cache_misses']} misses")
            assert metrics["cache_hits"] == 0
            assert metrics["cache_misses"] == 2
            print("  PASS")
        else:
            print("  FAIL: No metrics"); return

        # ── Test 2: Generate with cached [core] prefix ───────────
        print("\n" + "=" * 60)
        print("Test 2: Warm cache — import [core], fill [user2] (expect 1 hit)")
        print("=" * 60)

        inst = await client.launch_instance(INFERLET_NAME, arguments=[])
        await drain_startup(inst)
        t = time.monotonic()
        await inst.send(json.dumps({
            "mode": "generate",
            "import_key": cache_key_for_prefix([mod_core["hash"]]),
            "cache_hits": 1,
            "modules": [
                {"content": mod_user2["content"]},
            ],
            "first_module_is_system": False,
            "user_prompt": "Continue.",
            "max_tokens": 20,
            "program_id": "test_p1",
            "turn_index": 1,
            "total_modules": 2,
        }))
        text, metrics = await recv_response(inst)
        elapsed = (time.monotonic() - t) * 1000
        await inst.terminate()

        print(f"  Response: {text[:100]}")
        print(f"  Elapsed: {elapsed:.0f}ms")
        if metrics:
            print(f"  Cache: {metrics['cache_hits']} hits, {metrics['cache_misses']} misses")
            assert metrics["cache_hits"] == 1, f"Expected 1 hit, got {metrics['cache_hits']}"
            assert metrics["cache_misses"] == 1, f"Expected 1 miss, got {metrics['cache_misses']}"
            print("  PASS")
        else:
            print("  FAIL: No metrics"); return

        # ── Test 3: Skill switch — same [core], new [skill, user3] ─
        print("\n" + "=" * 60)
        print("Test 3: Skill switch — import [core], fill [skill, user3] (1 hit, 2 miss)")
        print("=" * 60)

        inst = await client.launch_instance(INFERLET_NAME, arguments=[])
        await drain_startup(inst)
        t = time.monotonic()
        await inst.send(json.dumps({
            "mode": "generate",
            "import_key": cache_key_for_prefix([mod_core["hash"]]),
            "cache_hits": 1,
            "modules": [
                {"content": mod_skill["content"]},
                {"content": mod_user3["content"]},
            ],
            "first_module_is_system": False,
            "user_prompt": "Continue.",
            "max_tokens": 30,
            "program_id": "test_p1",
            "turn_index": 2,
            "total_modules": 3,
        }))
        text, metrics = await recv_response(inst)
        elapsed = (time.monotonic() - t) * 1000
        await inst.terminate()

        print(f"  Response: {text[:100]}")
        print(f"  Elapsed: {elapsed:.0f}ms")
        if metrics:
            print(f"  Cache: {metrics['cache_hits']} hits, {metrics['cache_misses']} misses")
            assert metrics["cache_hits"] == 1, f"Expected 1 hit, got {metrics['cache_hits']}"
            assert metrics["cache_misses"] == 2, f"Expected 2 misses, got {metrics['cache_misses']}"
            print("  PASS")
        else:
            print("  FAIL: No metrics"); return

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_test())

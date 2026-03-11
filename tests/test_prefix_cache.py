"""Diagnostic: test which flush/fork/generate patterns work in server mode.

Run with: python tests/test_prefix_cache.py
Requires PIE server: cd /home/gc635/Documents/pie/pie && uv run pie serve --no-auth
"""

from __future__ import annotations

import asyncio

INFERLET_WASM = "/home/gc635/Documents/pie_openclaw/inferlet-test/target/wasm32-wasip2/release/test_fill_flush.wasm"
INFERLET_MANIFEST = "/home/gc635/Documents/pie_openclaw/inferlet-test/Pie.toml"
INFERLET_NAME = "test-fill-flush@0.8.0"
PIE_SERVER_URI = "ws://127.0.0.1:8080"

RECV_TIMEOUT = 30


async def drain_until_done(instance):
    """Receive all messages until __ALL_DONE__ or error."""
    while True:
        event, data = await asyncio.wait_for(instance.recv(), timeout=RECV_TIMEOUT)
        ename = event.name if hasattr(event, "name") else str(event)
        text = data if isinstance(data, str) else data.decode() if data else ""

        if ename == "Message":
            print(f"  {text.rstrip()}")
            if text.strip() == "__ALL_DONE__":
                return True
        elif ename in ("Stdout", "Stderr"):
            print(f"  [{ename}] {text.rstrip()}")
        elif ename in ("Completed", "Exception", "ServerError", "Aborted"):
            print(f"  [{ename}] {text[:300]}")
            return False


async def run_test():
    from pie_client import PieClient

    print("=" * 60)
    print("Flush/Fork/Generate Pattern Diagnostic")
    print("=" * 60)

    async with PieClient(PIE_SERVER_URI) as client:
        await client.authenticate("benchmark")

        print("\nInstalling diagnostic inferlet...")
        await client.install_program(INFERLET_WASM, INFERLET_MANIFEST)
        print("  Installed.")

        print("Launching instance...")
        instance = await client.launch_instance(INFERLET_NAME, arguments=[])
        print(f"  Instance: {instance.instance_id}")

        # Drain startup
        try:
            while True:
                await asyncio.wait_for(instance.recv(), timeout=2)
        except asyncio.TimeoutError:
            pass

        print("\nSending test trigger...")
        await instance.send("run")
        ok = await drain_until_done(instance)
        print(f"\nResult: {'ALL PASSED' if ok else 'FAILED (see output above)'}")

        await instance.send("__SHUTDOWN__")
        await instance.terminate()


if __name__ == "__main__":
    asyncio.run(run_test())

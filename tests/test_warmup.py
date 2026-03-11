"""Diagnostic: test if server warm-up affects fork+generate.

Run with: python tests/test_warmup.py [cold|warm|3pages]
  cold   = run fork test directly (no warm-up)
  warm   = run simple baseline first, then fork test
  3pages = no forks, just lots of tokens spanning 3+ pages
"""

from __future__ import annotations

import asyncio
import sys

INFERLET_WASM = "/home/gc635/Documents/pie_openclaw/inferlet-test/target/wasm32-wasip2/release/test_fill_flush.wasm"
INFERLET_MANIFEST = "/home/gc635/Documents/pie_openclaw/inferlet-test/Pie.toml"
INFERLET_NAME = "test-fill-flush@0.8.0"
PIE_SERVER_URI = "ws://127.0.0.1:8080"

RECV_TIMEOUT = 30


async def drain_messages(instance, sentinel="__ALL_DONE__"):
    """Receive all messages until sentinel or error."""
    while True:
        event, data = await asyncio.wait_for(instance.recv(), timeout=RECV_TIMEOUT)
        ename = event.name if hasattr(event, "name") else str(event)
        text = data if isinstance(data, str) else data.decode() if data else ""

        if ename == "Message":
            print(f"  {text.rstrip()}")
            if text.strip() == sentinel or text.strip() == "__WARMUP_DONE__":
                return True
        elif ename in ("Stdout", "Stderr"):
            print(f"  [{ename}] {text.rstrip()}")
        elif ename in ("Completed", "Exception", "ServerError", "Aborted"):
            print(f"  [{ename}] {text[:300]}")
            return False


async def run_test(mode):
    from pie_client import PieClient

    print(f"Mode: {mode}")
    print("=" * 60)

    async with PieClient(PIE_SERVER_URI) as client:
        await client.authenticate("benchmark")

        print("Installing diagnostic inferlet...")
        await client.install_program(INFERLET_WASM, INFERLET_MANIFEST)

        print("Launching instance...")
        instance = await client.launch_instance(INFERLET_NAME, arguments=[])
        print(f"  Instance: {instance.instance_id}")

        # Drain startup
        try:
            while True:
                await asyncio.wait_for(instance.recv(), timeout=2)
        except asyncio.TimeoutError:
            pass

        if mode == "warm":
            print("\n--- Warm-up phase ---")
            await instance.send("warmup")
            ok = await drain_messages(instance, "__WARMUP_DONE__")
            if not ok:
                print("WARM-UP FAILED"); return

            print("\n--- Fork test (after warm-up) ---")
            await instance.send("test_fork")
            ok = await drain_messages(instance)
            print(f"\nResult: {'PASSED' if ok else 'FAILED'}")

        elif mode == "cold":
            print("\n--- Cold fork test (no warm-up) ---")
            await instance.send("test_cold")
            ok = await drain_messages(instance)
            print(f"\nResult: {'PASSED' if ok else 'FAILED'}")

        elif mode == "3pages":
            print("\n--- 3-page test (no forks) ---")
            await instance.send("test_3pages")
            ok = await drain_messages(instance)
            print(f"\nResult: {'PASSED' if ok else 'FAILED'}")

        await instance.send("__SHUTDOWN__")
        await instance.terminate()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cold"
    asyncio.run(run_test(mode))

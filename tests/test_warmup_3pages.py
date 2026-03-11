"""Test: warm-up then 3-page test."""
from __future__ import annotations
import asyncio

WASM = "/home/gc635/Documents/pie_openclaw/inferlet-test/target/wasm32-wasip2/release/test_fill_flush.wasm"
MANIFEST = "/home/gc635/Documents/pie_openclaw/inferlet-test/Pie.toml"
NAME = "test-fill-flush@0.8.0"
URI = "ws://127.0.0.1:8080"

async def drain(instance, sentinel):
    while True:
        event, data = await asyncio.wait_for(instance.recv(), timeout=30)
        ename = event.name if hasattr(event, "name") else str(event)
        text = data if isinstance(data, str) else data.decode() if data else ""
        if ename == "Message":
            print(f"  {text.rstrip()}")
            if sentinel in text:
                return True
        elif ename in ("Stderr",):
            print(f"  [{ename}] {text.rstrip()}")
        elif ename in ("Exception", "ServerError", "Aborted"):
            print(f"  [{ename}] {text[:300]}")
            return False

async def run():
    from pie_client import PieClient
    async with PieClient(URI) as client:
        await client.authenticate("benchmark")
        await client.install_program(WASM, MANIFEST)
        instance = await client.launch_instance(NAME, arguments=[])
        try:
            while True: await asyncio.wait_for(instance.recv(), timeout=2)
        except asyncio.TimeoutError: pass

        print("--- Warm-up ---")
        await instance.send("warmup")
        ok = await drain(instance, "__WARMUP_DONE__")
        if not ok:
            print("WARM-UP FAILED"); return

        print("\n--- 3-page test (after warm-up) ---")
        await instance.send("test_3pages")
        ok = await drain(instance, "__ALL_DONE__")
        print(f"\nResult: {'PASSED' if ok else 'FAILED'}")

        await instance.send("__SHUTDOWN__")
        await instance.terminate()

if __name__ == "__main__":
    asyncio.run(run())

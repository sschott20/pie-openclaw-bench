"""Test: can the modular-kv-cache binary do basic forward passes?"""
from __future__ import annotations
import asyncio

WASM = "/home/gc635/Documents/pie_openclaw/inferlet/target/wasm32-wasip2/release/modular_kv_cache.wasm"
MANIFEST = "/home/gc635/Documents/pie_openclaw/inferlet/Pie.toml"
NAME = "modular-kv-cache@0.8.0"
URI = "ws://127.0.0.1:8080"

async def run():
    from pie_client import PieClient
    async with PieClient(URI) as client:
        await client.authenticate("benchmark")
        await client.install_program(WASM, MANIFEST)
        instance = await client.launch_instance(NAME, arguments=[])
        print(f"Instance: {instance.instance_id}")

        # Drain startup
        try:
            while True:
                await asyncio.wait_for(instance.recv(), timeout=2)
        except asyncio.TimeoutError:
            pass

        print("Sending warmup...")
        await instance.send("warmup")
        while True:
            event, data = await asyncio.wait_for(instance.recv(), timeout=30)
            ename = event.name if hasattr(event, "name") else str(event)
            text = data if isinstance(data, str) else data.decode() if data else ""
            if ename == "Message":
                print(f"  {text.rstrip()}")
                if "__WARMUP_DONE__" in text:
                    print("\nPASSED — modular-kv-cache binary can do forward passes")
                    break
            elif ename in ("Stderr",):
                print(f"  [{ename}] {text.rstrip()}")
            elif ename in ("Exception", "ServerError"):
                print(f"  [{ename}] {text[:300]}")
                print("\nFAILED — binary cannot do forward passes")
                break

        await instance.send("__SHUTDOWN__")
        await instance.terminate()

if __name__ == "__main__":
    asyncio.run(run())

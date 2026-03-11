"""Simple smoke test: submit text-completion inferlet to PIE serve and verify output.

Run with: ./venv/bin/python -I tests/test_submit_inferlets.py
Requires PIE server: cd /home/gc635/Documents/pie/pie && uv run pie serve --no-auth
"""

from __future__ import annotations

import asyncio
import sys

PIE_SERVER_URI = "ws://127.0.0.1:8080"
RECV_TIMEOUT = 30

TEXT_COMPLETION_WASM = "/home/gc635/Documents/pie/std/text-completion/target/wasm32-wasip2/release/text_completion.wasm"
TEXT_COMPLETION_MANIFEST = "/home/gc635/Documents/pie/std/text-completion/Pie.toml"
TEXT_COMPLETION_NAME = "text-completion@0.1.0"


async def main():
    from pie_client import PieClient

    print(f"Connecting to {PIE_SERVER_URI}...")
    async with PieClient(PIE_SERVER_URI) as client:
        await client.authenticate("benchmark")
        print("Authenticated.")

        print("Installing text-completion inferlet...")
        await client.install_program(TEXT_COMPLETION_WASM, TEXT_COMPLETION_MANIFEST)
        print("Installed.")

        instance = await client.launch_instance(
            TEXT_COMPLETION_NAME,
            arguments=["-p", "What is 2+2? Answer briefly.", "-n", "20"],
        )
        print(f"Launched instance: {instance.instance_id}")

        output = []
        while True:
            event, data = await asyncio.wait_for(instance.recv(), timeout=RECV_TIMEOUT)
            ename = event.name if hasattr(event, "name") else str(event)
            text = data if isinstance(data, str) else data.decode() if data else ""

            print(f"  [{ename}] {repr(text[:200])}")
            if text:
                output.append(text)
            if ename in ("Completed", "Aborted", "Exception", "ServerError"):
                break

        full = "".join(output)
        print(f"\nOutput: {full}")
        print("PASS" if full.strip() else "FAIL — no output received")
        sys.exit(0 if full.strip() else 1)


if __name__ == "__main__":
    asyncio.run(main())

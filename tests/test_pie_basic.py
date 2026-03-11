"""Minimal test: PIE server + text-completion inferlet via messaging.

Verifies that receive()/send() works in server mode at all.

Requires PIE server running:
  cd /home/gc635/Documents/pie/pie && uv run pie serve --no-auth
"""

import asyncio
import json


async def run_test():
    from pie_client import PieClient

    async with PieClient("ws://127.0.0.1:8080") as client:
        await client.authenticate("benchmark")

        # Use PIE's built-in text-completion (one-shot, no messaging)
        print("=== Test 1: text-completion via launch args (no messaging) ===")
        instance = await client.launch_instance(
            "text-completion",
            arguments=["-p", "What is 2+2? Answer briefly.", "-n", "10"],
        )
        print(f"  Instance: {instance.instance_id}")

        while True:
            event, data = await asyncio.wait_for(instance.recv(), timeout=30)
            ename = event.name if hasattr(event, "name") else str(event)
            text = data if isinstance(data, str) else data.decode() if data else ""
            print(f"  [{ename}] {repr(text[:200])}")
            if ename in ("Completed", "Aborted", "Exception", "ServerError"):
                break

        # Test 2: Try sending a message to a launched instance
        print("\n=== Test 2: launch instance and send message ===")
        instance2 = await client.launch_instance(
            "text-completion",
            arguments=["-p", "Say hello", "-n", "5"],
        )
        print(f"  Instance: {instance2.instance_id}")

        # Try sending - this should either work or be ignored
        try:
            await instance2.send("test message")
            print("  send() succeeded (no error)")
        except Exception as e:
            print(f"  send() failed: {e}")

        while True:
            event, data = await asyncio.wait_for(instance2.recv(), timeout=30)
            ename = event.name if hasattr(event, "name") else str(event)
            text = data if isinstance(data, str) else data.decode() if data else ""
            print(f"  [{ename}] {repr(text[:200])}")
            if ename in ("Completed", "Aborted", "Exception", "ServerError"):
                break

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(run_test())

"""Shared PIE connection helper wrapping the PIE Python client."""

from __future__ import annotations

from enum import Enum
from typing import AsyncIterator


class Event(str, Enum):
    """Events received from a PIE inferlet instance."""

    MESSAGE = "message"
    COMPLETED = "completed"
    ABORTED = "aborted"
    EXCEPTION = "exception"
    SERVER_ERROR = "server_error"
    OUT_OF_RESOURCES = "out_of_resources"
    BLOB = "blob"
    STDOUT = "stdout"
    STDERR = "stderr"


class PieConnection:
    """Async wrapper around the PIE Python client for inferlet communication.

    This abstraction handles:
    - Connecting to the PIE server via WebSocket
    - Launching/attaching to inferlet instances
    - Sending/receiving messages with event type parsing
    """

    def __init__(self, server_uri: str):
        self._server_uri = server_uri
        self._client = None
        self._instance = None

    async def connect(self) -> None:
        """Connect to the PIE server."""
        try:
            from pie_client import PieClient
        except ImportError:
            raise ImportError(
                "pie_client not installed. Install the PIE Python client: "
                "pip install pie-client (or install from PIE repo)"
            )

        self._client = PieClient(self._server_uri)
        await self._client.__aenter__()
        await self._client.authenticate("benchmark")

    async def launch_inferlet(
        self,
        inferlet_name: str,
        wasm_path: str | None = None,
        manifest_path: str | None = None,
        arguments: list[str] | None = None,
        detached: bool = False,
    ) -> None:
        """Launch an inferlet instance."""
        assert self._client is not None

        # Install if custom WASM provided
        if wasm_path and manifest_path:
            if not await self._client.program_exists(
                inferlet_name, wasm_path, manifest_path
            ):
                await self._client.install_program(wasm_path, manifest_path)

        self._instance = await self._client.launch_instance(
            inferlet_name,
            arguments=arguments or [],
            detached=detached,
        )

    async def send(self, message: str) -> None:
        """Send a string message to the inferlet."""
        assert self._instance is not None
        await self._instance.send(message)

    async def recv(self) -> tuple[Event, str]:
        """Receive a single event from the inferlet."""
        assert self._instance is not None
        event, data = await self._instance.recv()
        # Map PIE client event types to our Event enum
        event_name = event.name.lower() if hasattr(event, "name") else str(event).lower()
        try:
            return Event(event_name), data if isinstance(data, str) else data.decode()
        except ValueError:
            return Event.MESSAGE, str(data)

    async def recv_stream(self) -> AsyncIterator[tuple[Event, str]]:
        """Yield events until COMPLETED or error."""
        while True:
            event, data = await self.recv()
            yield event, data
            if event in (Event.COMPLETED, Event.ABORTED, Event.EXCEPTION, Event.SERVER_ERROR):
                break

    async def terminate(self) -> None:
        """Terminate the running inferlet instance."""
        if self._instance:
            await self._instance.terminate()
            self._instance = None

    async def close(self) -> None:
        """Close the connection."""
        await self.terminate()
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None

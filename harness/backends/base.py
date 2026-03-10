"""Abstract base class for all LLM serving backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from harness.models import ExperimentConfig, ModularRequest, StreamingResponse


class Backend(ABC):
    """Interface that all backends (vLLM, PIE-std, PIE-cache) implement."""

    @abstractmethod
    async def setup(self, config: ExperimentConfig) -> None:
        """Initialize connection to the backend service."""

    @abstractmethod
    async def send_request(self, request: ModularRequest) -> StreamingResponse:
        """Send a modular request and return the streaming response with timing."""

    @abstractmethod
    async def reset_state(self) -> None:
        """Clear caches between independent experiment runs."""

    @abstractmethod
    async def teardown(self) -> None:
        """Clean up resources."""

    @abstractmethod
    async def get_server_metrics(self) -> dict:
        """Retrieve backend-specific server metrics (GPU util, memory, etc.)."""

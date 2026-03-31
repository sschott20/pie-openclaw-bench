"""Abstract base class for workload generators."""

from __future__ import annotations

from abc import ABC, abstractmethod

from harness.models import ModularRequest
from harness.prompts import PromptSizes


class WorkloadGenerator(ABC):
    """Generates sequences of ModularRequests for a specific scenario."""

    def __init__(self, sizes: PromptSizes | None = None):
        self.sizes = sizes or PromptSizes()

    @abstractmethod
    def generate_program(self, program_id: str) -> list[ModularRequest]:
        """Generate all turns for a complete agent session.

        Returns an ordered list of ModularRequests representing each turn
        in the session, with the correct module composition and changes.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Scenario name for identification in results."""

"""Heartbeat workload generator.

Periodic requests with identical system prompts and long inter-request intervals.
Tests cache eviction immunity — modular caching retains modules explicitly
while prefix caching goes cold between beats.
"""

from __future__ import annotations

from harness.models import ModularRequest
from harness.prompts.synthetic import (
    PromptSizes,
    make_core_instructions,
    make_tool_schemas,
    make_skill,
    make_memory,
    make_heartbeat_prompt,
    make_updated_memory,
)
from harness.workloads.base import WorkloadGenerator


class HeartbeatWorkload(WorkloadGenerator):
    """Heartbeat: periodic identical requests with occasional memory updates."""

    def __init__(
        self,
        num_beats: int = 10,
        memory_update_every: int = 10,
        sizes: PromptSizes | None = None,
    ):
        super().__init__(sizes)
        self.num_beats = num_beats
        self.memory_update_every = memory_update_every

    @property
    def name(self) -> str:
        return "heartbeat"

    def generate_program(self, program_id: str) -> list[ModularRequest]:
        # Static modules
        core = make_core_instructions(self.sizes.core_tokens)
        tools = make_tool_schemas(self.sizes.tool_tokens)
        skill = make_skill("debugging", self.sizes.skill_tokens)
        heartbeat = make_heartbeat_prompt()
        base_memory = make_memory(self.sizes.memory_tokens)

        requests = []

        for beat in range(self.num_beats):
            # Occasionally update memory (every N beats)
            if beat > 0 and beat % self.memory_update_every == 0:
                memory = make_updated_memory(
                    self.sizes.memory_tokens, update_index=beat
                )
            else:
                memory = base_memory

            modules = [core, tools, skill, memory, heartbeat]

            requests.append(ModularRequest(
                program_id=program_id,
                turn_index=beat,
                modules=modules,
                max_response_tokens=self.sizes.response_tokens,
            ))

        return requests

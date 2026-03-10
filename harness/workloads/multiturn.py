"""Multi-turn conversation workload generator.

System prompt prefix is constant while conversation history grows linearly.
Tests whether modular caching helps when growing suffix dominates.
"""

from __future__ import annotations

from harness.models import ModularRequest, PromptModule
from harness.prompts.synthetic import (
    PromptSizes,
    make_core_instructions,
    make_tool_schemas,
    make_skill,
    make_memory,
    CONVERSATION_TURNS,
)
from harness.workloads.base import WorkloadGenerator


class MultiTurnWorkload(WorkloadGenerator):
    """Multi-turn conversation: static system prefix + growing history."""

    def __init__(
        self,
        num_turns: int = 20,
        sizes: PromptSizes | None = None,
    ):
        super().__init__(sizes)
        self.num_turns = num_turns

    @property
    def name(self) -> str:
        return "multiturn"

    def generate_program(self, program_id: str) -> list[ModularRequest]:
        # Static modules
        core = make_core_instructions(self.sizes.core_tokens)
        tools = make_tool_schemas(self.sizes.tool_tokens)
        skill = make_skill("code_review", self.sizes.skill_tokens)
        memory = make_memory(self.sizes.memory_tokens)

        requests = []
        history_entries: list[str] = []

        for turn in range(self.num_turns):
            # Build conversation history as a single growing module
            role = "user" if turn % 2 == 0 else "assistant"
            base_msg = CONVERSATION_TURNS[turn % len(CONVERSATION_TURNS)][1]
            history_entries.append(f"**{role}** (turn {turn}): {base_msg}")

            history_content = "## Conversation History\n\n" + "\n\n".join(history_entries)
            history_module = PromptModule(
                name="conversation_history",
                content=history_content,
            )

            modules = [core, tools, skill, memory, history_module]

            requests.append(ModularRequest(
                program_id=program_id,
                turn_index=turn,
                modules=modules,
                max_response_tokens=self.sizes.response_tokens,
            ))

        return requests

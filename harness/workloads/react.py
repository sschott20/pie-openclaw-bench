"""ReAct tool loop workload generator.

Generates sequences where stable system modules (core, tools, skills, memory)
are present on every turn, with growing tool call/result history. Tests
cache retention across tool execution pauses.
"""

from __future__ import annotations

from harness.models import ModularRequest, PromptModule
from harness.prompts import PromptSizes, get_module
from harness.workloads.base import WorkloadGenerator


class ReactWorkload(WorkloadGenerator):
    """ReAct tool loop: stable prefix + growing tool history."""

    def __init__(
        self,
        num_turns: int = 10,
        sizes: PromptSizes | None = None,
    ):
        super().__init__(sizes)
        self.num_turns = num_turns

    @property
    def name(self) -> str:
        return "react"

    def generate_program(self, program_id: str) -> list[ModularRequest]:
        p = get_module(self.sizes.prompt_source)
        # Static modules (same every turn)
        core = p.make_core_instructions(self.sizes.core_tokens)
        tools = p.make_tool_schemas(self.sizes.tool_tokens)
        skill = p.make_skill("code_review", self.sizes.skill_tokens)
        memory = p.make_memory(self.sizes.memory_tokens)
        user_msg = p.make_user_message("Fix the failing test in src/auth/login.py")

        requests = []
        tool_history: list[PromptModule] = []

        for turn in range(self.num_turns):
            # Build module list: static prefix + accumulated tool history + user msg
            modules = [core, tools, skill, memory]
            modules.extend(tool_history)
            modules.append(user_msg)

            requests.append(ModularRequest(
                program_id=program_id,
                turn_index=turn,
                modules=modules,
                max_response_tokens=self.sizes.response_tokens,
            ))

            # Simulate tool call + result for next turn
            tool_call = PromptModule(
                name=f"tool_call_{turn}",
                content=f"Assistant called bash: pytest tests/auth/ -v (turn {turn})",
            )
            tool_result = p.make_tool_result("bash")
            # Give each result a unique name so they're distinct modules
            tool_result = PromptModule(
                name=f"tool_result_{turn}",
                content=tool_result.content + f"\n(turn {turn})",
            )
            tool_history.extend([tool_call, tool_result])

        return requests

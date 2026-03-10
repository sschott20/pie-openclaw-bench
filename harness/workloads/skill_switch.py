"""Skill switch workload generator.

Mid-session module changes that break prefix caching. This is the strongest
scenario for modular caching — when skills change, prefix caching loses
everything after the changed module, while modular caching only re-encodes
the changed skill.
"""

from __future__ import annotations

from harness.models import ModularRequest
from harness.prompts.synthetic import (
    PromptSizes,
    make_core_instructions,
    make_tool_schemas,
    make_skill,
    make_memory,
    make_user_message,
)
from harness.workloads.base import WorkloadGenerator

# Skill rotation pattern: phases of different active skills
SKILL_PHASES = [
    ["code_review", "debugging"],
    ["web_search_research", "architecture_planning"],
    ["code_review", "test_writing"],
]


class SkillSwitchWorkload(WorkloadGenerator):
    """Skill switch: mid-session skill changes that break prefix caching."""

    def __init__(
        self,
        turns_per_phase: int = 5,
        num_phases: int = 3,
        sizes: PromptSizes | None = None,
    ):
        super().__init__(sizes)
        self.turns_per_phase = turns_per_phase
        self.num_phases = min(num_phases, len(SKILL_PHASES))

    @property
    def name(self) -> str:
        return "skill_switch"

    def generate_program(self, program_id: str) -> list[ModularRequest]:
        # Static modules (same every turn)
        core = make_core_instructions(self.sizes.core_tokens)
        memory = make_memory(self.sizes.memory_tokens)
        user_msg = make_user_message("Continue working on the current task.")

        requests = []
        turn = 0

        for phase_idx in range(self.num_phases):
            skill_names = SKILL_PHASES[phase_idx % len(SKILL_PHASES)]

            # Generate tool schemas appropriate for this phase
            tools = make_tool_schemas(self.sizes.tool_tokens)

            # Generate skill modules for this phase
            skills = [
                make_skill(name, self.sizes.skill_tokens // len(skill_names))
                for name in skill_names
            ]

            for _ in range(self.turns_per_phase):
                modules = [core, tools] + skills + [memory, user_msg]

                requests.append(ModularRequest(
                    program_id=program_id,
                    turn_index=turn,
                    modules=modules,
                    max_response_tokens=self.sizes.response_tokens,
                ))
                turn += 1

        return requests

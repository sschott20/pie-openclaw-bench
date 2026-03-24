"""Skill switch workload generator.

Mid-session module changes that break prefix caching. Skills rotate every turn
so that most requests hit the prefix-breaking case. Conversation history and
memory sit after the skill modules, so vLLM loses them on every switch while
modular caching reuses them independently.

Module order: [core, tools, skill, history, memory, user_msg]
"""

from __future__ import annotations

from harness.models import ModularRequest
from harness.prompts.synthetic import (
    PromptSizes,
    make_core_instructions,
    make_conversation_history,
    make_tool_schemas,
    make_skill,
    make_memory,
    make_user_message,
)
from harness.workloads.base import WorkloadGenerator

# Skills rotate round-robin every turn
SKILL_ROTATION = [
    "code_review",
    "web_search_research",
    "debugging",
    "architecture_planning",
    "test_writing",
]


class SkillSwitchWorkload(WorkloadGenerator):
    """Skill switch: per-turn skill rotation that breaks prefix caching."""

    def __init__(
        self,
        num_turns: int = 15,
        sizes: PromptSizes | None = None,
        # Keep old params accepted but ignored for backwards compat
        turns_per_phase: int | None = None,
        num_phases: int | None = None,
    ):
        super().__init__(sizes)
        self.num_turns = num_turns

    @property
    def name(self) -> str:
        return "skill_switch"

    def generate_program(self, program_id: str) -> list[ModularRequest]:
        # Static modules (same every turn)
        core = make_core_instructions(self.sizes.core_tokens)
        tools = make_tool_schemas(self.sizes.tool_tokens)
        memory = make_memory(self.sizes.memory_tokens)
        user_msg = make_user_message("Continue working on the current task.")

        # Pre-generate all skill variants
        skills = [
            make_skill(name, self.sizes.skill_tokens)
            for name in SKILL_ROTATION
        ]

        requests = []
        for turn in range(self.num_turns):
            # Rotate skill each turn
            skill = skills[turn % len(SKILL_ROTATION)]

            # Growing history (realistic: grows each turn)
            history = make_conversation_history(
                num_turns=min(turn + 1, 10),
                tokens_per_turn=self.sizes.history_tokens_per_turn,
            )

            # Order: [core, tools, SKILL, history, memory, user_msg]
            # Skill is in the middle — everything after it (history, memory,
            # user_msg) is lost by prefix caching on each switch but reusable
            # by modular caching.
            modules = [core, tools, skill, history, memory, user_msg]

            requests.append(ModularRequest(
                program_id=program_id,
                turn_index=turn,
                modules=modules,
                max_response_tokens=self.sizes.response_tokens,
            ))

        return requests

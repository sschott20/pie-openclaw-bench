"""Tests for synthetic prompt content generator."""

from harness.prompts.synthetic import (
    CHARS_PER_TOKEN,
    PromptSizes,
    make_core_instructions,
    make_tool_schemas,
    make_skill,
    make_memory,
    make_conversation_history,
    make_heartbeat_prompt,
    make_updated_memory,
    make_user_message,
)


class TestModuleGeneration:
    def test_core_instructions_default_size(self):
        m = make_core_instructions()
        # Should be approximately 500 tokens (500 * 4 = 2000 chars)
        assert abs(len(m.content) - 500 * CHARS_PER_TOKEN) < CHARS_PER_TOKEN * 10
        assert m.name == "core_instructions"

    def test_core_instructions_custom_size(self):
        m = make_core_instructions(1000)
        assert abs(len(m.content) - 1000 * CHARS_PER_TOKEN) < CHARS_PER_TOKEN * 10

    def test_tool_schemas(self):
        m = make_tool_schemas(2000)
        assert m.name == "tool_schemas"
        assert "bash" in m.content.lower()

    def test_tool_schemas_specific_tools(self):
        m = make_tool_schemas(1000, tool_names=["bash", "read"])
        assert "bash" in m.content.lower()
        assert "read" in m.content.lower()

    def test_skill(self):
        m = make_skill("code_review", 1500)
        assert m.name == "skill_code_review"
        assert "review" in m.content.lower()

    def test_memory(self):
        m = make_memory(200)
        assert m.name == "memory"
        assert "TypeScript" in m.content or "memory" in m.content.lower()

    def test_conversation_history(self):
        m = make_conversation_history(num_turns=4, tokens_per_turn=150)
        assert m.name == "conversation_history"
        assert "History" in m.content

    def test_heartbeat_prompt(self):
        m = make_heartbeat_prompt()
        assert m.name == "heartbeat_prompt"
        assert len(m.content) > 0

    def test_updated_memory_differs_from_base(self):
        base = make_memory(200)
        updated = make_updated_memory(200, update_index=0)
        assert base.name == updated.name == "memory"
        assert base.content_hash != updated.content_hash

    def test_user_message(self):
        m = make_user_message("test prompt")
        assert m.name == "user_message"
        assert m.content == "test prompt"


class TestHashConsistency:
    def test_same_content_same_hash(self):
        m1 = make_core_instructions(500)
        m2 = make_core_instructions(500)
        assert m1.content_hash == m2.content_hash

    def test_different_size_different_hash(self):
        m1 = make_core_instructions(500)
        m2 = make_core_instructions(1000)
        assert m1.content_hash != m2.content_hash

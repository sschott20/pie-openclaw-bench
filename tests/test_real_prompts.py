"""Tests for real Claude Code prompt content and prompt source switching."""

from harness.prompts.real import (
    CHARS_PER_TOKEN,
    CONVERSATION_TURNS,
    TOOL_SCHEMA_TEMPLATES,
    SKILL_TEMPLATES,
    MEMORY_ENTRIES,
    make_core_instructions,
    make_tool_schemas,
    make_skill,
    make_memory,
    make_conversation_history,
    make_heartbeat_prompt,
    make_updated_memory,
    make_user_message,
)
from harness.prompts import PromptSizes, get_module


class TestRealContentPresent:
    """Verify real content is substantially different from synthetic placeholders."""

    def test_core_instructions_mentions_claude_code(self):
        m = make_core_instructions(3000)
        assert "Claude Code" in m.content
        assert m.name == "core_instructions"

    def test_tool_schemas_have_real_tools(self):
        m = make_tool_schemas(15000)
        assert m.name == "tool_schemas"
        # Real tools should include Bash, Read, Edit, Write, Glob, Grep
        content_lower = m.content.lower()
        for tool in ["bash", "read", "edit", "write", "glob", "grep"]:
            assert tool in content_lower, f"Missing tool: {tool}"

    def test_tool_schemas_have_json_schemas(self):
        """Real tool definitions include actual JSON parameter schemas."""
        m = make_tool_schemas(15000)
        assert '"type": "object"' in m.content
        assert '"required"' in m.content

    def test_bash_tool_has_git_instructions(self):
        """The Bash tool in real Claude Code includes git commit/PR instructions."""
        assert "bash" in TOOL_SCHEMA_TEMPLATES
        desc = TOOL_SCHEMA_TEMPLATES["bash"]["description"]
        assert "git" in desc.lower()
        assert "commit" in desc.lower()
        assert "pull request" in desc.lower()

    def test_skills_are_substantial(self):
        """Real skills have multi-phase structure."""
        for name, content in SKILL_TEMPLATES.items():
            assert len(content) > 200, f"Skill {name} too short"
            # Real skills have structured phases/steps
            assert any(
                marker in content
                for marker in ["##", "Phase", "Step", "Agent"]
            ), f"Skill {name} lacks structure"

    def test_memory_entries_have_claude_md_structure(self):
        """Real memory entries include CLAUDE.md-style project instructions."""
        combined = "\n".join(MEMORY_ENTRIES)
        assert "CLAUDE.md" in combined
        assert "```" in combined  # Code blocks
        assert "##" in combined  # Headers

    def test_conversation_turns_are_realistic(self):
        """Real conversation turns reference real development tasks."""
        assert len(CONVERSATION_TURNS) > 10
        combined = " ".join(msg for _, msg in CONVERSATION_TURNS)
        # Should reference real dev activities
        assert "test" in combined.lower()
        assert "fix" in combined.lower()

    def test_all_tools_present(self):
        """All expected real tools are defined."""
        expected = {
            "bash", "read", "edit", "write", "glob", "grep",
            "agent", "web_search", "web_fetch", "todowrite", "multi_edit",
        }
        assert expected.issubset(set(TOOL_SCHEMA_TEMPLATES.keys()))

    def test_all_skills_present(self):
        """All expected real skills are defined."""
        expected = {
            "code_review", "debugging", "verification",
            "web_search_research", "architecture_planning", "test_writing",
        }
        assert expected.issubset(set(SKILL_TEMPLATES.keys()))


class TestRealFactoryFunctions:
    """Factory functions produce valid PromptModules."""

    def test_core_instructions_size(self):
        m = make_core_instructions(3000)
        assert abs(len(m.content) - 3000 * CHARS_PER_TOKEN) < CHARS_PER_TOKEN * 20

    def test_tool_schemas_size(self):
        m = make_tool_schemas(15000)
        assert abs(len(m.content) - 15000 * CHARS_PER_TOKEN) < CHARS_PER_TOKEN * 20

    def test_specific_tools_only(self):
        m = make_tool_schemas(5000, tool_names=["bash", "read"])
        assert "bash" in m.content.lower()
        assert "read" in m.content.lower()

    def test_skill_module(self):
        m = make_skill("debugging", 2000)
        assert m.name == "skill_debugging"
        assert "debug" in m.content.lower()

    def test_memory_module(self):
        m = make_memory(3500)
        assert m.name == "memory"

    def test_conversation_history(self):
        m = make_conversation_history(num_turns=4, tokens_per_turn=150)
        assert m.name == "conversation_history"

    def test_heartbeat(self):
        m = make_heartbeat_prompt()
        assert m.name == "heartbeat_prompt"

    def test_updated_memory_differs(self):
        base = make_memory(3500)
        updated = make_updated_memory(3500, update_index=0)
        assert base.content_hash != updated.content_hash

    def test_user_message(self):
        m = make_user_message("test")
        assert m.content == "test"


class TestPromptSourceSwitching:
    """get_module dispatches between real and synthetic correctly."""

    def test_get_real_module(self):
        mod = get_module("real")
        assert mod.__name__ == "harness.prompts.real"

    def test_get_synthetic_module(self):
        mod = get_module("synthetic")
        assert mod.__name__ == "harness.prompts.synthetic"

    def test_real_core_mentions_claude(self):
        mod = get_module("real")
        m = mod.make_core_instructions(3000)
        assert "Claude Code" in m.content

    def test_synthetic_core_no_claude(self):
        mod = get_module("synthetic")
        m = mod.make_core_instructions(500)
        assert "Claude Code" not in m.content

    def test_prompt_sizes_default_is_real(self):
        sizes = PromptSizes()
        assert sizes.prompt_source == "real"

    def test_prompt_sizes_switchable(self):
        sizes = PromptSizes(prompt_source="synthetic")
        assert sizes.prompt_source == "synthetic"

    def test_workload_uses_prompt_source(self):
        """Workloads respect prompt_source from PromptSizes."""
        from harness.workloads.react import ReactWorkload

        # Real prompts
        real_wl = ReactWorkload(
            num_turns=2,
            sizes=PromptSizes(prompt_source="real", core_tokens=500, tool_tokens=500, skill_tokens=500, memory_tokens=200),
        )
        real_reqs = real_wl.generate_program("test_real")
        real_core = real_reqs[0].modules[0].content

        # Synthetic prompts
        syn_wl = ReactWorkload(
            num_turns=2,
            sizes=PromptSizes(prompt_source="synthetic", core_tokens=500, tool_tokens=500, skill_tokens=500, memory_tokens=200),
        )
        syn_reqs = syn_wl.generate_program("test_syn")
        syn_core = syn_reqs[0].modules[0].content

        # They should differ (real mentions Claude Code, synthetic doesn't)
        assert real_core != syn_core


class TestRealTokenSizes:
    """Verify natural content sizes are in expected ranges."""

    def test_core_instructions_natural_size(self):
        """Core instructions should be ~3000 tokens naturally."""
        from harness.prompts.real import CORE_INSTRUCTIONS_TEMPLATE
        natural_tokens = len(CORE_INSTRUCTIONS_TEMPLATE) / CHARS_PER_TOKEN
        assert 2000 < natural_tokens < 5000, f"Got {natural_tokens} tokens"

    def test_tool_schemas_natural_size(self):
        """All tool schemas assembled should be ~4000-20000 tokens.

        Our excerpted versions are ~4500 tokens; the real Claude Code prompt
        has ~15K tokens of tools (including sandbox rules, more tools, etc.)
        but padded to target_tokens=15000 by the factory function.
        """
        import json
        text = ""
        for schema in TOOL_SCHEMA_TEMPLATES.values():
            text += schema["description"] + json.dumps(schema["parameters"])
        natural_tokens = len(text) / CHARS_PER_TOKEN
        assert 3000 < natural_tokens < 20000, f"Got {natural_tokens} tokens"

    def test_memory_natural_size(self):
        """Memory entries should be ~2000-5000 tokens."""
        text = "\n".join(MEMORY_ENTRIES)
        natural_tokens = len(text) / CHARS_PER_TOKEN
        assert 500 < natural_tokens < 6000, f"Got {natural_tokens} tokens"

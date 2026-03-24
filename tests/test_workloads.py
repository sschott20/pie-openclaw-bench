"""Tests for workload generators."""

from harness.prompts.synthetic import PromptSizes
from harness.workloads.react import ReactWorkload
from harness.workloads.multiturn import MultiTurnWorkload
from harness.workloads.skill_switch import SkillSwitchWorkload
from harness.workloads.heartbeat import HeartbeatWorkload


class TestReactWorkload:
    def test_generates_correct_num_turns(self):
        wl = ReactWorkload(num_turns=5)
        requests = wl.generate_program("test_p1")
        assert len(requests) == 5

    def test_static_modules_unchanged(self):
        wl = ReactWorkload(num_turns=5)
        requests = wl.generate_program("test_p1")
        # First 4 modules (core, tools, skill, memory) should have same hash across turns
        for i in range(1, len(requests)):
            for j in range(4):
                assert requests[i].modules[j].content_hash == requests[0].modules[j].content_hash

    def test_growing_tool_history(self):
        wl = ReactWorkload(num_turns=5)
        requests = wl.generate_program("test_p1")
        # Each turn should have more modules (tool call + result accumulate)
        for i in range(1, len(requests)):
            assert len(requests[i].modules) > len(requests[i - 1].modules)

    def test_program_id_set(self):
        wl = ReactWorkload(num_turns=3)
        requests = wl.generate_program("my_program")
        for r in requests:
            assert r.program_id == "my_program"

    def test_name(self):
        assert ReactWorkload().name == "react"


class TestMultiTurnWorkload:
    def test_generates_correct_num_turns(self):
        wl = MultiTurnWorkload(num_turns=10)
        requests = wl.generate_program("test_p1")
        assert len(requests) == 10

    def test_static_prefix_unchanged(self):
        wl = MultiTurnWorkload(num_turns=10)
        requests = wl.generate_program("test_p1")
        # First 4 modules should be stable
        for i in range(1, len(requests)):
            for j in range(4):
                assert requests[i].modules[j].content_hash == requests[0].modules[j].content_hash

    def test_history_changes_every_turn(self):
        wl = MultiTurnWorkload(num_turns=10)
        requests = wl.generate_program("test_p1")
        history_hashes = [r.modules[4].content_hash for r in requests]
        # Each turn's history should be unique (grows each turn)
        assert len(set(history_hashes)) == len(history_hashes)

    def test_name(self):
        assert MultiTurnWorkload().name == "multiturn"


class TestSkillSwitchWorkload:
    def test_generates_correct_num_turns(self):
        wl = SkillSwitchWorkload(num_turns=15)
        requests = wl.generate_program("test_p1")
        assert len(requests) == 15

    def test_skill_rotates_every_turn(self):
        wl = SkillSwitchWorkload(num_turns=6)
        requests = wl.generate_program("test_p1")

        # Each consecutive turn should have a different skill
        for i in range(len(requests) - 1):
            skill_i = [m for m in requests[i].modules if m.name.startswith("skill_")]
            skill_next = [m for m in requests[i + 1].modules if m.name.startswith("skill_")]
            assert skill_i[0].name != skill_next[0].name

    def test_skill_cycles_after_exhausting_rotation(self):
        from harness.workloads.skill_switch import SKILL_ROTATION
        n = len(SKILL_ROTATION)
        wl = SkillSwitchWorkload(num_turns=n + 1)
        requests = wl.generate_program("test_p1")
        # Turn n should wrap back to same skill as turn 0
        skill_0 = [m for m in requests[0].modules if m.name.startswith("skill_")][0]
        skill_n = [m for m in requests[n].modules if m.name.startswith("skill_")][0]
        assert skill_0.content_hash == skill_n.content_hash

    def test_core_stable_across_turns(self):
        wl = SkillSwitchWorkload(num_turns=6)
        requests = wl.generate_program("test_p1")
        core_hash = requests[0].modules[0].content_hash
        for r in requests:
            assert r.modules[0].content_hash == core_hash

    def test_has_history_module(self):
        wl = SkillSwitchWorkload(num_turns=3)
        requests = wl.generate_program("test_p1")
        for r in requests:
            history = [m for m in r.modules if m.name == "conversation_history"]
            assert len(history) == 1

    def test_memory_after_skill(self):
        """Memory sits after skill so vLLM loses it on skill switch."""
        wl = SkillSwitchWorkload(num_turns=3)
        requests = wl.generate_program("test_p1")
        for r in requests:
            names = [m.name for m in r.modules]
            skill_idx = next(i for i, n in enumerate(names) if n.startswith("skill_"))
            memory_idx = names.index("memory")
            assert memory_idx > skill_idx

    def test_name(self):
        assert SkillSwitchWorkload().name == "skill_switch"


class TestHeartbeatWorkload:
    def test_generates_correct_num_beats(self):
        wl = HeartbeatWorkload(num_beats=10)
        requests = wl.generate_program("test_p1")
        assert len(requests) == 10

    def test_identical_beats(self):
        wl = HeartbeatWorkload(num_beats=5, memory_update_every=10)
        requests = wl.generate_program("test_p1")
        # All beats should have identical module hashes (no memory update within 5 beats)
        for i in range(1, len(requests)):
            for j in range(len(requests[i].modules)):
                assert requests[i].modules[j].content_hash == requests[0].modules[j].content_hash

    def test_memory_update(self):
        wl = HeartbeatWorkload(num_beats=5, memory_update_every=2)
        requests = wl.generate_program("test_p1")
        # Memory module should change at beat 2 and 4
        memory_hashes = [
            next(m.content_hash for m in r.modules if m.name == "memory")
            for r in requests
        ]
        assert memory_hashes[0] == memory_hashes[1]
        assert memory_hashes[0] != memory_hashes[2]

    def test_name(self):
        assert HeartbeatWorkload().name == "heartbeat"

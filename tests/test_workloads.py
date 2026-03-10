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
        wl = SkillSwitchWorkload(turns_per_phase=5, num_phases=3)
        requests = wl.generate_program("test_p1")
        assert len(requests) == 15

    def test_skills_change_between_phases(self):
        wl = SkillSwitchWorkload(turns_per_phase=3, num_phases=2)
        requests = wl.generate_program("test_p1")

        # Get skill modules from first and second phase
        phase1_skills = [m for m in requests[0].modules if m.name.startswith("skill_")]
        phase2_skills = [m for m in requests[3].modules if m.name.startswith("skill_")]

        # Skill names should differ between phases
        phase1_names = {m.name for m in phase1_skills}
        phase2_names = {m.name for m in phase2_skills}
        assert phase1_names != phase2_names

    def test_core_stable_across_phases(self):
        wl = SkillSwitchWorkload(turns_per_phase=3, num_phases=2)
        requests = wl.generate_program("test_p1")
        # Core instructions should be same across all turns
        core_hash = requests[0].modules[0].content_hash
        for r in requests:
            assert r.modules[0].content_hash == core_hash

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

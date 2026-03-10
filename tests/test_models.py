"""Tests for core data models."""

from harness.models import (
    BackendType,
    ExperimentConfig,
    ModularRequest,
    PromptModule,
    ProgramMetrics,
    RequestMetrics,
    StreamingResponse,
)


class TestPromptModule:
    def test_auto_hash(self):
        m = PromptModule(name="test", content="hello world")
        assert m.content_hash != ""
        assert len(m.content_hash) == 64  # SHA-256 hex

    def test_same_content_same_hash(self):
        m1 = PromptModule(name="a", content="hello")
        m2 = PromptModule(name="b", content="hello")
        assert m1.content_hash == m2.content_hash

    def test_different_content_different_hash(self):
        m1 = PromptModule(name="a", content="hello")
        m2 = PromptModule(name="a", content="world")
        assert m1.content_hash != m2.content_hash

    def test_frozen(self):
        m = PromptModule(name="test", content="hello")
        try:
            m.name = "other"
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_token_count_estimate(self):
        m = PromptModule(name="test", content="a" * 400)
        assert m.token_count_estimate == 100


class TestModularRequest:
    def test_flat_prompt(self):
        modules = [
            PromptModule(name="a", content="hello"),
            PromptModule(name="b", content="world"),
        ]
        req = ModularRequest(program_id="p1", turn_index=0, modules=modules)
        assert req.flat_prompt() == "hello\nworld"

    def test_total_prompt_tokens(self):
        modules = [
            PromptModule(name="a", content="a" * 400),
            PromptModule(name="b", content="b" * 800),
        ]
        req = ModularRequest(program_id="p1", turn_index=0, modules=modules)
        assert req.total_prompt_tokens_estimate == 300  # 100 + 200


class TestRequestMetrics:
    def test_tpot(self):
        m = RequestMetrics(
            program_id="p1",
            turn_index=0,
            backend=BackendType.VLLM,
            ttft_ms=50.0,
            total_latency_ms=150.0,
            tokens_generated=11,
        )
        assert abs(m.tpot_ms - 10.0) < 0.001  # (150-50)/(11-1)

    def test_tpot_single_token(self):
        m = RequestMetrics(
            program_id="p1",
            turn_index=0,
            backend=BackendType.VLLM,
            ttft_ms=50.0,
            total_latency_ms=50.0,
            tokens_generated=1,
        )
        assert m.tpot_ms == 0.0

    def test_prefill_ratio(self):
        m = RequestMetrics(
            program_id="p1",
            turn_index=0,
            backend=BackendType.PIE_CACHE,
            ttft_ms=10.0,
            total_latency_ms=100.0,
            tokens_generated=50,
            cache_hits=3,
            cache_misses=1,
            tokens_saved=900,
            tokens_computed=100,
        )
        assert abs(m.prefill_ratio - 0.9) < 0.001


class TestProgramMetrics:
    def _make_program(self) -> ProgramMetrics:
        return ProgramMetrics(
            program_id="p1",
            backend=BackendType.PIE_CACHE,
            request_metrics=[
                RequestMetrics(
                    program_id="p1",
                    turn_index=0,
                    backend=BackendType.PIE_CACHE,
                    ttft_ms=50.0,
                    total_latency_ms=200.0,
                    tokens_generated=20,
                    cache_hits=0,
                    cache_misses=4,
                    tokens_saved=0,
                    tokens_computed=500,
                ),
                RequestMetrics(
                    program_id="p1",
                    turn_index=1,
                    backend=BackendType.PIE_CACHE,
                    ttft_ms=10.0,
                    total_latency_ms=150.0,
                    tokens_generated=20,
                    cache_hits=3,
                    cache_misses=1,
                    tokens_saved=450,
                    tokens_computed=50,
                ),
            ],
        )

    def test_mean_ttft(self):
        p = self._make_program()
        assert abs(p.mean_ttft_ms - 30.0) < 0.001

    def test_cache_hit_rate(self):
        p = self._make_program()
        # hits: 0+3=3, total modules: (0+4)+(3+1)=8
        assert abs(p.cache_hit_rate - 3 / 8) < 0.001

    def test_total_prefill_savings(self):
        p = self._make_program()
        assert p.total_prefill_savings_tokens == 450


class TestStreamingResponse:
    def test_text(self):
        r = StreamingResponse(tokens=["hello", " ", "world"])
        assert r.text == "hello world"

    def test_tokens_generated(self):
        r = StreamingResponse(tokens=["a", "b", "c"])
        assert r.tokens_generated == 3

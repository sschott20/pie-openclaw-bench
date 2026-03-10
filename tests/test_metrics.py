"""Tests for metrics collection and aggregation."""

from harness.metrics.aggregator import (
    aggregate_program_metrics,
    compute_experiment_summary,
    request_metrics_to_dataframe,
)
from harness.metrics.collector import collect_request_metrics
from harness.models import (
    BackendType,
    ModularRequest,
    PromptModule,
    RequestMetrics,
    StreamingResponse,
)


def _make_request_metrics(
    program_id: str = "p1",
    turn: int = 0,
    backend: BackendType = BackendType.VLLM,
    ttft: float = 50.0,
    total: float = 200.0,
    tokens: int = 20,
) -> RequestMetrics:
    return RequestMetrics(
        program_id=program_id,
        turn_index=turn,
        backend=backend,
        ttft_ms=ttft,
        total_latency_ms=total,
        tokens_generated=tokens,
    )


class TestCollector:
    def test_collect_from_response(self):
        request = ModularRequest(
            program_id="p1",
            turn_index=0,
            modules=[PromptModule(name="a", content="hello")],
        )
        response = StreamingResponse(
            tokens=["hello", " ", "world"],
            ttft_ms=42.0,
            total_latency_ms=100.0,
        )
        rm = collect_request_metrics(request, response, BackendType.VLLM)
        assert rm.program_id == "p1"
        assert rm.turn_index == 0
        assert rm.ttft_ms == 42.0
        assert rm.tokens_generated == 3


class TestAggregator:
    def test_aggregate_by_program(self):
        metrics = [
            _make_request_metrics("p1", 0),
            _make_request_metrics("p1", 1),
            _make_request_metrics("p2", 0),
        ]
        programs = aggregate_program_metrics(metrics)
        assert len(programs) == 2
        p1 = next(p for p in programs if p.program_id == "p1")
        assert len(p1.request_metrics) == 2

    def test_experiment_summary(self):
        metrics = [
            _make_request_metrics("p1", 0, ttft=50),
            _make_request_metrics("p1", 1, ttft=30),
            _make_request_metrics("p2", 0, ttft=40),
        ]
        programs = aggregate_program_metrics(metrics)
        summary = compute_experiment_summary(programs)
        assert summary["num_programs"] == 2
        assert summary["num_requests"] == 3
        assert 30 <= summary["mean_ttft_ms"] <= 50

    def test_empty_summary(self):
        assert compute_experiment_summary([]) == {}

    def test_to_dataframe(self):
        metrics = [
            _make_request_metrics("p1", 0, ttft=50),
            _make_request_metrics("p1", 1, ttft=30),
        ]
        df = request_metrics_to_dataframe(metrics)
        assert len(df) == 2
        assert "ttft_ms" in df.columns
        assert "backend" in df.columns
        assert df["ttft_ms"].tolist() == [50.0, 30.0]

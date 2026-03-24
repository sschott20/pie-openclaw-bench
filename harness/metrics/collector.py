"""Per-request metrics collection."""

from __future__ import annotations

from harness.models import (
    BackendType,
    ModularRequest,
    RequestMetrics,
    StreamingResponse,
)


def collect_request_metrics(
    request: ModularRequest,
    response: StreamingResponse,
    backend: BackendType,
) -> RequestMetrics:
    """Build a RequestMetrics from a completed request/response pair."""
    return RequestMetrics(
        program_id=request.program_id,
        turn_index=request.turn_index,
        backend=backend,
        ttft_ms=response.ttft_ms,
        total_latency_ms=response.total_latency_ms,
        tokens_generated=response.tokens_generated,
        generated_text=response.text,
        cache_hits=response.cache_hits,
        cache_misses=response.cache_misses,
        tokens_saved=response.tokens_saved,
        tokens_computed=response.tokens_computed,
    )

"""Metrics aggregation at program and experiment levels, with CSV output."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from harness.models import BackendType, ProgramMetrics, RequestMetrics


def aggregate_program_metrics(
    request_metrics: list[RequestMetrics],
) -> list[ProgramMetrics]:
    """Group request metrics by (program_id, backend) into ProgramMetrics."""
    groups: dict[tuple[str, BackendType], list[RequestMetrics]] = defaultdict(list)
    for rm in request_metrics:
        groups[(rm.program_id, rm.backend)].append(rm)

    programs = []
    for (pid, backend), metrics in groups.items():
        # Sort by turn index
        metrics.sort(key=lambda m: m.turn_index)
        programs.append(ProgramMetrics(
            program_id=pid,
            backend=backend,
            request_metrics=metrics,
        ))
    return programs


def compute_experiment_summary(
    programs: list[ProgramMetrics],
) -> dict:
    """Compute experiment-level aggregated metrics."""
    if not programs:
        return {}

    jct = [p.job_completion_time_ms for p in programs]
    ttfts = [
        rm.ttft_ms
        for p in programs
        for rm in p.request_metrics
    ]

    return {
        "num_programs": len(programs),
        "num_requests": sum(len(p.request_metrics) for p in programs),
        "mean_job_completion_ms": float(np.mean(jct)),
        "p50_job_completion_ms": float(np.percentile(jct, 50)),
        "p90_job_completion_ms": float(np.percentile(jct, 90)),
        "p95_job_completion_ms": float(np.percentile(jct, 95)),
        "mean_ttft_ms": float(np.mean(ttfts)),
        "p50_ttft_ms": float(np.percentile(ttfts, 50)),
        "p90_ttft_ms": float(np.percentile(ttfts, 90)),
        "p95_ttft_ms": float(np.percentile(ttfts, 95)),
        "mean_cache_hit_rate": float(np.mean([p.cache_hit_rate for p in programs])),
        "total_prefill_savings_tokens": sum(
            p.total_prefill_savings_tokens for p in programs
        ),
    }


def request_metrics_to_dataframe(
    metrics: list[RequestMetrics],
) -> pd.DataFrame:
    """Convert request metrics to a DataFrame for analysis."""
    rows = []
    for m in metrics:
        rows.append({
            "program_id": m.program_id,
            "turn_index": m.turn_index,
            "backend": m.backend.value,
            "ttft_ms": m.ttft_ms,
            "total_latency_ms": m.total_latency_ms,
            "tokens_generated": m.tokens_generated,
            "tpot_ms": m.tpot_ms,
            "cache_hits": m.cache_hits,
            "cache_misses": m.cache_misses,
            "tokens_saved": m.tokens_saved,
            "tokens_computed": m.tokens_computed,
            "prefill_ratio": m.prefill_ratio,
        })
    return pd.DataFrame(rows)


def save_metrics_csv(
    metrics: list[RequestMetrics],
    path: Path,
) -> None:
    """Save request metrics to CSV."""
    df = request_metrics_to_dataframe(metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_metrics_csv(path: Path) -> pd.DataFrame:
    """Load request metrics from CSV."""
    return pd.read_csv(path)

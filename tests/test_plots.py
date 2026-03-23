"""Tests for the plotting module."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from harness.metrics.plots import (
    generate_all_plots,
    generate_quality_plots,
    generate_report_summary,
    plot_cache_hit_rate,
    plot_job_completion_time,
    plot_latency_percentiles,
    plot_parameter_sensitivity,
    plot_prefill_breakdown,
    plot_quality_by_turn,
    plot_quality_comparison,
    plot_speedup_vs_quality,
    plot_tpot_comparison,
    plot_ttft_by_turn,
    plot_ttft_comparison,
)


# ---------------------------------------------------------------------------
# Helpers to generate synthetic DataFrames
# ---------------------------------------------------------------------------

def _make_perf_df(
    backends: list[str] | None = None,
    n_programs: int = 3,
    n_turns: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic performance DataFrame matching the aggregator schema."""
    rng = np.random.RandomState(seed)
    if backends is None:
        backends = ["vllm", "pie_std", "pie_cache"]

    rows = []
    for backend in backends:
        for pid in range(n_programs):
            for turn in range(n_turns):
                ttft = rng.uniform(20, 200)
                tpot = rng.uniform(1, 10)
                tokens = rng.randint(10, 100)
                total = ttft + tpot * tokens
                hits = rng.randint(0, 6) if backend == "pie_cache" else 0
                misses = rng.randint(0, 4) if backend == "pie_cache" else 0
                saved = rng.randint(0, 500) if backend == "pie_cache" else 0
                computed = rng.randint(100, 600)
                total_tokens = saved + computed
                rows.append({
                    "program_id": f"p{pid}",
                    "turn_index": turn,
                    "backend": backend,
                    "ttft_ms": ttft,
                    "total_latency_ms": total,
                    "tokens_generated": tokens,
                    "tpot_ms": tpot,
                    "cache_hits": hits,
                    "cache_misses": misses,
                    "tokens_saved": saved,
                    "tokens_computed": computed,
                    "prefill_ratio": saved / total_tokens if total_tokens else 0.0,
                })
    return pd.DataFrame(rows)


def _make_quality_workload_df() -> pd.DataFrame:
    """Quality DataFrame with workload-level aggregates."""
    return pd.DataFrame([
        {
            "workload": "react",
            "exact_match_rate": 0.85,
            "mean_bleu": 0.92,
            "mean_rouge_l": 0.90,
            "mean_edit_distance": 2.1,
        },
        {
            "workload": "multiturn",
            "exact_match_rate": 0.70,
            "mean_bleu": 0.80,
            "mean_rouge_l": 0.78,
            "mean_edit_distance": 5.3,
        },
        {
            "workload": "skill_switch",
            "exact_match_rate": 0.60,
            "mean_bleu": 0.75,
            "mean_rouge_l": 0.72,
            "mean_edit_distance": 8.0,
        },
    ])


def _make_quality_turn_df(n_turns: int = 8, seed: int = 42) -> pd.DataFrame:
    """Quality DataFrame with per-turn metrics."""
    rng = np.random.RandomState(seed)
    rows = []
    for t in range(n_turns):
        rows.append({
            "turn_index": t,
            "bleu_score": max(0, 0.95 - 0.03 * t + rng.normal(0, 0.02)),
            "rouge_l_f1": max(0, 0.93 - 0.025 * t + rng.normal(0, 0.02)),
            "exact_match": rng.random() > (0.2 + 0.05 * t),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests for existing plot functions
# ---------------------------------------------------------------------------

class TestTTFTComparison:
    def test_returns_figure(self):
        df = _make_perf_df()
        fig = plot_ttft_comparison(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self):
        df = _make_perf_df()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ttft.png"
            fig = plot_ttft_comparison(df, output_path=path)
            assert path.exists()
            plt.close(fig)

    def test_single_backend(self):
        df = _make_perf_df(backends=["vllm"])
        fig = plot_ttft_comparison(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestTTFTByTurn:
    def test_returns_figure(self):
        df = _make_perf_df()
        fig = plot_ttft_by_turn(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestCacheHitRate:
    def test_returns_figure(self):
        df = _make_perf_df()
        fig = plot_cache_hit_rate(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_pie_cache_data(self):
        df = _make_perf_df(backends=["vllm"])
        fig = plot_cache_hit_rate(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPrefillBreakdown:
    def test_returns_figure(self):
        df = _make_perf_df()
        fig = plot_prefill_breakdown(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_pie_cache_data(self):
        df = _make_perf_df(backends=["vllm"])
        fig = plot_prefill_breakdown(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests for new plot functions
# ---------------------------------------------------------------------------

class TestLatencyPercentiles:
    def test_returns_figure(self):
        df = _make_perf_df()
        fig = plot_latency_percentiles(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_backend(self):
        df = _make_perf_df(backends=["pie_cache"])
        fig = plot_latency_percentiles(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self):
        df = _make_perf_df()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "latency_pct.png"
            fig = plot_latency_percentiles(df, output_path=path)
            assert path.exists()
            plt.close(fig)

    def test_empty_df(self):
        df = pd.DataFrame(columns=[
            "program_id", "turn_index", "backend", "ttft_ms",
            "total_latency_ms", "tokens_generated", "tpot_ms",
            "cache_hits", "cache_misses", "tokens_saved",
            "tokens_computed", "prefill_ratio",
        ])
        fig = plot_latency_percentiles(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestJobCompletionTime:
    def test_returns_figure(self):
        df = _make_perf_df()
        fig = plot_job_completion_time(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_backend(self):
        df = _make_perf_df(backends=["vllm"])
        fig = plot_job_completion_time(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self):
        df = _make_perf_df()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jct.png"
            fig = plot_job_completion_time(df, output_path=path)
            assert path.exists()
            plt.close(fig)

    def test_empty_df(self):
        df = pd.DataFrame(columns=[
            "program_id", "turn_index", "backend", "ttft_ms",
            "total_latency_ms", "tokens_generated", "tpot_ms",
            "cache_hits", "cache_misses", "tokens_saved",
            "tokens_computed", "prefill_ratio",
        ])
        fig = plot_job_completion_time(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestTPOTComparison:
    def test_returns_figure(self):
        df = _make_perf_df()
        fig = plot_tpot_comparison(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_backend(self):
        df = _make_perf_df(backends=["pie_std"])
        fig = plot_tpot_comparison(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self):
        df = _make_perf_df()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tpot.png"
            fig = plot_tpot_comparison(df, output_path=path)
            assert path.exists()
            plt.close(fig)


class TestParameterSensitivity:
    def test_returns_figure(self):
        results = {
            5: _make_perf_df(n_turns=5),
            10: _make_perf_df(n_turns=10),
            15: _make_perf_df(n_turns=15),
        }
        fig = plot_parameter_sensitivity(
            results, param_name="num_turns", metric_name="ttft_ms",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_param_value(self):
        results = {1: _make_perf_df()}
        fig = plot_parameter_sensitivity(
            results, param_name="batch_size", metric_name="ttft_ms",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self):
        results = {5: _make_perf_df(), 10: _make_perf_df()}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sensitivity.png"
            fig = plot_parameter_sensitivity(
                results, param_name="x", metric_name="ttft_ms",
                output_path=path,
            )
            assert path.exists()
            plt.close(fig)

    def test_empty_results(self):
        empty_df = pd.DataFrame(columns=[
            "program_id", "turn_index", "backend", "ttft_ms",
        ])
        results = {1: empty_df}
        fig = plot_parameter_sensitivity(
            results, param_name="x", metric_name="ttft_ms",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestQualityComparison:
    def test_returns_figure(self):
        qdf = _make_quality_workload_df()
        fig = plot_quality_comparison(qdf)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self):
        qdf = _make_quality_workload_df()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quality.png"
            fig = plot_quality_comparison(qdf, output_path=path)
            assert path.exists()
            plt.close(fig)

    def test_missing_metric_columns(self):
        """Gracefully handles DataFrame missing some quality metric columns."""
        qdf = pd.DataFrame([
            {"workload": "react", "exact_match_rate": 0.9},
            {"workload": "multiturn", "exact_match_rate": 0.7},
        ])
        fig = plot_quality_comparison(qdf)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestQualityByTurn:
    def test_returns_figure(self):
        qdf = _make_quality_turn_df()
        fig = plot_quality_by_turn(qdf)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self):
        qdf = _make_quality_turn_df()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "quality_turn.png"
            fig = plot_quality_by_turn(qdf, output_path=path)
            assert path.exists()
            plt.close(fig)

    def test_partial_columns(self):
        """Only bleu_score present, no rouge or exact_match."""
        qdf = pd.DataFrame({
            "turn_index": [0, 1, 2],
            "bleu_score": [0.9, 0.85, 0.8],
        })
        fig = plot_quality_by_turn(qdf)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestSpeedupVsQuality:
    def test_returns_figure(self):
        perf_df = _make_perf_df()
        quality_df = _make_quality_turn_df(n_turns=5)
        fig = plot_speedup_vs_quality(perf_df, quality_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_missing_vllm(self):
        perf_df = _make_perf_df(backends=["pie_cache"])
        quality_df = _make_quality_turn_df()
        fig = plot_speedup_vs_quality(perf_df, quality_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_missing_pie_cache(self):
        perf_df = _make_perf_df(backends=["vllm"])
        quality_df = _make_quality_turn_df()
        fig = plot_speedup_vs_quality(perf_df, quality_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_bleu_column(self):
        perf_df = _make_perf_df()
        quality_df = pd.DataFrame({
            "turn_index": [0, 1, 2],
            "some_other_metric": [0.5, 0.6, 0.7],
        })
        fig = plot_speedup_vs_quality(perf_df, quality_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self):
        perf_df = _make_perf_df()
        quality_df = _make_quality_turn_df(n_turns=5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "speedup_quality.png"
            fig = plot_speedup_vs_quality(
                perf_df, quality_df, output_path=path,
            )
            assert path.exists()
            plt.close(fig)


# ---------------------------------------------------------------------------
# Tests for orchestrator functions
# ---------------------------------------------------------------------------

class TestGenerateAllPlots:
    def test_creates_files(self):
        df = _make_perf_df()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "plots"
            generate_all_plots(df, output_dir, experiment_name="test")
            expected_files = [
                "test_ttft_comparison.png",
                "test_ttft_by_turn.png",
                "test_cache_hit_rate.png",
                "test_prefill_breakdown.png",
                "test_latency_percentiles.png",
                "test_job_completion_time.png",
                "test_tpot_comparison.png",
            ]
            for fname in expected_files:
                assert (output_dir / fname).exists(), f"Missing: {fname}"

    def test_single_backend(self):
        df = _make_perf_df(backends=["vllm"])
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "plots"
            generate_all_plots(df, output_dir)
            assert output_dir.exists()


class TestGenerateQualityPlots:
    def test_workload_quality(self):
        qdf = _make_quality_workload_df()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "quality"
            generate_quality_plots(qdf, output_dir, experiment_name="test")
            assert (output_dir / "test_quality_comparison.png").exists()

    def test_turn_quality(self):
        qdf = _make_quality_turn_df()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "quality"
            generate_quality_plots(qdf, output_dir, experiment_name="test")
            assert (output_dir / "test_quality_by_turn.png").exists()

    def test_combined_quality(self):
        """DataFrame with both workload and turn_index columns."""
        qdf = pd.DataFrame({
            "workload": ["react"] * 5,
            "turn_index": list(range(5)),
            "exact_match_rate": [0.9, 0.85, 0.8, 0.75, 0.7],
            "mean_bleu": [0.95, 0.9, 0.85, 0.8, 0.75],
            "mean_rouge_l": [0.93, 0.88, 0.83, 0.78, 0.73],
            "bleu_score": [0.95, 0.9, 0.85, 0.8, 0.75],
            "rouge_l_f1": [0.93, 0.88, 0.83, 0.78, 0.73],
            "exact_match": [True, True, True, False, False],
        })
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "quality"
            generate_quality_plots(qdf, output_dir, experiment_name="test")
            assert (output_dir / "test_quality_comparison.png").exists()
            assert (output_dir / "test_quality_by_turn.png").exists()


class TestGenerateReportSummary:
    def test_with_quality(self):
        df = _make_perf_df()
        qdf = _make_quality_workload_df()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "report"
            generate_report_summary(df, qdf, output_dir, experiment_name="test")
            assert (output_dir / "test_summary.png").exists()

    def test_without_quality(self):
        df = _make_perf_df()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "report"
            generate_report_summary(df, None, output_dir, experiment_name="test")
            assert (output_dir / "test_summary.png").exists()

    def test_no_pie_cache(self):
        df = _make_perf_df(backends=["vllm", "pie_std"])
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "report"
            generate_report_summary(df, None, output_dir, experiment_name="test")
            assert (output_dir / "test_summary.png").exists()

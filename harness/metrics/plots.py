"""Matplotlib/seaborn figure generation for benchmark results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_style():
    """Configure consistent plot style."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 150


BACKEND_COLORS = {
    "vllm": "#4C72B0",
    "pie_std": "#DD8452",
    "pie_cache": "#55A868",
}

BACKEND_LABELS = {
    "vllm": "vLLM (prefix cache)",
    "pie_std": "PIE std (no cache)",
    "pie_cache": "PIE modular cache",
}


def plot_ttft_comparison(
    df: pd.DataFrame,
    title: str = "TTFT by Backend",
    output_path: Path | None = None,
) -> plt.Figure:
    """Bar chart comparing mean TTFT across backends with error bars."""
    setup_style()

    fig, ax = plt.subplots()

    backends = df["backend"].unique()
    stats = []
    for b in backends:
        subset = df[df["backend"] == b]["ttft_ms"]
        stats.append({
            "backend": b,
            "mean": subset.mean(),
            "std": subset.std(),
            "label": BACKEND_LABELS.get(b, b),
            "color": BACKEND_COLORS.get(b, "#888888"),
        })

    x = np.arange(len(stats))
    bars = ax.bar(
        x,
        [s["mean"] for s in stats],
        yerr=[s["std"] for s in stats],
        color=[s["color"] for s in stats],
        capsize=5,
        width=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([s["label"] for s in stats], rotation=15, ha="right")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title(title)

    # Add value labels on bars
    for bar, s in zip(bars, stats):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s["std"] + 1,
            f"{s['mean']:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    return fig


def plot_ttft_by_turn(
    df: pd.DataFrame,
    title: str = "TTFT Across Turns",
    output_path: Path | None = None,
) -> plt.Figure:
    """Line chart of TTFT across turns for each backend."""
    setup_style()

    fig, ax = plt.subplots()

    for backend in df["backend"].unique():
        subset = df[df["backend"] == backend]
        by_turn = subset.groupby("turn_index")["ttft_ms"].agg(["mean", "std"]).reset_index()
        ax.plot(
            by_turn["turn_index"],
            by_turn["mean"],
            label=BACKEND_LABELS.get(backend, backend),
            color=BACKEND_COLORS.get(backend, "#888888"),
            marker="o",
            markersize=4,
        )
        ax.fill_between(
            by_turn["turn_index"],
            by_turn["mean"] - by_turn["std"],
            by_turn["mean"] + by_turn["std"],
            alpha=0.2,
            color=BACKEND_COLORS.get(backend, "#888888"),
        )

    ax.set_xlabel("Turn Index")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    return fig


def plot_cache_hit_rate(
    df: pd.DataFrame,
    title: str = "Cache Hit Rate by Turn",
    output_path: Path | None = None,
) -> plt.Figure:
    """Line chart of cache hit rate across turns (PIE-cache only)."""
    setup_style()

    fig, ax = plt.subplots()

    cache_df = df[df["backend"] == "pie_cache"].copy()
    if cache_df.empty:
        ax.text(0.5, 0.5, "No PIE-cache data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        return fig

    cache_df["hit_rate"] = cache_df["cache_hits"] / (
        cache_df["cache_hits"] + cache_df["cache_misses"]
    ).replace(0, np.nan)

    by_turn = cache_df.groupby("turn_index")["hit_rate"].agg(["mean", "std"]).reset_index()
    ax.plot(
        by_turn["turn_index"],
        by_turn["mean"],
        color=BACKEND_COLORS["pie_cache"],
        marker="o",
        markersize=4,
    )
    ax.fill_between(
        by_turn["turn_index"],
        (by_turn["mean"] - by_turn["std"]).clip(0),
        (by_turn["mean"] + by_turn["std"]).clip(upper=1),
        alpha=0.2,
        color=BACKEND_COLORS["pie_cache"],
    )

    ax.set_xlabel("Turn Index")
    ax.set_ylabel("Cache Hit Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    return fig


def plot_prefill_breakdown(
    df: pd.DataFrame,
    title: str = "Prefill Token Breakdown",
    output_path: Path | None = None,
) -> plt.Figure:
    """Stacked bar showing cached vs computed tokens per turn (PIE-cache)."""
    setup_style()

    fig, ax = plt.subplots()

    cache_df = df[df["backend"] == "pie_cache"].copy()
    if cache_df.empty:
        ax.text(0.5, 0.5, "No PIE-cache data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        return fig

    by_turn = cache_df.groupby("turn_index")[["tokens_saved", "tokens_computed"]].mean().reset_index()

    ax.bar(
        by_turn["turn_index"],
        by_turn["tokens_saved"],
        label="Cached (imported)",
        color=BACKEND_COLORS["pie_cache"],
    )
    ax.bar(
        by_turn["turn_index"],
        by_turn["tokens_computed"],
        bottom=by_turn["tokens_saved"],
        label="Computed (GPU prefill)",
        color=BACKEND_COLORS["pie_std"],
    )

    ax.set_xlabel("Turn Index")
    ax.set_ylabel("Tokens")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    return fig


def plot_latency_percentiles(
    df: pd.DataFrame,
    title: str = "TTFT Latency Percentiles",
    output_path: Path | None = None,
) -> plt.Figure:
    """Grouped bar chart of P50, P90, P95, P99 TTFT for each backend."""
    setup_style()

    fig, ax = plt.subplots()

    percentiles = [50, 90, 95, 99]
    percentile_labels = ["P50", "P90", "P95", "P99"]
    backends = sorted(df["backend"].unique())

    if len(backends) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path)
        return fig

    n_percentiles = len(percentiles)
    n_backends = len(backends)
    bar_width = 0.8 / max(n_backends, 1)
    x = np.arange(n_percentiles)

    for i, backend in enumerate(backends):
        subset = df[df["backend"] == backend]["ttft_ms"]
        values = [float(np.percentile(subset, p)) for p in percentiles]
        offset = (i - (n_backends - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            values,
            width=bar_width,
            label=BACKEND_LABELS.get(backend, backend),
            color=BACKEND_COLORS.get(backend, "#888888"),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(percentile_labels)
    ax.set_xlabel("Percentile")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    return fig


def plot_job_completion_time(
    df: pd.DataFrame,
    title: str = "Job Completion Time",
    output_path: Path | None = None,
) -> plt.Figure:
    """Box plot of total job (program) completion time per backend."""
    setup_style()

    fig, ax = plt.subplots()

    backends = sorted(df["backend"].unique())
    if len(backends) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path)
        return fig

    jct_data = []
    labels = []
    colors = []
    for backend in backends:
        subset = df[df["backend"] == backend]
        program_jct = subset.groupby("program_id")["total_latency_ms"].sum()
        jct_data.append(program_jct.values)
        labels.append(BACKEND_LABELS.get(backend, backend))
        colors.append(BACKEND_COLORS.get(backend, "#888888"))

    bp = ax.boxplot(jct_data, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Job Completion Time (ms)")
    ax.set_title(title)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    return fig


def plot_tpot_comparison(
    df: pd.DataFrame,
    title: str = "Time Per Output Token",
    output_path: Path | None = None,
) -> plt.Figure:
    """Bar chart of mean time-per-output-token across backends."""
    setup_style()

    fig, ax = plt.subplots()

    backends = df["backend"].unique()
    stats = []
    for b in backends:
        subset = df[df["backend"] == b]["tpot_ms"]
        stats.append({
            "backend": b,
            "mean": subset.mean(),
            "std": subset.std(),
            "label": BACKEND_LABELS.get(b, b),
            "color": BACKEND_COLORS.get(b, "#888888"),
        })

    x = np.arange(len(stats))
    bars = ax.bar(
        x,
        [s["mean"] for s in stats],
        yerr=[s["std"] for s in stats],
        color=[s["color"] for s in stats],
        capsize=5,
        width=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([s["label"] for s in stats], rotation=15, ha="right")
    ax.set_ylabel("TPOT (ms)")
    ax.set_title(title)

    for bar, s in zip(bars, stats):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s["std"] + 0.1,
            f"{s['mean']:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    return fig


def plot_parameter_sensitivity(
    results_dict: dict[str | int | float, pd.DataFrame],
    param_name: str,
    metric_name: str,
    title: str = "Parameter Sensitivity",
    output_path: Path | None = None,
) -> plt.Figure:
    """Line chart showing how a metric changes as a parameter varies.

    Args:
        results_dict: Mapping from parameter value to DataFrame.
        param_name: Name of the swept parameter (for x-axis label).
        metric_name: Column name in each DataFrame to aggregate.
        title: Plot title.
        output_path: Optional path to save the figure.
    """
    setup_style()

    fig, ax = plt.subplots()

    # Collect all backends across all DataFrames
    all_backends: set[str] = set()
    for param_df in results_dict.values():
        if not param_df.empty:
            all_backends.update(param_df["backend"].unique())
    all_backends_sorted = sorted(all_backends)

    param_values = sorted(results_dict.keys())

    for backend in all_backends_sorted:
        means = []
        valid_params = []
        for pv in param_values:
            param_df = results_dict[pv]
            subset = param_df[param_df["backend"] == backend]
            if not subset.empty and metric_name in subset.columns:
                means.append(subset[metric_name].mean())
                valid_params.append(pv)
        if valid_params:
            ax.plot(
                valid_params,
                means,
                label=BACKEND_LABELS.get(backend, backend),
                color=BACKEND_COLORS.get(backend, "#888888"),
                marker="o",
                markersize=5,
            )

    ax.set_xlabel(param_name)
    ax.set_ylabel(f"Mean {metric_name}")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    return fig


def plot_quality_comparison(
    quality_df: pd.DataFrame,
    title: str = "Quality Metrics by Workload",
    output_path: Path | None = None,
) -> plt.Figure:
    """Grouped bar chart of quality metrics by workload.

    quality_df columns: workload, exact_match_rate, mean_bleu, mean_rouge_l,
                        mean_edit_distance
    """
    setup_style()

    fig, ax = plt.subplots()

    metrics = ["exact_match_rate", "mean_bleu", "mean_rouge_l"]
    metric_labels = ["Exact Match Rate", "Mean BLEU", "Mean ROUGE-L"]
    metric_colors = ["#4C72B0", "#DD8452", "#55A868"]

    workloads = quality_df["workload"].unique()
    n_workloads = len(workloads)
    n_metrics = len(metrics)

    if n_workloads == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path)
        return fig

    bar_width = 0.8 / max(n_metrics, 1)
    x = np.arange(n_workloads)

    for i, (metric, label, color) in enumerate(
        zip(metrics, metric_labels, metric_colors)
    ):
        if metric not in quality_df.columns:
            continue
        values = [
            quality_df[quality_df["workload"] == w][metric].iloc[0]
            if not quality_df[quality_df["workload"] == w].empty
            else 0.0
            for w in workloads
        ]
        offset = (i - (n_metrics - 1) / 2) * bar_width
        ax.bar(x + offset, values, width=bar_width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    return fig


def plot_quality_by_turn(
    quality_df: pd.DataFrame,
    title: str = "Quality Metrics by Turn",
    output_path: Path | None = None,
) -> plt.Figure:
    """Line chart of quality metrics across turn indices.

    quality_df columns: turn_index, bleu_score, rouge_l_f1, exact_match (bool)
    """
    setup_style()

    fig, ax = plt.subplots()

    metric_map = {
        "bleu_score": ("BLEU", "#4C72B0"),
        "rouge_l_f1": ("ROUGE-L F1", "#DD8452"),
        "exact_match": ("Exact Match", "#55A868"),
    }

    for col, (label, color) in metric_map.items():
        if col not in quality_df.columns:
            continue
        by_turn = quality_df.groupby("turn_index")[col].mean().reset_index()
        ax.plot(
            by_turn["turn_index"],
            by_turn[col].astype(float),
            label=label,
            color=color,
            marker="o",
            markersize=4,
        )

    ax.set_xlabel("Turn Index")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    return fig


def plot_speedup_vs_quality(
    perf_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    title: str = "Speedup vs Quality",
    output_path: Path | None = None,
) -> plt.Figure:
    """Scatter plot of TTFT speedup vs BLEU score.

    Each point is one turn index. x-axis = vllm_ttft / pie_cache_ttft,
    y-axis = BLEU score.
    """
    setup_style()

    fig, ax = plt.subplots()

    # Compute per-turn mean TTFT for vllm and pie_cache
    vllm_df = perf_df[perf_df["backend"] == "vllm"]
    pie_cache_df = perf_df[perf_df["backend"] == "pie_cache"]

    if vllm_df.empty or pie_cache_df.empty:
        ax.text(0.5, 0.5, "Need both vllm and pie_cache data",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path)
        return fig

    vllm_by_turn = vllm_df.groupby("turn_index")["ttft_ms"].mean()
    pie_by_turn = pie_cache_df.groupby("turn_index")["ttft_ms"].mean()

    # Determine the BLEU column
    bleu_col = None
    for candidate in ("bleu_score", "mean_bleu"):
        if candidate in quality_df.columns:
            bleu_col = candidate
            break

    if bleu_col is None:
        ax.text(0.5, 0.5, "No BLEU column found",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path)
        return fig

    quality_by_turn = quality_df.groupby("turn_index")[bleu_col].mean()

    # Align on common turn indices
    common_turns = sorted(
        set(vllm_by_turn.index) & set(pie_by_turn.index) & set(quality_by_turn.index)
    )

    if not common_turns:
        ax.text(0.5, 0.5, "No overlapping turn indices",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        fig.tight_layout()
        if output_path:
            fig.savefig(output_path)
        return fig

    speedups = []
    bleu_scores = []
    for t in common_turns:
        pie_ttft = pie_by_turn[t]
        if pie_ttft > 0:
            speedups.append(vllm_by_turn[t] / pie_ttft)
            bleu_scores.append(quality_by_turn[t])

    ax.scatter(speedups, bleu_scores, color=BACKEND_COLORS["pie_cache"],
               s=60, alpha=0.7, edgecolors="black", linewidths=0.5)

    # Reference lines
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect BLEU")
    ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5, label="No speedup")

    ax.set_xlabel("TTFT Speedup (vLLM / PIE-cache)")
    ax.set_ylabel("BLEU Score")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    return fig


def generate_all_plots(
    df: pd.DataFrame,
    output_dir: Path,
    experiment_name: str = "experiment",
) -> None:
    """Generate all standard performance plots for an experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_ttft_comparison(
        df,
        title=f"TTFT Comparison — {experiment_name}",
        output_path=output_dir / f"{experiment_name}_ttft_comparison.png",
    )
    plot_ttft_by_turn(
        df,
        title=f"TTFT by Turn — {experiment_name}",
        output_path=output_dir / f"{experiment_name}_ttft_by_turn.png",
    )
    plot_cache_hit_rate(
        df,
        title=f"Cache Hit Rate — {experiment_name}",
        output_path=output_dir / f"{experiment_name}_cache_hit_rate.png",
    )
    plot_prefill_breakdown(
        df,
        title=f"Prefill Breakdown — {experiment_name}",
        output_path=output_dir / f"{experiment_name}_prefill_breakdown.png",
    )
    plot_latency_percentiles(
        df,
        title=f"Latency Percentiles — {experiment_name}",
        output_path=output_dir / f"{experiment_name}_latency_percentiles.png",
    )
    plot_job_completion_time(
        df,
        title=f"Job Completion Time — {experiment_name}",
        output_path=output_dir / f"{experiment_name}_job_completion_time.png",
    )
    plot_tpot_comparison(
        df,
        title=f"TPOT Comparison — {experiment_name}",
        output_path=output_dir / f"{experiment_name}_tpot_comparison.png",
    )
    plt.close("all")


def generate_quality_plots(
    quality_df: pd.DataFrame,
    output_dir: Path,
    experiment_name: str = "experiment",
) -> None:
    """Generate quality-specific plots for an experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Quality comparison by workload (needs workload column)
    if "workload" in quality_df.columns:
        plot_quality_comparison(
            quality_df,
            title=f"Quality Comparison — {experiment_name}",
            output_path=output_dir / f"{experiment_name}_quality_comparison.png",
        )

    # Quality by turn (needs turn_index column)
    if "turn_index" in quality_df.columns:
        plot_quality_by_turn(
            quality_df,
            title=f"Quality by Turn — {experiment_name}",
            output_path=output_dir / f"{experiment_name}_quality_by_turn.png",
        )

    plt.close("all")


def generate_report_summary(
    df: pd.DataFrame,
    quality_df: pd.DataFrame | None,
    output_dir: Path,
    experiment_name: str = "experiment",
) -> None:
    """Generate a single-page 2x2 summary figure.

    Top-left: TTFT comparison bars
    Top-right: Cache hit rate by turn
    Bottom-left: Job completion time boxes
    Bottom-right: Quality metrics bars (if quality_df provided, else prefill breakdown)
    """
    setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Experiment Summary — {experiment_name}", fontsize=16, y=0.98)

    # --- Top-left: TTFT comparison ---
    ax = axes[0, 0]
    backends = sorted(df["backend"].unique())
    stats = []
    for b in backends:
        subset = df[df["backend"] == b]["ttft_ms"]
        stats.append({
            "mean": subset.mean(),
            "std": subset.std(),
            "label": BACKEND_LABELS.get(b, b),
            "color": BACKEND_COLORS.get(b, "#888888"),
        })
    x = np.arange(len(stats))
    ax.bar(
        x,
        [s["mean"] for s in stats],
        yerr=[s["std"] for s in stats],
        color=[s["color"] for s in stats],
        capsize=5,
        width=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([s["label"] for s in stats], rotation=15, ha="right",
                       fontsize=9)
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT Comparison")

    # --- Top-right: Cache hit rate by turn ---
    ax = axes[0, 1]
    cache_df = df[df["backend"] == "pie_cache"].copy()
    if cache_df.empty:
        ax.text(0.5, 0.5, "No PIE-cache data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
    else:
        total = cache_df["cache_hits"] + cache_df["cache_misses"]
        cache_df["hit_rate"] = cache_df["cache_hits"] / total.replace(0, np.nan)
        by_turn = cache_df.groupby("turn_index")["hit_rate"].mean().reset_index()
        ax.plot(by_turn["turn_index"], by_turn["hit_rate"],
                color=BACKEND_COLORS["pie_cache"], marker="o", markersize=4)
        ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Turn Index")
    ax.set_ylabel("Cache Hit Rate")
    ax.set_title("Cache Hit Rate by Turn")

    # --- Bottom-left: Job completion time boxes ---
    ax = axes[1, 0]
    jct_data = []
    labels = []
    colors = []
    for backend in backends:
        subset = df[df["backend"] == backend]
        program_jct = subset.groupby("program_id")["total_latency_ms"].sum()
        jct_data.append(program_jct.values)
        labels.append(BACKEND_LABELS.get(backend, backend))
        colors.append(BACKEND_COLORS.get(backend, "#888888"))
    if jct_data:
        bp = ax.boxplot(jct_data, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.tick_params(axis="x", rotation=15)
    ax.set_ylabel("Job Completion Time (ms)")
    ax.set_title("Job Completion Time")

    # --- Bottom-right: Quality or prefill breakdown ---
    ax = axes[1, 1]
    if quality_df is not None and "workload" in quality_df.columns:
        metrics = ["exact_match_rate", "mean_bleu", "mean_rouge_l"]
        metric_labels = ["Exact Match", "BLEU", "ROUGE-L"]
        metric_colors = ["#4C72B0", "#DD8452", "#55A868"]
        workloads = quality_df["workload"].unique()
        n_workloads = len(workloads)
        n_metrics = len(metrics)
        bar_width = 0.8 / max(n_metrics, 1)
        x = np.arange(n_workloads)
        for i, (metric, label, color) in enumerate(
            zip(metrics, metric_labels, metric_colors)
        ):
            if metric not in quality_df.columns:
                continue
            values = [
                quality_df[quality_df["workload"] == w][metric].iloc[0]
                if not quality_df[quality_df["workload"] == w].empty
                else 0.0
                for w in workloads
            ]
            offset = (i - (n_metrics - 1) / 2) * bar_width
            ax.bar(x + offset, values, width=bar_width, label=label, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(workloads, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.set_title("Quality Metrics")
        ax.legend(fontsize=8)
    else:
        # Fallback: prefill breakdown
        if not cache_df.empty:
            by_turn = cache_df.groupby("turn_index")[
                ["tokens_saved", "tokens_computed"]
            ].mean().reset_index()
            ax.bar(by_turn["turn_index"], by_turn["tokens_saved"],
                   label="Cached", color=BACKEND_COLORS["pie_cache"])
            ax.bar(by_turn["turn_index"], by_turn["tokens_computed"],
                   bottom=by_turn["tokens_saved"],
                   label="Computed", color=BACKEND_COLORS["pie_std"])
            ax.set_xlabel("Turn Index")
            ax.set_ylabel("Tokens")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
        ax.set_title("Prefill Breakdown")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / f"{experiment_name}_summary.png"
    fig.savefig(output_path)
    plt.close(fig)

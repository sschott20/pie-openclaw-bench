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


def generate_all_plots(
    df: pd.DataFrame,
    output_dir: Path,
    experiment_name: str = "experiment",
) -> None:
    """Generate all standard plots for an experiment."""
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
    plt.close("all")

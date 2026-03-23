"""Data models for quality evaluation results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class PairResult:
    """Quality comparison for a single (reference, candidate) pair."""

    turn_index: int
    reference_text: str
    candidate_text: str
    exact_match: bool
    bleu_score: float
    rouge_l_f1: float
    edit_distance: float  # normalized 0-1
    first_divergence: int | None  # None if identical


@dataclass
class QualityReport:
    """Aggregated quality evaluation across all pairs."""

    workload: str
    reference_backend: str
    candidate_backend: str
    num_pairs: int
    pairs: list[PairResult] = field(default_factory=list)
    # Aggregate metrics
    exact_match_rate: float = 0.0
    mean_bleu: float = 0.0
    mean_rouge_l: float = 0.0
    mean_edit_distance: float = 0.0

    def to_dict(self) -> dict:
        """Serialize report to a dictionary."""
        return {
            "workload": self.workload,
            "reference_backend": self.reference_backend,
            "candidate_backend": self.candidate_backend,
            "num_pairs": self.num_pairs,
            "exact_match_rate": self.exact_match_rate,
            "mean_bleu": self.mean_bleu,
            "mean_rouge_l": self.mean_rouge_l,
            "mean_edit_distance": self.mean_edit_distance,
            "pairs": [
                {
                    "turn_index": p.turn_index,
                    "exact_match": p.exact_match,
                    "bleu_score": p.bleu_score,
                    "rouge_l_f1": p.rouge_l_f1,
                    "edit_distance": p.edit_distance,
                    "first_divergence": p.first_divergence,
                    "reference_text": p.reference_text,
                    "candidate_text": p.candidate_text,
                }
                for p in self.pairs
            ],
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert per-pair results to a DataFrame."""
        rows = []
        for p in self.pairs:
            rows.append({
                "turn_index": p.turn_index,
                "exact_match": p.exact_match,
                "bleu_score": p.bleu_score,
                "rouge_l_f1": p.rouge_l_f1,
                "edit_distance": p.edit_distance,
                "first_divergence": p.first_divergence,
            })
        return pd.DataFrame(rows)

    def save_json(self, path: Path) -> None:
        """Save report as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

"""Core quality evaluation: BLEU, ROUGE-L, edit distance, exact match."""

from __future__ import annotations

from collections import Counter
from math import exp, log

from harness.quality.models import PairResult, QualityReport


def _tokenize(text: str) -> list[str]:
    """Whitespace tokenization."""
    return text.split()


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from a token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _compute_bleu(reference: str, candidate: str, max_n: int = 4) -> float:
    """Compute sentence-level BLEU with 1..max_n gram precisions and brevity penalty.

    Uses smoothed precision (add-1) to avoid zero scores on short sentences.
    """
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)

    if not cand_tokens:
        return 0.0
    if not ref_tokens:
        return 0.0

    # Clipped n-gram precisions with add-1 smoothing
    log_precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(_ngrams(ref_tokens, n))
        cand_ngrams = Counter(_ngrams(cand_tokens, n))

        clipped = 0
        total = 0
        for ng, count in cand_ngrams.items():
            clipped += min(count, ref_ngrams.get(ng, 0))
            total += count

        # Add-1 smoothing for n > 1 to handle short sentences
        if n == 1:
            if total == 0:
                return 0.0
            precision = clipped / total
            if precision == 0:
                return 0.0
        else:
            precision = (clipped + 1) / (total + 1)

        log_precisions.append(log(precision))

    # Geometric mean of precisions (uniform weights)
    avg_log_precision = sum(log_precisions) / len(log_precisions)

    # Brevity penalty
    bp = 1.0
    if len(cand_tokens) < len(ref_tokens):
        bp = exp(1 - len(ref_tokens) / len(cand_tokens))

    return bp * exp(avg_log_precision)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute length of longest common subsequence using O(min(m,n)) space."""
    if len(a) < len(b):
        a, b = b, a

    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[len(b)]


def _compute_rouge_l(reference: str, candidate: str) -> float:
    """Compute ROUGE-L F1 score based on longest common subsequence."""
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)

    if not ref_tokens or not cand_tokens:
        return 0.0

    lcs_len = _lcs_length(ref_tokens, cand_tokens)
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / len(cand_tokens)
    recall = lcs_len / len(ref_tokens)

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def _levenshtein_distance(a: list[str], b: list[str]) -> int:
    """Compute Levenshtein edit distance between two token sequences."""
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        curr = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[len(b)]


def _compute_edit_distance(reference: str, candidate: str) -> float:
    """Normalized token-level Levenshtein distance (0=identical, 1=completely different)."""
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)

    max_len = max(len(ref_tokens), len(cand_tokens))
    if max_len == 0:
        return 0.0

    dist = _levenshtein_distance(ref_tokens, cand_tokens)
    return dist / max_len


def _find_first_divergence(reference: str, candidate: str) -> int | None:
    """Find the character index where reference and candidate first differ.

    Returns None if the strings are identical.
    """
    if reference == candidate:
        return None

    min_len = min(len(reference), len(candidate))
    for i in range(min_len):
        if reference[i] != candidate[i]:
            return i

    # One is a prefix of the other
    return min_len


class QualityEvaluator:
    """Evaluates quality degradation between reference and candidate outputs."""

    def compare_outputs(
        self,
        reference_responses: list[str],
        candidate_responses: list[str],
    ) -> QualityReport:
        """Compare paired reference and candidate responses.

        Args:
            reference_responses: Outputs from full-prefill backend (e.g. vLLM).
            candidate_responses: Outputs from modular-cache backend (e.g. PIE cache).

        Returns:
            QualityReport with per-pair and aggregate metrics.
        """
        if len(reference_responses) != len(candidate_responses):
            raise ValueError(
                f"Mismatched lengths: {len(reference_responses)} references "
                f"vs {len(candidate_responses)} candidates"
            )

        pairs: list[PairResult] = []
        for i, (ref, cand) in enumerate(
            zip(reference_responses, candidate_responses)
        ):
            exact = ref == cand
            bleu = _compute_bleu(ref, cand)
            rouge_l = _compute_rouge_l(ref, cand)
            edit_dist = _compute_edit_distance(ref, cand)
            first_div = _find_first_divergence(ref, cand)

            pairs.append(PairResult(
                turn_index=i,
                reference_text=ref,
                candidate_text=cand,
                exact_match=exact,
                bleu_score=bleu,
                rouge_l_f1=rouge_l,
                edit_distance=edit_dist,
                first_divergence=first_div,
            ))

        num_pairs = len(pairs)
        if num_pairs == 0:
            return QualityReport(
                workload="",
                reference_backend="",
                candidate_backend="",
                num_pairs=0,
            )

        return QualityReport(
            workload="",
            reference_backend="",
            candidate_backend="",
            num_pairs=num_pairs,
            pairs=pairs,
            exact_match_rate=sum(1 for p in pairs if p.exact_match) / num_pairs,
            mean_bleu=sum(p.bleu_score for p in pairs) / num_pairs,
            mean_rouge_l=sum(p.rouge_l_f1 for p in pairs) / num_pairs,
            mean_edit_distance=sum(p.edit_distance for p in pairs) / num_pairs,
        )

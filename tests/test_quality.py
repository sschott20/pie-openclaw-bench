"""Tests for quality evaluation pipeline."""

from harness.quality.evaluator import (
    QualityEvaluator,
    _compute_bleu,
    _compute_edit_distance,
    _compute_rouge_l,
    _find_first_divergence,
)
from harness.quality.models import QualityReport


class TestExactMatch:
    def test_identical_strings(self):
        evaluator = QualityEvaluator()
        report = evaluator.compare_outputs(
            ["hello world foo bar"],
            ["hello world foo bar"],
        )
        assert report.exact_match_rate == 1.0
        assert report.pairs[0].exact_match is True
        assert report.pairs[0].first_divergence is None

    def test_different_strings(self):
        evaluator = QualityEvaluator()
        report = evaluator.compare_outputs(
            ["hello world"],
            ["goodbye world"],
        )
        assert report.exact_match_rate == 0.0
        assert report.pairs[0].exact_match is False

    def test_multiple_pairs_mixed(self):
        evaluator = QualityEvaluator()
        report = evaluator.compare_outputs(
            ["same", "different", "same"],
            ["same", "changed", "same"],
        )
        assert report.num_pairs == 3
        assert report.exact_match_rate == 2.0 / 3.0


class TestBLEU:
    def test_identical_gives_one(self):
        text = "the cat sat on the mat"
        score = _compute_bleu(text, text)
        assert abs(score - 1.0) < 1e-6

    def test_completely_different_gives_low(self):
        score = _compute_bleu(
            "the cat sat on the mat",
            "xyz abc def ghi jkl mno",
        )
        assert score < 0.1

    def test_partial_overlap(self):
        score = _compute_bleu(
            "the cat sat on the mat",
            "the cat sat on a rug",
        )
        assert 0.2 < score < 0.9

    def test_empty_candidate(self):
        score = _compute_bleu("hello world", "")
        assert score == 0.0

    def test_empty_reference(self):
        score = _compute_bleu("", "hello world")
        assert score == 0.0

    def test_both_empty(self):
        score = _compute_bleu("", "")
        assert score == 0.0

    def test_brevity_penalty(self):
        # Short candidate relative to reference should be penalized
        ref = "the cat sat on the mat in the house"
        short_cand = "the cat"
        long_cand = "the cat sat on the mat in the house by the door"
        short_score = _compute_bleu(ref, short_cand)
        long_score = _compute_bleu(ref, long_cand)
        # Long candidate (superset of ref) should score higher than short
        assert long_score > short_score

    def test_known_example(self):
        # A well-known BLEU test case: one word substitution in a 7-word sentence
        ref = "the cat is on the mat now"
        cand = "the cat is on the rug now"
        score = _compute_bleu(ref, cand)
        # Should be high but not perfect
        assert 0.4 < score < 0.95


class TestROUGEL:
    def test_identical_gives_one(self):
        text = "the cat sat on the mat"
        score = _compute_rouge_l(text, text)
        assert abs(score - 1.0) < 1e-6

    def test_completely_different_gives_zero(self):
        score = _compute_rouge_l(
            "the cat sat on the mat",
            "xyz abc def ghi jkl mno",
        )
        assert score == 0.0

    def test_partial_overlap(self):
        score = _compute_rouge_l(
            "the cat sat on the mat",
            "the dog sat on the rug",
        )
        # LCS: "the", "sat", "on", "the" -> 4 tokens
        # precision = 4/6, recall = 4/6, F1 = 4/6
        assert abs(score - 4.0 / 6.0) < 1e-6

    def test_empty_inputs(self):
        assert _compute_rouge_l("", "hello") == 0.0
        assert _compute_rouge_l("hello", "") == 0.0
        assert _compute_rouge_l("", "") == 0.0

    def test_subsequence_not_substring(self):
        # LCS should find non-contiguous common subsequence
        ref = "A B C D E"
        cand = "A X C X E"
        score = _compute_rouge_l(ref, cand)
        # LCS = A, C, E -> length 3
        # precision = 3/5, recall = 3/5, F1 = 3/5
        assert abs(score - 3.0 / 5.0) < 1e-6


class TestEditDistance:
    def test_identical_gives_zero(self):
        dist = _compute_edit_distance("hello world", "hello world")
        assert dist == 0.0

    def test_completely_different(self):
        dist = _compute_edit_distance("aaa bbb ccc", "xxx yyy zzz")
        assert dist == 1.0

    def test_one_substitution(self):
        dist = _compute_edit_distance("the cat sat", "the dog sat")
        # 1 substitution out of 3 tokens
        assert abs(dist - 1.0 / 3.0) < 1e-6

    def test_empty_both(self):
        assert _compute_edit_distance("", "") == 0.0

    def test_one_empty(self):
        # "hello world" has 2 tokens, max_len=2, dist=2 -> normalized=1.0
        assert _compute_edit_distance("hello world", "") == 1.0

    def test_insertion(self):
        dist = _compute_edit_distance("a b c", "a b c d")
        # 1 insertion, max_len=4
        assert abs(dist - 1.0 / 4.0) < 1e-6


class TestFirstDivergence:
    def test_identical(self):
        assert _find_first_divergence("hello", "hello") is None

    def test_first_char_differs(self):
        assert _find_first_divergence("abc", "xbc") == 0

    def test_middle_differs(self):
        assert _find_first_divergence("abcdef", "abcxef") == 3

    def test_prefix(self):
        # One is a prefix of the other
        assert _find_first_divergence("abc", "abcdef") == 3

    def test_empty_vs_nonempty(self):
        assert _find_first_divergence("", "abc") == 0
        assert _find_first_divergence("abc", "") == 0

    def test_both_empty(self):
        assert _find_first_divergence("", "") is None


class TestQualityEvaluatorIntegration:
    def test_identical_outputs(self):
        evaluator = QualityEvaluator()
        texts = [
            "The function returns a list of integers.",
            "Error handling is done via try-except blocks.",
            "The database query uses an index on user_id.",
        ]
        report = evaluator.compare_outputs(texts, texts)

        assert report.num_pairs == 3
        assert report.exact_match_rate == 1.0
        assert abs(report.mean_bleu - 1.0) < 1e-6
        assert abs(report.mean_rouge_l - 1.0) < 1e-6
        assert report.mean_edit_distance == 0.0

    def test_completely_different(self):
        evaluator = QualityEvaluator()
        refs = ["alpha beta gamma delta"]
        cands = ["one two three four"]
        report = evaluator.compare_outputs(refs, cands)

        assert report.exact_match_rate == 0.0
        assert report.mean_bleu < 0.1
        assert report.mean_rouge_l == 0.0
        assert report.mean_edit_distance == 1.0

    def test_partial_overlap(self):
        evaluator = QualityEvaluator()
        refs = ["the quick brown fox jumps over the lazy dog"]
        cands = ["the quick red fox leaps over the lazy cat"]
        report = evaluator.compare_outputs(refs, cands)

        assert report.exact_match_rate == 0.0
        assert 0.2 < report.mean_bleu < 0.9
        assert 0.5 < report.mean_rouge_l < 1.0
        assert 0.0 < report.mean_edit_distance < 0.5

    def test_empty_lists(self):
        evaluator = QualityEvaluator()
        report = evaluator.compare_outputs([], [])
        assert report.num_pairs == 0

    def test_mismatched_lengths_raises(self):
        evaluator = QualityEvaluator()
        try:
            evaluator.compare_outputs(["a"], ["b", "c"])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_empty_string_pairs(self):
        evaluator = QualityEvaluator()
        report = evaluator.compare_outputs(["", ""], ["", ""])
        assert report.num_pairs == 2
        assert report.exact_match_rate == 1.0
        # BLEU and ROUGE-L on empty strings return 0
        assert report.mean_bleu == 0.0
        assert report.mean_rouge_l == 0.0
        assert report.mean_edit_distance == 0.0


class TestQualityReportSerialization:
    def test_to_dict(self):
        evaluator = QualityEvaluator()
        report = evaluator.compare_outputs(["hello world"], ["hello world"])
        report.workload = "test"
        report.reference_backend = "vllm"
        report.candidate_backend = "pie_cache"

        d = report.to_dict()
        assert d["workload"] == "test"
        assert d["num_pairs"] == 1
        assert d["exact_match_rate"] == 1.0
        assert len(d["pairs"]) == 1

    def test_to_dataframe(self):
        evaluator = QualityEvaluator()
        report = evaluator.compare_outputs(
            ["hello world", "foo bar"],
            ["hello world", "foo baz"],
        )
        df = report.to_dataframe()
        assert len(df) == 2
        assert "bleu_score" in df.columns
        assert "rouge_l_f1" in df.columns

    def test_save_json(self, tmp_path):
        evaluator = QualityEvaluator()
        report = evaluator.compare_outputs(["a b c"], ["a b c"])
        report.workload = "test"
        report.reference_backend = "vllm"
        report.candidate_backend = "pie_cache"

        out = tmp_path / "report.json"
        report.save_json(out)
        assert out.exists()

        import json
        loaded = json.loads(out.read_text())
        assert loaded["exact_match_rate"] == 1.0

    def test_turn_indices(self):
        evaluator = QualityEvaluator()
        report = evaluator.compare_outputs(
            ["a", "b", "c"],
            ["a", "b", "c"],
        )
        assert [p.turn_index for p in report.pairs] == [0, 1, 2]

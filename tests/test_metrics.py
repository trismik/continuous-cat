"""Tests for metrics module."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics.evaluation import (
    TieMetrics,
    GroundTruthRanking,
    bootstrap_ci,
    compute_confident_accuracy,
    compute_ground_truth,
    compute_kendall_tau,
    compute_tie_metrics,
    detect_ties_from_cis,
)


class TestKendallTau:
    """Test Kendall's tau computation."""

    def test_perfect_agreement(self):
        """Identical rankings should have tau = 1.0."""
        ranking = ["a", "b", "c", "d"]
        tau = compute_kendall_tau(ranking, ranking)
        assert abs(tau - 1.0) < 1e-10

    def test_perfect_disagreement(self):
        """Reversed rankings should have tau = -1.0."""
        pred = ["a", "b", "c", "d"]
        gt = ["d", "c", "b", "a"]
        tau = compute_kendall_tau(pred, gt)
        assert abs(tau - (-1.0)) < 1e-10

    def test_partial_agreement(self):
        """Partially matching rankings should have tau between -1 and 1."""
        pred = ["a", "b", "c", "d"]
        gt = ["a", "c", "b", "d"]  # One swap
        tau = compute_kendall_tau(pred, gt)
        assert -1 < tau < 1

    def test_different_models(self):
        """Rankings with different models should only compare common ones."""
        pred = ["a", "b", "c"]
        gt = ["b", "a", "d"]  # Only a, b in common
        tau = compute_kendall_tau(pred, gt)
        # a, b are swapped, so tau = -1
        assert abs(tau - (-1.0)) < 1e-10


class TestTieMetrics:
    """Test tie detection metrics."""

    def test_perfect_prediction(self):
        """Perfect tie prediction should have P=R=F1=1."""
        pred = {("a", "b"), ("c", "d")}
        gt = {("a", "b"), ("c", "d")}

        metrics = compute_tie_metrics(pred, gt)

        assert metrics.tp == 2
        assert metrics.fp == 0
        assert metrics.fn == 0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0

    def test_no_predictions(self):
        """No predicted ties when there are GT ties."""
        pred = set()
        gt = {("a", "b")}

        metrics = compute_tie_metrics(pred, gt)

        assert metrics.tp == 0
        assert metrics.fp == 0
        assert metrics.fn == 1
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0

    def test_false_positives(self):
        """Predicting ties that don't exist."""
        pred = {("a", "b"), ("c", "d")}
        gt = {("a", "b")}

        metrics = compute_tie_metrics(pred, gt)

        assert metrics.tp == 1
        assert metrics.fp == 1
        assert metrics.fn == 0
        assert metrics.precision == 0.5
        assert metrics.recall == 1.0

    def test_tuple_order_independent(self):
        """Tie pairs should match regardless of order."""
        pred = {("b", "a")}  # Reversed order
        gt = {("a", "b")}

        metrics = compute_tie_metrics(pred, gt)

        assert metrics.tp == 1
        assert metrics.fp == 0
        assert metrics.fn == 0


class TestBootstrapCI:
    """Test bootstrap confidence intervals."""

    def test_basic_ci(self):
        """Basic CI computation."""
        np.random.seed(42)
        scores = list(np.random.normal(0.5, 0.1, 100))

        mean, lower, upper = bootstrap_ci(scores, n_bootstrap=1000)

        # Mean should be close to 0.5
        assert abs(mean - 0.5) < 0.05

        # CI should contain the mean
        assert lower < mean < upper

        # CI should be narrower than the data range
        assert upper - lower < 0.5

    def test_deterministic_with_seed(self):
        """Same seed should give same results."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]

        result1 = bootstrap_ci(scores, seed=42)
        result2 = bootstrap_ci(scores, seed=42)

        assert result1 == result2

    def test_confidence_level(self):
        """Higher confidence should give wider CI."""
        scores = list(np.random.normal(0.5, 0.1, 100))

        _, lower_90, upper_90 = bootstrap_ci(scores, confidence_level=0.90)
        _, lower_95, upper_95 = bootstrap_ci(scores, confidence_level=0.95)
        _, lower_99, upper_99 = bootstrap_ci(scores, confidence_level=0.99)

        width_90 = upper_90 - lower_90
        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99

        assert width_90 < width_95 < width_99


class TestTieDetection:
    """Test tie detection from CIs."""

    def test_overlapping_cis(self):
        """Overlapping CIs should be detected as ties."""
        ranking = ["a", "b", "c"]
        ci_lowers = {"a": 0.5, "b": 0.4, "c": 0.1}
        ci_uppers = {"a": 0.7, "b": 0.6, "c": 0.3}

        # a and b overlap (0.5-0.7 and 0.4-0.6)
        # c doesn't overlap with a or b

        ties = detect_ties_from_cis(ranking, ci_lowers, ci_uppers)

        assert ("a", "b") in ties or ("b", "a") in ties
        assert ("a", "c") not in ties and ("c", "a") not in ties
        assert ("b", "c") not in ties and ("c", "b") not in ties

    def test_no_overlap(self):
        """Non-overlapping CIs should not be tied."""
        ranking = ["a", "b", "c"]
        ci_lowers = {"a": 0.7, "b": 0.4, "c": 0.1}
        ci_uppers = {"a": 0.9, "b": 0.6, "c": 0.3}

        ties = detect_ties_from_cis(ranking, ci_lowers, ci_uppers)

        assert len(ties) == 0


class TestGroundTruth:
    """Test ground truth computation."""

    def test_basic_ranking(self):
        """Test basic ground truth ranking."""
        model_scores = {
            "good": [0.9, 0.85, 0.88, 0.92, 0.87],
            "bad": [0.3, 0.35, 0.28, 0.32, 0.31],
        }

        gt = compute_ground_truth(model_scores, n_bootstrap=100)

        # Good should rank first
        assert gt.models[0] == "good"
        assert gt.means["good"] > gt.means["bad"]

        # CIs should not overlap (clearly different)
        assert gt.ci_lowers["good"] > gt.ci_uppers["bad"]

        # No ties expected
        assert len(gt.ties) == 0

    def test_ties_detected(self):
        """Test that similar models are detected as tied."""
        model_scores = {
            "a": [0.5, 0.52, 0.48, 0.51, 0.49],
            "b": [0.5, 0.51, 0.49, 0.50, 0.52],
        }

        gt = compute_ground_truth(model_scores, n_bootstrap=100)

        # Should be tied (very similar distributions)
        assert len(gt.ties) > 0


class TestConfidentAccuracy:
    """Test confident accuracy computation."""

    def test_perfect_confident_accuracy(self):
        """Perfect ordering on confident pairs."""
        pred = ["a", "b", "c"]
        gt = ["a", "b", "c"]
        pred_ties = set()  # All pairs are confident

        acc = compute_confident_accuracy(pred, gt, pred_ties)
        assert acc == 1.0

    def test_tied_pairs_excluded(self):
        """Tied pairs should be excluded from accuracy."""
        pred = ["a", "b", "c"]
        gt = ["b", "a", "c"]  # a and b swapped
        pred_ties = {("a", "b")}  # But we're not confident about a vs b

        acc = compute_confident_accuracy(pred, gt, pred_ties)

        # Only comparing a-c and b-c, both correct
        assert acc == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

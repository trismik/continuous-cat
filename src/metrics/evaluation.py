"""Evaluation metrics for ranking comparison.

This module provides:
- Kendall's tau-b correlation for ranking comparison
- Tie detection metrics (precision, recall, F1)
- Bootstrap confidence intervals for ground truth
- Ground truth tie detection based on CI overlap
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy import stats


def compute_kendall_tau(
    predicted_ranking: List[str],
    gt_ranking: List[str],
) -> float:
    """Compute Kendall's tau-b between two rankings.

    Uses scipy.stats.kendalltau which handles ties properly.

    Args:
        predicted_ranking: Predicted ranking (list of model names, best first)
        gt_ranking: Ground truth ranking (list of model names, best first)

    Returns:
        Kendall's tau-b correlation coefficient in [-1, 1]
    """
    # Create position maps
    gt_positions = {model: i for i, model in enumerate(gt_ranking)}
    pred_positions = {model: i for i, model in enumerate(predicted_ranking)}

    # Get common models
    common_models = list(set(gt_positions.keys()) & set(pred_positions.keys()))

    if len(common_models) < 2:
        return 0.0

    # Convert to position arrays
    gt_pos_array = [gt_positions[m] for m in common_models]
    pred_pos_array = [pred_positions[m] for m in common_models]

    # Compute Kendall's tau-b
    tau, _ = stats.kendalltau(gt_pos_array, pred_pos_array)

    return float(tau) if not np.isnan(tau) else 0.0


# =============================================================================
# TIE DETECTION METRICS
# =============================================================================

@dataclass
class TieMetrics:
    """Tie detection metrics."""

    tp: int  # True positives (correctly predicted ties)
    fp: int  # False positives (predicted ties that aren't in GT)
    fn: int  # False negatives (GT ties that weren't predicted)
    precision: float  # TP / (TP + FP)
    recall: float  # TP / (TP + FN)
    f1: float  # 2 * P * R / (P + R)


def compute_tie_metrics(
    predicted_ties: Set[Tuple[str, str]],
    gt_ties: Set[Tuple[str, str]],
) -> TieMetrics:
    """Compute tie detection metrics (precision, recall, F1).

    Args:
        predicted_ties: Set of predicted tied pairs
        gt_ties: Set of ground truth tied pairs

    Returns:
        TieMetrics with TP, FP, FN, precision, recall, F1
    """
    # Normalize tie tuples to sorted order for comparison
    pred_normalized = {tuple(sorted(pair)) for pair in predicted_ties}
    gt_normalized = {tuple(sorted(pair)) for pair in gt_ties}

    tp = len(pred_normalized & gt_normalized)
    fp = len(pred_normalized - gt_normalized)
    fn = len(gt_normalized - pred_normalized)

    # Compute precision
    if tp + fp == 0:
        precision = 0.0 if fn > 0 else 1.0
    else:
        precision = tp / (tp + fp)

    # Compute recall
    if tp + fn == 0:
        recall = 1.0
    else:
        recall = tp / (tp + fn)

    # Compute F1
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return TieMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for mean score.

    Args:
        scores: List of scores
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (mean, ci_lower, ci_upper)
    """
    scores_arr = np.array(scores)
    n = len(scores_arr)

    if n == 0:
        return 0.0, 0.0, 0.0

    # Calculate observed mean
    observed_mean = float(np.mean(scores_arr))

    # Bootstrap sampling
    rng = np.random.RandomState(seed)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores_arr, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    # Calculate percentile CI
    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return observed_mean, ci_lower, ci_upper


# =============================================================================
# GROUND TRUTH RANKING
# =============================================================================

@dataclass
class GroundTruthRanking:
    """Ground truth ranking with confidence intervals and ties."""

    models: List[str]  # Ordered by mean score (descending)
    means: Dict[str, float]  # Mean score per model
    ci_lowers: Dict[str, float]  # Lower CI bound per model
    ci_uppers: Dict[str, float]  # Upper CI bound per model
    ties: Set[Tuple[str, str]]  # Set of tied model pairs (CI overlap)


def detect_ties_from_cis(
    ranking: List[str],
    ci_lowers: Dict[str, float],
    ci_uppers: Dict[str, float],
) -> Set[Tuple[str, str]]:
    """Detect ties between model pairs based on CI overlap.

    Two models are tied if their confidence intervals overlap.

    Args:
        ranking: List of model names ordered by mean score (descending)
        ci_lowers: Dict of model name to lower CI bound
        ci_uppers: Dict of model name to upper CI bound

    Returns:
        Set of (model_a, model_b) tuples that are tied
    """
    ties = set()

    # Check ALL pairs
    for i, model_a in enumerate(ranking):
        for model_b in ranking[i + 1:]:
            # Check if CIs overlap
            overlap = (ci_lowers[model_a] <= ci_uppers[model_b] and
                       ci_lowers[model_b] <= ci_uppers[model_a])

            if overlap:
                ties.add(tuple(sorted([model_a, model_b])))

    return ties


def compute_ground_truth(
    model_scores: Dict[str, List[float]],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
) -> GroundTruthRanking:
    """Compute ground truth ranking with bootstrap CIs and tie detection.

    Args:
        model_scores: Dict mapping model_name to list of scores
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for CIs

    Returns:
        GroundTruthRanking with ordered models, means, CIs, and ties
    """
    means = {}
    ci_lowers = {}
    ci_uppers = {}

    for model, scores in model_scores.items():
        if not scores:
            continue

        mean, ci_lower, ci_upper = bootstrap_ci(
            scores,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        )

        means[model] = mean
        ci_lowers[model] = ci_lower
        ci_uppers[model] = ci_upper

    # Rank by mean score (descending)
    ranking = sorted(means.keys(), key=lambda m: means[m], reverse=True)

    # Detect ties based on CI overlap
    ties = detect_ties_from_cis(ranking, ci_lowers, ci_uppers)

    return GroundTruthRanking(
        models=ranking,
        means=means,
        ci_lowers=ci_lowers,
        ci_uppers=ci_uppers,
        ties=ties,
    )


def compute_confident_accuracy(
    pred_ranking: List[str],
    gt_ranking: List[str],
    pred_ties: Set[Tuple[str, str]],
) -> float:
    """Accuracy on pairs where adaptive ranker is confident (non-ties).

    For pairs where the ranker is confident (not tied), check if the
    predicted ordering matches the ground truth ordering.

    Args:
        pred_ranking: Predicted ranking (list of model names)
        gt_ranking: Ground truth ranking (list of model names)
        pred_ties: Set of predicted tied pairs

    Returns:
        Accuracy in [0, 1] on confident pairs
    """
    pred_positions = {model: i for i, model in enumerate(pred_ranking)}
    gt_positions = {model: i for i, model in enumerate(gt_ranking)}

    common_models = list(set(pred_positions.keys()) & set(gt_positions.keys()))

    correct = 0
    total = 0

    # Check all pairs
    for i, model_a in enumerate(common_models):
        for model_b in common_models[i + 1:]:
            # Skip if this pair is tied in prediction
            if tuple(sorted([model_a, model_b])) in pred_ties:
                continue

            total += 1

            # Check if ordering matches
            pred_a_better = pred_positions[model_a] < pred_positions[model_b]
            gt_a_better = gt_positions[model_a] < gt_positions[model_b]

            if pred_a_better == gt_a_better:
                correct += 1

    if total == 0:
        return 1.0  # No confident pairs, perfect by default

    return correct / total

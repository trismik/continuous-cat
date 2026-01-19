"""Item difficulty calibration for Continuous CAT.

This module implements IRT parameter estimation:
- Item difficulties (b) from mean scores
- Model abilities (theta) from method of moments
- Noise parameter (k) from residual variance
- Negative discrimination filtering

These parameters must be estimated from held-in models and then
used to evaluate held-out models.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def logit(p: float, epsilon: float = 1e-6) -> float:
    """Convert probability to logit scale.

    Args:
        p: Probability in (0, 1)
        epsilon: Small value to avoid infinity

    Returns:
        Logit value
    """
    p_clamped = np.clip(p, epsilon, 1 - epsilon)
    return float(np.log(p_clamped / (1 - p_clamped)))


def inv_logit(x: float) -> float:
    """Convert logit to probability scale.

    Args:
        x: Logit value

    Returns:
        Probability in (0, 1)
    """
    return float(1 / (1 + np.exp(-x)))


def scale_to_unit_interval(
    values: Dict[int, float],
    epsilon: float = 0.001,
) -> Dict[int, float]:
    """Scale values linearly to [epsilon, 1-epsilon].

    This ensures that continuous scores (like ROUGE) which may cluster
    in a narrow range get spread across [0,1] before logit transformation.

    Args:
        values: Dict mapping item_id to raw score
        epsilon: Small margin to avoid exact 0 or 1

    Returns:
        Dict mapping item_id to scaled score in [epsilon, 1-epsilon]
    """
    if not values:
        return {}

    vals = list(values.values())
    min_val, max_val = min(vals), max(vals)

    # If all values are the same, return 0.5 for all
    if max_val - min_val < 1e-9:
        return {k: 0.5 for k in values}

    scale = (1 - 2 * epsilon) / (max_val - min_val)
    return {
        item_id: (v - min_val) * scale + epsilon
        for item_id, v in values.items()
    }


# =============================================================================
# CALIBRATION RESULT
# =============================================================================

@dataclass
class CalibrationResult:
    """Result of difficulty calibration.

    Attributes:
        difficulties: Dict mapping item_id to logit difficulty
        k: Noise parameter (k=1 for standard Rasch, k>1 for noisier metrics)
        discrimination: Global discrimination = 1/sqrt(k)
        abilities: Dict mapping model name to ability estimate (theta)
        n_filtered: Number of items filtered for negative discrimination
    """

    difficulties: Dict[int, float]
    k: float
    discrimination: float
    abilities: Dict[str, float]
    n_filtered: int = 0


# =============================================================================
# ABILITY ESTIMATION
# =============================================================================

def estimate_abilities(
    scores: Dict[str, Dict[int, float]],
    min_score: float,
    max_score: float,
    epsilon: float = 1e-6,
) -> Dict[str, float]:
    """Estimate model abilities (theta) from average performance.

    Uses method of moments: theta_j = logit(mean(scaled_y_ij) across all items i)

    IMPORTANT: Scores are scaled using global min/max before computing abilities
    to be consistent with difficulty estimation.

    Args:
        scores: Dict mapping model_name -> {item_id -> score}
        min_score: Minimum score in the dataset (for scaling)
        max_score: Maximum score in the dataset (for scaling)
        epsilon: Small value for numerical stability

    Returns:
        Dict mapping model_name to ability estimate (theta)
    """
    abilities = {}
    score_range = max_score - min_score

    for model_name, model_scores in scores.items():
        if not model_scores:
            continue

        # Scale each score to [epsilon, 1-epsilon] using global range
        scaled_scores = []
        for score in model_scores.values():
            if score_range < 1e-9:
                scaled = 0.5
            else:
                scaled = (score - min_score) / score_range
                scaled = np.clip(scaled, epsilon, 1 - epsilon)
            scaled_scores.append(scaled)

        # Compute mean of scaled scores
        p_mean = np.mean(scaled_scores)
        p_clamped = np.clip(p_mean, epsilon, 1 - epsilon)
        abilities[model_name] = logit(p_clamped, epsilon)

    return abilities


# =============================================================================
# NOISE PARAMETER ESTIMATION
# =============================================================================

def estimate_noise_k(
    scores: Dict[str, Dict[int, float]],
    difficulties: Dict[int, float],
    abilities: Dict[str, float],
    min_score: float,
    max_score: float,
    epsilon: float = 1e-6,
) -> float:
    """Estimate noise parameter k from residuals (single score per item).

    k = Σ(y_ij - μ_ij)² / Σ μ_ij(1-μ_ij)
    where μ_ij = logistic(θ_j - b_i)

    NOTE: This function expects one score per (model, item). For k estimation
    using ALL individual observations, use estimate_noise_k_all_observations()
    instead.

    Args:
        scores: Dict mapping model_name -> {item_id -> score}
        difficulties: Dict mapping item_id to logit difficulty
        abilities: Dict mapping model_name to ability estimate
        min_score: Minimum score in the dataset (for scaling)
        max_score: Maximum score in the dataset (for scaling)
        epsilon: Small value for numerical stability

    Returns:
        Noise parameter k (k=1 for standard Rasch)
    """
    sum_squared_residuals = 0.0
    sum_bernoulli_variance = 0.0
    n_obs = 0
    score_range = max_score - min_score

    for model_name, model_scores in scores.items():
        if model_name not in abilities:
            continue

        theta_j = abilities[model_name]

        for item_id, score in model_scores.items():
            if item_id not in difficulties:
                continue

            # Scale score using global range
            if score_range < 1e-9:
                y = 0.5
            else:
                y = (score - min_score) / score_range
                y = np.clip(y, epsilon, 1 - epsilon)

            b_i = difficulties[item_id]

            # Predicted mean: μ_ij = logistic(θ_j - b_i)
            mu = inv_logit(theta_j - b_i)
            mu = np.clip(mu, epsilon, 1 - epsilon)

            sum_squared_residuals += (y - mu) ** 2
            sum_bernoulli_variance += mu * (1 - mu)
            n_obs += 1

    if n_obs == 0 or sum_bernoulli_variance < epsilon:
        return 1.0

    k = sum_squared_residuals / sum_bernoulli_variance
    return float(k)


def estimate_noise_k_all_observations(
    all_observations: Dict[str, Dict[int, List[float]]],
    difficulties: Dict[int, float],
    abilities: Dict[str, float],
    min_score: float,
    max_score: float,
    epsilon: float = 1e-6,
) -> float:
    """Estimate noise parameter k from ALL individual observations.

    Uses every (model, item, temp) observation rather than pre-averaged scores.

    k = Σ(y_ij - μ_ij)² / Σ μ_ij(1-μ_ij)
    where μ_ij = logistic(θ_j - b_i)

    Args:
        all_observations: Dict mapping model_name -> {item_id -> [list of scores]}
        difficulties: Dict mapping item_id to logit difficulty
        abilities: Dict mapping model_name to ability estimate
        min_score: Minimum score in the dataset (for scaling)
        max_score: Maximum score in the dataset (for scaling)
        epsilon: Small value for numerical stability

    Returns:
        Noise parameter k (k=1 for standard Rasch)
    """
    sum_squared_residuals = 0.0
    sum_bernoulli_variance = 0.0
    n_obs = 0
    score_range = max_score - min_score

    for model_name, model_items in all_observations.items():
        if model_name not in abilities:
            continue

        theta_j = abilities[model_name]

        for item_id, score_list in model_items.items():
            if item_id not in difficulties:
                continue

            b_i = difficulties[item_id]

            # Process each individual observation
            for score in score_list:
                # Scale score using global range
                if score_range < 1e-9:
                    y = 0.5
                else:
                    y = (score - min_score) / score_range
                    y = np.clip(y, epsilon, 1 - epsilon)

                # Predicted mean: μ_ij = logistic(θ_j - b_i)
                mu = inv_logit(theta_j - b_i)
                mu = np.clip(mu, epsilon, 1 - epsilon)

                sum_squared_residuals += (y - mu) ** 2
                sum_bernoulli_variance += mu * (1 - mu)
                n_obs += 1

    if n_obs == 0 or sum_bernoulli_variance < epsilon:
        return 1.0

    k = sum_squared_residuals / sum_bernoulli_variance
    return float(k)


# =============================================================================
# NEGATIVE DISCRIMINATION FILTERING
# =============================================================================

def filter_negative_discrimination(
    scores: Dict[str, Dict[int, float]],
    difficulties: Dict[int, float],
    abilities: Dict[str, float],
) -> Tuple[Dict[int, float], int]:
    """Filter out items where scores anti-correlate with model abilities.

    For each item, we compute the Pearson correlation between:
    - The item scores across all calibration models
    - The model abilities (theta values)

    Items with negative correlation are "discriminating in the wrong direction"
    (easier for weaker models, harder for stronger models) and should be removed.

    Args:
        scores: Dict mapping model_name -> {item_id -> score}
        difficulties: Dict mapping item_id to logit difficulty
        abilities: Dict mapping model_name to ability estimate

    Returns:
        Tuple of (filtered_difficulties, n_removed)
    """
    # Build item scores by model
    item_scores_by_model: Dict[int, Dict[str, float]] = {}

    for model_name, model_scores in scores.items():
        if model_name not in abilities:
            continue

        for item_id, score in model_scores.items():
            if item_id not in difficulties:
                continue

            if item_id not in item_scores_by_model:
                item_scores_by_model[item_id] = {}
            item_scores_by_model[item_id][model_name] = score

    # Compute correlation for each item
    filtered_difficulties = {}
    n_removed = 0

    for item_id, difficulty in difficulties.items():
        if item_id not in item_scores_by_model:
            # No scores for this item, keep it
            filtered_difficulties[item_id] = difficulty
            continue

        model_scores = item_scores_by_model[item_id]

        # Need at least 3 models to compute meaningful correlation
        if len(model_scores) < 3:
            filtered_difficulties[item_id] = difficulty
            continue

        # Build arrays for correlation
        item_score_arr = []
        ability_arr = []
        for model, score in model_scores.items():
            if model in abilities:
                item_score_arr.append(score)
                ability_arr.append(abilities[model])

        if len(item_score_arr) < 3:
            filtered_difficulties[item_id] = difficulty
            continue

        # Compute Pearson correlation
        item_score_arr = np.array(item_score_arr)
        ability_arr = np.array(ability_arr)

        # Check for zero variance
        if np.std(item_score_arr) < 1e-9 or np.std(ability_arr) < 1e-9:
            filtered_difficulties[item_id] = difficulty
            continue

        correlation = np.corrcoef(item_score_arr, ability_arr)[0, 1]

        if np.isnan(correlation):
            filtered_difficulties[item_id] = difficulty
            continue

        # Filter out items with negative correlation
        if correlation < 0:
            n_removed += 1
        else:
            filtered_difficulties[item_id] = difficulty

    return filtered_difficulties, n_removed


# =============================================================================
# MAIN CALIBRATION FUNCTION
# =============================================================================

def calibrate(
    scores: Dict[str, Dict[int, float]],
    estimate_k: bool = True,
    filter_negative: bool = True,
    global_min_score: float = None,
    global_max_score: float = None,
    all_observations: Dict[str, Dict[int, List[float]]] = None,
) -> CalibrationResult:
    """Calibrate item difficulties and discrimination from scores.

    Algorithm:
    1. Compute mean score per item across all models
    2. Scale mean scores to [epsilon, 1-epsilon] using mean scores' min/max
    3. Convert to logit difficulties: b_i = logit(1 - scaled_p_i)
    4. Estimate model abilities (using global min/max for scaling)
    5. Optionally filter negative discrimination items
    6. Optionally estimate noise parameter k (using global min/max for scaling)

    IMPORTANT: Different min/max values are used for different steps:
    - Difficulties: min/max of per-item mean scores
    - Abilities/k: min/max of ALL individual (model, temp, item) observations

    Args:
        scores: Dict mapping model_name -> {item_id -> score in [0,1]}
        estimate_k: Whether to estimate noise parameter (default: True)
        filter_negative: Whether to filter negative discrimination items (default: True)
        global_min_score: Optional global min from all individual observations
        global_max_score: Optional global max from all individual observations
        all_observations: Optional dict mapping model -> {item_id -> [list of scores]}
            for k estimation using ALL individual observations

    Returns:
        CalibrationResult with difficulties, k, discrimination, and abilities
    """
    # Get all scores from input (these may be pre-averaged per model)
    all_raw_scores = []
    for model_name, model_scores in scores.items():
        all_raw_scores.extend(model_scores.values())

    if not all_raw_scores:
        raise ValueError("No scores found for calibration")

    # Use provided global min/max if available, otherwise compute from input scores
    # NOTE: global_min/max should come from ALL individual (model, temp, item)
    # observations, not from pre-averaged scores
    min_score = global_min_score if global_min_score is not None else min(all_raw_scores)
    max_score = global_max_score if global_max_score is not None else max(all_raw_scores)

    # Step 2: Aggregate scores per item (mean across models)
    item_scores: Dict[int, List[float]] = {}
    for model_name, model_scores in scores.items():
        for item_id, score in model_scores.items():
            if item_id not in item_scores:
                item_scores[item_id] = []
            item_scores[item_id].append(score)

    # Compute mean per item
    mean_scores = {
        item_id: np.mean(score_list)
        for item_id, score_list in item_scores.items()
    }

    # Step 3: Scale mean scores to unit interval using MEAN SCORES min/max
    # NOTE: min/max of mean_scores is used for difficulty scaling,
    # while min/max of all individual scores is used for abilities/k estimation
    scaled_scores = scale_to_unit_interval(mean_scores, epsilon=0.001)

    # Step 4: Convert to logit difficulties
    # difficulty = logit(1 - scaled_p_correct)
    difficulties = {
        item_id: logit(1.0 - p_scaled)
        for item_id, p_scaled in scaled_scores.items()
    }

    # Step 5: Estimate abilities (using scaled scores)
    abilities = estimate_abilities(scores, min_score, max_score)

    # Step 6: Optionally filter negative discrimination items
    n_filtered = 0
    if filter_negative:
        difficulties, n_filtered = filter_negative_discrimination(
            scores, difficulties, abilities
        )

    # Step 7: Optionally estimate k
    if estimate_k:
        if all_observations is not None:
            # Use ALL individual observations
            k = estimate_noise_k_all_observations(
                all_observations, difficulties, abilities, min_score, max_score
            )
        else:
            # Use pre-averaged scores
            k = estimate_noise_k(scores, difficulties, abilities, min_score, max_score)
        discrimination = 1.0 / np.sqrt(k)
    else:
        k = 1.0
        discrimination = 1.0

    return CalibrationResult(
        difficulties=difficulties,
        k=k,
        discrimination=discrimination,
        abilities=abilities,
        n_filtered=n_filtered,
    )

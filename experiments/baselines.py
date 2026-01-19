"""Baseline implementations for ranking experiments.

This module provides:
- run_random_baseline: Random sampling with matched budget
- run_fixed_length_cat: Fixed-length CAT giving all models same number of items
"""

import copy
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from continuous_cat.cat import (
    CATSettings,
    Item,
    run_cat,
)


def run_random_baseline(
    holdout_scores: Dict[str, Dict[int, float]],
    token_counts: Dict[int, Dict[str, int]],
    model_costs: Dict[str, Dict[str, float]],
    holdout_models: List[str],
    matched_cost: float,
    seed: int,
) -> Tuple[List[str], Dict[str, int], float]:
    """Run random sampling baseline with matched budget.

    Simple random sampling:
    - Pick random model, pick random item
    - If we can afford it, sample it
    - Stop as soon as we hit an item we can't afford

    Args:
        holdout_scores: Dict of model -> {item_id: score}
        token_counts: Dict of item_id -> {input_tokens, output_tokens}
        model_costs: Dict of model -> {input, output}
        holdout_models: Models to rank
        matched_cost: Budget matched to adaptive's actual cost
        seed: Random seed

    Returns:
        (ranking, items_per_model, total_cost)
    """
    rng = np.random.RandomState(seed)

    # Track sampled items per model (to avoid re-sampling)
    sampled_items: Dict[str, Set[int]] = {m: set() for m in holdout_models}
    scores_per_model: Dict[str, List[float]] = {m: [] for m in holdout_models}
    remaining_budget = matched_cost
    total_cost = 0.0

    # Compute cost for a specific item and model
    def get_item_cost(model: str, item_id: int) -> float:
        if model not in model_costs:
            return 0.001
        input_rate = model_costs[model]["input"] / 1_000_000
        output_rate = model_costs[model]["output"] / 1_000_000
        tokens = token_counts[item_id]
        return tokens["input_tokens"] * input_rate + tokens["output_tokens"] * output_rate

    # Track available models (those with items left)
    # Use sorted list for deterministic iteration order across runs
    available_models = sorted(holdout_models)

    while remaining_budget > 0 and available_models:
        # Pick a random model that still has items
        models_with_items = [
            m for m in available_models
            if any(item_id not in sampled_items[m] for item_id in holdout_scores[m].keys())
        ]

        if not models_with_items:
            break

        model = rng.choice(models_with_items)

        # Find remaining items for this model
        # Sort item IDs for deterministic order
        remaining_items = sorted([
            item_id for item_id in holdout_scores[model].keys()
            if item_id not in sampled_items[model]
        ])

        if not remaining_items:
            available_models.remove(model)
            continue

        # Sample one item
        item_id = rng.choice(remaining_items)

        # Calculate exact cost
        item_cost = get_item_cost(model, item_id)

        # Stop if we can't afford this item
        if item_cost > remaining_budget:
            break

        # Record sample
        sampled_items[model].add(item_id)
        scores_per_model[model].append(holdout_scores[model][item_id])
        remaining_budget -= item_cost
        total_cost += item_cost

    # Compute mean scores and ranking
    mean_scores = {}
    items_per_model = {}
    for model in holdout_models:
        items_per_model[model] = len(scores_per_model[model])
        if scores_per_model[model]:
            mean_scores[model] = float(np.mean(scores_per_model[model]))
        else:
            mean_scores[model] = 0.0

    ranking = sorted(holdout_models, key=lambda m: mean_scores[m], reverse=True)

    return ranking, items_per_model, total_cost


def run_fixed_length_cat(
    items: List[Item],
    holdout_scores: Dict[str, Dict[int, float]],
    token_counts: Dict[int, Dict[str, int]],
    model_costs: Dict[str, Dict[str, float]],
    holdout_models: List[str],
    items_per_model: int,
    seed: int,
) -> Tuple[List[str], Dict[str, int], float]:
    """Run fixed-length CAT giving all models the same number of items.

    Uses run_cat with modified settings to ensure consistent behavior
    (same initialization, jitter, and Bayesian updates as adaptive ranker).

    Args:
        items: Calibrated items with difficulties
        holdout_scores: Dict of model -> {item_id: score}
        token_counts: Dict of item_id -> {input_tokens, output_tokens}
        model_costs: Dict of model -> {input, output}
        holdout_models: Models to rank
        items_per_model: Fixed number of items for each model
        seed: Random seed

    Returns:
        (ranking, items_per_model_dict, total_cost)
    """
    np.random.seed(seed)

    # Configure settings for fixed-length stopping
    settings = CATSettings(
        max_items=items_per_model,
        std_error_threshold=0.0,  # Disable SE-based stopping
        theta_tolerance=0.0,  # Disable theta stability stopping
    )

    # Add costs to items for cost tracking
    items_with_costs = copy.deepcopy(items)
    for item in items_with_costs:
        if item.id in token_counts:
            tokens = token_counts[item.id]
            item.input_tokens = tokens["input_tokens"]
            item.output_tokens = tokens["output_tokens"]

    # Run CAT for each model independently
    final_thetas = {}
    items_used = {}
    total_cost = 0.0

    for model in holdout_models:
        model_scores = holdout_scores[model]

        def get_score(item_id: int) -> float:
            return model_scores.get(item_id, 0.5)

        # Run CAT
        final_state = run_cat(items_with_costs, get_score, settings)

        final_thetas[model] = final_state.theta
        items_used[model] = final_state.iteration

        # Compute cost for this model
        for item_id in final_state.items_used:
            if model not in model_costs:
                total_cost += 0.001
                continue
            input_rate = model_costs[model]["input"] / 1_000_000
            output_rate = model_costs[model]["output"] / 1_000_000
            tokens = token_counts.get(item_id, {"input_tokens": 0, "output_tokens": 0})
            total_cost += tokens["input_tokens"] * input_rate + tokens["output_tokens"] * output_rate

    # Rank by theta
    ranking = sorted(holdout_models, key=lambda m: final_thetas[m], reverse=True)

    return ranking, items_used, total_cost

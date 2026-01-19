"""Multi-model adaptive ranker for continuous scores.

This module implements the two-phase adaptive ranking algorithm:
1. Warm-up: Administer min_items to each model to get initial estimates
2. Adaptive selection: Greedily select next model based on marginal value
   per dollar (SE^2 / ((n+1) * cost))

Stopping criteria:
- All adjacent pairs in ranking meet confidence threshold, OR
- Total budget exhausted
"""

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats

from .cat import (
    CATSettings,
    CATState,
    Item,
    bayesian_update,
    check_stopping_criteria,
    initialize_theta,
    select_item_mfi,
)


@dataclass
class RankingResult:
    """Result of multi-model ranking."""

    ranking: List[str]  # Model names in order (best to worst)
    states: Dict[str, CATState]  # Final CAT state per model
    confidence_matrix: Dict[Tuple[str, str], float]  # P(a > b) for pairs
    items_per_model: Dict[str, int]  # Items administered per model
    cost_per_model: Dict[str, float]  # Cost incurred per model
    total_items: int
    total_cost: float


def compute_pairwise_probability(state_a: CATState, state_b: CATState) -> float:
    """Compute P(theta_a > theta_b) using normal approximation.

    Given two models with normal posterior distributions:
    - theta_a ~ N(mu_a, SE_a^2)
    - theta_b ~ N(mu_b, SE_b^2)

    The difference theta_a - theta_b ~ N(mu_a - mu_b, SE_a^2 + SE_b^2).
    We compute P(theta_a - theta_b > 0) using the standard normal CDF.

    Args:
        state_a: CATState for model A
        state_b: CATState for model B

    Returns:
        Probability that theta_a > theta_b (between 0 and 1)
    """
    mu_a = state_a.theta if state_a.theta is not None else 0.0
    mu_b = state_b.theta if state_b.theta is not None else 0.0
    se_a = state_a.std_error if state_a.std_error is not None else float("inf")
    se_b = state_b.std_error if state_b.std_error is not None else float("inf")

    # Handle edge cases
    if np.isinf(se_a) or np.isinf(se_b):
        return 0.5

    # Compute variance of difference
    var_diff = se_a**2 + se_b**2

    if var_diff <= 0:
        return 0.5

    # Compute z-score and probability
    z = (mu_a - mu_b) / np.sqrt(var_diff)
    prob = stats.norm.cdf(z)

    return float(prob)


def get_current_ranking(states: Dict[str, CATState]) -> List[str]:
    """Get current ranking of models sorted by theta (high to low).

    Args:
        states: Dictionary mapping model names to CATStates

    Returns:
        List of model names sorted by theta in descending order
    """
    def get_theta(item: Tuple[str, CATState]) -> float:
        name, state = item
        return state.theta if state.theta is not None else float("-inf")

    sorted_items = sorted(states.items(), key=get_theta, reverse=True)
    return [name for name, _ in sorted_items]


def get_uncertain_models(
    states: Dict[str, CATState],
    ranking: List[str],
    confidence_threshold: float = 0.95,
) -> List[str]:
    """Get models involved in uncertain adjacent comparisons.

    A model is uncertain if it's directly involved in an adjacent pair
    where P is not decisively > upper_bound or < lower_bound.

    Args:
        states: Dictionary mapping model names to CATStates
        ranking: Current ranking (ordered list of model names)
        confidence_threshold: Confidence level (default 0.95 for 95%)

    Returns:
        List of model names involved in uncertain comparisons
    """
    upper_bound = 1 - (1 - confidence_threshold) / 2
    lower_bound = (1 - confidence_threshold) / 2

    uncertain_models = set()

    for i in range(len(ranking) - 1):
        model_a = ranking[i]
        model_b = ranking[i + 1]

        prob = compute_pairwise_probability(states[model_a], states[model_b])

        # If uncertain, add both models to the set
        if not (prob > upper_bound or prob < lower_bound):
            uncertain_models.add(model_a)
            uncertain_models.add(model_b)

    return list(uncertain_models)



def compute_item_cost(
    item: Item,
    model_cost: Dict[str, float],
) -> float:
    """Compute cost of evaluating an item with a model.

    Args:
        item: Item with input_tokens and output_tokens
        model_cost: Dict with 'input' and 'output' prices per 1M tokens

    Returns:
        Cost in dollars
    """
    input_cost = item.input_tokens * model_cost['input'] / 1_000_000
    output_cost = item.output_tokens * model_cost['output'] / 1_000_000
    return input_cost + output_cost


def add_costs_to_items(
    items: List[Item],
    model_cost: Dict[str, float],
) -> None:
    """Add cost attribute to items in-place.

    Args:
        items: List of items to augment
        model_cost: Dict with 'input' and 'output' prices per 1M tokens
    """
    for item in items:
        item.cost = compute_item_cost(item, model_cost)


def estimate_avg_item_cost(items: List[Item]) -> float:
    """Estimate average cost of items.

    Args:
        items: List of items with cost attribute

    Returns:
        Average cost
    """
    if not items:
        return 0.0
    return float(np.mean([item.cost for item in items]))



def rank_models(
    items: List[Item],
    model_callbacks: Dict[str, Callable[[int], float]],
    model_costs: Dict[str, Dict[str, float]],
    confidence_threshold: float = 0.95,
    min_items: int = 5,
    total_budget: Optional[float] = None,
    max_items_per_model: Optional[int] = None,
    settings: CATSettings = None,
    relative_costs: Optional[Dict[str, float]] = None,
) -> RankingResult:
    """Run two-phase adaptive ranking with total budget constraint.

    Algorithm:
    1. Warm-up: Administer min_items to each model
    2. Loop until all pairs confident OR budget exhausted:
       - Identify uncertain adjacent pairs
       - For each model in uncertain pairs, compute marginal value = SE^2 / ((n+1) * cost)
       - Select model with highest marginal value
       - Administer one item, deduct cost from budget
    3. Return ranking with per-model cost breakdown

    Args:
        items: List of items with IRT parameters and token counts
        model_callbacks: Dict mapping model_name to callback(item_id) -> score
        model_costs: Dict mapping model_name to {'input': ..., 'output': ...} per 1M tokens
            (used for BUDGET TRACKING)
        confidence_threshold: Confidence level for pairwise stopping (default 0.95)
        min_items: Minimum items per model in warm-up phase (default 5)
        total_budget: Total cost budget across all models (required)
        max_items_per_model: Maximum items per model (default: len(items) // 2)
        settings: CAT settings (default: CATSettings())
        relative_costs: Optional dict of relative costs for MODEL SELECTION preference.
            If None, all models are treated equally (1.0 each). This is separate from
            model_costs which tracks actual budget consumption.

    Returns:
        RankingResult with ranking, states, and cost breakdown
    """
    if settings is None:
        settings = CATSettings()

    # Override courier stopping criteria to be permissive
    # We want to control stopping at the multi-model level, not per-model
    settings = copy.deepcopy(settings)
    settings.std_error_threshold = 1e-9  # Effectively disabled
    settings.theta_tolerance = 1e-9  # Disable theta stability stopping

    model_names = list(model_callbacks.keys())
    n_models = len(model_names)

    if n_models < 2:
        raise ValueError("Need at least 2 models to rank")

    if max_items_per_model is None:
        max_items_per_model = len(items) // 2

    # Initialize per-model state and item pools
    states: Dict[str, CATState] = {}
    remaining_items: Dict[str, List[Item]] = {}

    for name in model_names:
        # Initialize state
        initial_theta = initialize_theta(items)
        states[name] = CATState(
            theta=initial_theta,
            std_error=settings.initial_std_error,
        )

        # Create copy of items with model-specific costs (for budget tracking)
        model_items = copy.deepcopy(items)
        add_costs_to_items(model_items, model_costs[name])
        remaining_items[name] = model_items

    # Use provided relative costs or default to equal (1.0) for all models
    if relative_costs is None:
        model_relative_costs = {name: 1.0 for name in model_names}
    else:
        # Normalize relative costs so minimum is 1.0
        min_cost = min(relative_costs.values())
        model_relative_costs = {name: relative_costs[name] / min_cost for name in model_names}

    # Track exhausted models (no items left)
    exhausted_models: Set[str] = set()

    # Phase 1: Warm-up - administer min_items to each model
    for _ in range(min_items):
        for name in model_names:
            if name in exhausted_models:
                continue

            state = states[name]
            items_pool = remaining_items[name]

            if not items_pool:
                exhausted_models.add(name)
                continue

            # Select next item
            selected_item, diff_effective = select_item_mfi(
                state.theta, state.std_error, items_pool, settings
            )

            # Get score from model
            score = model_callbacks[name](selected_item.id)

            # Update state
            state.items_used.append(selected_item.id)
            state.scores.append(score)
            state.total_cost += selected_item.cost

            state = bayesian_update(
                state, score, diff_effective, selected_item.discrimination, settings
            )
            states[name] = state

            # Remove used item
            remaining_items[name] = [
                item for item in items_pool if item.id != selected_item.id
            ]

    # Phase 2: Adaptive selection
    max_iterations = max_items_per_model * n_models

    for iteration in range(max_iterations):
        # Get current ranking
        ranking = get_current_ranking(states)

        # Get uncertain models
        uncertain_models = get_uncertain_models(states, ranking, confidence_threshold)

        if not uncertain_models:
            # All adjacent pairs are confident
            break

        # Check budget and filter available models
        current_total_cost = sum(states[name].total_cost for name in model_names)

        if total_budget is not None:
            available_models = []
            for name in uncertain_models:
                if name in exhausted_models:
                    continue
                if states[name].iteration >= max_items_per_model:
                    continue

                # Check if we can afford next item
                avg_cost = estimate_avg_item_cost(remaining_items[name])
                if current_total_cost + avg_cost <= total_budget:
                    available_models.append(name)

            if not available_models:
                # Budget exhausted
                break
        else:
            # No budget constraint - just check max_items and exhaustion
            available_models = [
                name for name in uncertain_models
                if name not in exhausted_models
                and states[name].iteration < max_items_per_model
            ]

            if not available_models:
                break

        # Compute marginal value for each available model
        # value = SE^2 / ((n+1) * relative_cost)
        values: Dict[str, float] = {}
        for name in available_models:
            state = states[name]
            n_items = state.iteration
            se = state.std_error if state.std_error is not None else float("inf")
            rel_cost = model_relative_costs[name]

            if np.isinf(se):
                value = float("inf")
            else:
                value = (se ** 2) / ((n_items + 1) * rel_cost)

            values[name] = value

        # Select model with highest marginal value
        next_model = max(values, key=values.__getitem__)

        # Administer one item to selected model
        state = states[next_model]
        items_pool = remaining_items[next_model]

        if not items_pool:
            exhausted_models.add(next_model)
            continue

        # Select next item
        selected_item, diff_effective = select_item_mfi(
            state.theta, state.std_error, items_pool, settings
        )

        # Get score from model
        score = model_callbacks[next_model](selected_item.id)

        # Update state
        state.items_used.append(selected_item.id)
        state.scores.append(score)
        state.total_cost += selected_item.cost

        state = bayesian_update(
            state, score, diff_effective, selected_item.discrimination, settings
        )
        states[next_model] = state

        # Remove used item
        remaining_items[next_model] = [
            item for item in items_pool if item.id != selected_item.id
        ]

    # Compute final ranking and confidence matrix
    ranking = get_current_ranking(states)

    confidence_matrix: Dict[Tuple[str, str], float] = {}
    for i, model_a in enumerate(ranking):
        for model_b in ranking[i + 1:]:
            prob = compute_pairwise_probability(states[model_a], states[model_b])
            confidence_matrix[(model_a, model_b)] = prob

    # Compute statistics
    items_per_model = {name: states[name].iteration for name in model_names}
    cost_per_model = {name: states[name].total_cost for name in model_names}
    total_items = sum(items_per_model.values())
    total_cost = sum(cost_per_model.values())

    return RankingResult(
        ranking=ranking,
        states=states,
        confidence_matrix=confidence_matrix,
        items_per_model=items_per_model,
        cost_per_model=cost_per_model,
        total_items=total_items,
        total_cost=total_cost,
    )

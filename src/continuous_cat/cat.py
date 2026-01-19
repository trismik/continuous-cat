"""Continuous Computerized Adaptive Testing (CAT) for continuous-valued outcomes.

This module implements single-model CAT using:
- Normal likelihood with heteroskedastic variance: σ² = μ(1-μ)/a²
- Maximum Fisher Information (MFI) item selection with windowing
- Bayesian ability estimation with posterior updates

Designed for real-valued scores in [0, 1] such as ROUGE scores or
normalized LLM-as-Judge ratings.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class CATSettings:
    """Settings for continuous CAT."""

    # Stopping criteria
    max_items: int = 150
    std_error_threshold: float = 0.20
    theta_stability_window: int = 15

    # Theta/ability estimation
    initial_std_error: float = 5.0
    theta_tolerance: float = 0.005
    theta_delta: float = 1.3338  # Window width multiplier
    epsilon: float = 1e-15  # Numerical stability threshold

    # Item selection jitter
    max_jitter: float = 0.00148
    jitter_factor: float = 0.093

    # Multi-model ranker overrides (set by ranker to disable per-model stopping)
    disable_per_model_stopping: bool = False


DEFAULT_SETTINGS = CATSettings()


@dataclass
class Item:
    """A test item with IRT parameters."""
    id: int
    difficulty: float  # b parameter (logit scale)
    discrimination: float  # a parameter
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


@dataclass
class CATState:
    """State of the CAT algorithm for a single model."""

    theta: Optional[float] = None  # Current ability estimate (logit scale)
    std_error: Optional[float] = None  # Current standard error
    items_used: List[int] = field(default_factory=list)  # Item IDs administered
    scores: List[float] = field(default_factory=list)  # Observed scores
    thetas: List[float] = field(default_factory=list)  # Theta history
    std_error_history: List[float] = field(default_factory=list)
    effective_difficulties: List[float] = field(default_factory=list)  # With jitter
    discriminations: List[float] = field(default_factory=list)
    total_cost: float = 0.0

    @property
    def iteration(self) -> int:
        """Number of items administered."""
        return len(self.items_used)


def logistic(theta: float, b: float) -> float:
    """Calculate expected score using logistic function.

    μ(θ, b) = 1 / (1 + exp(-(θ - b)))

    Note: Discrimination does NOT appear in the mean function for continuous
    outcomes. It only affects variance: σ² = μ(1-μ)/a²

    Args:
        theta: Ability estimate (logit scale)
        b: Item difficulty (logit scale)

    Returns:
        Expected score in [0, 1]
    """
    return 1.0 / (1.0 + np.exp(-(theta - b)))


def fisher_info(
    theta: float,
    difficulties: np.ndarray,
    discriminations: np.ndarray,
    epsilon: float = 1e-15,
) -> np.ndarray:
    """Calculate Fisher Information for items.

    FI(θ, b, a) = a² × μ(1-μ)

    where μ = logistic(θ - b)

    Args:
        theta: Ability estimate (logit scale)
        difficulties: Array of item difficulties (logit scale)
        discriminations: Array of item discrimination parameters
        epsilon: Small value for numerical stability

    Returns:
        Array of Fisher Information values
    """
    mu = 1.0 / (1.0 + np.exp(difficulties - theta))
    mu = np.clip(mu, epsilon, 1 - epsilon)
    fi = (discriminations ** 2) * mu * (1 - mu)
    return np.maximum(fi, epsilon)


def select_item_mfi(
    theta: float,
    std_error: float,
    items: List[Item],
    settings: CATSettings = DEFAULT_SETTINGS,
) -> Tuple[Item, float]:
    """Select next item using Maximum Fisher Information with windowing.

    Algorithm:
    1. Create window around current theta: [theta - delta*SE, theta + delta*SE]
    2. If no items in window, double window width and retry (up to 4 attempts)
    3. Apply jitter to item difficulties
    4. Select item with maximum Fisher Information

    Args:
        theta: Current ability estimate
        std_error: Current standard error
        items: Available items to choose from
        settings: CAT settings

    Returns:
        Tuple of (selected item, effective difficulty with jitter)
    """
    if not items:
        raise ValueError("No items available for selection")

    # Apply windowing to create item pool
    win = settings.theta_delta * std_error
    pool = []

    for _ in range(4):  # Try up to 4 times, doubling window width each retry
        pool = [item for item in items if abs(item.difficulty - theta) < win]
        if pool:
            break
        win = win * 2

    if not pool:
        pool = items  # Fall back to all items

    # Extract arrays for vectorized computation
    pool_difficulties = np.array([item.difficulty for item in pool])
    pool_discriminations = np.array([item.discrimination for item in pool])

    # Calculate jitter magnitude based on standard error
    jitter_magnitude = min(
        settings.max_jitter,
        settings.jitter_factor * std_error,
    )

    # Generate random jitters (in logit space)
    random_jitters = np.random.uniform(
        -jitter_magnitude, jitter_magnitude, size=len(pool_difficulties)
    )

    # Apply jitter to difficulties
    diff_effective = pool_difficulties + random_jitters

    # Calculate Fisher Information for all items
    fi_vec = fisher_info(
        theta, diff_effective, pool_discriminations, settings.epsilon
    )

    # Select item with maximum Fisher Information
    best_idx = np.argmax(fi_vec)
    best_item = pool[best_idx]
    best_diff_eff = diff_effective[best_idx]

    return best_item, float(best_diff_eff)



def bayesian_update(
    state: CATState,
    score: float,
    diff_effective: float,
    discrimination: float,
    settings: CATSettings = DEFAULT_SETTINGS,
) -> CATState:
    """Update ability estimate using Bayesian posterior update.

    Uses normal approximation to combine prior (current estimate) with
    likelihood from observed score.

    Args:
        state: Current CAT state
        score: Observed score in [0, 1]
        diff_effective: Effective difficulty (with jitter applied)
        discrimination: Item discrimination parameter
        settings: CAT settings

    Returns:
        New state with updated theta and standard error
    """
    theta = state.theta if state.theta is not None else 0.0
    std_error = state.std_error if state.std_error is not None else settings.initial_std_error

    # Calculate expected score μ
    mu = logistic(theta, diff_effective)
    mu = np.clip(mu, settings.epsilon, 1 - settings.epsilon)

    # Calculate Fisher Information: FI = a² × μ(1-μ)
    item_fi = (discrimination ** 2) * mu * (1 - mu)
    item_fi = max(item_fi, settings.epsilon)

    # Bayesian update
    prior_precision = 1.0 / (std_error ** 2)
    posterior_precision = prior_precision + item_fi
    new_posterior_var = 1.0 / posterior_precision
    new_std_error = math.sqrt(new_posterior_var)

    # Likelihood mean via Newton-Raphson step
    # θ_like = θ + (score - μ) / (μ(1-μ))
    # Note: discrimination cancels out in the gradient/FI ratio
    like_mean = theta + (score - mu) / (mu * (1 - mu))

    # Posterior mean (precision-weighted average)
    new_theta = (prior_precision * theta + item_fi * like_mean) / posterior_precision

    # Create updated state
    return CATState(
        theta=new_theta,
        std_error=new_std_error,
        items_used=state.items_used.copy(),
        scores=state.scores.copy(),
        thetas=state.thetas + [new_theta],
        std_error_history=state.std_error_history + [new_std_error],
        effective_difficulties=state.effective_difficulties + [diff_effective],
        discriminations=state.discriminations + [discrimination],
        total_cost=state.total_cost,
    )



def check_stopping_criteria(state: CATState, settings: CATSettings = DEFAULT_SETTINGS) -> bool:
    """Check if any stopping criterion is met.

    Stopping conditions:
    1. Maximum items reached
    2. Standard error below threshold
    3. Theta stability (convergence)

    Args:
        state: Current CAT state
        settings: CAT settings

    Returns:
        True if should stop, False otherwise
    """
    # Check max items
    if state.iteration >= settings.max_items:
        return True

    # Check standard error threshold
    if state.std_error is not None and state.std_error < settings.std_error_threshold:
        return True

    # Check theta stability
    if len(state.thetas) >= settings.theta_stability_window + 1:
        recent_thetas = state.thetas[-settings.theta_stability_window:]
        theta_changes = np.abs(np.diff(recent_thetas))
        if np.mean(theta_changes) < settings.theta_tolerance:
            return True

    return False


def initialize_theta(items: List[Item]) -> float:
    """Initialize ability estimate from item difficulties.

    Uses median difficulty as starting point.

    Args:
        items: Available items

    Returns:
        Initial theta estimate
    """
    if not items:
        return 0.0
    return float(np.median([item.difficulty for item in items]))


def run_cat(
    items: List[Item],
    get_score: Callable[[int], float],
    settings: CATSettings = DEFAULT_SETTINGS,
) -> CATState:
    """Run single-model CAT until stopping criterion.

    Args:
        items: List of items with IRT parameters
        get_score: Callback that takes item_id and returns score in [0, 1]
        settings: CAT settings

    Returns:
        Final CAT state with ability estimate and history
    """
    # Make a copy of items to track remaining
    remaining_items = items.copy()

    # Initialize state
    initial_theta = initialize_theta(items)
    state = CATState(
        theta=initial_theta,
        std_error=settings.initial_std_error,
    )

    while not check_stopping_criteria(state, settings) and remaining_items:
        # Select next item
        selected_item, diff_effective = select_item_mfi(
            state.theta, state.std_error, remaining_items, settings
        )

        # Get score from model
        score = get_score(selected_item.id)

        # Update state with item info
        state.items_used.append(selected_item.id)
        state.scores.append(score)
        state.total_cost += selected_item.cost

        # Perform Bayesian update
        state = bayesian_update(
            state, score, diff_effective, selected_item.discrimination, settings
        )

        # Remove used item from remaining
        remaining_items = [item for item in remaining_items if item.id != selected_item.id]

    return state


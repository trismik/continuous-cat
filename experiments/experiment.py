"""Main ranking experiment logic.

This module provides:
- run_dataset_experiment: Full experiment orchestration
- summarize_results: Compute summary statistics
- Utility functions for holdout sets and budget computation
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calibration.difficulty import calibrate
from continuous_cat.cat import Item
from continuous_cat.ranker import rank_models
from metrics.evaluation import (
    compute_confident_accuracy,
    compute_ground_truth,
    compute_kendall_tau,
    compute_tie_metrics,
)

from baselines import run_fixed_length_cat, run_random_baseline
from config import DEFAULT_CONFIG, ExperimentConfig
from data_loader import (
    get_all_calibration_observations,
    get_all_scores,
    get_all_scores_normalized,
    get_calibration_global_min_max,
    get_calibration_scores,
    get_model_costs,
    get_model_names,
    get_token_counts,
)


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""

    dataset: str
    holdout_set: int
    seed: int

    # Dataset info
    n_items: int  # Total items in dataset
    discrimination: float  # a = 1/sqrt(k)

    # Adaptive results
    adaptive_ranking: List[str]
    adaptive_tau: float
    adaptive_items: Dict[str, int]
    adaptive_cost: float
    adaptive_ties: Set[Tuple[str, str]]

    # Baseline results (random sampling with matched budget)
    baseline_ranking: List[str]
    baseline_tau: float
    baseline_items: Dict[str, int]
    baseline_cost: float

    # Fixed-length CAT results (all models get max_items)
    fixed_ranking: List[str]
    fixed_tau: float
    fixed_items: Dict[str, int]
    fixed_cost: float

    # Ground truth
    gt_ranking: List[str]
    gt_ties: Set[Tuple[str, str]]

    # Confident accuracy (accuracy on non-tie predictions)
    confident_accuracy: float


@dataclass
class AdaptiveResult:
    """Intermediate result from adaptive phase (before baseline)."""
    dataset: str
    holdout_set: int
    seed: int
    n_items: int
    discrimination: float
    adaptive_ranking: List[str]
    adaptive_tau: float
    adaptive_items: Dict[str, int]
    adaptive_cost: float
    adaptive_ties: Set[Tuple[str, str]]
    fixed_ranking: List[str]
    fixed_tau: float
    fixed_items: Dict[str, int]
    fixed_cost: float
    gt_ranking: List[str]
    gt_ties: Set[Tuple[str, str]]
    confident_accuracy: float
    # Cached data for baseline phase
    holdout_scores: Dict[str, Dict[int, float]]
    token_counts: Dict[int, Dict[str, int]]
    model_costs: Dict[str, Dict[str, float]]
    holdout_models: List[str]
    items: List[Item]


def create_holdout_sets(
    models: List[str],
    n_sets: int,
    n_per_set: int,
    seed: int = 42,
) -> List[List[str]]:
    """Create disjoint holdout sets of models.

    Args:
        models: Full list of model names
        n_sets: Number of holdout sets
        n_per_set: Models per set
        seed: Random seed for partitioning

    Returns:
        List of holdout sets (each is a list of model names)
    """
    rng = np.random.RandomState(seed)
    shuffled = list(models)
    rng.shuffle(shuffled)

    holdout_sets = []
    for i in range(n_sets):
        start = i * n_per_set
        end = start + n_per_set
        if end <= len(shuffled):
            holdout_sets.append(shuffled[start:end])

    return holdout_sets


def compute_budget(
    token_counts: Dict[int, Dict[str, int]],
    model_costs: Dict[str, Dict[str, float]],
    holdout_models: List[str],
    target_items: int,
) -> float:
    """Compute budget for a holdout set.

    budget = avg_cost_per_item × n_models × target_items_per_model

    Args:
        token_counts: Dict of item_id -> {input_tokens, output_tokens}
        model_costs: Dict of model_name -> {input, output}
        holdout_models: Models in this holdout set
        target_items: Target items per model

    Returns:
        Total budget in dollars
    """
    # Compute average cost per item per model
    avg_costs = []
    for model in holdout_models:
        if model not in model_costs:
            continue

        input_cost = model_costs[model]["input"] / 1_000_000
        output_cost = model_costs[model]["output"] / 1_000_000

        item_costs = []
        for item_id, tokens in token_counts.items():
            cost = tokens["input_tokens"] * input_cost + tokens["output_tokens"] * output_cost
            item_costs.append(cost)

        avg_costs.append(np.mean(item_costs))

    if not avg_costs:
        return float("inf")

    avg_cost_per_item = np.mean(avg_costs)
    return avg_cost_per_item * len(holdout_models) * target_items


def run_adaptive_phase(
    dataset_name: str,
    holdout_models: List[str],
    held_in_models: List[str],
    seed: int,
    config: ExperimentConfig,
) -> AdaptiveResult:
    """Run adaptive phase of experiment (before baseline).

    Args:
        dataset_name: Name of the dataset
        holdout_models: Models to rank
        held_in_models: Models for calibration
        seed: Random seed
        config: Experiment configuration

    Returns:
        AdaptiveResult with adaptive and fixed rankings (baseline pending)
    """
    # Load data
    token_counts = get_token_counts(dataset_name)
    model_costs = get_model_costs()

    # Get calibration scores (held-in models, all temps)
    calibration_scores = get_calibration_scores(
        dataset_name, held_in_models, config.calibration_temps
    )

    # Get global min/max and all observations for calibration
    global_min, global_max = get_calibration_global_min_max(
        dataset_name, held_in_models, config.calibration_temps
    )
    all_observations = get_all_calibration_observations(
        dataset_name, held_in_models, config.calibration_temps
    )

    # Calibrate difficulties
    cal_result = calibrate(
        calibration_scores,
        estimate_k=True,
        filter_negative=config.filter_negative_discrimination,
        global_min_score=global_min,
        global_max_score=global_max,
        all_observations=all_observations,
    )

    # Create items with calibrated difficulties
    items = []
    for item_id, diff in cal_result.difficulties.items():
        if item_id not in token_counts:
            continue
        items.append(Item(
            id=item_id,
            difficulty=diff,
            discrimination=cal_result.discrimination,
            input_tokens=token_counts[item_id]["input_tokens"],
            output_tokens=token_counts[item_id]["output_tokens"],
        ))

    # Compute budget
    budget = compute_budget(
        token_counts, model_costs, holdout_models, config.target_items_per_model
    )

    # Get normalized scores for CAT (preserves relative differences between models)
    # Normalization maps scores to [0, 1] using GLOBAL min/max across all models
    holdout_scores_normalized = get_all_scores_normalized(
        dataset_name, holdout_models, config.eval_temp
    )

    def make_callback(model: str):
        scores = holdout_scores_normalized[model]
        def callback(item_id: int) -> float:
            return scores.get(item_id, 0.5)
        return callback

    callbacks = {model: make_callback(model) for model in holdout_models}

    # Also get raw scores for ground truth computation and caching
    holdout_scores = get_all_scores(dataset_name, holdout_models, config.eval_temp)

    # Filter model costs to only holdout models
    holdout_costs = {
        model: model_costs[model]
        for model in holdout_models
        if model in model_costs
    }

    # Compute relative costs for model selection
    # Uses simple average: (input_per_1m + output_per_1m) / 2
    # Then normalizes so min = 1.0
    model_simple_costs = {}
    for model in holdout_models:
        if model not in model_costs:
            model_simple_costs[model] = 1.0
            continue
        # Simple average of input and output rates
        model_simple_costs[model] = (model_costs[model]["input"] + model_costs[model]["output"]) / 2

    min_cost = min(model_simple_costs.values())
    relative_costs = {m: c / min_cost for m, c in model_simple_costs.items()}

    # Run adaptive ranking
    np.random.seed(seed)
    result = rank_models(
        items=items,
        model_callbacks=callbacks,
        model_costs=holdout_costs,
        confidence_threshold=config.confidence_level,
        min_items=config.min_items_per_model,
        total_budget=budget,
        max_items_per_model=config.max_items_per_model,
        relative_costs=relative_costs,
    )

    # Compute ground truth
    gt_scores = {
        model: list(holdout_scores[model].values())
        for model in holdout_models
    }
    gt = compute_ground_truth(
        gt_scores,
        n_bootstrap=config.bootstrap_n,
        confidence_level=config.confidence_level,
    )

    # Compute Kendall's tau for adaptive
    adaptive_tau = compute_kendall_tau(result.ranking, gt.models)

    # Extract ties from adaptive result
    adaptive_ties = set()
    for (a, b), prob in result.confidence_matrix.items():
        upper = 1 - (1 - config.confidence_level) / 2
        lower = (1 - config.confidence_level) / 2
        if not (prob > upper or prob < lower):
            adaptive_ties.add(tuple(sorted([a, b])))

    # Run fixed-length CAT (all models get max items used by adaptive)
    # Use normalized scores for fair comparison with adaptive
    max_items_used = max(result.items_per_model.values())
    fixed_ranking, fixed_items, fixed_cost = run_fixed_length_cat(
        items=items,
        holdout_scores=holdout_scores_normalized,
        token_counts=token_counts,
        model_costs=model_costs,
        holdout_models=holdout_models,
        items_per_model=max_items_used,
        seed=seed,
    )
    fixed_tau = compute_kendall_tau(fixed_ranking, gt.models)

    # Compute confident accuracy (accuracy on non-tie predictions)
    conf_acc = compute_confident_accuracy(result.ranking, gt.models, adaptive_ties)

    return AdaptiveResult(
        dataset=dataset_name,
        holdout_set=0,  # Will be set by caller
        seed=seed,
        n_items=len(token_counts),
        discrimination=cal_result.discrimination,
        adaptive_ranking=result.ranking,
        adaptive_tau=adaptive_tau,
        adaptive_items=result.items_per_model,
        adaptive_cost=result.total_cost,
        adaptive_ties=adaptive_ties,
        fixed_ranking=fixed_ranking,
        fixed_tau=fixed_tau,
        fixed_items=fixed_items,
        fixed_cost=fixed_cost,
        gt_ranking=gt.models,
        gt_ties=gt.ties,
        confident_accuracy=conf_acc,
        # Cache for baseline phase - use normalized scores for consistency
        holdout_scores=holdout_scores_normalized,
        token_counts=token_counts,
        model_costs=model_costs,
        holdout_models=holdout_models,
        items=items,
    )


def run_baseline_phase(
    adaptive_result: AdaptiveResult,
    matched_cost: float,
    config: ExperimentConfig,
) -> ExperimentResult:
    """Run baseline phase with matched cost budget.

    Args:
        adaptive_result: Result from adaptive phase
        matched_cost: Cost budget for baseline (averaged across seeds)
        config: Experiment configuration

    Returns:
        Complete ExperimentResult with baseline
    """
    # Run baseline with matched budget
    baseline_ranking, baseline_items, baseline_cost = run_random_baseline(
        holdout_scores=adaptive_result.holdout_scores,
        token_counts=adaptive_result.token_counts,
        model_costs=adaptive_result.model_costs,
        holdout_models=adaptive_result.holdout_models,
        matched_cost=matched_cost,
        seed=adaptive_result.seed,
    )
    baseline_tau = compute_kendall_tau(baseline_ranking, adaptive_result.gt_ranking)

    return ExperimentResult(
        dataset=adaptive_result.dataset,
        holdout_set=adaptive_result.holdout_set,
        seed=adaptive_result.seed,
        n_items=adaptive_result.n_items,
        discrimination=adaptive_result.discrimination,
        adaptive_ranking=adaptive_result.adaptive_ranking,
        adaptive_tau=adaptive_result.adaptive_tau,
        adaptive_items=adaptive_result.adaptive_items,
        adaptive_cost=adaptive_result.adaptive_cost,
        adaptive_ties=adaptive_result.adaptive_ties,
        baseline_ranking=baseline_ranking,
        baseline_tau=baseline_tau,
        baseline_items=baseline_items,
        baseline_cost=baseline_cost,
        fixed_ranking=adaptive_result.fixed_ranking,
        fixed_tau=adaptive_result.fixed_tau,
        fixed_items=adaptive_result.fixed_items,
        fixed_cost=adaptive_result.fixed_cost,
        gt_ranking=adaptive_result.gt_ranking,
        gt_ties=adaptive_result.gt_ties,
        confident_accuracy=adaptive_result.confident_accuracy,
    )


def run_dataset_experiment(
    dataset_name: str,
    config: ExperimentConfig = None,
) -> List[ExperimentResult]:
    """Run full experiment for a dataset.

    Uses two-phase matched budget mode:
    1. Run all adaptive seeds first, collect costs
    2. Compute average cost across seeds
    3. Run all baselines with that averaged cost

    Args:
        dataset_name: Name of the dataset
        config: Experiment configuration

    Returns:
        List of ExperimentResult for all holdout sets and seeds
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Get all models
    all_models = get_model_names(dataset_name)
    print(f"Dataset: {dataset_name}")
    print(f"  Total models: {len(all_models)}")

    # Create holdout sets
    holdout_sets = create_holdout_sets(
        all_models,
        config.n_holdout_sets,
        config.n_holdout_models,
        config.partition_seed,
    )
    print(f"  Holdout sets: {len(holdout_sets)}")

    results = []

    for holdout_idx, holdout_models in enumerate(holdout_sets):
        # Held-in models are all models not in holdout
        held_in_models = [m for m in all_models if m not in holdout_models]

        print(f"  Holdout set {holdout_idx + 1}: {holdout_models}")

        # Phase 1: Run all adaptive seeds first
        print(f"    Phase 1: Running adaptive for all {config.n_seeds} seeds...")
        adaptive_results = []
        for seed in range(config.n_seeds):
            adaptive_result = run_adaptive_phase(
                dataset_name=dataset_name,
                holdout_models=holdout_models,
                held_in_models=held_in_models,
                seed=seed,
                config=config,
            )
            adaptive_result.holdout_set = holdout_idx
            adaptive_results.append(adaptive_result)

            if (seed + 1) % 5 == 0:
                print(f"      Seed {seed + 1}/{config.n_seeds}: adaptive_tau={adaptive_result.adaptive_tau:.3f}")

        # Compute average cost across all adaptive seeds
        avg_cost = np.mean([r.adaptive_cost for r in adaptive_results])
        avg_items = np.mean([sum(r.adaptive_items.values()) for r in adaptive_results])
        print(f"    Matched budget: {avg_items:.1f} items, ${avg_cost:.4f}")

        # Phase 2: Run all baselines with matched cost
        print(f"    Phase 2: Running baseline for all {config.n_seeds} seeds...")
        for seed, adaptive_result in enumerate(adaptive_results):
            result = run_baseline_phase(
                adaptive_result=adaptive_result,
                matched_cost=avg_cost,  # Use averaged cost across all seeds
                config=config,
            )
            results.append(result)

            if (seed + 1) % 5 == 0:
                print(f"      Seed {seed + 1}/{config.n_seeds}: baseline_tau={result.baseline_tau:.3f}")

    return results


def summarize_results(results: List[ExperimentResult]) -> Dict:
    """Summarize experiment results.

    Args:
        results: List of experiment results

    Returns:
        Summary statistics
    """
    # Dataset info (same across all results)
    n_items = results[0].n_items
    discrimination = np.mean([r.discrimination for r in results])

    # Adaptive stats
    adaptive_taus = [r.adaptive_tau for r in results]
    adaptive_items = [sum(r.adaptive_items.values()) for r in results]
    adaptive_costs = [r.adaptive_cost for r in results]

    # Baseline stats
    baseline_taus = [r.baseline_tau for r in results]
    baseline_items = [sum(r.baseline_items.values()) for r in results]
    baseline_costs = [r.baseline_cost for r in results]

    # Fixed-length stats
    fixed_taus = [r.fixed_tau for r in results]
    fixed_items = [sum(r.fixed_items.values()) for r in results]
    fixed_costs = [r.fixed_cost for r in results]

    # Tie detection metrics
    all_tie_metrics = []
    for r in results:
        tm = compute_tie_metrics(r.adaptive_ties, r.gt_ties)
        all_tie_metrics.append(tm)

    # Tie percentages
    n_models = len(results[0].adaptive_ranking)
    n_pairs = n_models * (n_models - 1) // 2
    adaptive_tie_pcts = [len(r.adaptive_ties) / n_pairs for r in results]
    gt_tie_pcts = [len(r.gt_ties) / n_pairs for r in results]

    # Confident accuracy
    conf_accs = [r.confident_accuracy for r in results if r.confident_accuracy is not None]

    # % Used relative to exhaustive evaluation (items × models)
    exhaustive_items = n_items * n_models

    # Item and cost savings (adaptive vs fixed)
    item_savings = [1 - a / f if f > 0 else 0 for a, f in zip(adaptive_items, fixed_items)]
    cost_savings = [1 - a / f if f > 0 else 0 for a, f in zip(adaptive_costs, fixed_costs)]

    return {
        "n_runs": len(results),
        # Dataset info
        "n_items": n_items,
        "discrimination": discrimination,
        # Adaptive
        "adaptive_tau_mean": np.mean(adaptive_taus),
        "adaptive_tau_std": np.std(adaptive_taus),
        "adaptive_items_mean": np.mean(adaptive_items),
        "adaptive_items_std": np.std(adaptive_items),
        "adaptive_cost_mean": np.mean(adaptive_costs),
        "adaptive_cost_std": np.std(adaptive_costs),
        "pct_used_mean": np.mean(adaptive_items) / exhaustive_items * 100,
        "pct_used_std": np.std(adaptive_items) / exhaustive_items * 100,
        # Baseline
        "baseline_tau_mean": np.mean(baseline_taus),
        "baseline_tau_std": np.std(baseline_taus),
        "baseline_items_mean": np.mean(baseline_items),
        "baseline_items_std": np.std(baseline_items),
        "baseline_cost_mean": np.mean(baseline_costs),
        "baseline_cost_std": np.std(baseline_costs),
        # Fixed-length
        "fixed_tau_mean": np.mean(fixed_taus),
        "fixed_tau_std": np.std(fixed_taus),
        "fixed_items_mean": np.mean(fixed_items),
        "fixed_items_std": np.std(fixed_items),
        "fixed_cost_mean": np.mean(fixed_costs),
        "fixed_cost_std": np.std(fixed_costs),
        # Tau improvement (adaptive vs baseline)
        "tau_improvement": np.mean(adaptive_taus) - np.mean(baseline_taus),
        # Tau difference (adaptive vs fixed)
        "tau_diff_vs_fixed": np.mean(adaptive_taus) - np.mean(fixed_taus),
        # Item and cost savings (adaptive vs fixed)
        "item_savings_mean": np.mean(item_savings) * 100,
        "cost_savings_mean": np.mean(cost_savings) * 100,
        # Tie detection
        "adaptive_tie_pct_mean": np.mean(adaptive_tie_pcts) * 100,
        "gt_tie_pct_mean": np.mean(gt_tie_pcts) * 100,
        "tie_precision_mean": np.mean([tm.precision for tm in all_tie_metrics]),
        "tie_recall_mean": np.mean([tm.recall for tm in all_tie_metrics]),
        "tie_f1_mean": np.mean([tm.f1 for tm in all_tie_metrics]),
        "confident_accuracy_mean": np.mean(conf_accs) if conf_accs else None,
    }

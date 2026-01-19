"""Family holdout experiment.

Calibrates on large model families (OpenAI, Meta, Google) and
evaluates on held-out smaller families (Mistral, Qwen, Amazon).
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calibration.difficulty import calibrate
from continuous_cat.cat import Item
from continuous_cat.ranker import rank_models
from metrics.evaluation import compute_ground_truth, compute_kendall_tau

from baselines import run_random_baseline
from config import (
    CALIBRATION_FAMILIES,
    DEFAULT_CONFIG,
    ExperimentConfig,
    HOLDOUT_FAMILY_MODELS,
    MODEL_FAMILIES,
)
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
from experiment import compute_budget


@dataclass
class FamilyHoldoutResult:
    """Result of family holdout experiment."""
    dataset: str
    seed: int
    adaptive_tau: float
    baseline_tau: float
    discrimination: float


def run_family_holdout_experiment(
    dataset_name: str,
    n_seeds: int = 20,
    config: ExperimentConfig = None,
) -> List[FamilyHoldoutResult]:
    """Run family holdout experiment.

    Calibrates on OpenAI, Meta, Google models and evaluates on
    held-out Mistral, Qwen, Amazon models.

    Args:
        dataset_name: Name of the dataset
        n_seeds: Number of random seeds
        config: Experiment configuration

    Returns:
        List of FamilyHoldoutResult for all seeds
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Get all models
    all_models = get_model_names(dataset_name)

    # Get holdout models (1 from each small family)
    holdout_models = [
        m for m in HOLDOUT_FAMILY_MODELS.values()
        if m in all_models
    ]

    # Get calibration models (from large families)
    calibration_models = []
    for family in CALIBRATION_FAMILIES:
        for model in MODEL_FAMILIES.get(family, []):
            if model in all_models:
                calibration_models.append(model)

    print(f"Dataset: {dataset_name}")
    print(f"  Holdout models ({len(holdout_models)}): {holdout_models}")
    print(f"  Calibration models ({len(calibration_models)})")

    if len(holdout_models) < 2:
        print("  Not enough holdout models, skipping")
        return []

    # Load data
    token_counts = get_token_counts(dataset_name)
    model_costs = get_model_costs()

    # Get calibration scores
    calibration_scores = get_calibration_scores(
        dataset_name, calibration_models, config.calibration_temps
    )

    # Get global min/max and all observations for calibration
    global_min, global_max = get_calibration_global_min_max(
        dataset_name, calibration_models, config.calibration_temps
    )
    all_observations = get_all_calibration_observations(
        dataset_name, calibration_models, config.calibration_temps
    )

    # Calibrate difficulties using only calibration models
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

    # Get raw scores for ground truth
    holdout_scores_raw = get_all_scores(dataset_name, holdout_models, config.eval_temp)

    # Get normalized scores for CAT
    holdout_scores_normalized = get_all_scores_normalized(
        dataset_name, holdout_models, config.eval_temp
    )

    # Compute ground truth (using raw scores)
    gt_scores = {
        model: list(holdout_scores_raw[model].values())
        for model in holdout_models
    }
    gt = compute_ground_truth(
        gt_scores,
        n_bootstrap=config.bootstrap_n,
        confidence_level=config.confidence_level,
    )

    # Compute budget
    budget = compute_budget(
        token_counts, model_costs, holdout_models, config.target_items_per_model
    )

    # Filter model costs
    holdout_costs = {
        model: model_costs[model]
        for model in holdout_models
        if model in model_costs
    }

    # Compute relative costs for model selection
    # Uses simple average: (input_per_1m + output_per_1m) / 2
    model_simple_costs = {}
    for model in holdout_models:
        if model not in model_costs:
            model_simple_costs[model] = 1.0
            continue
        model_simple_costs[model] = (model_costs[model]["input"] + model_costs[model]["output"]) / 2

    min_cost = min(model_simple_costs.values())
    relative_costs = {m: c / min_cost for m, c in model_simple_costs.items()}

    results = []

    for seed in range(n_seeds):
        # Create score callbacks (using normalized scores)
        def make_callback(model: str):
            scores = holdout_scores_normalized[model]
            def callback(item_id: int) -> float:
                return scores.get(item_id, 0.5)
            return callback

        callbacks = {model: make_callback(model) for model in holdout_models}

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

        adaptive_tau = compute_kendall_tau(result.ranking, gt.models)

        # Run baseline (using normalized scores for consistency)
        baseline_ranking, _, _ = run_random_baseline(
            holdout_scores=holdout_scores_normalized,
            token_counts=token_counts,
            model_costs=model_costs,
            holdout_models=holdout_models,
            matched_cost=result.total_cost,
            seed=seed,
        )
        baseline_tau = compute_kendall_tau(baseline_ranking, gt.models)

        results.append(FamilyHoldoutResult(
            dataset=dataset_name,
            seed=seed,
            adaptive_tau=adaptive_tau,
            baseline_tau=baseline_tau,
            discrimination=cal_result.discrimination,
        ))

        if (seed + 1) % 5 == 0:
            print(f"  Seed {seed + 1}/{n_seeds}: adaptive={adaptive_tau:.3f}, baseline={baseline_tau:.3f}")

    return results


def summarize_family_holdout(results: List[FamilyHoldoutResult]) -> Dict:
    """Summarize family holdout results."""
    if not results:
        return {}

    adaptive_taus = [r.adaptive_tau for r in results]
    baseline_taus = [r.baseline_tau for r in results]

    return {
        "n_runs": len(results),
        "adaptive_tau_mean": np.mean(adaptive_taus),
        "adaptive_tau_std": np.std(adaptive_taus),
        "baseline_tau_mean": np.mean(baseline_taus),
        "baseline_tau_std": np.std(baseline_taus),
        "discrimination": np.mean([r.discrimination for r in results]),
    }

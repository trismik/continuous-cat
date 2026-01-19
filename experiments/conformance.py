"""Conformance analysis for heteroskedastic model.

Tests how well the heteroskedastic model predicts score variance:
σ² = k × μ(1-μ) where μ is predicted mean and k is noise parameter.
"""

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calibration.difficulty import calibrate
from config import DEFAULT_CONFIG, ExperimentConfig
from data_loader import (
    get_all_calibration_observations,
    get_calibration_global_min_max,
    get_calibration_scores,
    get_model_names,
    load_data,
)


@dataclass
class ConformanceResult:
    """Result of conformance analysis."""
    dataset: str
    discrimination: float
    k: float
    r_squared: float
    tau: float  # Kendall's tau between predicted and observed variance


def run_conformance_analysis(
    dataset_name: str,
    config: ExperimentConfig = None,
) -> ConformanceResult:
    """Run conformance analysis for a dataset.

    Tests how well the heteroskedastic model predicts score variance:
    σ² = k × μ(1-μ) where μ is predicted mean and k is noise parameter.

    The variance is computed per (item, model) pair across temperatures,
    then compared to the predicted variance based on the mean score.

    Args:
        dataset_name: Name of the dataset
        config: Experiment configuration

    Returns:
        ConformanceResult with R² and tau metrics
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Get all models
    all_models = get_model_names(dataset_name)

    # Get scores for all temperatures (to compute variance)
    all_temps = config.calibration_temps

    # Calibrate to get k and discrimination
    calibration_scores = get_calibration_scores(
        dataset_name, all_models, all_temps
    )

    # Get global min/max and all observations for calibration
    global_min, global_max = get_calibration_global_min_max(
        dataset_name, all_models, all_temps
    )
    all_observations = get_all_calibration_observations(
        dataset_name, all_models, all_temps
    )

    cal_result = calibrate(
        calibration_scores,
        estimate_k=True,
        filter_negative=False,  # Include all items for conformance
        global_min_score=global_min,
        global_max_score=global_max,
        all_observations=all_observations,
    )

    # Collect per-(item, model) scores across temperatures
    item_model_scores = defaultdict(list)  # (item_id, model) -> list of scores

    data = load_data()
    dataset = data["datasets"][dataset_name]

    for model_name, model_data in dataset["scores"].items():
        if model_name not in all_models:
            continue
        for temp, temp_scores in model_data.items():
            if temp not in all_temps:
                continue
            for item_id_str, score in temp_scores.items():
                item_id = int(item_id_str)
                if item_id in cal_result.difficulties:
                    item_model_scores[(item_id, model_name)].append(score)

    # Compute observed and predicted variance per (item, model) pair
    observed_vars = []
    predicted_vars = []

    for (item_id, model), scores in item_model_scores.items():
        if len(scores) < 2:
            continue

        # Observed variance across temperatures
        obs_var = np.var(scores, ddof=1)

        # Predicted variance: k × μ(1-μ)
        # Use mean score as estimate of μ
        mu = np.mean(scores)
        mu = np.clip(mu, 0.01, 0.99)  # Avoid extreme values
        pred_var = cal_result.k * mu * (1 - mu)

        observed_vars.append(obs_var)
        predicted_vars.append(pred_var)

    if len(observed_vars) < 5:
        return ConformanceResult(
            dataset=dataset_name,
            discrimination=cal_result.discrimination,
            k=cal_result.k,
            r_squared=0.0,
            tau=0.0,
        )

    observed_vars = np.array(observed_vars)
    predicted_vars = np.array(predicted_vars)

    # Compute correlation-based R² (Pearson r²)
    # This measures how well predicted variance tracks observed variance,
    # regardless of absolute scale differences
    correlation = np.corrcoef(predicted_vars, observed_vars)[0, 1]
    r_squared = correlation ** 2 if not np.isnan(correlation) else 0.0

    # Compute Kendall's tau between predicted and observed rankings
    tau, _ = sp_stats.kendalltau(predicted_vars, observed_vars)

    return ConformanceResult(
        dataset=dataset_name,
        discrimination=cal_result.discrimination,
        k=cal_result.k,
        r_squared=r_squared,
        tau=tau if not np.isnan(tau) else 0.0,
    )

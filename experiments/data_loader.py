"""Data loading utilities for experiments."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_PATH = Path(__file__).parent.parent / "data" / "scores.json"


def load_data() -> Dict[str, Any]:
    """Load the bundled score data."""
    with open(DATA_PATH) as f:
        return json.load(f)


def get_dataset_names() -> List[str]:
    """Get list of available dataset names."""
    data = load_data()
    return list(data["datasets"].keys())


def get_model_names(dataset_name: str, exclude: Optional[List[str]] = None) -> List[str]:
    """Get list of model names for a dataset."""
    data = load_data()
    dataset = data["datasets"][dataset_name]
    models = list(dataset["scores"].keys())

    if exclude:
        models = [m for m in models if m not in exclude]

    return sorted(models)


def get_scores(
    dataset_name: str,
    model_name: str,
    temp: str = "temp_0.0",
) -> Dict[int, float]:
    """Get scores for a model on a dataset.

    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model
        temp: Temperature config (default: temp_0.0)

    Returns:
        Dict mapping item_id to score
    """
    data = load_data()
    dataset = data["datasets"][dataset_name]
    model_scores = dataset["scores"].get(model_name, {})
    temp_scores = model_scores.get(temp, {})

    # Convert string keys to int
    return {int(k): v for k, v in temp_scores.items()}


def get_all_scores(
    dataset_name: str,
    models: List[str],
    temp: str = "temp_0.0",
) -> Dict[str, Dict[int, float]]:
    """Get scores for multiple models.

    Args:
        dataset_name: Name of the dataset
        models: List of model names
        temp: Temperature config

    Returns:
        Dict mapping model_name to {item_id: score}
    """
    return {
        model: get_scores(dataset_name, model, temp)
        for model in models
    }


def get_token_counts(dataset_name: str) -> Dict[int, Dict[str, int]]:
    """Get token counts for a dataset.

    Returns:
        Dict mapping item_id to {input_tokens, output_tokens}
    """
    data = load_data()
    dataset = data["datasets"][dataset_name]
    items = dataset["items"]

    # Convert string keys to int
    return {int(k): v for k, v in items.items()}


def get_model_costs() -> Dict[str, Dict[str, float]]:
    """Get model costs (per 1M tokens).

    Returns:
        Dict mapping model_name to {input, output}
    """
    data = load_data()
    return data["model_costs"]


def get_calibration_scores(
    dataset_name: str,
    models: List[str],
    temps: List[str] = None,
) -> Dict[str, Dict[int, float]]:
    """Get scores for calibration (averaged across temperatures).

    Args:
        dataset_name: Name of the dataset
        models: List of model names
        temps: List of temperature configs (default: all temps)

    Returns:
        Dict mapping model_name to {item_id: avg_score}
    """
    if temps is None:
        temps = ["temp_0.0", "temp_0.4", "temp_0.7", "temp_1.0"]

    data = load_data()
    dataset = data["datasets"][dataset_name]

    result = {}
    for model in models:
        model_scores = dataset["scores"].get(model, {})

        # Aggregate scores across temperatures
        item_scores = {}
        for temp in temps:
            temp_scores = model_scores.get(temp, {})
            for item_id, score in temp_scores.items():
                item_id = int(item_id)
                if item_id not in item_scores:
                    item_scores[item_id] = []
                item_scores[item_id].append(score)

        # Average across temperatures
        result[model] = {
            item_id: sum(scores) / len(scores)
            for item_id, scores in item_scores.items()
        }

    return result


def get_calibration_global_min_max(
    dataset_name: str,
    models: List[str],
    temps: List[str] = None,
) -> tuple:
    """Get global min/max from all individual (model, temp, item) observations.

    Args:
        dataset_name: Name of the dataset
        models: List of model names
        temps: List of temperature configs (default: all temps)

    Returns:
        Tuple of (global_min, global_max)
    """
    if temps is None:
        temps = ["temp_0.0", "temp_0.4", "temp_0.7", "temp_1.0"]

    data = load_data()
    dataset = data["datasets"][dataset_name]

    all_scores = []
    for model in models:
        model_scores = dataset["scores"].get(model, {})
        for temp in temps:
            temp_scores = model_scores.get(temp, {})
            all_scores.extend(temp_scores.values())

    return min(all_scores), max(all_scores)


def get_all_calibration_observations(
    dataset_name: str,
    models: List[str],
    temps: List[str] = None,
) -> Dict[str, Dict[int, List[float]]]:
    """Get all individual (model, temp, item) observations for k estimation.

    Args:
        dataset_name: Name of the dataset
        models: List of model names
        temps: List of temperature configs (default: all temps)

    Returns:
        Dict mapping model_name to {item_id: [list of scores across temps]}
    """
    if temps is None:
        temps = ["temp_0.0", "temp_0.4", "temp_0.7", "temp_1.0"]

    data = load_data()
    dataset = data["datasets"][dataset_name]

    result = {}
    for model in models:
        model_scores = dataset["scores"].get(model, {})
        item_observations: Dict[int, List[float]] = {}

        for temp in temps:
            temp_scores = model_scores.get(temp, {})
            for item_id, score in temp_scores.items():
                item_id = int(item_id)
                if item_id not in item_observations:
                    item_observations[item_id] = []
                item_observations[item_id].append(score)

        result[model] = item_observations

    return result


def compute_global_score_range(
    dataset_name: str,
    models: List[str],
    temp: str = "temp_0.0",
) -> tuple:
    """Compute global min/max scores across all models.

    Args:
        dataset_name: Name of the dataset
        models: List of model names
        temp: Temperature config

    Returns:
        Tuple of (global_min, global_max)
    """
    all_scores = []
    for model in models:
        scores = get_scores(dataset_name, model, temp)
        all_scores.extend(scores.values())

    return min(all_scores), max(all_scores)


def get_all_scores_normalized(
    dataset_name: str,
    models: List[str],
    temp: str = "temp_0.0",
) -> Dict[str, Dict[int, float]]:
    """Get normalized scores for multiple models.

    Scores are normalized to [0, 1] using global min/max across all models.

    Args:
        dataset_name: Name of the dataset
        models: List of model names
        temp: Temperature config

    Returns:
        Dict mapping model_name to {item_id: normalized_score}
    """
    global_min, global_max = compute_global_score_range(dataset_name, models, temp)
    score_range = global_max - global_min

    result = {}
    for model in models:
        scores = get_scores(dataset_name, model, temp)

        if score_range < 1e-9:
            # All scores are the same
            result[model] = {item_id: 0.5 for item_id in scores}
        else:
            result[model] = {
                item_id: (score - global_min) / score_range
                for item_id, score in scores.items()
            }

    return result

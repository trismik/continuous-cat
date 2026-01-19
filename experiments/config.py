"""Experiment configuration."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Configuration for ranking experiments."""

    # Holdout configuration
    n_seeds: int = 20  # Number of random seeds per holdout set
    n_holdout_models: int = 4  # Models per holdout set
    n_holdout_sets: int = 5  # Number of disjoint holdout sets
    partition_seed: int = 42  # Fixed seed for creating holdout sets

    # CAT configuration
    min_items_per_model: int = 10  # Warm-up items before adaptive selection
    target_items_per_model: int = 20  # Target for auto-budget calculation
    max_items_per_model: Optional[int] = None  # Hard cap (None = no cap)

    # Budget mode
    matched_budget_mode: bool = True  # Baseline matches adaptive's actual usage

    # Calibration
    calibration_temps: List[str] = field(default_factory=lambda: [
        "temp_0.0", "temp_0.4", "temp_0.7", "temp_1.0"
    ])
    filter_negative_discrimination: bool = True

    # Evaluation
    eval_temp: str = "temp_0.0"
    confidence_level: float = 0.95
    bootstrap_n: int = 10000


# Default configuration
DEFAULT_CONFIG = ExperimentConfig()

# Dataset names
# Note: govreport_bertscore excluded because all models achieved near-identical scores
DATASET_NAMES = [
    "biolaysumm_rougel",
    "biolaysumm_bertscore",
    "biolaysumm_fkgl",
    "govreport_rougel",
    # "govreport_bertscore",  # excluded: all models near-identical
    "truthfulqa_judge",
    "truthfulqa_bertscore",
    "flores_bleu",
    "flores_comet",
    "nemotron_pii",
]

# Model family definitions for family holdout experiment
MODEL_FAMILIES = {
    "openai": [
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o-mini",
    ],
    "meta": [
        "us.meta.llama4-maverick-17b-instruct-v1:0",
        "us.meta.llama4-scout-17b-instruct-v1:0",
        "us.meta.llama3-3-70b-instruct-v1:0",
        "us.meta.llama3-2-3b-instruct-v1:0",
        "us.meta.llama3-1-8b-instruct-v1:0",
    ],
    "google": [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ],
    "amazon": [
        "amazon.nova-pro-v1:0",
        "amazon.nova-lite-v1:0",
        "amazon.nova-micro-v1:0",
    ],
    "mistral": [
        "mistral.mistral-7b-instruct-v0:2",
        "mistral.mixtral-8x7b-instruct-v0:1",
    ],
    "qwen": [
        "qwen.qwen3-32b-v1:0",
        "qwen.qwen3-coder-30b-a3b-v1:0",
    ],
}

# Holdout models: 1 from each of the 3 smallest families (best model from each)
HOLDOUT_FAMILY_MODELS = {
    "mistral": "mistral.mixtral-8x7b-instruct-v0:1",
    "qwen": "qwen.qwen3-32b-v1:0",
    "amazon": "amazon.nova-pro-v1:0",
}

# Calibration families (the 3 largest)
CALIBRATION_FAMILIES = ["openai", "meta", "google"]

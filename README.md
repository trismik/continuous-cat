# Continuous CAT for LLM Evaluation

Reproducibility package for the paper "Continuous Computerized Adaptive Testing for LLM Evaluation".

This repository contains a self-contained implementation of Continuous CAT and all experiments needed to reproduce the paper's results.

## Quick Start

```bash
# Create and activate conda environment
conda create -n continuous-cat python=3.11
conda activate continuous-cat

# Install dependencies
pip install -r requirements.txt

# Run all experiments (generates Tables 2-6)
python experiments/run_all.py --n-seeds 20 --output-dir experiments/results

# Run tests
pytest tests/
```

## Repository Structure

```
continuous-cat-paper/
├── data/
│   └── scores.json              # Pre-computed scores for 10 dataset-metric combinations
│
├── src/
│   ├── continuous_cat/          # Core CAT algorithm
│   │   ├── cat.py               # Single-model CAT: MFI selection, Bayesian updates
│   │   └── ranker.py            # Multi-model adaptive ranker with pairwise stopping
│   │
│   ├── calibration/             # IRT parameter estimation
│   │   └── difficulty.py        # Item difficulties, noise k, discrimination filtering
│   │
│   └── metrics/                 # Evaluation metrics
│       └── evaluation.py        # Kendall tau, bootstrap CIs, tie detection
│
├── experiments/
│   ├── config.py                # Dataset configs, model families, hyperparameters
│   ├── data_loader.py           # Load scores from data/scores.json
│   ├── run_ranking.py           # Core experiment runner
│   └── run_all.py               # Generate all paper tables
│
├── tests/
│   └── test_metrics.py                 # Tests for evaluation metrics
│
└── requirements.txt
```

## Core Modules

### `src/continuous_cat/cat.py`
Single-model Continuous CAT implementation:
- **Logistic IRT model**: `P(Y|θ,b) = logistic(θ - b)` with heteroskedastic variance `σ² = k × μ(1-μ)`
- **Maximum Fisher Information (MFI)** item selection with windowing
- **Bayesian ability estimation** with posterior updates after each item
- **Jittered difficulties** to prevent item selection degeneracy

Key hyperparameters (preserved for reproducibility):
```python
initial_std_error = 5.0      # Prior uncertainty
theta_delta = 1.3338         # Window width multiplier
max_jitter = 0.00148         # Maximum difficulty jitter
jitter_factor = 0.093        # SE-scaled jitter
```

### `src/continuous_cat/ranker.py`
Multi-model adaptive ranker:
- Runs CAT independently for each model
- **Pairwise confidence stopping**: stops when all adjacent pairs in ranking are confident
- **Cost-aware allocation**: prioritizes uncertain pairs with high marginal value
- **Budget constraints**: respects total budget across all models

### `src/calibration/difficulty.py`
IRT parameter estimation from held-in models:
- **Item difficulties**: `b_i = logit(1 - mean_score_i)`
- **Noise parameter k**: estimated from residual variance
- **Discrimination**: `a = 1/√k`
- **Negative discrimination filtering**: removes items that anti-correlate with ability

### `src/metrics/evaluation.py`
Evaluation metrics:
- **Kendall's tau-b**: rank correlation with tie handling
- **Bootstrap confidence intervals**: for ground truth tie detection
- **Tie detection metrics**: precision, recall, F1
- **Confident accuracy**: accuracy on non-tie predictions

## Data Format

`data/scores.json` contains pre-computed evaluation scores:

```python
{
  "datasets": {
    "biolaysumm_rougel": {
      "items": {
        "0": {"input_tokens": 1234, "output_tokens": 567},
        ...
      },
      "scores": {
        "gpt-4o-mini": {
          "temp_0.0": {"0": 0.45, "1": 0.52, ...},
          "temp_0.4": {...},
          "temp_0.7": {...},
          "temp_1.0": {...}
        },
        ...
      }
    },
    ...
  },
  "model_costs": {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    ...
  }
}
```

### Datasets (10 total)

| Dataset | Metric | Items | Description |
|---------|--------|-------|-------------|
| biolaysumm_rougel | ROUGE-L | 1376 | Biomedical lay summarization |
| biolaysumm_bertscore | BERTScore | 1376 | Biomedical lay summarization |
| biolaysumm_fkgl | FKGL | 1376 | Biomedical lay summarization |
| govreport_rougel | ROUGE-L | 973 | Government report summarization |
| govreport_bertscore | BERTScore | 973 | Government report summarization |
| truthfulqa_judge | LLM-Judge | 817 | Truthfulness evaluation |
| truthfulqa_bertscore | BERTScore | 817 | Truthfulness evaluation |
| flores_bleu | BLEU | 1012 | Turkish-English translation |
| flores_comet | COMET | 1012 | Turkish-English translation |
| nemotron_pii | F1 | 2000 | PII detection |

### Models (21 total)

From 6 model families:
- **OpenAI**: gpt-5-mini, gpt-5-nano, gpt-4.1-mini, gpt-4.1-nano, gpt-4o-mini
- **Meta**: llama4-maverick-17b, llama4-scout-17b, llama3-3-70b, llama3-2-3b, llama3-1-8b
- **Google**: gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.0-flash, gemini-2.0-flash-lite
- **Amazon**: nova-pro, nova-lite, nova-micro
- **Mistral**: mistral-7b, mixtral-8x7b
- **Qwen**: qwen3-32b, qwen3-coder-30b

## Experiments

### Running All Experiments

```bash
# Full run (20 seeds, ~30 minutes)
python experiments/run_all.py --n-seeds 20

# Quick test (2 seeds, ~3 minutes)
python experiments/run_all.py --n-seeds 2 --output-dir experiments/results-test

# Specific datasets only
python experiments/run_all.py --datasets biolaysumm_rougel govreport_rougel
```

### Generated Tables

| Table | File | Description |
|-------|------|-------------|
| Table 2 | `table2_main_results.md` | Main ranking results: Adaptive τ vs Baseline τ |
| Table 3 | `table3_tie_detection.md` | Tie detection: Precision, Recall, F1, Confident Accuracy |
| Table 4 | `table4_adaptive_vs_fixed.md` | Adaptive vs Fixed-length CAT: Item/Cost savings |
| Table 5 | `table5_family_holdout.md` | Cross-family generalization |
| Table 6 | `table6_conformance.md` | Distributional conformance: R², τ |

### Experiment Configuration

Key settings in `experiments/config.py`:

```python
ExperimentConfig(
    n_seeds=20,                    # Random seeds per holdout set
    n_holdout_models=4,            # Models per holdout set
    n_holdout_sets=5,              # Disjoint holdout sets
    min_items_per_model=10,        # Warm-up items before adaptive selection
    target_items_per_model=20,     # For auto-budget calculation
    confidence_level=0.95,         # Pairwise stopping threshold
    bootstrap_n=10000,             # Bootstrap samples for ground truth CIs
    partition_seed=42,             # Fixed seed for reproducible holdout sets
)
```

### Family Holdout Experiment (Table 5)

Tests cross-family generalization:
- **Calibration families**: OpenAI, Meta, Google (14 models)
- **Holdout families**: Mistral, Qwen, Amazon (3 models, 1 per family)

## Tests

```bash
# Run all tests
pytest tests/ -v
```

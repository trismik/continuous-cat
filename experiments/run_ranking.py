#!/usr/bin/env python3
"""Run ranking experiment for a single dataset.

This script runs the adaptive ranking experiment and compares to baselines.
"""

import argparse

from config import ExperimentConfig
from experiment import run_dataset_experiment, summarize_results


def main():
    parser = argparse.ArgumentParser(description="Run ranking experiment")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("--n-seeds", type=int, default=20, help="Number of seeds")
    parser.add_argument("--n-holdout", type=int, default=4, help="Models per holdout")
    args = parser.parse_args()

    config = ExperimentConfig(
        n_seeds=args.n_seeds,
        n_holdout_models=args.n_holdout,
    )

    results = run_dataset_experiment(args.dataset, config)
    summary = summarize_results(results)

    print("\n=== Summary ===")
    print(f"Runs: {summary['n_runs']}")
    print(f"\nAdaptive:")
    print(f"  Tau: {summary['adaptive_tau_mean']:.3f} ± {summary['adaptive_tau_std']:.3f}")
    print(f"  Items: {summary['adaptive_items_mean']:.1f} ± {summary['adaptive_items_std']:.1f}")
    print(f"  Cost: ${summary['adaptive_cost_mean']:.4f} ± ${summary['adaptive_cost_std']:.4f}")
    print(f"\nBaseline (random, matched cost):")
    print(f"  Tau: {summary['baseline_tau_mean']:.3f} ± {summary['baseline_tau_std']:.3f}")
    print(f"  Items: {summary['baseline_items_mean']:.1f} ± {summary['baseline_items_std']:.1f}")
    print(f"  Cost: ${summary['baseline_cost_mean']:.4f} ± ${summary['baseline_cost_std']:.4f}")
    print(f"\nTau improvement: {summary['tau_improvement']:+.3f}")
    print(f"Tie P/R/F1: {summary['tie_precision_mean']:.3f}/{summary['tie_recall_mean']:.3f}/{summary['tie_f1_mean']:.3f}")


if __name__ == "__main__":
    main()

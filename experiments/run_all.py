#!/usr/bin/env python3
"""Run all experiments and generate paper tables.

This script runs the full experiment suite and outputs markdown tables.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
from typing import Dict, List, Tuple

import numpy as np

from config import DATASET_NAMES, ExperimentConfig
from conformance import run_conformance_analysis
from experiment import run_dataset_experiment, summarize_results
from family_holdout import run_family_holdout_experiment, summarize_family_holdout


# Mapping from dataset_name to (Dataset, Metric) for display
DATASET_METRIC_MAP = {
    "biolaysumm_rougel": ("BioLaySumm", "ROUGE-L"),
    "biolaysumm_bertscore": ("BioLaySumm", "BERTScore"),
    "biolaysumm_fkgl": ("BioLaySumm", "FKGL"),
    "govreport_rougel": ("GovReport", "ROUGE-L"),
    "govreport_bertscore": ("GovReport", "BERTScore"),
    "truthfulqa_judge": ("TruthfulQA", "LLM-Judge"),
    "truthfulqa_bertscore": ("TruthfulQA", "BERTScore"),
    "flores_bleu": ("FLORES", "BLEU"),
    "flores_comet": ("FLORES", "COMET"),
    "nemotron_pii": ("Nemotron", "F1"),
}


def get_dataset_metric(dataset_name: str) -> Tuple[str, str]:
    """Get display names for dataset and metric."""
    return DATASET_METRIC_MAP.get(dataset_name, (dataset_name, ""))


def generate_table2_markdown(results: Dict) -> str:
    """Generate Table 2: Main ranking results.

    Paper columns: Dataset, Metric, Size, a, Adaptive τ, Baseline τ, Items, % Used
    """
    lines = [
        "## Table 2: Main Ranking Results",
        "",
        "| Dataset | Metric | Size | a | Adaptive τ | Baseline τ | Items | % Used |",
        "|---------|--------|------|---|------------|------------|-------|--------|",
    ]

    for dataset_name, summary in results.items():
        dataset, metric = get_dataset_metric(dataset_name)
        adapt_tau = summary['adaptive_tau_mean']
        base_tau = summary['baseline_tau_mean']
        # Bold the higher tau value
        adapt_fmt = f"**{adapt_tau:.3f}**" if adapt_tau >= base_tau else f"{adapt_tau:.3f}"
        base_fmt = f"**{base_tau:.3f}**" if base_tau > adapt_tau else f"{base_tau:.3f}"
        lines.append(
            f"| {dataset} | {metric} | "
            f"{summary['n_items']} | "
            f"{summary['discrimination']:.2f} | "
            f"{adapt_fmt}±{summary['adaptive_tau_std']:.2f} | "
            f"{base_fmt}±{summary['baseline_tau_std']:.2f} | "
            f"{summary['adaptive_items_mean']:.0f}±{summary['adaptive_items_std']:.0f} | "
            f"{summary['pct_used_mean']:.1f}±{summary['pct_used_std']:.1f}% |"
        )

    # Compute averages
    avg_discrimination = np.mean([s['discrimination'] for s in results.values()])
    avg_adaptive_tau = np.mean([s['adaptive_tau_mean'] for s in results.values()])
    avg_baseline_tau = np.mean([s['baseline_tau_mean'] for s in results.values()])
    avg_items = np.mean([s['adaptive_items_mean'] for s in results.values()])
    avg_pct_used = np.mean([s['pct_used_mean'] for s in results.values()])

    # Bold the higher tau value
    avg_adapt_fmt = f"**{avg_adaptive_tau:.2f}**" if avg_adaptive_tau >= avg_baseline_tau else f"{avg_adaptive_tau:.2f}"
    avg_base_fmt = f"**{avg_baseline_tau:.2f}**" if avg_baseline_tau > avg_adaptive_tau else f"{avg_baseline_tau:.2f}"
    lines.append(
        f"| *Overall* | *(mean)* | | {avg_discrimination:.2f} | "
        f"{avg_adapt_fmt} | {avg_base_fmt} | "
        f"{avg_items:.0f} | {avg_pct_used:.1f}% |"
    )

    return "\n".join(lines)


def generate_table3_markdown(results: Dict) -> str:
    """Generate Table 3: Tie detection analysis.

    Paper columns: Dataset, Metric, Adapt Tie%, GT Tie%, Tie P, Tie R, Tie F1, Conf Acc
    """
    lines = [
        "## Table 3: Tie Detection Analysis",
        "",
        "| Dataset | Metric | Adapt Tie% | GT Tie% | Tie P | Tie R | Tie F1 | Conf Acc |",
        "|---------|--------|------------|---------|-------|-------|--------|----------|",
    ]

    for dataset_name, summary in results.items():
        dataset, metric = get_dataset_metric(dataset_name)
        conf_acc = summary['confident_accuracy_mean']
        conf_acc_str = f"{conf_acc:.2f}" if conf_acc is not None else "--"
        lines.append(
            f"| {dataset} | {metric} | "
            f"{summary['adaptive_tie_pct_mean']:.0f}% | "
            f"{summary['gt_tie_pct_mean']:.0f}% | "
            f"{summary['tie_precision_mean']:.2f} | "
            f"{summary['tie_recall_mean']:.2f} | "
            f"{summary['tie_f1_mean']:.2f} | "
            f"{conf_acc_str} |"
        )

    # Compute averages
    avg_adapt_tie = np.mean([s['adaptive_tie_pct_mean'] for s in results.values()])
    avg_gt_tie = np.mean([s['gt_tie_pct_mean'] for s in results.values()])
    avg_precision = np.mean([s['tie_precision_mean'] for s in results.values()])
    avg_recall = np.mean([s['tie_recall_mean'] for s in results.values()])
    avg_f1 = np.mean([s['tie_f1_mean'] for s in results.values()])
    conf_accs = [s['confident_accuracy_mean'] for s in results.values() if s['confident_accuracy_mean'] is not None]
    avg_conf_acc = np.mean(conf_accs) if conf_accs else None

    avg_conf_acc_str = f"{avg_conf_acc:.2f}" if avg_conf_acc else "--"
    lines.append(
        f"| *Overall* | *(mean)* | {avg_adapt_tie:.0f}% | {avg_gt_tie:.0f}% | "
        f"{avg_precision:.2f} | {avg_recall:.2f} | {avg_f1:.2f} | "
        f"{avg_conf_acc_str} |"
    )

    return "\n".join(lines)


def generate_table4_markdown(results: Dict) -> str:
    """Generate Table 4: Adaptive vs Fixed-length CAT.

    Paper columns: Dataset, Metric, Adapt τ, Fixed τ, Δτ, Item Savings, Cost Savings
    """
    lines = [
        "## Table 4: Adaptive vs Fixed-Length CAT",
        "",
        "| Dataset | Metric | Adapt τ | Fixed τ | Δτ | Item Savings | Cost Savings |",
        "|---------|--------|---------|---------|-----|--------------|--------------|",
    ]

    for dataset_name, summary in results.items():
        dataset, metric = get_dataset_metric(dataset_name)
        delta_tau = summary['tau_diff_vs_fixed']
        delta_sign = "+" if delta_tau >= 0 else ""
        lines.append(
            f"| {dataset} | {metric} | "
            f"{summary['adaptive_tau_mean']:.2f} | "
            f"{summary['fixed_tau_mean']:.2f} | "
            f"{delta_sign}{delta_tau:.2f} | "
            f"{summary['item_savings_mean']:.0f}% | "
            f"{summary['cost_savings_mean']:.0f}% |"
        )

    # Compute averages
    avg_adaptive_tau = np.mean([s['adaptive_tau_mean'] for s in results.values()])
    avg_fixed_tau = np.mean([s['fixed_tau_mean'] for s in results.values()])
    avg_delta_tau = avg_adaptive_tau - avg_fixed_tau
    avg_item_savings = np.mean([s['item_savings_mean'] for s in results.values()])
    avg_cost_savings = np.mean([s['cost_savings_mean'] for s in results.values()])

    delta_sign = "+" if avg_delta_tau >= 0 else ""
    lines.append(
        f"| *Overall* | | {avg_adaptive_tau:.2f} | {avg_fixed_tau:.2f} | "
        f"{delta_sign}{avg_delta_tau:.2f} | {avg_item_savings:.0f}% | {avg_cost_savings:.0f}% |"
    )

    return "\n".join(lines)


def generate_table5_markdown(results: Dict) -> str:
    """Generate Table 5: Family holdout evaluation.

    Paper columns: Dataset, Metric, Base τ, Adapt τ
    """
    lines = [
        "## Table 5: Family Holdout Evaluation",
        "",
        "Calibrated on OpenAI, Meta, Google. Evaluated on Mistral, Qwen, Amazon Nova.",
        "",
        "| Dataset | Metric | Baseline τ | Adaptive τ |",
        "|---------|--------|------------|------------|",
    ]

    for dataset_name, summary in results.items():
        if not summary:
            continue
        dataset, metric = get_dataset_metric(dataset_name)
        adapt_tau = summary['adaptive_tau_mean']
        base_tau = summary['baseline_tau_mean']
        # Bold the higher tau value
        adapt_fmt = f"**{adapt_tau:.2f}**" if adapt_tau >= base_tau else f"{adapt_tau:.2f}"
        base_fmt = f"**{base_tau:.2f}**" if base_tau > adapt_tau else f"{base_tau:.2f}"
        lines.append(
            f"| {dataset} | {metric} | "
            f"{base_fmt}±{summary['baseline_tau_std']:.2f} | "
            f"{adapt_fmt}±{summary['adaptive_tau_std']:.2f} |"
        )

    # Compute averages
    valid_results = [s for s in results.values() if s]
    if valid_results:
        avg_baseline = np.mean([s['baseline_tau_mean'] for s in valid_results])
        avg_adaptive = np.mean([s['adaptive_tau_mean'] for s in valid_results])
        # Bold the higher tau value
        avg_adapt_fmt = f"**{avg_adaptive:.2f}**" if avg_adaptive >= avg_baseline else f"{avg_adaptive:.2f}"
        avg_base_fmt = f"**{avg_baseline:.2f}**" if avg_baseline > avg_adaptive else f"{avg_baseline:.2f}"
        lines.append(
            f"| *Overall* | | {avg_base_fmt} | {avg_adapt_fmt} |"
        )

    return "\n".join(lines)


def generate_table6_markdown(results: Dict) -> str:
    """Generate Table 6: Distributional conformance analysis.

    Paper columns: Dataset, Metric, a, R², τ
    """
    lines = [
        "## Table 6: Distributional Conformance",
        "",
        "Tests how well heteroskedastic model σ² = k×μ(1-μ) fits observed variance.",
        "",
        "| Dataset | Metric | a | R² | τ |",
        "|---------|--------|---|-----|---|",
    ]

    for dataset_name, result in results.items():
        dataset, metric = get_dataset_metric(dataset_name)
        lines.append(
            f"| {dataset} | {metric} | "
            f"{result.discrimination:.2f} | "
            f"{result.r_squared:.2f} | "
            f"{result.tau:.2f} |"
        )

    # Compute averages
    if results:
        avg_a = np.mean([r.discrimination for r in results.values()])
        avg_r2 = np.mean([r.r_squared for r in results.values()])
        avg_tau = np.mean([r.tau for r in results.values()])
        lines.append(
            f"| *Overall* | | {avg_a:.2f} | {avg_r2:.2f} | {avg_tau:.2f} |"
        )

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--n-seeds", type=int, default=20, help="Number of seeds")
    parser.add_argument("--n-holdout", type=int, default=4, help="Models per holdout")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--datasets", nargs="+", default=None, help="Specific datasets to run")
    args = parser.parse_args()

    config = ExperimentConfig(
        n_seeds=args.n_seeds,
        n_holdout_models=args.n_holdout,
    )

    # Override dataset list if specified
    if args.datasets:
        datasets_to_run = args.datasets
    else:
        datasets_to_run = DATASET_NAMES

    # Create output directory
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(exist_ok=True)

    # Run experiments
    print("Running main ranking experiment...")
    all_results = {}

    for dataset_name in datasets_to_run:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print('='*60)

        results = run_dataset_experiment(dataset_name, config)
        summary = summarize_results(results)
        all_results[dataset_name] = summary

        dataset, metric = get_dataset_metric(dataset_name)
        print(f"\n{dataset} / {metric}:")
        print(f"  Size: {summary['n_items']}, a={summary['discrimination']:.2f}")
        print(f"  Adaptive tau: {summary['adaptive_tau_mean']:.3f} ± {summary['adaptive_tau_std']:.3f}")
        print(f"  Baseline tau: {summary['baseline_tau_mean']:.3f} ± {summary['baseline_tau_std']:.3f}")
        print(f"  Items: {summary['adaptive_items_mean']:.0f} ({summary['pct_used_mean']:.1f}%)")

    # Save raw results
    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Run family holdout experiment (Table 5)
    print("\n" + "="*60)
    print("Running family holdout experiment...")
    print("="*60)
    family_results = {}
    for dataset_name in datasets_to_run:
        print(f"\n--- {dataset_name} ---")
        results = run_family_holdout_experiment(dataset_name, n_seeds=args.n_seeds, config=config)
        family_results[dataset_name] = summarize_family_holdout(results)

    # Run conformance analysis (Table 6)
    print("\n" + "="*60)
    print("Running conformance analysis...")
    print("="*60)
    conformance_results = {}
    for dataset_name in datasets_to_run:
        print(f"\n--- {dataset_name} ---")
        result = run_conformance_analysis(dataset_name, config=config)
        conformance_results[dataset_name] = result
        print(f"  a={result.discrimination:.2f}, R²={result.r_squared:.2f}, τ={result.tau:.2f}")

    # Generate tables
    table2 = generate_table2_markdown(all_results)
    table3 = generate_table3_markdown(all_results)
    table4 = generate_table4_markdown(all_results)
    table5 = generate_table5_markdown(family_results)
    table6 = generate_table6_markdown(conformance_results)

    # Save tables
    with open(output_dir / "table2_main_results.md", "w") as f:
        f.write(table2)

    with open(output_dir / "table3_tie_detection.md", "w") as f:
        f.write(table3)

    with open(output_dir / "table4_adaptive_vs_fixed.md", "w") as f:
        f.write(table4)

    with open(output_dir / "table5_family_holdout.md", "w") as f:
        f.write(table5)

    with open(output_dir / "table6_conformance.md", "w") as f:
        f.write(table6)

    # Print tables
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(table2)
    print()
    print(table3)
    print()
    print(table4)
    print()
    print(table5)
    print()
    print(table6)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()

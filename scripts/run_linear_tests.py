#!/usr/bin/env python
"""
Run linear regression tests with Fama-French factor randomization.

This script tests the sensitivity of anomaly significance tests to decimal
rounding in Fama-French factors using the linear regression methodology
from Fama-French (1992).
"""

import argparse
import pandas as pd
from pathlib import Path

from ff_decimals.data import (
    load_fama_french_factors,
    load_chen_zimmerman_predictors,
)
from ff_decimals.analysis import run_linear_simulations
from ff_decimals.utils import analyze_sensitivity, identify_fragile_predictors
from ff_decimals.config import SIMULATIONS_DIR, TABLES_DIR, FIGURES_DIR, N_SIMULATIONS


def main():
    """Run linear regression tests with factor randomization."""
    parser = argparse.ArgumentParser(
        description="Test sensitivity of linear regressions to FF factor decimals"
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=N_SIMULATIONS,
        help=f"Number of simulations (default: {N_SIMULATIONS})",
    )
    parser.add_argument(
        "--ff-spec",
        choices=["3-factor", "5-factor"],
        default="5-factor",
        help="Fama-French specification (default: 5-factor)",
    )
    parser.add_argument(
        "--noise-type",
        choices=["uniform", "truncnorm"],
        default="truncnorm",
        help="Type of noise distribution (default: truncnorm)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save simulation results to CSV",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Linear Regression Tests with FF Factor Randomization")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading Fama-French factors...")
    ff_factors = load_fama_french_factors(specification=args.ff_spec)
    print(f"  Loaded {len(ff_factors)} months of {args.ff_spec} factors")
    print(f"  Factors: {', '.join(ff_factors.columns)}")

    print("\n[2/5] Loading Chen-Zimmerman predictors...")
    predictors = load_chen_zimmerman_predictors()
    print(f"  Loaded {len(predictors.columns)} predictors")
    print(f"  Sample period: {predictors.index[0]} to {predictors.index[-1]}")

    # Run simulations
    print(f"\n[3/5] Running {args.n_simulations} simulations...")
    print(f"  Noise type: {args.noise_type}")
    print(f"  Noise range: ±0.005")
    print(f"  Random seed: {args.seed}")
    print("  This may take several minutes...")

    results = run_linear_simulations(
        factors=ff_factors,
        predictors=predictors,
        n_simulations=args.n_simulations,
        noise_type=args.noise_type,
        seed=args.seed,
    )

    # Analyze sensitivity
    print("\n[4/5] Analyzing sensitivity...")
    sensitivity_metrics = analyze_sensitivity(results)

    print(f"\n  T-statistic range statistics:")
    print(f"    Mean:   {sensitivity_metrics['t_range'].mean():.3f}")
    print(f"    Median: {sensitivity_metrics['t_range'].median():.3f}")
    print(f"    Max:    {sensitivity_metrics['t_range'].max():.3f}")
    print(f"    Min:    {sensitivity_metrics['t_range'].min():.3f}")

    print(f"\n  Most sensitive predictors (top 10):")
    for i, (predictor, t_range) in enumerate(
        sensitivity_metrics["t_range"].head(10).items(), 1
    ):
        print(f"    {i:2d}. {predictor:20s}: {t_range:.3f}")

    # Identify fragile predictors
    fragile = identify_fragile_predictors(sensitivity_metrics)
    print(f"\n  Fragile predictors (cross significance threshold): {len(fragile)}")

    # Save results
    if args.save_results:
        print("\n[5/5] Saving results...")

        # Save simulation results
        results_file = SIMULATIONS_DIR / f"linear_simulations_{args.ff_spec}.csv"
        results.to_csv(results_file)
        print(f"  Simulation results: {results_file}")

        # Save sensitivity metrics
        sensitivity_file = TABLES_DIR / f"linear_sensitivity_{args.ff_spec}.csv"
        sensitivity_metrics.to_csv(sensitivity_file)
        print(f"  Sensitivity metrics: {sensitivity_file}")

        # Save fragile predictors
        if len(fragile) > 0:
            fragile_file = TABLES_DIR / f"linear_fragile_{args.ff_spec}.csv"
            fragile.to_csv(fragile_file)
            print(f"  Fragile predictors: {fragile_file}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

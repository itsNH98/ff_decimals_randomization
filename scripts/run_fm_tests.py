#!/usr/bin/env python
"""
Run Fama-Macbeth two-pass regression tests with FF factor randomization.

This script tests the sensitivity of Fama-Macbeth risk premium estimates to
decimal rounding in Fama-French factors. Due to the errors-in-variables problem,
Fama-Macbeth is expected to be more sensitive than linear regressions.
"""

import argparse
import pandas as pd
from pathlib import Path

from ff_decimals.data import (
    load_fama_french_factors,
    load_chen_zimmerman_predictors,
    load_crsp_monthly_returns,
)
from ff_decimals.analysis import run_fama_macbeth, run_fm_simulations
from ff_decimals.utils import analyze_sensitivity
from ff_decimals.config import SIMULATIONS_DIR, TABLES_DIR, N_SIMULATIONS


def main():
    """Run Fama-Macbeth tests with factor randomization."""
    parser = argparse.ArgumentParser(
        description="Test sensitivity of Fama-Macbeth to FF factor decimals"
    )
    parser.add_argument(
        "--predictor",
        type=str,
        default="Accruals",
        help="Predictor to test (default: Accruals)",
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
        default="3-factor",
        help="Fama-French specification (default: 3-factor for FM)",
    )
    parser.add_argument(
        "--noise-type",
        choices=["uniform", "truncnorm"],
        default="uniform",
        help="Type of noise distribution (default: uniform)",
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
    print("Fama-Macbeth Two-Pass Regression Tests with FF Factor Randomization")
    print("=" * 80)

    # Load data
    print("\n[1/6] Loading Fama-French factors...")
    ff_factors = load_fama_french_factors(
        specification=args.ff_spec,
        include_rf=True,  # FM needs RF for excess returns
    )
    print(f"  Loaded {len(ff_factors)} months of {args.ff_spec} factors")
    print(f"  Factors: {', '.join(ff_factors.columns)}")

    print("\n[2/6] Loading Chen-Zimmerman predictors...")
    predictors = load_chen_zimmerman_predictors()
    if args.predictor not in predictors.columns:
        raise ValueError(
            f"Predictor '{args.predictor}' not found. "
            f"Available: {', '.join(predictors.columns[:10])}..."
        )
    predictor = predictors[args.predictor]
    print(f"  Testing predictor: {args.predictor}")

    print("\n[3/6] Loading CRSP monthly returns...")
    try:
        crsp_returns = load_crsp_monthly_returns()
        print(f"  Loaded {len(crsp_returns)} asset-months")
        print(f"  Unique assets: {crsp_returns['PERMNO'].nunique()}")
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        print("\n  To run Fama-Macbeth tests, you need CRSP returns data.")
        print("  Options:")
        print("    1. Place returns CSV at: data/processed/returns.csv")
        print("    2. Use research_data package to download from WRDS")
        return

    # Run baseline (no randomization)
    print("\n[4/6] Running baseline Fama-Macbeth (no randomization)...")
    baseline_betas, baseline_results = run_fama_macbeth(
        returns=crsp_returns,
        factors=ff_factors,
        predictor=predictor,
        predictor_name=args.predictor,
        add_noise=False,
    )
    print("\n  Baseline risk premium estimates:")
    print(baseline_results.summary())

    # Run simulations
    print(f"\n[5/6] Running {args.n_simulations} simulations...")
    print(f"  Noise type: {args.noise_type}")
    print(f"  Noise range: ±0.005")
    print(f"  Random seed: {args.seed}")
    print("  This will take a while (FM is computationally intensive)...")

    sim_results = run_fm_simulations(
        returns=crsp_returns,
        factors=ff_factors,
        predictor=predictor,
        predictor_name=args.predictor,
        n_simulations=args.n_simulations,
        noise_type=args.noise_type,
        seed=args.seed,
    )

    # Analyze sensitivity
    print("\n[6/6] Analyzing sensitivity...")

    print(f"\n  T-statistic distributions across simulations:")
    for factor in sim_results.index:
        t_stats = sim_results.loc[factor]
        print(f"\n    {factor}:")
        print(f"      Baseline: {baseline_results.tvalues[factor]:.3f}")
        print(f"      Mean:     {t_stats.mean():.3f}")
        print(f"      Std:      {t_stats.std():.3f}")
        print(f"      Range:    [{t_stats.min():.3f}, {t_stats.max():.3f}]")

    # Check if significance flips
    baseline_t = baseline_results.tvalues[args.predictor]
    sim_t_stats = sim_results.loc[args.predictor]

    if abs(baseline_t) > 1.96:
        pct_significant = (abs(sim_t_stats) > 1.96).mean() * 100
        print(f"\n  Significance analysis for {args.predictor}:")
        print(f"    Baseline is significant (|t| = {abs(baseline_t):.2f} > 1.96)")
        print(f"    Remains significant in {pct_significant:.1f}% of simulations")

        if pct_significant < 95:
            print(f"    WARNING: Fragile significance! Can flip with decimal rounding.")

    # Save results
    if args.save_results:
        print("\n  Saving results...")

        # Save simulation results
        results_file = (
            SIMULATIONS_DIR / f"fm_simulations_{args.predictor}_{args.ff_spec}.csv"
        )
        sim_results.to_csv(results_file)
        print(f"    Simulation results: {results_file}")

        # Save baseline results
        baseline_file = TABLES_DIR / f"fm_baseline_{args.predictor}_{args.ff_spec}.csv"
        baseline_results.summary2().tables[1].to_csv(baseline_file)
        print(f"    Baseline results: {baseline_file}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

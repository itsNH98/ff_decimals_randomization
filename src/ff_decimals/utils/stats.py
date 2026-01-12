"""Statistical utility functions for sensitivity analysis."""

import pandas as pd
import numpy as np
from typing import Optional


def compute_t_value_range(simulation_results: pd.DataFrame) -> pd.Series:
    """
    Compute the range of t-statistics across simulations for each predictor.

    Parameters
    ----------
    simulation_results : pd.DataFrame
        DataFrame with shape (n_simulations, n_predictors) or (n_predictors, n_simulations)
        containing t-statistics from simulation runs.

    Returns
    -------
    pd.Series
        Series with t-statistic ranges (max - min) for each predictor,
        sorted in descending order.

    Examples
    --------
    >>> results = run_linear_simulations(ff, predictors, n_simulations=100)
    >>> t_ranges = compute_t_value_range(results)
    >>> t_ranges.head(10)  # Most sensitive predictors
    """
    # Compute range (max - min) across simulations
    if simulation_results.shape[0] < simulation_results.shape[1]:
        # Transpose if needed (simulations should be rows)
        simulation_results = simulation_results.T

    t_range = simulation_results.max() - simulation_results.min()

    return t_range.sort_values(ascending=False)


def analyze_sensitivity(
    simulation_results: pd.DataFrame,
    baseline_t_stats: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Comprehensive sensitivity analysis of simulation results.

    Parameters
    ----------
    simulation_results : pd.DataFrame
        DataFrame with shape (n_simulations, n_predictors) containing
        t-statistics from simulations.
    baseline_t_stats : pd.Series, optional
        Baseline t-statistics without randomization. If provided, computes
        deviations from baseline.

    Returns
    -------
    pd.DataFrame
        DataFrame with sensitivity metrics for each predictor:
        - t_mean: Mean t-statistic
        - t_std: Standard deviation of t-statistics
        - t_min: Minimum t-statistic
        - t_max: Maximum t-statistic
        - t_range: Range (max - min)
        - cv: Coefficient of variation (std / |mean|)
        - baseline_t: Baseline t-statistic (if provided)
        - max_deviation: Max absolute deviation from baseline (if provided)

    Examples
    --------
    >>> results = run_linear_simulations(ff, predictors, n_simulations=100)
    >>> sensitivity = analyze_sensitivity(results)
    >>> sensitivity.sort_values('t_range', ascending=False).head(10)
    """
    # Compute basic statistics
    metrics = pd.DataFrame(
        {
            "t_mean": simulation_results.mean(),
            "t_std": simulation_results.std(),
            "t_min": simulation_results.min(),
            "t_max": simulation_results.max(),
            "t_range": simulation_results.max() - simulation_results.min(),
        }
    )

    # Coefficient of variation (relative dispersion)
    metrics["cv"] = metrics["t_std"] / np.abs(metrics["t_mean"])

    # Add baseline comparison if provided
    if baseline_t_stats is not None:
        metrics["baseline_t"] = baseline_t_stats
        metrics["mean_deviation"] = metrics["t_mean"] - baseline_t_stats
        metrics["max_deviation"] = (
            simulation_results.subtract(baseline_t_stats, axis=1).abs().max()
        )

    return metrics.sort_values("t_range", ascending=False)


def identify_fragile_predictors(
    sensitivity_metrics: pd.DataFrame,
    significance_threshold: float = 1.96,
) -> pd.DataFrame:
    """
    Identify predictors whose statistical significance is fragile to decimal rounding.

    A predictor is considered "fragile" if small decimal variations can flip its
    significance status (e.g., from significant to insignificant).

    Parameters
    ----------
    sensitivity_metrics : pd.DataFrame
        Output from analyze_sensitivity() with columns: t_mean, t_min, t_max, etc.
    significance_threshold : float, default 1.96
        Absolute t-statistic threshold for significance (default: 5% two-tailed).

    Returns
    -------
    pd.DataFrame
        DataFrame containing fragile predictors with additional columns:
        - crosses_threshold: Whether the predictor crosses significance threshold
        - sign_flips: Whether the predictor changes sign across simulations
        - pct_significant: Percentage of simulations where |t| > threshold

    Examples
    --------
    >>> sensitivity = analyze_sensitivity(results)
    >>> fragile = identify_fragile_predictors(sensitivity)
    >>> print(f"Found {len(fragile)} fragile predictors")
    """
    fragile_metrics = sensitivity_metrics.copy()

    # Check if predictor crosses significance threshold
    fragile_metrics["crosses_threshold"] = (
        fragile_metrics["t_min"] < significance_threshold
    ) & (fragile_metrics["t_max"] > significance_threshold)

    # Check for sign flips
    fragile_metrics["sign_flips"] = (fragile_metrics["t_min"] < 0) & (
        fragile_metrics["t_max"] > 0
    )

    # Filter to only fragile predictors
    fragile = fragile_metrics[
        fragile_metrics["crosses_threshold"] | fragile_metrics["sign_flips"]
    ]

    return fragile.sort_values("t_range", ascending=False)


def compute_factor_impact(
    factors: pd.DataFrame,
    threshold: float = 0.10,
) -> pd.DataFrame:
    """
    Identify periods where factor values are small (high % impact from rounding).

    When factor values are close to zero, a fixed decimal error (±0.005) represents
    a larger percentage error, amplifying the impact on regression results.

    Parameters
    ----------
    factors : pd.DataFrame
        DataFrame of Fama-French factors.
    threshold : float, default 0.10
        Absolute threshold for identifying "small" factor values.

    Returns
    -------
    pd.DataFrame
        DataFrame showing periods where each factor is below threshold.

    Examples
    --------
    >>> ff = load_fama_french_factors()
    >>> small_factors = compute_factor_impact(ff, threshold=0.10)
    >>> small_factors.sum()  # Count of months below threshold for each factor
    """
    # Identify where absolute values are below threshold
    small_values = factors.abs() < threshold

    # Return DataFrame of periods with small values
    return factors[small_values.any(axis=1)]


def summarize_simulation_results(
    simulation_results: pd.DataFrame,
    top_n: int = 20,
) -> dict:
    """
    Generate summary statistics and identify key findings from simulations.

    Parameters
    ----------
    simulation_results : pd.DataFrame
        DataFrame with simulation results.
    top_n : int, default 20
        Number of top results to include in summary.

    Returns
    -------
    dict
        Dictionary containing:
        - n_predictors: Total number of predictors tested
        - n_simulations: Number of simulations run
        - t_range_stats: Summary statistics of t-statistic ranges
        - most_sensitive: Top N most sensitive predictors
        - least_sensitive: Top N least sensitive predictors

    Examples
    --------
    >>> results = run_linear_simulations(ff, predictors, n_simulations=100)
    >>> summary = summarize_simulation_results(results)
    >>> print(f"Tested {summary['n_predictors']} predictors")
    >>> print(f"Mean t-range: {summary['t_range_stats']['mean']:.3f}")
    """
    # Compute t-statistic ranges
    t_ranges = compute_t_value_range(simulation_results)

    # Generate summary
    summary = {
        "n_predictors": len(simulation_results.columns),
        "n_simulations": len(simulation_results),
        "t_range_stats": {
            "mean": t_ranges.mean(),
            "std": t_ranges.std(),
            "min": t_ranges.min(),
            "max": t_ranges.max(),
            "median": t_ranges.median(),
        },
        "most_sensitive": t_ranges.head(top_n).to_dict(),
        "least_sensitive": t_ranges.tail(top_n).to_dict(),
    }

    return summary

"""Linear regression tests for anomaly significance (Fama-French 1992 style)."""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from typing import Optional

from ..data import load_signal_documentation, prepare_study_dates
from .randomization import add_decimal_noise


def run_linear_regression(
    factors: pd.DataFrame,
    predictor: pd.Series,
    predictor_name: str,
    study_dates: pd.DataFrame = None,
    randomized_factors: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """
    Run linear regression of predictor returns on Fama-French factors.

    Tests whether a new anomaly/predictor remains significant when controlling
    for Fama-French factors. This follows the Fama-French (1992) methodology.

    Parameters
    ----------
    factors : pd.DataFrame
        DataFrame of Fama-French factors (e.g., Mkt-RF, SMB, HML).
    predictor : pd.Series
        Series containing returns for the predictor/anomaly being tested.
    predictor_name : str
        Name of the predictor (used to lookup sample period in study_dates).
    study_dates : pd.DataFrame, optional
        DataFrame with sample start/end dates for each predictor.
        If None, loads from signal documentation.
    randomized_factors : np.ndarray, optional
        Array of random noise to add to factors. If None, uses factors as-is.

    Returns
    -------
    tuple[float, float]
        (t-statistic, coefficient) for the predictor in the regression.

    Notes
    -----
    The regression specification is:
        predictor_return = α + β₁·Mkt-RF + β₂·SMB + β₃·HML + ... + ε

    The t-statistic on α tests whether the predictor has significant abnormal
    returns after controlling for factor exposures.

    Examples
    --------
    >>> ff = load_fama_french_factors()
    >>> predictors = load_chen_zimmerman_predictors()
    >>> t_stat, coef = run_linear_regression(ff, predictors['Accruals'], 'Accruals')
    """
    # Load study dates if not provided
    if study_dates is None:
        signal_doc = load_signal_documentation()
        study_dates = prepare_study_dates(signal_doc)

    # Prepare factors (add randomization if provided)
    if randomized_factors is not None:
        factors_to_use = factors + randomized_factors
    else:
        factors_to_use = factors.copy()

    # Merge factors and predictor
    predictor_df = pd.DataFrame(predictor)
    merged_df = pd.merge(
        factors_to_use, predictor_df, left_index=True, right_index=True
    )

    # Drop missing values
    merged_df = merged_df.dropna()

    # Slice to predictor's sample period
    start_date = study_dates.loc[predictor_name, "SampleStartYear"]
    end_date = study_dates.loc[predictor_name, "SampleEndYear"]
    merged_df = merged_df[
        (merged_df.index >= start_date) & (merged_df.index <= end_date)
    ]

    # Set up regression variables
    # Y is the predictor (last column), X is all factors
    X = merged_df.iloc[:, :-1]
    Y = merged_df.iloc[:, -1]

    # Run OLS regression
    reg = smf.ols("Y ~ 1 + X", data=merged_df).fit()

    # Return t-statistic and coefficient for intercept
    return reg.tvalues[0], reg.params[0]


def run_linear_simulations(
    factors: pd.DataFrame,
    predictors: pd.DataFrame,
    n_simulations: int = 100,
    noise_type: str = "truncnorm",
    seed: int = None,
) -> pd.DataFrame:
    """
    Run linear regression tests with randomized factors across multiple simulations.

    For each predictor, runs n_simulations regressions with different random
    noise added to the factors, measuring the sensitivity of t-statistics to
    decimal rounding.

    Parameters
    ----------
    factors : pd.DataFrame
        DataFrame of Fama-French factors.
    predictors : pd.DataFrame
        DataFrame of Chen-Zimmerman predictors (326 columns).
    n_simulations : int, default 100
        Number of simulations per predictor.
    noise_type : str, default "truncnorm"
        Type of noise distribution ("uniform" or "truncnorm").
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with shape (n_simulations, n_predictors) containing
        t-statistics from each simulation.

    Examples
    --------
    >>> ff = load_fama_french_factors()
    >>> predictors = load_chen_zimmerman_predictors()
    >>> results = run_linear_simulations(ff, predictors, n_simulations=100)
    >>> results.shape
    (100, 326)

    >>> # Analyze sensitivity
    >>> t_stat_ranges = results.max() - results.min()
    >>> most_sensitive = t_stat_ranges.nlargest(10)
    """
    if seed is not None:
        np.random.seed(seed)

    # Load study dates
    signal_doc = load_signal_documentation()
    study_dates = prepare_study_dates(signal_doc)

    # Initialize results DataFrame
    results = pd.DataFrame(
        index=range(n_simulations),
        columns=predictors.columns,
        dtype=float,
    )

    # Run simulations for each predictor
    for predictor_name in predictors.columns:
        print(f"Running simulations for {predictor_name}...")

        for i in range(n_simulations):
            # Generate random noise
            sim_seed = None if seed is None else seed + i
            noise = add_decimal_noise(
                factors,
                noise_type=noise_type,
                seed=sim_seed,
            )

            # Run regression with noise
            t_stat, _ = run_linear_regression(
                factors=factors,
                predictor=predictors[predictor_name],
                predictor_name=predictor_name,
                study_dates=study_dates,
                randomized_factors=noise - factors,  # Just the noise component
            )

            results.at[i, predictor_name] = t_stat

    return results


def compute_sensitivity_metrics(simulation_results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sensitivity metrics from simulation results.

    Parameters
    ----------
    simulation_results : pd.DataFrame
        DataFrame from run_linear_simulations() with shape (n_simulations, n_predictors).

    Returns
    -------
    pd.DataFrame
        DataFrame with sensitivity metrics for each predictor:
        - t_range: Range of t-statistics (max - min)
        - t_std: Standard deviation of t-statistics
        - t_mean: Mean t-statistic across simulations
        - t_min: Minimum t-statistic
        - t_max: Maximum t-statistic

    Examples
    --------
    >>> results = run_linear_simulations(ff, predictors, n_simulations=100)
    >>> metrics = compute_sensitivity_metrics(results)
    >>> metrics.sort_values('t_range', ascending=False).head(10)
    """
    metrics = pd.DataFrame(
        {
            "t_range": simulation_results.max() - simulation_results.min(),
            "t_std": simulation_results.std(),
            "t_mean": simulation_results.mean(),
            "t_min": simulation_results.min(),
            "t_max": simulation_results.max(),
        }
    )

    return metrics.sort_values("t_range", ascending=False)

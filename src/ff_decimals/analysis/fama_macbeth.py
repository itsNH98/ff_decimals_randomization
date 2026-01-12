"""Fama-Macbeth two-pass regression tests (Fama-Macbeth 1973 style)."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Optional

from .randomization import add_decimal_noise


def estimate_factor_loadings(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    predictor: pd.Series,
    predictor_name: str,
    add_noise: bool = False,
    noise_type: str = "uniform",
    seed: int = None,
) -> pd.DataFrame:
    """
    First stage: Estimate factor loadings (betas) for each asset via time-series regression.

    For each asset i, runs the regression:
        ExRet_it = α_i + β_i1·Factor1_t + β_i2·Factor2_t + ... + ε_it

    Parameters
    ----------
    returns : pd.DataFrame
        Cross-sectional stock returns with columns: PERMNO, MthRet, etc.
        Index should be PeriodIndex (monthly).
    factors : pd.DataFrame
        DataFrame of Fama-French factors (must include 'RF' column).
    predictor : pd.Series
        Returns for the predictor/anomaly being tested.
    predictor_name : str
        Name of the predictor column.
    add_noise : bool, default False
        Whether to add random noise to factors (for simulations).
    noise_type : str, default "uniform"
        Type of noise ("uniform" or "truncnorm").
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with index=PERMNO, columns=factor loadings (betas).

    Notes
    -----
    This implements the first stage of Fama-Macbeth (1973) two-pass regression.
    The estimated betas will be used as regressors in the second stage.

    Examples
    --------
    >>> betas = estimate_factor_loadings(crsp, ff, predictor, 'Accruals')
    >>> betas.head()
    """
    # Merge returns with factors
    merged_df = pd.merge(returns, factors, left_index=True, right_index=True)

    # Merge with predictor
    predictor_df = pd.DataFrame({predictor_name: predictor})
    merged_df = pd.merge(merged_df, predictor_df, left_index=True, right_index=True)

    # Calculate excess returns
    merged_df["ExRet"] = merged_df["MthRet"] - (merged_df["RF"] / 100)

    # Drop RF and rearrange columns
    merged_df = merged_df.drop(columns=["RF"])

    # Group by asset (PERMNO)
    groups = merged_df.groupby("PERMNO")

    # Define regression function for each asset
    def estimate_betas(group):
        """Estimate factor loadings for a single asset."""
        # Add noise to factors if requested
        if add_noise:
            factor_cols = ["Mkt-RF", "SMB", "HML"]  # Adjust based on factors used
            group[factor_cols] = group[factor_cols] + np.random.uniform(
                -0.005, 0.005, size=group[factor_cols].shape
            )

        # Set up regression: ExRet ~ const + factors + predictor
        # X includes factors and predictor (all columns after ExRet)
        X = sm.add_constant(group.iloc[:, 4:])  # Skip PERMNO, MthRet, MthCalDt, ExRet
        Y = group["ExRet"]

        # Run OLS
        try:
            model = sm.OLS(Y, X, missing="drop")
            results = model.fit()
            return results.params
        except Exception:
            # Return NaN if regression fails
            return pd.Series(index=X.columns, dtype=float)

    # Apply regression to each group
    betas = groups.apply(estimate_betas).unstack()

    return betas


def estimate_risk_premia(
    returns: pd.DataFrame,
    betas: pd.DataFrame,
    factors: pd.DataFrame,
    predictor: pd.Series,
    predictor_name: str,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Second stage: Estimate risk premia via cross-sectional regression.

    For each time period t, runs the regression:
        R_it = γ_0t + γ_1t·β_i1 + γ_2t·β_i2 + ... + u_it

    Then tests whether the average γ coefficients are significant.

    Parameters
    ----------
    returns : pd.DataFrame
        Cross-sectional stock returns.
    betas : pd.DataFrame
        Estimated factor loadings from first stage.
    factors : pd.DataFrame
        DataFrame of Fama-French factors (must include 'RF' column).
    predictor : pd.Series
        Returns for the predictor being tested.
    predictor_name : str
        Name of the predictor.

    Returns
    -------
    RegressionResultsWrapper
        OLS regression results with risk premium estimates and t-statistics.

    Notes
    -----
    This implements the second stage of Fama-Macbeth (1973) two-pass regression.

    The key econometric issue: If factors in stage 1 have measurement error
    (decimal rounding), the estimated betas will have errors, creating an
    errors-in-variables problem in stage 2.

    Examples
    --------
    >>> results = estimate_risk_premia(crsp, betas, ff, predictor, 'Accruals')
    >>> print(results.summary())
    """
    # Merge returns with factors
    all_df = pd.merge(returns, factors, left_index=True, right_index=True)

    # Merge with predictor
    predictor_df = pd.DataFrame({predictor_name: predictor})
    all_df = pd.merge(all_df, predictor_df, left_index=True, right_index=True)

    # Calculate excess returns
    all_df["ExRet"] = all_df["MthRet"] - (all_df["RF"] / 100)

    # Merge with betas
    merged_2sls_df = pd.merge(all_df, betas, left_on="PERMNO", right_index=True)

    # Rename columns to avoid conflicts (factors_y are the betas)
    beta_cols = [col for col in betas.columns if col != "const"]
    rename_dict = {f"{col}_y": col for col in beta_cols}
    merged_2sls_df = merged_2sls_df.rename(columns=rename_dict)

    # Select relevant columns for regression
    cols_to_keep = ["PERMNO", "ExRet"] + beta_cols
    merged_2sls_df = merged_2sls_df[cols_to_keep]

    # Set up cross-sectional regression
    Y = merged_2sls_df["ExRet"]
    X = sm.add_constant(merged_2sls_df[beta_cols])

    # Run OLS
    results = sm.OLS(Y, X, missing="drop").fit()

    return results


def run_fama_macbeth(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    predictor: pd.Series,
    predictor_name: str,
    add_noise: bool = False,
    seed: int = None,
) -> tuple[pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:
    """
    Run complete Fama-Macbeth two-pass regression.

    Parameters
    ----------
    returns : pd.DataFrame
        Cross-sectional stock returns.
    factors : pd.DataFrame
        DataFrame of Fama-French factors (must include 'RF' column).
    predictor : pd.Series
        Returns for the predictor/anomaly being tested.
    predictor_name : str
        Name of the predictor.
    add_noise : bool, default False
        Whether to add random noise to factors in stage 1.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, RegressionResultsWrapper]
        (betas, risk_premium_results)
        - betas: Factor loadings from stage 1
        - risk_premium_results: OLS results from stage 2

    Examples
    --------
    >>> betas, results = run_fama_macbeth(crsp, ff, predictor, 'Accruals')
    >>> print(results.summary())
    >>> print(f"Predictor t-stat: {results.tvalues['Accruals']:.2f}")
    """
    # Stage 1: Estimate factor loadings
    betas = estimate_factor_loadings(
        returns=returns,
        factors=factors,
        predictor=predictor,
        predictor_name=predictor_name,
        add_noise=add_noise,
        seed=seed,
    )

    # Stage 2: Estimate risk premia
    results = estimate_risk_premia(
        returns=returns,
        betas=betas,
        factors=factors,
        predictor=predictor,
        predictor_name=predictor_name,
    )

    return betas, results


def run_fm_simulations(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    predictor: pd.Series,
    predictor_name: str,
    n_simulations: int = 100,
    noise_type: str = "uniform",
    seed: int = None,
) -> pd.DataFrame:
    """
    Run Fama-Macbeth regressions with randomized factors across multiple simulations.

    For each simulation, adds random noise to factors in stage 1 and measures
    the sensitivity of risk premium estimates to decimal rounding.

    Parameters
    ----------
    returns : pd.DataFrame
        Cross-sectional stock returns.
    factors : pd.DataFrame
        DataFrame of Fama-French factors.
    predictor : pd.Series
        Returns for the predictor being tested.
    predictor_name : str
        Name of the predictor.
    n_simulations : int, default 100
        Number of simulations to run.
    noise_type : str, default "uniform"
        Type of noise distribution.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with shape (n_factors + 1, n_simulations) containing
        t-statistics from stage 2 for each simulation.

    Notes
    -----
    This is where the errors-in-variables problem is most apparent:
    - Stage 1: Measurement error in factors → errors in estimated betas
    - Stage 2: Errors in betas → biased/inconsistent risk premium estimates

    Examples
    --------
    >>> results = run_fm_simulations(crsp, ff, predictor, 'Accruals', n_simulations=100)
    >>> # Analyze distribution of t-statistics
    >>> import seaborn as sns
    >>> sns.histplot(results.loc['Accruals'])
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize results dictionary
    results_dict = {}

    # Run simulations
    for i in range(n_simulations):
        print(f"Running simulation {i+1}/{n_simulations}...")

        # Generate random seed for this simulation
        sim_seed = None if seed is None else seed + i

        # Run Fama-Macbeth with noise
        _, fm_results = run_fama_macbeth(
            returns=returns,
            factors=factors,
            predictor=predictor,
            predictor_name=predictor_name,
            add_noise=True,
            seed=sim_seed,
        )

        # Store t-statistics
        results_dict[i] = fm_results.tvalues

    # Convert to DataFrame
    results_df = pd.DataFrame(results_dict)

    return results_df

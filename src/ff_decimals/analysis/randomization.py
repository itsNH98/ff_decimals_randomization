"""Add decimal noise to Fama-French factors to simulate rounding errors."""

import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from typing import Literal

from ..config import NOISE_LOWER, NOISE_UPPER


def add_decimal_noise(
    factors: pd.DataFrame,
    noise_type: Literal["uniform", "truncnorm"] = "truncnorm",
    lower: float = NOISE_LOWER,
    upper: float = NOISE_UPPER,
    seed: int = None,
) -> pd.DataFrame:
    """
    Add random decimal noise to Fama-French factors.

    This simulates the measurement error introduced by decimal rounding
    when factors are reported with different levels of precision across
    data sources.

    Parameters
    ----------
    factors : pd.DataFrame
        DataFrame of Fama-French factors (e.g., Mkt-RF, SMB, HML).
    noise_type : {"uniform", "truncnorm"}, default "truncnorm"
        Distribution to draw noise from:
        - "uniform": Uniform distribution on [lower, upper]
        - "truncnorm": Truncated normal distribution on [lower, upper]
    lower : float, default -0.005
        Lower bound of noise distribution (in percentage points).
    upper : float, default 0.005
        Upper bound of noise distribution (in percentage points).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with same shape as input, containing randomized factors.

    Notes
    -----
    The default range of ±0.005 represents the maximum rounding error when
    rounding to 2 decimal places. For example:
    - True value: 0.0234 → Rounded to 2dp: 0.02 → Error: -0.0034
    - True value: 0.0267 → Rounded to 2dp: 0.03 → Error: +0.0033

    Examples
    --------
    >>> ff = load_fama_french_factors()
    >>> ff_noisy = add_decimal_noise(ff, seed=42)
    >>> (ff_noisy - ff).abs().max().max()  # Check max noise magnitude
    0.005
    """
    if seed is not None:
        np.random.seed(seed)

    if noise_type == "uniform":
        # Uniform distribution on [lower, upper]
        noise = np.random.uniform(lower, upper, size=factors.shape)

    elif noise_type == "truncnorm":
        # Truncated normal distribution on [lower, upper]
        # Standardize bounds for scipy's truncnorm
        a, b = lower, upper
        noise = truncnorm.rvs(a, b, size=factors.shape)

    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    # Add noise to factors
    noisy_factors = factors + noise

    return noisy_factors


def generate_randomized_factors(
    factors: pd.DataFrame,
    n_simulations: int = 100,
    noise_type: Literal["uniform", "truncnorm"] = "truncnorm",
    lower: float = NOISE_LOWER,
    upper: float = NOISE_UPPER,
    seed: int = None,
) -> list[pd.DataFrame]:
    """
    Generate multiple randomized versions of Fama-French factors.

    Parameters
    ----------
    factors : pd.DataFrame
        DataFrame of Fama-French factors.
    n_simulations : int, default 100
        Number of randomized factor sets to generate.
    noise_type : {"uniform", "truncnorm"}, default "truncnorm"
        Distribution to draw noise from.
    lower : float, default -0.005
        Lower bound of noise distribution.
    upper : float, default 0.005
        Upper bound of noise distribution.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list[pd.DataFrame]
        List of DataFrames, each containing one randomized factor set.

    Examples
    --------
    >>> ff = load_fama_french_factors()
    >>> randomized_factors = generate_randomized_factors(ff, n_simulations=10, seed=42)
    >>> len(randomized_factors)
    10
    """
    if seed is not None:
        np.random.seed(seed)

    randomized_factors = []
    for i in range(n_simulations):
        # Use different seed for each simulation
        sim_seed = None if seed is None else seed + i
        noisy_factors = add_decimal_noise(
            factors,
            noise_type=noise_type,
            lower=lower,
            upper=upper,
            seed=sim_seed,
        )
        randomized_factors.append(noisy_factors)

    return randomized_factors

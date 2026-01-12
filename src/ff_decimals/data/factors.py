"""Load Fama-French factors using research_data package."""

import pandas as pd
from typing import Literal

# Try to import research_data, provide helpful error if not available
try:
    from research_data import load_monthly_base
    from research_data.sources.open_asset_pricing import download_oap_factors
except ImportError as e:
    raise ImportError(
        "research_data package not found. Install with:\n"
        "  pip install -e ../research_data\n"
        f"Original error: {e}"
    )


def load_fama_french_factors(
    specification: Literal["3-factor", "5-factor"] = "5-factor",
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Load Fama-French factors from research_data.

    This is a thin wrapper around research_data.load_monthly_base() that extracts
    only the FF factor columns.

    Parameters
    ----------
    specification : {"3-factor", "5-factor"}
        Which Fama-French specification to use.
        - "3-factor": mktrf, smb, hml (+ rf)
        - "5-factor": mktrf, smb, hml, rmw, cma (+ rf)
    start_date : str, optional
        Start date in YYYY-MM-DD format.
    end_date : str, optional
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        DataFrame with date index and factor returns (in percentage points).
        Does NOT include rf by default (use load_monthly_base for that).

    Examples
    --------
    >>> ff3 = load_fama_french_factors(specification="3-factor")
    >>> ff5 = load_fama_french_factors(specification="5-factor")
    >>> ff5.columns
    Index(['mktrf', 'smb', 'hml', 'rmw', 'cma'], dtype='object')
    """
    # Load monthly base data (has everything)
    df = load_monthly_base(start_date=start_date, end_date=end_date)

    # Extract factor columns based on specification
    if specification == "3-factor":
        factor_cols = ["mktrf", "smb", "hml"]
    elif specification == "5-factor":
        factor_cols = ["mktrf", "smb", "hml", "rmw", "cma"]
    else:
        raise ValueError(f"Unknown specification: {specification}")

    # Get unique dates and factors
    ff = df[["date"] + factor_cols].drop_duplicates(subset=["date"]).set_index("date")

    return ff


def load_fama_french_with_rf(
    specification: Literal["3-factor", "5-factor"] = "5-factor",
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Load Fama-French factors including risk-free rate.

    Same as load_fama_french_factors() but includes the 'rf' column.
    Useful for Fama-Macbeth regressions that need excess returns.

    Parameters
    ----------
    specification : {"3-factor", "5-factor"}
        Which Fama-French specification to use.
    start_date : str, optional
        Start date in YYYY-MM-DD format.
    end_date : str, optional
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        DataFrame with date index, factor returns, and rf.

    Examples
    --------
    >>> ff_with_rf = load_fama_french_with_rf(specification="3-factor")
    >>> ff_with_rf.columns
    Index(['mktrf', 'smb', 'hml', 'rf'], dtype='object')
    """
    # Load factors
    ff = load_fama_french_factors(specification, start_date, end_date)

    # Add rf back
    df = load_monthly_base(start_date=start_date, end_date=end_date)
    rf = df[["date", "rf"]].drop_duplicates(subset=["date"]).set_index("date")

    ff = ff.join(rf)

    return ff

"""Load CRSP stock returns using research_data package."""

import pandas as pd

try:
    from research_data import load_monthly_base
except ImportError as e:
    raise ImportError(
        "research_data package not found. Install with:\n"
        "  pip install -e ../research_data\n"
        f"Original error: {e}"
    )


def load_crsp_monthly_returns(
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Load CRSP monthly stock returns from research_data.

    This is a thin wrapper around research_data.load_monthly_base().
    Returns the full dataset with returns, market caps, Fama-French factors, etc.

    Parameters
    ----------
    start_date : str, optional
        Start date in YYYY-MM-DD format.
    end_date : str, optional
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        DataFrame with CRSP monthly returns and additional fields:
        - permno: CRSP permanent security identifier
        - date: Month-end date
        - ret: Monthly return
        - mktrf, smb, hml, rmw, cma: Fama-French factors
        - rf: Risk-free rate
        - mcap, size_decile, exchcd, shrcd, etc.

    Examples
    --------
    >>> crsp = load_crsp_monthly_returns(start_date="2000-01-01")
    >>> crsp[['permno', 'date', 'ret', 'mktrf']].head()

    Notes
    -----
    This loads the pre-built monthly panel from research_data. Make sure you've
    run 'rd build monthly' first to generate the dataset.
    """
    return load_monthly_base(start_date=start_date, end_date=end_date)

"""Load Chen-Zimmerman predictors from Open Asset Pricing."""

import pandas as pd
from pathlib import Path

from ..config import CHEN_PREDICTORS_CSV, SIGNAL_DOC_CSV

# Try to import research_data
try:
    from research_data.sources.open_asset_pricing import (
        download_oap_factors,
        download_oap_signal,
        list_oap_signals,
    )

    RESEARCH_DATA_AVAILABLE = True
except ImportError:
    RESEARCH_DATA_AVAILABLE = False


def load_chen_zimmerman_from_oap(
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Load Chen-Zimmerman predictor returns from Open Asset Pricing via research_data.

    This downloads the factor returns (long-short portfolios) for all anomalies
    from the Open Asset Pricing project.

    Parameters
    ----------
    start_date : str, optional
        Start date in YYYY-MM-DD format.
    end_date : str, optional
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        DataFrame with date and predictor returns (factor portfolios).

    Examples
    --------
    >>> predictors = load_chen_zimmerman_from_oap()
    >>> predictors.shape
    (1200+, 326)
    """
    # Download OAP factors (these are the Chen-Zimmerman predictor returns)
    df = download_oap_factors()

    # Filter dates if requested
    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]

    # Set date as index
    df = df.set_index("date")

    return df

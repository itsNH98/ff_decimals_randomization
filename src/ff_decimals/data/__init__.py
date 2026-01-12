"""Data loading modules - thin wrappers around research_data."""

from .factors import load_fama_french_factors, load_fama_french_with_rf
from .predictors import load_chen_zimmerman_from_oap
from .crsp import load_crsp_monthly_returns

__all__ = [
    "load_fama_french_factors",
    "load_fama_french_with_rf",
    "load_chen_zimmerman_from_oap",
    "load_crsp_monthly_returns",
]

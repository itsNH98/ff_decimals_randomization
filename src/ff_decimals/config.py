"""Configuration settings for FF decimals randomization project."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
SIMULATIONS_DIR = RESULTS_DIR / "simulations"

# Ensure directories exist
for directory in [DATA_RAW, DATA_PROCESSED, FIGURES_DIR, TABLES_DIR, SIMULATIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Local data files (legacy)
CHEN_PREDICTORS_CSV = DATA_DIR / "chen_predictors.csv"
SIGNAL_DOC_CSV = DATA_DIR / "signal_doc.csv"

# Research data package path (from environment variable or default)
RESEARCH_DATA_PATH = os.getenv(
    "RESEARCH_DATA_PATH",
    str(PROJECT_ROOT.parent / "research_data"),
)

# Randomization parameters
NOISE_LOWER = -0.005  # Lower bound for decimal noise
NOISE_UPPER = 0.005   # Upper bound for decimal noise
N_SIMULATIONS = 100   # Default number of simulations

# Fama-French factor specifications
FF_3_FACTORS = ["Mkt-RF", "SMB", "HML"]
FF_5_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

# Data sources
FAMA_FRENCH_DATASET = "F-F_Research_Data_5_Factors_2x3"
FAMA_FRENCH_START = "1900"

# Claude Instructions for ff_decimals_randomization

## Package Management

**ALWAYS use `uv` for running Python scripts and managing dependencies.**

```bash
# Run Python scripts
uv run python script.py

# Install packages
uv pip install package_name

# Install this package in editable mode
uv pip install -e .
```

## Project Setup

This project depends on `research_data` which must be installed first:

```bash
# Install research_data
cd ../research_data
uv pip install -e .

# Then install this package
cd ../ff_decimals_randomization
uv pip install -e .
```

## Running Analysis

```bash
# Run POC analysis
uv run python run_poc_analysis.py
```

## Beads Issue Tracking

This project uses beads for issue tracking. The beads database is initialized at `.beads/`.

Current issues can be viewed with:
```bash
bd list
```

## Key Dependencies

- `research_data` - Provides CRSP, FF factors, and Open Asset Pricing integration
- `numpy`, `pandas`, `scipy`, `statsmodels` - Core numerical libraries
- `matplotlib`, `seaborn` - Visualization (if needed)

## Research Context

This project tests the sensitivity of asset pricing tests to decimal rounding in Fama-French factors (±0.005). The core contribution is the randomization analysis in `src/ff_decimals/analysis/randomization.py`.

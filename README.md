# Effect of Decimals on Methodologies Employing Fama-French Factors

Research on the impact of decimal rounding in Fama-French factors on asset pricing test conclusions.

**Status:** Ongoing research project.

## Research Question

How sensitive are asset pricing tests to small decimal approximations (±0.005) in Fama-French factors? Could researchers reach different conclusions about anomaly significance simply due to rounding differences in factor data?

## Key Insight: Why Fama-Macbeth is More Sensitive

Fama-Macbeth two-pass regressions are expected to show greater sensitivity due to the **errors-in-variables problem**:

1. **Stage 1**: Measurement error in factors → errors in estimated betas
2. **Stage 2**: Errors in betas (used as regressors) → classic errors-in-variables
   - Causes attenuation bias (coefficients biased toward zero)
   - Inflates standard errors
   - Can flip statistical significance

In contrast, linear regressions test factors directly without this compounding error.

## Installation

```bash
# Install research_data first (provides CRSP, FF factors, Open Asset Pricing)
cd ../research_data
pip install -e .

# Then install this package
cd ../ff_decimals_randomization
pip install -e .

# Install Jupyter for notebooks (optional)
pip install jupyter
```

## Quick Start

### Using Notebooks (Recommended)

```bash
# Launch Jupyter
jupyter notebook

# Open notebooks:
# - notebooks/01_linear_regression_tests.ipynb
# - notebooks/02_fama_macbeth_tests.ipynb
```

### Using Python

```python
from research_data import load_monthly_base
from ff_decimals.data import load_fama_french_factors, load_chen_zimmerman_from_oap
from ff_decimals.analysis import add_decimal_noise

# Load data (research_data gives us CRSP + FF factors merged)
monthly = load_monthly_base()

# Load FF factors separately
ff = load_fama_french_factors(specification="5-factor")

# Load Chen-Zimmerman predictors from Open Asset Pricing
predictors = load_chen_zimmerman_from_oap()

# Add decimal noise to factors
ff_noisy = add_decimal_noise(ff, noise_type='truncnorm', seed=42)

# Run your tests with noisy factors...
```

## Project Structure

```
ff_decimals_randomization/
├── src/ff_decimals/              # Thin wrappers around research_data
│   ├── data/                     # Data loaders
│   │   ├── factors.py            # Load FF factors
│   │   ├── predictors.py         # Load Chen-Zimmerman from OAP
│   │   └── crsp.py               # Load CRSP returns
│   ├── analysis/
│   │   └── randomization.py      # Core: Add decimal noise to factors
│   └── utils/
│       └── stats.py              # Sensitivity analysis helpers
├── notebooks/                    # Analysis notebooks
│   ├── 01_linear_regression_tests.ipynb
│   └── 02_fama_macbeth_tests.ipynb
├── results/                      # Outputs
│   ├── figures/
│   ├── tables/
│   └── simulations/
├── data/                         # Legacy data files (optional)
│   ├── chen_predictors.csv
│   └── signal_doc.csv
├── archive/                      # Old notebook versions
└── README.md
```

## How It Works

### 1. Randomization Approach

The default noise range of ±0.005 represents the maximum rounding error when rounding to 2 decimal places:
- True value: 0.0234 → Rounded: 0.02 → Error: -0.0034
- True value: 0.0267 → Rounded: 0.03 → Error: +0.0033
- Maximum error: ±0.005

We simulate this by adding random noise drawn from a truncated normal distribution on [-0.005, 0.005] to each factor return.

### 2. Linear Regression Tests (Fama-French 1992)

For each predictor/anomaly:
```
predictor_return_t = α + β₁·Mkt-RF_t + β₂·SMB_t + β₃·HML_t + ε_t
```

The t-statistic on α tests whether the predictor has significant abnormal returns. We measure how this changes across 100+ simulations with randomized factors.

### 3. Fama-Macbeth Two-Pass Tests (Fama-Macbeth 1973)

Uses `research_data.methods.fama_macbeth()` for the implementation.

**Stage 1 (Time-series):** Estimate factor loadings:
```
R_it = α_i + β_i·Factors_t + ε_it
```

**Stage 2 (Cross-sectional):** Estimate risk premia:
```
R_it = γ₀ + γ·β̂_i + u_it
```

If factors have measurement error in Stage 1, the estimated β̂_i will have errors, creating an errors-in-variables problem in Stage 2.

## Data Sources (via research_data)

- **Fama-French Factors**: WRDS `ff.fivefactors_monthly`
- **Chen-Zimmerman Predictors**: Open Asset Pricing (200+ anomalies)
- **CRSP Returns**: WRDS `crsp.msf`

All data access is handled through the `research_data` package.

## Key Functions

### Data Loading
```python
from ff_decimals.data import (
    load_fama_french_factors,      # FF factors from research_data
    load_fama_french_with_rf,      # FF factors + risk-free rate
    load_chen_zimmerman_from_oap,  # Chen-Zimmerman from OAP
    load_crsp_monthly_returns,     # CRSP from research_data
)
```

### Analysis
```python
from ff_decimals.analysis import add_decimal_noise

# Add noise to factors
ff_noisy = add_decimal_noise(
    factors=ff,
    noise_type='truncnorm',  # or 'uniform'
    lower=-0.005,
    upper=0.005,
    seed=42,
)
```

### Methods (from research_data)
```python
from research_data.methods import fama_macbeth, portfolio_sorts, rolling_ols
```

## Results Interpretation

### Sensitivity Metrics

- **t_range**: Range of t-statistics (max - min) across simulations
- **t_std**: Standard deviation of t-statistics
- **cv**: Coefficient of variation (relative dispersion)

### Fragile Predictors

Predictors are "fragile" if decimal rounding can:
1. Flip significance status (cross the |t| > 1.96 threshold)
2. Change the sign of the coefficient

## References

### Core Papers

Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*, 81(3), 607-636.

Fama, E. F., & French, K. R. (1992). The cross‐section of expected stock returns. *The Journal of Finance*, 47(2), 427-465.

Chen, A. Y., & Zimmermann, T. (2021). Open source cross-sectional asset pricing. *Critical Finance Review*, Forthcoming.

### Measurement Error & Econometrics

Akey, P., Robertson, A., & Simutin, M. (2022). Noisy factors. *Rotman School of Management Working Paper*, Forthcoming.

Shanken, J. (1992). On the estimation of beta-pricing models. *The Review of Financial Studies*, 5(1), 1-33.

Kim, D. (1995). The errors in the variables problem in the cross‐section of expected stock returns. *The Journal of Finance*, 50(5), 1605-1634.

Kan, R., Robotti, C., & Shanken, J. (2013). Pricing model performance and the two‐pass cross‐sectional regression methodology. *The Journal of Finance*, 68(6), 2617-2649.

## License

Research code - academic use only.

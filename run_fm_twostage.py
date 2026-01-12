#!/usr/bin/env python
"""
Two-Pass Fama-Macbeth Analysis with Decimal Noise in FF Factors
OPTIMIZED VERSION USING POLARS

Design:
- Test assets: 25 portfolios (5×5 size × BM)
- Factors: FF3 (Mkt-RF, SMB, HML) + tested predictor
- Stage 1: Time-series regression to get betas (VECTORIZED)
- Stage 2: Cross-sectional regression using research_data.methods.fama_macbeth
- Noise: Added to FF3 factors only (±0.005)
- Track: Risk premium (lambda) on predictor across simulations
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
import pandas_datareader as pdr
import sys

print("="*80)
print("Two-Pass Fama-Macbeth: FF Factor Noise Sensitivity (OPTIMIZED)")
print("="*80)

# Create results directory
Path("results/tables").mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. Load 25 Portfolios (5×5 size × BM)
# ============================================================================
print("\n[1/5] Loading 25 FF portfolios (5×5 size × BM)...")
sys.stdout.flush()

portfolios_raw = pdr.get_data_famafrench('25_Portfolios_5x5', start='1926')[0]
portfolios_pd = portfolios_raw.copy()
portfolios_pd.index = portfolios_pd.index.to_timestamp()
portfolios_pd = portfolios_pd.reset_index()
portfolios_pd.columns = ['date'] + [str(c) for c in portfolios_pd.columns[1:]]

# Convert to polars
portfolios = pl.from_pandas(portfolios_pd)
print(f"  ✓ Downloaded {len(portfolios)} months, {len(portfolios.columns)-1} portfolios")
sys.stdout.flush()

# ============================================================================
# 2. Load FF3 Factors
# ============================================================================
print("\n[2/5] Loading FF3 factors...")
sys.stdout.flush()

ff3_raw = pdr.get_data_famafrench('F-F_Research_Data_Factors', start='1926')[0]
ff3_pd = ff3_raw[['Mkt-RF', 'SMB', 'HML', 'RF']].copy()
ff3_pd.columns = ['mktrf', 'smb', 'hml', 'rf']
ff3_pd.index = ff3_pd.index.to_timestamp()
ff3_pd = ff3_pd.reset_index()
ff3_pd.columns = ['date', 'mktrf', 'smb', 'hml', 'rf']

# Convert to polars
ff3 = pl.from_pandas(ff3_pd)
print(f"  ✓ Loaded FF3 factors ({len(ff3)} months)")
sys.stdout.flush()

# Merge and compute excess returns for portfolios
data = portfolios.join(ff3, on='date', how='inner')
portfolio_cols = [c for c in portfolios.columns if c != 'date']

# Subtract RF from each portfolio
for col in portfolio_cols:
    data = data.with_columns((pl.col(col) - pl.col('rf')).alias(f"{col}_excess"))

print(f"  ✓ Computed excess returns for {len(portfolio_cols)} portfolios")
sys.stdout.flush()

# ============================================================================
# 3. Load Predictor Factors (REAL Chen-Zimmerman from OAP)
# ============================================================================
print("\n[3/5] Loading predictor factors...")
sys.stdout.flush()

predictors_to_test = [
    'Mom12m', 'AssetGrowth', 'Accruals', 'GP',
    'ROE', 'Beta', 'IdioVol3F', 'Illiquidity'
]

# Load REAL Chen-Zimmerman predictors
try:
    predictors_pd = pd.read_csv('data/chen_predictors.csv', parse_dates=['date'])
    predictors_pd['date'] = pd.to_datetime(predictors_pd['date']).dt.to_period('M').dt.to_timestamp()
    predictors = pl.from_pandas(predictors_pd)
    print(f"  ✓ Loaded Chen-Zimmerman predictors from CSV")
    print(f"    {len(predictors)} months, {len(predictors.columns)-1} predictors available")
    sys.stdout.flush()
except Exception as e:
    print(f"  ✗ Failed to load predictors: {e}")
    sys.stdout.flush()
    raise

# ============================================================================
# 4. Efficient Two-Pass Fama-Macbeth Function
# ============================================================================

def stage1_betas_vectorized(portfolio_returns: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """
    Vectorized Stage 1: Estimate betas for all portfolios at once.

    Args:
        portfolio_returns: (T x N) array of excess returns
        factors: (T x K) array of factor returns

    Returns:
        betas: (N x K) array of factor loadings
    """
    # Add constant to factors
    T = factors.shape[0]
    X = np.column_stack([np.ones(T), factors])

    # Solve all regressions at once: (X'X)^-1 X'Y
    XtX_inv = np.linalg.inv(X.T @ X)
    betas_with_const = XtX_inv @ X.T @ portfolio_returns

    # Return betas (exclude constant)
    return betas_with_const[1:, :].T  # (N x K)


def two_pass_fama_macbeth_fast(data_pl: pl.DataFrame, portfolio_cols: list,
                                factor_cols: list, predictor_col: str) -> dict:
    """
    Fast two-pass FM using vectorized Stage 1 + existing fama_macbeth for Stage 2.

    Returns risk premium (lambda) and t-stat for predictor.
    """
    from research_data.methods import fama_macbeth

    # Prepare data for Stage 1
    data_clean = data_pl.select(['date'] + portfolio_cols + factor_cols + [predictor_col]).drop_nulls()

    if len(data_clean) < 120:
        raise ValueError("Insufficient data")

    # Get numpy arrays
    portfolio_returns = data_clean.select(portfolio_cols).to_numpy()  # (T x N)
    factors = data_clean.select(factor_cols + [predictor_col]).to_numpy()  # (T x K+1)

    # Stage 1: Vectorized beta estimation
    betas = stage1_betas_vectorized(portfolio_returns, factors)  # (N x K+1)

    # Prepare data for Stage 2
    # We need panel data: (date, portfolio_id, return, beta_mkt, beta_smb, beta_hml, beta_pred)
    T, N = portfolio_returns.shape
    K_plus_1 = factors.shape[1]

    dates = data_clean['date'].to_list()

    # Build panel
    panel_data = []
    for t in range(T):
        for i, port_col in enumerate(portfolio_cols):
            row = {
                'date': dates[t],
                'portfolio': port_col,
                'ret_excess': portfolio_returns[t, i],
            }
            # Add betas
            for k, factor_col in enumerate(factor_cols + [predictor_col]):
                row[f'beta_{factor_col}'] = betas[i, k]

            panel_data.append(row)

    panel = pl.DataFrame(panel_data)

    # Stage 2: Cross-sectional regression using existing FM function
    beta_cols = [f'beta_{col}' for col in factor_cols + [predictor_col]]

    result = fama_macbeth(
        df=panel,
        y_col='ret_excess',
        x_cols=beta_cols,
        date_col='date',
        add_constant=True,
    )

    # Extract predictor results
    pred_beta_col = f'beta_{predictor_col}'

    return {
        'lambda': result.mean_coeffs[pred_beta_col],
        'tstat': result.tstat[pred_beta_col],
        'pval': result.pval[pred_beta_col],
    }


# ============================================================================
# 5. Run Analysis for Each Predictor
# ============================================================================

print("\n[4/5] Running two-pass FM with noise simulations...")
print(f"  Testing {len(predictors_to_test)} predictors")
print(f"  Running 100 simulations per predictor")
sys.stdout.flush()

n_simulations = 100
all_results = []

# Merge predictors into main data
data = data.join(predictors, on='date', how='inner')

# Portfolio excess return columns
portfolio_excess_cols = [f"{c}_excess" for c in portfolio_cols]

for pred_name in predictors_to_test:
    if pred_name not in data.columns:
        print(f"\n  ⚠ Skipping {pred_name} (not available)")
        sys.stdout.flush()
        continue

    print(f"\n  Testing {pred_name}...")
    sys.stdout.flush()

    # Baseline FM (no noise)
    try:
        baseline = two_pass_fama_macbeth_fast(
            data_pl=data,
            portfolio_cols=portfolio_excess_cols,
            factor_cols=['mktrf', 'smb', 'hml'],
            predictor_col=pred_name
        )
        print(f"    Baseline: λ={baseline['lambda']:.4f}, t={baseline['tstat']:.2f}")
        sys.stdout.flush()
    except Exception as e:
        print(f"    ✗ Baseline failed: {e}")
        sys.stdout.flush()
        continue

    # Simulations with noise
    lambdas_sim = []
    tstats_sim = []

    for sim in range(n_simulations):
        if (sim + 1) % 20 == 0:
            print(f"      Simulation {sim+1}/{n_simulations}...")
            sys.stdout.flush()

        # Add noise to FF3 factors (±0.005)
        data_sim = data.clone()
        for factor_col in ['mktrf', 'smb', 'hml']:
            noise = np.random.uniform(-0.005, 0.005, size=len(data_sim))
            data_sim = data_sim.with_columns(
                (pl.col(factor_col) + pl.lit(noise)).alias(factor_col)
            )

        try:
            result = two_pass_fama_macbeth_fast(
                data_pl=data_sim,
                portfolio_cols=portfolio_excess_cols,
                factor_cols=['mktrf', 'smb', 'hml'],
                predictor_col=pred_name
            )
            lambdas_sim.append(result['lambda'])
            tstats_sim.append(result['tstat'])
        except:
            continue

    lambdas_sim = np.array(lambdas_sim)
    tstats_sim = np.array(tstats_sim)

    # Compute statistics
    lambda_range = lambdas_sim.max() - lambdas_sim.min()
    tstat_range = tstats_sim.max() - tstats_sim.min()

    baseline_sig = abs(baseline['tstat']) > 1.96
    pct_sig = (np.abs(tstats_sim) > 1.96).mean() * 100

    print(f"    Simulations ({len(tstats_sim)} successful):")
    print(f"      λ range: [{lambdas_sim.min():.4f}, {lambdas_sim.max():.4f}] (width={lambda_range:.4f})")
    print(f"      t-stat range: [{tstats_sim.min():.2f}, {tstats_sim.max():.2f}] (width={tstat_range:.2f})")

    if baseline_sig:
        print(f"      Baseline SIGNIFICANT → {pct_sig:.1f}% remain significant")
        if pct_sig < 95:
            print(f"      ⚠ FRAGILE! Can flip with decimal noise")

    sys.stdout.flush()

    # Store results
    all_results.append({
        'predictor': pred_name,
        'baseline_lambda': baseline['lambda'],
        'baseline_tstat': baseline['tstat'],
        'lambda_mean': lambdas_sim.mean(),
        'lambda_std': lambdas_sim.std(),
        'lambda_min': lambdas_sim.min(),
        'lambda_max': lambdas_sim.max(),
        'lambda_range': lambda_range,
        'tstat_mean': tstats_sim.mean(),
        'tstat_std': tstats_sim.std(),
        'tstat_min': tstats_sim.min(),
        'tstat_max': tstats_sim.max(),
        'tstat_range': tstat_range,
        'baseline_significant': baseline_sig,
        'pct_remain_significant': pct_sig,
        'n_simulations': len(tstats_sim),
    })

# ============================================================================
# 6. Save Results
# ============================================================================

print("\n[5/5] Saving results...")
sys.stdout.flush()

import pandas as pd
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('tstat_range', ascending=False)

output_path = 'results/tables/fm_twostage_sensitivity.csv'
results_df.to_csv(output_path, index=False)
print(f"  ✓ Saved to: {output_path}")
sys.stdout.flush()

print("\n" + "="*80)
print("SUMMARY: TWO-PASS FM SENSITIVITY TO FF FACTOR NOISE")
print("="*80)

print(f"\nTested {len(results_df)} predictors with 25 portfolios (5×5) and FF3 factors")
print(f"Added ±0.005 noise to FF3 factors across {n_simulations} simulations")
print("\nMost sensitive predictors (by t-stat range):")
print(results_df[['predictor', 'baseline_tstat', 'tstat_range', 'pct_remain_significant']].head(5).to_string(index=False))

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

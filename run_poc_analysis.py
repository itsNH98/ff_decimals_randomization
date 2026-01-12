#!/usr/bin/env python
"""
POC Analysis: Test decimal sensitivity with 10 predictors and FF portfolios
Run 1000 simulations to measure robustness.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import pandas_datareader as pdr

print("="*80)
print("FF Decimals Randomization - Proof of Concept Analysis")
print("="*80)

# Create results directories
Path("results/tables").mkdir(parents=True, exist_ok=True)

# ============================================================================
# PART 1: Linear Regression Tests (10 Predictors)
# ============================================================================

print("\n" + "="*80)
print("PART 1: Linear Regression Tests")
print("="*80)

print("\n[1/5] Loading Fama-French factors...")
# Try research_data first, fall back to pandas_datareader
try:
    from research_data import load_monthly_base
    monthly = load_monthly_base()
    ff_factors = monthly[['date', 'mktrf', 'smb', 'hml', 'rmw', 'cma']].drop_duplicates(subset=['date']).set_index('date')
    print(f"  ✓ Loaded from research_data")
except:
    print(f"  ℹ research_data not available, using pandas_datareader...")
    ff_raw = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3', start='1900')[0]
    ff_factors = ff_raw[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].copy()
    ff_factors.columns = ['mktrf', 'smb', 'hml', 'rmw', 'cma']
    ff_factors.index.name = 'date'

print(f"  Loaded {len(ff_factors)} months of FF 5-factor data")
print(f"  Date range: {ff_factors.index.min()} to {ff_factors.index.max()}")

print("\n[2/5] Loading Chen-Zimmerman predictors...")
# Try Open Asset Pricing first, fall back to local CSV
try:
    from research_data.sources.open_asset_pricing import download_oap_factors
    predictors_full = download_oap_factors()
    predictors_full['date'] = pd.to_datetime(predictors_full['date'])
    predictors_full = predictors_full.set_index('date')
    print(f"  ✓ Loaded from Open Asset Pricing")
except Exception as e:
    print(f"  ℹ Using local CSV file (OAP error: {e})")
    predictors_full = pd.read_csv('data/chen_predictors.csv', parse_dates=['date'], index_col='date')
    predictors_full.index = predictors_full.index.to_period('M')

# Select 10 well-known predictors
predictor_names = [
    'Size', 'BM', 'Mom12m', 'Accruals', 'GP',
    'AssetGrowth', 'IdioVol3F', 'Beta', 'ROE', 'Illiquidity'
]

# Filter to available predictors
available = [p for p in predictor_names if p in predictors_full.columns]
if len(available) < 10:
    print(f"  Warning: Only {len(available)}/{len(predictor_names)} predictors available")
    print(f"  Available: {available}")
    predictor_names = available[:10] if len(available) >= 10 else available

predictors = predictors_full[predictor_names]
print(f"  Loaded {len(predictors.columns)} predictors: {', '.join(predictors.columns)}")

print("\n[3/5] Running 1000 simulations with decimal noise...")
from ff_decimals.analysis.randomization import add_decimal_noise

n_simulations = 1000
results = {}

for pred_name in predictors.columns:
    print(f"  Testing {pred_name}...", end=' ', flush=True)

    pred = predictors[pred_name]
    t_stats_sim = []

    for i in range(n_simulations):
        # Add decimal noise to FF factors
        ff_noisy = add_decimal_noise(ff_factors, noise_type='truncnorm', seed=42+i)

        # Merge with predictor
        df_sim = ff_noisy.join(pred, how='inner').dropna()

        if len(df_sim) < 50:  # Skip if not enough data
            continue

        # Run regression
        X_sim = sm.add_constant(df_sim[['mktrf', 'smb', 'hml', 'rmw', 'cma']])
        y_sim = df_sim[pred_name]

        try:
            model_sim = sm.OLS(y_sim, X_sim).fit()
            t_stats_sim.append(model_sim.tvalues['const'])
        except:
            continue

    if len(t_stats_sim) > 10:
        t_stats_sim = np.array(t_stats_sim)
        results[pred_name] = {
            'n_simulations': len(t_stats_sim),
            'mean': t_stats_sim.mean(),
            'std': t_stats_sim.std(),
            'min': t_stats_sim.min(),
            'max': t_stats_sim.max(),
            'range': t_stats_sim.max() - t_stats_sim.min(),
        }
        print(f"✓ (range: {results[pred_name]['range']:.3f})")
    else:
        print(f"✗ (insufficient data)")

print("\n[4/5] Analyzing sensitivity...")
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('range', ascending=False)

print("\n" + "-"*80)
print("LINEAR REGRESSION SENSITIVITY RESULTS")
print("-"*80)
print(f"\nTested {len(results_df)} predictors with ~{n_simulations} simulations each")
print(f"\nOverall Statistics:")
print(f"  Mean t-stat range:   {results_df['range'].mean():.3f}")
print(f"  Median t-stat range: {results_df['range'].median():.3f}")
print(f"  Max t-stat range:    {results_df['range'].max():.3f}")
print(f"  Min t-stat range:    {results_df['range'].min():.3f}")

print(f"\nMost Sensitive Predictors:")
for i, (pred, row) in enumerate(results_df.head(10).iterrows(), 1):
    print(f"  {i:2d}. {pred:15s}: range={row['range']:.3f}, mean={row['mean']:.3f}, std={row['std']:.3f}")

print("\n[5/5] Saving results...")
results_df.to_csv('results/tables/linear_sensitivity_poc.csv')
print(f"  Saved to: results/tables/linear_sensitivity_poc.csv")

# ============================================================================
# PART 2: Fama-Macbeth Tests (FF Portfolios)
# ============================================================================

print("\n" + "="*80)
print("PART 2: Fama-Macbeth Tests with FF Portfolios")
print("="*80)

print("\n[1/5] Loading FF 6 portfolios (2×3 size × BM)...")
from research_data.sources.fama_french import download_ff_portfolios

# Download 6 portfolios (2 size x 3 BM)
portfolios = download_ff_portfolios("6_2x3_monthly")
print(f"  Loaded {len(portfolios)} months of 6 portfolio returns")
print(f"  Portfolios: {[c for c in portfolios.columns if c != 'date']}")

# Convert to pandas
portfolios_pd = portfolios.to_pandas()
portfolios_pd['date'] = pd.to_datetime(portfolios_pd['date'])
portfolios_pd = portfolios_pd.set_index('date')

print("\n[2/5] Preparing data for Fama-Macbeth...")
# Get RF for excess returns
try:
    from research_data import load_monthly_base
    monthly = load_monthly_base()
    ff_with_rf = monthly[['date', 'rf']].drop_duplicates(subset=['date'])
except:
    ff_raw = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3', start='1900')[0]
    ff_with_rf = ff_raw[['RF']].reset_index()
    ff_with_rf.columns = ['date', 'rf']
    ff_with_rf['date'] = pd.to_datetime(ff_with_rf['date'])

# Melt portfolios to long format for FM
import polars as pl
from research_data.methods import fama_macbeth

portfolio_cols = [c for c in portfolios_pd.columns]
port_long_list = []

for port_name in portfolio_cols:
    df_port = portfolios_pd[[port_name]].copy()
    df_port = df_port.reset_index()
    df_port['portfolio'] = port_name
    df_port = df_port.rename(columns={port_name: 'ret'})
    port_long_list.append(df_port)

port_long = pd.concat(port_long_list, ignore_index=True)

# Merge with FF factors to get RF
port_long = port_long.merge(ff_with_rf, on='date', how='inner')
port_long['ret_excess'] = port_long['ret'] - port_long['rf']

# Create size and value dummies from portfolio names
port_long['is_small'] = port_long['portfolio'].str.contains('SMALL').astype(float)
port_long['is_value'] = port_long['portfolio'].str.contains('Hi').astype(float)

print(f"  Prepared {len(port_long)} portfolio-months for FM")

print("\n[3/5] Running baseline Fama-Macbeth (no noise)...")
port_pl = pl.from_pandas(port_long)
fm_baseline = fama_macbeth(
    df=port_pl,
    y_col='ret_excess',
    x_cols=['is_small', 'is_value'],
    date_col='date',
    add_constant=True,
)

print(f"\n  Baseline FM Results:")
print(f"    Size premium (is_small):  coef={fm_baseline.mean_coeffs['is_small']:.4f}, t={fm_baseline.tstat['is_small']:.2f}")
print(f"    Value premium (is_value): coef={fm_baseline.mean_coeffs['is_value']:.4f}, t={fm_baseline.tstat['is_value']:.2f}")

print("\n[4/5] Running 1000 FM simulations with decimal noise...")
t_stats_size = []
t_stats_value = []
coefs_size = []
coefs_value = []

for i in range(n_simulations):
    if (i+1) % 100 == 0:
        print(f"  Simulation {i+1}/{n_simulations}...")

    # Add noise to risk-free rate (affects excess returns)
    port_sim = port_long.copy()
    rf_noisy = port_sim['rf'] + np.random.uniform(-0.005, 0.005, size=len(port_sim))
    port_sim['ret_excess'] = port_sim['ret'] - rf_noisy

    # Convert to Polars
    port_sim_pl = pl.from_pandas(port_sim)

    try:
        fm_sim = fama_macbeth(
            df=port_sim_pl,
            y_col='ret_excess',
            x_cols=['is_small', 'is_value'],
            date_col='date',
            add_constant=True,
        )

        t_stats_size.append(fm_sim.tstat['is_small'])
        t_stats_value.append(fm_sim.tstat['is_value'])
        coefs_size.append(fm_sim.mean_coeffs['is_small'])
        coefs_value.append(fm_sim.mean_coeffs['is_value'])
    except Exception as e:
        continue

t_stats_size = np.array(t_stats_size)
t_stats_value = np.array(t_stats_value)
coefs_size = np.array(coefs_size)
coefs_value = np.array(coefs_value)

print("\n" + "-"*80)
print("FAMA-MACBETH SENSITIVITY RESULTS")
print("-"*80)

print(f"\nSize Premium (is_small):")
print(f"  Baseline:  coef={fm_baseline.mean_coeffs['is_small']:.4f}, t={fm_baseline.tstat['is_small']:.2f}")
print(f"  Simulations ({len(t_stats_size)} successful):")
print(f"    Mean t-stat:   {t_stats_size.mean():.2f}")
print(f"    Std t-stat:    {t_stats_size.std():.2f}")
print(f"    Range:         [{t_stats_size.min():.2f}, {t_stats_size.max():.2f}]")
print(f"    t-stat range:  {t_stats_size.max() - t_stats_size.min():.2f}")

print(f"\nValue Premium (is_value):")
print(f"  Baseline:  coef={fm_baseline.mean_coeffs['is_value']:.4f}, t={fm_baseline.tstat['is_value']:.2f}")
print(f"  Simulations ({len(t_stats_value)} successful):")
print(f"    Mean t-stat:   {t_stats_value.mean():.2f}")
print(f"    Std t-stat:    {t_stats_value.std():.2f}")
print(f"    Range:         [{t_stats_value.min():.2f}, {t_stats_value.max():.2f}]")
print(f"    t-stat range:  {t_stats_value.max() - t_stats_value.min():.2f}")

print("\n[5/5] Saving results...")
fm_results = pd.DataFrame({
    'simulation': range(len(t_stats_size)),
    't_stat_size': t_stats_size,
    't_stat_value': t_stats_value,
    'coef_size': coefs_size,
    'coef_value': coefs_value,
})
fm_results.to_csv('results/tables/fm_sensitivity_poc.csv', index=False)
print(f"  Saved to: results/tables/fm_sensitivity_poc.csv")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nResults saved:")
print(f"  - results/tables/linear_sensitivity_poc.csv")
print(f"  - results/tables/fm_sensitivity_poc.csv")
print("\n" + "="*80)

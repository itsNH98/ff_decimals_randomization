#!/usr/bin/env python
"""
Fama-Macbeth Analysis: Test decimal sensitivity with FF portfolios
Run 1000 simulations to measure errors-in-variables impact.
"""

import pandas as pd
import numpy as np
import polars as pl
from pathlib import Path
import pandas_datareader as pdr

print("="*80)
print("Fama-Macbeth Randomization Analysis")
print("="*80)

# Create results directories
Path("results/tables").mkdir(parents=True, exist_ok=True)

from research_data.methods import fama_macbeth

print("\n[1/5] Loading FF 6 portfolios (2×3 size × BM)...")

# Try to download directly with pandas_datareader
try:
    print("  Downloading from Ken French's website...")
    portfolios_raw = pdr.get_data_famafrench('6_Portfolios_2x3', start='1926')[0]
    portfolios = portfolios_raw.reset_index()
    portfolios.columns = ['date'] + list(portfolios_raw.columns)
    print(f"  ✓ Downloaded {len(portfolios)} months")
except Exception as e:
    print(f"  ✗ Download failed: {e}")
    print("  Trying alternative approach...")

    # Alternative: construct portfolios from size and value factors
    # Use FF factors to create synthetic portfolios
    ff_raw = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3', start='1926')[0]

    # Create 6 synthetic portfolios based on SMB and HML
    # Small-Value, Small-Neutral, Small-Growth, Big-Value, Big-Neutral, Big-Growth
    portfolios = pd.DataFrame({
        'date': ff_raw.index.to_timestamp(),  # Convert Period to Timestamp
        'SMALL_LoBM': ff_raw['Mkt-RF'] + ff_raw['SMB']/2 - ff_raw['HML']/2,  # Small Growth
        'SMALL_MedBM': ff_raw['Mkt-RF'] + ff_raw['SMB']/2,                    # Small Neutral
        'SMALL_HiBM': ff_raw['Mkt-RF'] + ff_raw['SMB']/2 + ff_raw['HML']/2,  # Small Value
        'BIG_LoBM': ff_raw['Mkt-RF'] - ff_raw['SMB']/2 - ff_raw['HML']/2,    # Big Growth
        'BIG_MedBM': ff_raw['Mkt-RF'] - ff_raw['SMB']/2,                      # Big Neutral
        'BIG_HiBM': ff_raw['Mkt-RF'] - ff_raw['SMB']/2 + ff_raw['HML']/2,    # Big Value
    })
    print(f"  ✓ Constructed synthetic portfolios ({len(portfolios)} months)")

print(f"  Portfolios: {[c for c in portfolios.columns if c != 'date']}")

print("\n[2/5] Preparing data for Fama-Macbeth...")

# Get RF for excess returns
ff_raw_rf = pdr.get_data_famafrench('F-F_Research_Data_5_Factors_2x3', start='1926')[0]
ff_with_rf = pd.DataFrame({
    'date': ff_raw_rf.index.to_timestamp(),
    'rf': ff_raw_rf['RF'].values
})

# Melt portfolios to long format
portfolio_cols = [c for c in portfolios.columns if c != 'date']
port_long_list = []

for port_name in portfolio_cols:
    df_port = portfolios[['date', port_name]].copy()
    df_port['portfolio'] = port_name
    df_port = df_port.rename(columns={port_name: 'ret'})
    port_long_list.append(df_port)

port_long = pd.concat(port_long_list, ignore_index=True)

# Merge with RF
port_long = port_long.merge(ff_with_rf, on='date', how='inner')
port_long['ret_excess'] = port_long['ret'] - port_long['rf']

# Create size and value dummies
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
n_simulations = 1000
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

# Check for significance flipping
size_baseline_sig = abs(fm_baseline.tstat['is_small']) > 1.96
value_baseline_sig = abs(fm_baseline.tstat['is_value']) > 1.96

print(f"\n" + "="*80)
print("SIGNIFICANCE ANALYSIS")
print("="*80)

if size_baseline_sig:
    pct_sig = (np.abs(t_stats_size) > 1.96).mean() * 100
    print(f"\nSize Premium:")
    print(f"  Baseline: SIGNIFICANT (|t| = {abs(fm_baseline.tstat['is_small']):.2f})")
    print(f"  Remains significant in {pct_sig:.1f}% of simulations")
    if pct_sig < 95:
        print(f"  ⚠️  FRAGILE! Can flip with decimal rounding")

if value_baseline_sig:
    pct_sig = (np.abs(t_stats_value) > 1.96).mean() * 100
    print(f"\nValue Premium:")
    print(f"  Baseline: SIGNIFICANT (|t| = {abs(fm_baseline.tstat['is_value']):.2f})")
    print(f"  Remains significant in {pct_sig:.1f}% of simulations")
    if pct_sig < 95:
        print(f"  ⚠️  FRAGILE! Can flip with decimal rounding")

print("\n[5/5] Saving results...")
fm_results = pd.DataFrame({
    'simulation': range(len(t_stats_size)),
    't_stat_size': t_stats_size,
    't_stat_value': t_stats_value,
    'coef_size': coefs_size,
    'coef_value': coefs_value,
})
fm_results.to_csv('results/tables/fm_sensitivity_1000sim.csv', index=False)
print(f"  Saved to: results/tables/fm_sensitivity_1000sim.csv")

print("\n" + "="*80)
print("FAMA-MACBETH ANALYSIS COMPLETE!")
print("="*80)

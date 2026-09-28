#!/usr/bin/env python3
# plot_modis_spei_density.py


#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
from pathlib import Path

#%%
# Load data
data_file = Path('/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/MCD19A1_analysis/MODIS_SPEI12_values_JAS.parquet')
print(f"Loading data from {data_file}...")
df = pd.read_parquet(data_file)

# Remove any remaining NaN values
initial_count = len(df)
df = df.dropna(subset=['NDVI', 'CCI', 'NIRv', 'NDMI', 'SPEI'])
final_count = len(df)

print(f"Loaded {final_count:,} valid data points ({initial_count - final_count:,} removed)")
print(f"Years: {df['year'].min():.0f} - {df['year'].max():.0f}")
print(f"Tiles: {sorted(df['tile'].unique())}")

# Plotting setup
plt.rcParams.update({
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 13,
})

bins_2dhist = 100
plot_dir = Path('/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/drought/scatter_annual/QCfilt_CloudFree/')
plot_dir.mkdir(parents=True, exist_ok=True)

# Define vegetation indices to plot
indices = {
    'CCI': {'ylim': (-0.1, 0.3), 'ylabel': 'CCI'},
    'NDVI': {'ylim': (0.7, 1.0), 'ylabel': 'NDVI'},
    'NDMI': {'ylim': (0.5, 0.9), 'ylabel': 'NDMI'},
    'NIRv': {'ylim': (0.1, 0.3), 'ylabel': 'NIRv'},
}

# Create plots for each index
for index_name, params in indices.items():
    print(f"\n{'='*60}")
    print(f"Plotting SPEI-12 vs {index_name}...")
    print(f"{'='*60}")
    
    # Prepare data
    X = df['SPEI'].values.reshape(-1, 1)
    y = df[index_name].values
    
    # Linear regression
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    r2 = r2_score(y, y_pred)
    
    # Spearman correlation
    spearman_r, p_val = stats.spearmanr(X.flatten(), y)
    
    print(f"  R² = {r2:.4f}")
    print(f"  Spearman r = {spearman_r:.4f} (p = {p_val:.2e})")
    print(f"  Slope = {reg.coef_[0]:.6f}")
    print(f"  Intercept = {reg.intercept_:.6f}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(4.75, 4))
    h = ax.hist2d(X.flatten(), y, bins=bins_2dhist, cmap='bone_r')
    ax.set_xlabel('SPEI-12')
    ax.set_ylabel(params['ylabel'])
    ax.grid(True, alpha=0.3)
    ax.set_ylim(params['ylim'])
    
    # Plot regression line
    ax.plot(X.flatten(), y_pred, color='red', linewidth=2, alpha=0.8, 
            label=f"R²={r2:.3f}, ρ={spearman_r:.3f}")
    ax.legend(loc='best', framealpha=0.9)
    
    fig.tight_layout()
    
    # Save figure
    output_file = plot_dir / f'SPEI12_vs_{index_name}_JAS.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.show()
    plt.close()

print(f"\n{'='*60}")
print("All plots complete!")
print(f"{'='*60}")
# %%

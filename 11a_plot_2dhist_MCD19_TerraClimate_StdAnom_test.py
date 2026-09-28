#!/usr/bin/env python3
# plot_modis_terraclimate_density.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
from pathlib import Path


# Load data
data_file = Path('/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/MCD19A1_analysis/MCD19_TerraClimate_StdAnom_values_JAS.parquet')
print(f"Loading data from {data_file}...")
df = pd.read_parquet(data_file)

# Remove any remaining NaN values
initial_count = len(df)
df = df.dropna()
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
# NOTE: Not using these ylims because these are std anomalies. trying 10th and 90th percentiles later on...
veg_indices = {
    'CCI_std_anomaly': {'ylim': (-0.1, 0.3), 'ylabel': 'CCI'},
    'NDVI_std_anomaly': {'ylim': (0.7, 1.0), 'ylabel': 'NDVI'},
    'NDMI_std_anomaly': {'ylim': (0.5, 0.9), 'ylabel': 'NDMI'},
    'NIRv_std_anomaly': {'ylim': (0.1, 0.3), 'ylabel': 'NIRv'},
}

# Define climate variables to plot
climate_vars = {
    'PDSI': {'xlabel': 'PDSI', 'xlim': None},
    'CWD': {'xlabel': 'CWD (mm)', 'xlim': None},
    'PET': {'xlabel': 'PET (mm)', 'xlim': None},
    'SoilM': {'xlabel': 'Soil Moisture (mm)', 'xlim': None},
    'VPD': {'xlabel': 'VPD (kPa)', 'xlim': None},
}

# Create plots for each combination
for climate_name, climate_params in climate_vars.items():
    print(f"\n{'='*60}")
    print(f"Processing {climate_name}...")
    print(f"{'='*60}")
    
    for veg_name, veg_params in veg_indices.items():
        print(f"  Plotting {climate_name} vs {veg_name}...")

        output_file = plot_dir / f'TerraClimate_{climate_name}_vs_{veg_name}_JAS.png'
        # if os.path.exists(output_file):
        #     print(f"    File {output_file} already exists, skipping...")
            # continue

        # Prepare data
        X = df[climate_name].values.reshape(-1, 1)
        y = df[veg_name].values
        ymin = np.nanquantile(y, 0.04)
        ymax = np.nanquantile(y, 0.96)

        xmin = np.nanquantile(X, 0.4)
        xmax = np.nanquantile(X, 0.96)

        # Linear regression
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        r2 = r2_score(y, y_pred)
        
        # Spearman correlation
        spearman_r, p_val = stats.spearmanr(X.flatten(), y)
        
        print(f"    R² = {r2:.4f}, Spearman r = {spearman_r:.4f}")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(4.75, 4))
        h = ax.hist2d(X.flatten(), y, bins=bins_2dhist, cmap='bone_r')
        ax.set_xlabel(climate_params['xlabel'])
        ax.set_ylabel(veg_params['ylabel'])
        ax.grid(True, alpha=0.3)
        ax.set_ylim([ymin, ymax])
        ax.set_xlim([xmin, xmax])
        # if climate_params['xlim']:
        #     ax.set_xlim(climate_params['xlim'])

        # Plot regression line
        ax.plot(X.flatten(), y_pred, color='red', linewidth=2, alpha=0.8, 
                label=f"R²={r2:.3f}, ρ={spearman_r:.3f}")
        ax.legend(loc='best', framealpha=0.9)

        fig.tight_layout()

        # Save figure
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_file}")

        # Free memory
        plt.close(fig)
        del h  # Delete histogram object explicitly

print(f"\n{'='*60}")
print("All plots complete!")
print(f"Total plots created: {len(climate_vars) * len(veg_indices)}")
print(f"{'='*60}")
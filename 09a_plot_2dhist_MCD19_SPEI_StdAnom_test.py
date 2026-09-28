#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#

#########################################
# Load packages
#########################################
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
from pathlib import Path

# from tensorboard import summary

#########################################
# Define Global Filepaths
#########################################

# set up directories
dat_basedir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/MCD19A1_analysis/by_ecoregion/' 
out_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/MCD19A1_analysis/by_ecoregion/summary_data'


########################################
# Define Global Variables and constants
#########################################


# Plotting setup
plt.rcParams.update({
    'axes.labelsize': 17,
    'axes.titlesize': 17,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 13,
    'agg.path.chunksize': 10000,  # Add this line
})

bins_2dhist = 100


# Define vegetation indices to plot
# NOTE: Not using these ylims because these are std anomalies. trying 10th and 90th percentiles later on...
veg_indices = {
    'CCI_std_anomaly': {'ylim': (-0.1, 0.3), 'ylabel': 'CCI Std Anomaly'},
    'NDVI_std_anomaly': {'ylim': (0.7, 1.0), 'ylabel': 'NDVI Std Anomaly'},
    'NDMI_std_anomaly': {'ylim': (0.5, 0.9), 'ylabel': 'NDMI Std Anomaly'},
    'NIRv_std_anomaly': {'ylim': (0.1, 0.3), 'ylabel': 'NIRv Std Anomaly'},
}

# Define climate variables to plot
climate_vars = {
    'PDSI': {'xlabel': 'PDSI', 'xlim': None},
    'CWD': {'xlabel': 'CWD Anomaly (mm)', 'xlim': None},
    'PET': {'xlabel': 'PET Anomaly (mm)', 'xlim': None},
    'SoilM': {'xlabel': 'Soil Moisture Anomaly (mm)', 'xlim': None},
    'VPD': {'xlabel': 'VPD Anomaly (kPa)', 'xlim': None},
    'SPEI01': {'xlabel': 'SPEI-01', 'xlim': None},
    'SPEI12': {'xlabel': 'SPEI-12', 'xlim': None},
}

#########################################
# Define Global Functions
#########################################


#########################################
# Begin Main
#########################################

# def main():

ecoregion = 'Colorado_Plateaus'
climate_name = 'SPEI12'
veg_name = 'NDVI_std_anomaly'
timing_abbrev = 'JJA'

ecoregion_withSpaceSlash = ecoregion.replace('_', ' ').replace("~", "/").replace(",", "") #Arizona/New Mexico Mountains

data_file = os.path.join(dat_basedir, f'MCD19_SPEI_values_{timing_abbrev}_{ecoregion}.parquet')

if not os.path.exists(data_file):
    print(f"Data file {data_file} not found, skipping...")
    sys.exit(0)

plot_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/drought/scatter_annual/QCfilt_CloudFree/std_anomaly_{timing_abbrev}'
climate_params = climate_vars[climate_name]
veg_params = veg_indices[veg_name]
# Create plots for each combination

print(f"Loading data from {data_file}...")
df = pd.read_parquet(data_file)

# Remove any remaining NaN values
initial_count = len(df)
df = df.dropna()
final_count = len(df)

print(f"Loaded {final_count:,} valid data points ({initial_count - final_count:,} removed)")
print(f"Years: {df['year'].min():.0f} - {df['year'].max():.0f}")
print(f"Tiles: {sorted(df['tile'].unique())}")
print(f"\n{'='*60}")
print(f"Processing {climate_name}...")
print(f"{'='*60}")

print(f"  Plotting {climate_name} vs {veg_name}...")

plot_dir = os.path.join(plot_basedir, f'{climate_name}_{veg_name}')
Path(plot_dir).mkdir(parents=True, exist_ok=True)

out_file = os.path.join(out_dir, f'lm_SPEIbase_{climate_name}_vs_{veg_name}_{timing_abbrev}_{ecoregion}.csv')
plot_file = os.path.join(plot_dir, f'SPEIbase_{climate_name}_vs_{veg_name}_{timing_abbrev}_{ecoregion}.png')
# if os.path.exists(plot_file):
#     print(f"    File {plot_file} already exists, skipping...")
    # continue

# Prepare data
X = df[climate_name].values.reshape(-1, 1)
y = df[veg_name].values
ymin = np.nanquantile(y, 0.04)
ymax = np.nanquantile(y, 0.96)

xmin = np.nanquantile(X, 0.04)
xmax = np.nanquantile(X, 0.96)

# Linear regression
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)
r2 = r2_score(y, y_pred)
slope = reg.coef_[0]
lm_p_val = stats.linregress(X.flatten(), y).pvalue

# Spearman correlation
spearman_r, spearman_p_val = stats.spearmanr(X.flatten(), y)

print(f"    R² = {r2:.4f}, Spearman r = {spearman_r:.4f}")
print(f"    Linear regression p-value = {lm_p_val:.4f}")
print(f"    Spearman correlation p-value = {spearman_p_val:.4f}")

summary_df = pd.DataFrame({
    'ecoregion': [ecoregion_withSpaceSlash],
    'climate_variable': [climate_name],
    'vegetation_index': [veg_name],
    'slope': [slope],
    'r_squared': [r2],
    'lm_p_value': [lm_p_val],
    'spearman_r': [spearman_r],
    'spearman_p_value': [spearman_p_val]
})
summary_df.to_csv(out_file, index=False)
print(f"    Saved regression summary to {out_file}")

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
        label=f"slope={reg.coef_[0]:.3e}\nρ={spearman_r:.3f}")
ax.legend(loc='best', framealpha=0.9)

fig.tight_layout()

# Save figure
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"    Saved: {plot_file}")

# Free memory
plt.close(fig)
del h  # Delete histogram object explicitly

print(f"\n{'='*60}")
print("plotting complete!")
print(f"{'='*60}")

# if __name__ == "__main__":
#     main()
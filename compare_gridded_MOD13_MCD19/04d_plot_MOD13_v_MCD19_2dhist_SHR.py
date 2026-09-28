#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Step 2 of 2 for the MCD19-vs-MOD13 2D histogram figure (ESSD manuscript, Sec 3.2 /
# planned Fig. 3.2): load the precomputed histogram + regression summary written by
# 08a_compute_mcd19_mod13_histogram_stats.py and plot it. Nothing here re-reads the
# Parquet dataset or re-bins any data, so this is meant to be re-run over and over while
# iterating on plot aesthetics (colormap, fonts, figure size, line colors, etc.).
#
# Axis convention, bias definition, regression type, and NDVI floor filter were all decided
# upstream in 08a - see that script's header for the full record of those decisions.
#

#########################################
# Load packages
#########################################
#%%
import os

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

#%%
#########################################
# Config
#########################################

OUTPUT_ROOT = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/MCD19_MOD13_comparison'
SUMMARY_DIR = os.path.join(OUTPUT_ROOT, 'histogram_summaries')

# Select land cover class to plot for this run - must match a summary file already written
# by 08a_compute_mcd19_mod13_histogram_stats.py
land_cover = 'Shrubland'

summary_file = os.path.join(SUMMARY_DIR, f'{land_cover}_ndvi_hist2d_summary.npz')

#########################################
# Plot aesthetics - iterate on these freely
#########################################

plt.rcParams.update({
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 13,
})


#########################################
# Begin Main
#########################################
#%%

print(f"Loading histogram summary for land_cover = {land_cover}...")
summary = np.load(summary_file)

H = summary['H']
xedges = summary['xedges']
yedges = summary['yedges']
lo = float(summary['lo'])
hi = float(summary['hi'])
land_cover_label = str(summary['land_cover'])

land_cover_stats = {
    'n': int(summary['n']),
    'slope': float(summary['slope']),
    'intercept': float(summary['intercept']),
    'r2': float(summary['r2']),
    'bias': float(summary['bias']),
    'rmse': float(summary['rmse']),
    'mae': float(summary['mae']),
}

#%%

FIGSIZE = (4.5, 4)
DENSITY_CMAP = 'inferno'
FIT_LINE_COLOR = '#3D8BFF'
ONE_TO_ONE_COLOR = '#8F8F8F'


fig, ax = plt.subplots(figsize=FIGSIZE)
ax.set_facecolor('black')

# H has shape (n_xbins, n_ybins) from np.histogram2d; pcolormesh wants (n_ybins, n_xbins) -
# same transpose hist2d applies internally (matplotlib.Axes.hist2d calls pcolormesh(xedges,
# yedges, h.T, ...)), so this reproduces exactly what hist2d(x, y, ...) would have plotted.
pcm = ax.pcolormesh(xedges, yedges, H.T, cmap=DENSITY_CMAP, norm=LogNorm())
cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Pixel count', fontsize=12)
cbar.ax.tick_params(labelsize=12)

# 1:1 line
ax.plot([lo, hi], [lo, hi], color=ONE_TO_ONE_COLOR, linestyle='--', linewidth=1.5, label='1:1 line')

# OLS fit line
fit_x = np.array([lo, hi])
fit_y = land_cover_stats['slope'] * fit_x + land_cover_stats['intercept']
ax.plot(fit_x, fit_y, color=FIT_LINE_COLOR, linestyle='-', linewidth=2, label='OLS fit')

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect('equal')
ax.set_xlabel('MOD13 NDVI')
ax.set_ylabel('MCD19 NDVI')
ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
# ax.set_title(f'MCD19 vs. MOD13 NDVI: {land_cover_label}')
# ax.legend(loc='upper left', frameon=False, fontsize=9, labelcolor='white')

stats_text = (
    # f"N = {land_cover_stats['n']:,}\n"
    f"R² = {land_cover_stats['r2']:.3f}\n"
    # f"slope = {land_cover_stats['slope']:.3f}\n"
    # f"intercept = {land_cover_stats['intercept']:.3f}\n"
    f"bias = {land_cover_stats['bias']:.3f}\n"
    f"RMSE = {land_cover_stats['rmse']:.3f}"
    # f"MAE = {land_cover_stats['mae']:.3f}"
)
ax.text(
    0.98, 0.03, stats_text,
    transform=ax.transAxes,
    ha='right', va='bottom',
    fontsize=12,
    color='white',
    bbox=dict(boxstyle='round', facecolor='none', edgecolor='none', alpha=0.0),
)

fig.tight_layout()
plt.show()

# %%



### Version 2 - White to dark color scheme



FIGSIZE = (4.5, 4)
DENSITY_CMAP = 'viridis'
FIT_LINE_COLOR = 'red'
ONE_TO_ONE_COLOR = 'black'


fig, ax = plt.subplots(figsize=FIGSIZE)
ax.set_facecolor('#01002B')

# H has shape (n_xbins, n_ybins) from np.histogram2d; pcolormesh wants (n_ybins, n_xbins) -
# same transpose hist2d applies internally (matplotlib.Axes.hist2d calls pcolormesh(xedges,
# yedges, h.T, ...)), so this reproduces exactly what hist2d(x, y, ...) would have plotted.
pcm = ax.pcolormesh(xedges, yedges, H.T, cmap=DENSITY_CMAP, norm=LogNorm())
cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Pixel count', fontsize=12)
cbar.ax.tick_params(labelsize=12)

# 1:1 line
ax.plot([lo, hi], [lo, hi], color=ONE_TO_ONE_COLOR, linestyle='--', linewidth=1.5, label='1:1 line')

# OLS fit line
fit_x = np.array([lo, hi])
fit_y = land_cover_stats['slope'] * fit_x + land_cover_stats['intercept']
ax.plot(fit_x, fit_y, color=FIT_LINE_COLOR, linestyle='-', linewidth=2, label='OLS fit')

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect('equal')
ax.set_xlabel('MOD13 NDVI')
ax.set_ylabel('MCD19 NDVI')
ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
# ax.set_title(f'MCD19 vs. MOD13 NDVI: {land_cover_label}')
# ax.legend(loc='upper left', frameon=False, fontsize=9, labelcolor='white')

stats_text = (
    # f"N = {land_cover_stats['n']:,}\n"
    f"R² = {land_cover_stats['r2']:.3f}\n"
    # f"slope = {land_cover_stats['slope']:.3f}\n"
    # f"intercept = {land_cover_stats['intercept']:.3f}\n"
    f"bias = {land_cover_stats['bias']:.3f}\n"
    f"RMSE = {land_cover_stats['rmse']:.3f}"
    # f"MAE = {land_cover_stats['mae']:.3f}"
)
ax.text(
    0.98, 0.03, stats_text,
    transform=ax.transAxes,
    ha='right', va='bottom',
    fontsize=12,
    color='white',
    bbox=dict(boxstyle='round', facecolor='none', edgecolor='none', alpha=0.0),
)

fig.tight_layout()
plt.show()

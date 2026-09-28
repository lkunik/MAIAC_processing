#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load per-site MCD19 vs. PhotoSpec/FloX QC fit-stat CSVs and plot heatmaps
of QC-scheme effect (site x delta-from-row-mean) for bias, R2, Spearman_r,
and CRMSE, as one figure: NDVI in the left column, CCI in the right column,
metrics top to bottom in that order.

Each input CSV is the fit_stats_df written out by the QC-comparison script
(out_stats_descr = "QC_Compare"), with columns:
    QC, QC_descr, element, R2, bias, CRMSE, Spearman_r, Spearman_pval

Each heatmap shows QC1/QC2/QC3/QC4 minus that site's own mean across QC1-4,
so site-to-site differences in absolute fit quality (driven by spatial
heterogeneity, not QC choice) are removed and what's left is the effect of
QC choice relative to that site's own average.
"""

#%%
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

DATA_DIR = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/output/CVMVC_QC_regression_stats'

SITES = ['US-NR1', 'OSBS', 'Ca-Obs', 'DEJU', 'US-UMB', 'US-Ne3']

METRIC_ORDER = ['bias', 'R2', 'Spearman_r', 'CRMSE']  # figure rows, top to bottom
ELEMENT_ORDER = ['NDVI', 'CCI']  # figure columns, left to right

QC_ORDER = ['QC1', 'QC2', 'QC3', 'QC4']
QC_LABELS = {
    'QC1': 'AllAOD,\nAllAdj',
    'QC2': 'LowAOD,\nAllAdj',
    'QC3': 'AllAOD,\nClearAdj',
    'QC4': 'LowAOD,\nClearAdj',
}

# vmin/vmax per metric for the diverging color scale (delta values), applied
# symmetrically as [-vmax, vmax]. Set to None to auto-scale to that metric's
# own max abs delta. Edit these while experimenting.
DELTA_VMAX = {
    'R2': None,
    'bias': None,
    'Spearman_r': None,
    'CRMSE': None,
}

#%%
############################################
# Load each site's fit-stats CSV and stack into one long dataframe
############################################

# Filename product label varies (PhotoSpec vs FloX for US-Ne3), so the glob
# wildcards that segment rather than hard-coding it.
site_dfs = []
for site_name in SITES:
    pattern = os.path.join(DATA_DIR, f'MCD19_v_*_fit_stats_QC_Compare_{site_name}.csv')
    matches = glob.glob(pattern)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file for site '{site_name}' matching {pattern}, "
            f"found {len(matches)}: {matches}"
        )
    site_df = pd.read_csv(matches[0])
    # First three columns are the former MultiIndex (QC, QC_descr, element)
    site_df = site_df.rename(columns={site_df.columns[0]: 'QC', site_df.columns[1]: 'QC_descr', site_df.columns[2]: 'element'})
    site_df['site'] = site_name
    site_dfs.append(site_df)

combined = pd.concat(site_dfs, ignore_index=True)

#%%
############################################
# Plot heatmap function (called once per metric x element subplot below)
############################################

def plot_delta_heatmap(ax, mat, metric, element, vmax=None):
    """mat: site x [QC1, QC2, QC3, QC4] dataframe of (QC_i - row mean) deltas."""
    values = mat.values.astype(float)

    finite = values[np.isfinite(values)]
    if vmax is None:
        vmax = np.nanmax(np.abs(finite)) if finite.size else 1.0
    vmax = vmax if vmax > 0 else 1.0

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    if element == 'NDVI':
        if metric == 'bias':
            cmap = 'RdBu'
        elif metric in ['R2', 'Spearman_r']:
            cmap = 'YlGn'
        else:
            cmap = 'YlGn_r'
    else:  # CCI
        if metric == 'bias':
            cmap = 'RdBu'
        elif metric in ['R2', 'Spearman_r']:
            cmap = 'YlOrBr'
        else:
            cmap = 'YlOrBr_r'
    im = ax.imshow(values, aspect='auto', cmap=cmap, norm=norm)

    site_labels = mat.index.to_series().replace({'OSBS': 'US-xSB', 'DEJU': 'US-xDJ'})

    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([QC_LABELS[q] for q in mat.columns])
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(site_labels)
    ax.set_title(f'{metric} -- {element}')

    # annotate cell values
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isfinite(val):
                if metric == 'bias':
                    text_color = 'white' if abs(val) > 0.7 * vmax else 'black'
                elif metric in ['R2', 'Spearman_r']:
                    text_color = 'white' if val > 0 else 'black'
                else:
                    text_color = 'white' if val < 0 else 'black'
                ax.text(j, i, f'{val:+.4f}', ha='center', va='center', color=text_color, fontsize=10)

    cbar = ax.figure.colorbar(im, ax=ax, label=f'Δ{metric} from site mean')
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label(f'Δ{metric} from site mean', fontsize=11)

#%%
############################################
# Pivot to site x QC matrices, subtract each row's own mean, and plot
# everything as one figure: rows = metrics (bias, R2, Spearman_r, CRMSE),
# columns = elements (NDVI, CCI)
############################################

# Set global matplotlib font sizes for axes and legends
plt.rcParams.update({
   'axes.labelsize': 15,
   'axes.titlesize': 15,
   'xtick.labelsize': 12,
   'ytick.labelsize': 14,
   'legend.fontsize': 10,
})

fig, axes = plt.subplots(
    nrows=len(METRIC_ORDER),
    ncols=len(ELEMENT_ORDER),
    figsize=(6 * len(ELEMENT_ORDER), 4.2 * len(METRIC_ORDER)),
)

for row, metric in enumerate(METRIC_ORDER):
    for col, element in enumerate(ELEMENT_ORDER):
        ax = axes[row, col]

        sub = combined[combined['element'] == element]
        mat = sub.pivot(index='site', columns='QC', values=metric)
        mat = mat.reindex(index=SITES, columns=QC_ORDER)

        row_mean = mat.mean(axis=1)
        delta_mat = mat.subtract(row_mean, axis=0)

        plot_delta_heatmap(ax, delta_mat, metric, element, vmax=DELTA_VMAX[metric])

plt.tight_layout()
plt.show()
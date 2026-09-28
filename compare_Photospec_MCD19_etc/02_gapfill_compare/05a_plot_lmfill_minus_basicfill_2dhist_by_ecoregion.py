#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Heatmaps of lm-fill minus basic-fill (y) by time of year (x) at gap-filled pixels, per ecoregion,
# with the median per composite period. One multi-panel figure per index (CCI, NDVI).
# Reads the .npz summaries from 05_calc_2dhist_lmfill_minus_basicfill_by_ecoregion_batch.py.
# Same layout as 03a_plot_diff_from_median_2dhist_by_ecoregion.py.
#

#%%
import os
import numpy as np
from matplotlib import pyplot as plt

#########################################
# Settings
#########################################

summary_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/summary_data'
ecoregions_list = ['Boreal', 'Eastern', 'Western', 'Southwest US Mexico', 'GREAT PLAINS', 'NORTH AMERICAN DESERTS']
ecoregion_titles = {
    'Boreal': 'Boreal Forests (ENF)',
    'Eastern': 'Eastern Forests (Mixed/DBF)',
    'Western': 'Western Cordillera (ENF)',
    'Southwest US Mexico': 'Temperate Sierras (ENF)',
    'GREAT PLAINS': 'Great Plains (Grassland)',
    'NORTH AMERICAN DESERTS': 'Deserts (Shrubland)'
}
# panel position (row, column) of each ecoregion in the multi-panel figure
panel_layout = {
    'Boreal': (0, 0),
    'GREAT PLAINS': (1, 0),
    'NORTH AMERICAN DESERTS': (2, 0),
    'Western': (0, 1),
    'Eastern': (1, 1),
    'Southwest US Mexico': (2, 1)
}
file_suffix = '_lmfill_minus_basicfill_hist_LCmask_only.npz'

pct_lo, pct_hi = 0.5, 99.5   # y-axis limits = these percentiles of all gap-filled values (all composite periods), per ecoregion
n_bins = 30              # bins between the limits; values outside are stacked into the bottom/top bins
cmap_name = 'magma_r'

# colorbar position in each panel's axes coordinates [x0, y0, width, height]
cbar_pos = [1.02, 0, 0.025, 1]

# x axis in day of year, with month letter labels (same as the seasonality script)
DOY_composite_starts = np.arange(1, 366, 16)
x_edges = np.append(DOY_composite_starts, 366)   # composite period edges in day of year (last period ends Dec 31)
months_begin = np.array([1, 32, 61, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365])
months_doy = (months_begin[1:] + months_begin[:-1]) / 2
months_labs = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

plt.rcParams.update({
   'axes.labelsize': 14,
   'axes.titlesize': 14,
   'xtick.labelsize': 12,
   'ytick.labelsize': 12,
})


def percentile_from_hist(counts_1d, fine_edges, pct):
    # counts_1d: (n_fine,) counts in slots [underflow, fine bins..., overflow]
    # returns the fine bin edge at which the cumulative fraction first reaches pct/100
    # (lower edge of that bin for the low percentile, upper edge for the high percentile; accurate to the fine bin width)
    cdf = np.cumsum(counts_1d) / counts_1d.sum()
    k = int(np.searchsorted(cdf, pct / 100))   # slot index
    k = min(max(k, 1), len(fine_edges) - 1)    # underflow/overflow slots -> first/last fine bin
    return fine_edges[k - 1] if pct < 50 else fine_edges[k]


def median_from_hist(counts_1d, fine_edges):
    # median from a fine histogram: center of the fine bin where the cumulative fraction first reaches 0.5
    # (NaN if there are no counts; accurate to the fine bin width)
    if counts_1d.sum() == 0:
        return np.nan
    cdf = np.cumsum(counts_1d) / counts_1d.sum()
    k = int(np.searchsorted(cdf, 0.5))
    k = min(max(k, 1), len(fine_edges) - 1)
    return (fine_edges[k - 1] + fine_edges[k]) / 2


#%%
# load every ecoregion's summary once, then build one figure per index
dat_dict = {}
for ecoregion in ecoregions_list:
    npz_file = os.path.join(summary_dir, f'{ecoregion.replace(" ", "_")}{file_suffix}')
    if not os.path.exists(npz_file):
        print(f'File not found: {npz_file}. Skipping {ecoregion}.')
        continue
    with np.load(npz_file) as dat:
        dat_dict[ecoregion] = {k: dat[k] for k in dat.files}

indices = list(next(iter(dat_dict.values()))['indices'])

#%%
for i_idx, idx in enumerate(indices):

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    for ecoregion, (row, col) in panel_layout.items():
        ax = axes[row, col]
        if ecoregion not in dat_dict:
            ax.set_title(ecoregion_titles[ecoregion])
            ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
            continue

        dat = dat_dict[ecoregion]
        counts = dat['counts'][i_idx]              # (composite_period, n_fine)
        fine_edges = dat['fine_edges']
        n_years = len(dat['years'])

        # position of each fine slot (underflow = -inf, fine bin centers, overflow = +inf), used to assign slots to plot bins
        slot_pos = np.concatenate([[-np.inf], (fine_edges[:-1] + fine_edges[1:]) / 2, [np.inf]])

        # y-axis limits from all gap-filled values of this index (all composite periods)
        all_counts = counts.sum(axis=0)
        if all_counts.sum() == 0:
            print(f'No gap-filled {idx} data for {ecoregion}. Skipping.')
            ax.set_title(ecoregion_titles[ecoregion])
            ax.text(0.5, 0.5, 'no gap-filled pixels', ha='center', va='center', transform=ax.transAxes)
            continue
        y_lo = percentile_from_hist(all_counts, fine_edges, pct_lo)
        y_hi = percentile_from_hist(all_counts, fine_edges, pct_hi)
        y_edges = np.linspace(y_lo, y_hi, n_bins + 1)
        print(f'{ecoregion} {idx}: {pct_lo}th pct = {y_lo:.4f}, {pct_hi}th pct = {y_hi:.4f}')

        # assign every fine slot to a plot bin; slots outside the limits go to the bottom/top bin
        plot_bin = np.clip(np.searchsorted(y_edges, slot_pos, side='right') - 1, 0, n_bins - 1)
        M = np.zeros((len(slot_pos), n_bins))
        M[np.arange(len(slot_pos)), plot_bin] = 1
        H = counts @ M / n_years               # (composite_period, n_bins): mean annual count of gap-filled pixels

        H_plot = np.ma.masked_equal(H.T, 0)    # empty bins transparent
        pm = ax.pcolormesh(x_edges, y_edges, H_plot, cmap=cmap_name, shading='flat', vmin=0, vmax=H_plot.max())

        cax = ax.inset_axes(cbar_pos)
        cb = fig.colorbar(pm, cax=cax)
        cb.ax.tick_params(labelsize=9)

        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

        # median lm-fill minus basic-fill per composite period, as a step line spanning each composite period's column
        med = np.array([median_from_hist(counts[i_cp], fine_edges) for i_cp in range(counts.shape[0])])
        med_step = np.append(med, med[-1])                  # repeat the last value so the final column is covered
        ax.step(x_edges, med_step, where='post', color='cyan', linewidth=1.5, label='median')
        if (row, col) == (0, 0):   # legend in the first panel only
            ax.legend(loc='upper right', fontsize=9, frameon=False)

        ax.set_xticks(months_doy)
        ax.set_xticklabels(months_labs)
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(ecoregion_titles[ecoregion])

    fig.supylabel(f'{idx}: lm-fill minus basic-fill')
    fig.suptitle(f'{idx} lm-fill minus basic-fill at gap-filled pixels (colorbars: pixels per year)')
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.05, top=0.93, wspace=0.35, hspace=0.35)
    plt.show()
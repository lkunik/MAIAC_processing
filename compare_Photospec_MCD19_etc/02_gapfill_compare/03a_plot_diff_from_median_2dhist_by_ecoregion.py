#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Heatmaps of the difference from the historical median (y) by time of year (x), per ecoregion,
# with outlier-flagged and kept values overlaid. One multi-panel figure per index (CCI, NDVI).
# Reads the .npz summaries from 03_calc_2dhist_diff_from_median_by_ecoregion_batch.py.
#

#%%
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

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
# CHANGED: panel position (row, column) of each ecoregion in the multi-panel figure
panel_layout = {
    'Boreal': (0, 0),
    'GREAT PLAINS': (1, 0),
    'NORTH AMERICAN DESERTS': (2, 0),
    'Western': (0, 1),
    'Eastern': (1, 1),
    'Southwest US Mexico': (2, 1)
}
file_suffix = '_diff_from_median_hist_LCmask_only.npz'

pct_lo, pct_hi = 0.5, 99.5   # y-axis limits = these percentiles of all values (flagged + kept, all composite periods)
n_bins = 30              # bins between the limits; values outside are stacked into the bottom/top bins
cmap_names = {'kept': 'bone_r', 'flagged': 'YlOrRd'}   # colormap per category
draw_order = ['kept', 'flagged']                        # kept drawn first, flagged drawn on top
K_MAD = 3.0   # outlier filter threshold multiplier (K in the outlier filter script); lines drawn at +/- K * median MAD

# CHANGED: colorbar positions in each panel's axes coordinates [x0, y0, width, height]; kept closest to the plot area
cbar_pos = {'kept': [1.02, 0, 0.025, 1],
            'flagged': [1.20, 0, 0.025, 1]}

# CHANGED: x axis in day of year, with month letter labels (same as the seasonality script)
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


def alpha_ramp_cmap(base_name, alpha_min=0.0, alpha_max=1.0, n=256):
    # copy of an existing colormap whose opacity rises linearly from alpha_min (low end) to alpha_max (high end),
    # so low-count bins are transparent and the other layer shows through
    base = plt.get_cmap(base_name)
    colors = base(np.linspace(0, 1, n))
    colors[:, 3] = np.linspace(alpha_min, alpha_max, n)
    return mcolors.ListedColormap(colors, name=f'{base_name}_alpha')


def percentile_from_hist(counts_1d, fine_edges, pct):
    # counts_1d: (n_fine,) counts in slots [underflow, fine bins..., overflow]
    # returns the fine bin edge at which the cumulative fraction first reaches pct/100
    # (lower edge of that bin for the low percentile, upper edge for the high percentile; accurate to the fine bin width)
    cdf = np.cumsum(counts_1d) / counts_1d.sum()
    k = int(np.searchsorted(cdf, pct / 100))   # slot index
    k = min(max(k, 1), len(fine_edges) - 1)    # underflow/overflow slots -> first/last fine bin
    return fine_edges[k - 1] if pct < 50 else fine_edges[k]


#%%
# CHANGED: load every ecoregion's summary once, then build one figure per index
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

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    for ecoregion, (row, col) in panel_layout.items():
        ax = axes[row, col]
        if ecoregion not in dat_dict:
            ax.set_title(ecoregion_titles[ecoregion])
            ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
            continue

        dat = dat_dict[ecoregion]
        counts = dat['counts']                     # (index, category, composite_period, n_fine)
        fine_edges = dat['fine_edges']
        categories = list(dat['categories'])
        n_years = len(dat['years'])

        # position of each fine slot (underflow = -inf, fine bin centers, overflow = +inf), used to assign slots to plot bins
        slot_pos = np.concatenate([[-np.inf], (fine_edges[:-1] + fine_edges[1:]) / 2, [np.inf]])

        # y-axis limits from all values of this index (both categories, all composite periods)
        all_counts = counts[i_idx].sum(axis=(0, 1))
        if all_counts.sum() == 0:
            print(f'No {idx} data for {ecoregion}. Skipping.')
            continue
        y_lo = percentile_from_hist(all_counts, fine_edges, pct_lo)
        y_hi = percentile_from_hist(all_counts, fine_edges, pct_hi)
        y_edges = np.linspace(y_lo, y_hi, n_bins + 1)
        print(f'{ecoregion} {idx}: {pct_lo}th pct = {y_lo:.4f}, {pct_hi}th pct = {y_hi:.4f}')

        # assign every fine slot to a plot bin; slots outside the limits go to the bottom/top bin
        plot_bin = np.clip(np.searchsorted(y_edges, slot_pos, side='right') - 1, 0, n_bins - 1)
        M = np.zeros((len(slot_pos), n_bins))
        M[np.arange(len(slot_pos)), plot_bin] = 1
        H = counts[i_idx] @ M / n_years        # (category, composite_period, n_bins): mean annual count of pixels

        # both categories on one axis: kept first, flagged on top; the flagged colormap fades from fully transparent at 0
        # to fully opaque at its max, so low-count flagged bins let the kept layer show through (zero bins are also masked)
        pms = {}
        for cat in draw_order:
            i_cat = categories.index(cat)
            H_cat = np.ma.masked_equal(H[i_cat].T, 0)
            cmap_cat = alpha_ramp_cmap(cmap_names[cat]) if cat == 'flagged' else cmap_names[cat]   # alpha ramp on the flagged layer only
            pms[cat] = ax.pcolormesh(x_edges, y_edges, H_cat, cmap=cmap_cat, shading='flat',
                                     vmin=0, vmax=H_cat.max())   # separate color limits per category

        # CHANGED: colorbars in fixed positions to the right of each panel (kept closest, flagged further out)
        for cat in ['kept', 'flagged']:
            cax = ax.inset_axes(cbar_pos[cat])
            cb = fig.colorbar(pms[cat], cax=cax)
            cb.ax.set_title(cat, fontsize=10)
            cb.ax.tick_params(labelsize=9)

        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

        # +/- K * median MAD per composite period, as step lines spanning each composite period's column
        if 'MAD_median' in dat:
            thr = K_MAD * dat['MAD_median'][i_idx]              # (composite_period,)
            thr_step = np.append(thr, thr[-1])                  # repeat the last value so the final column is covered
            for sign in (1, -1):
                ax.step(x_edges, sign * thr_step, where='post', color='black', linewidth=1.2, linestyle='--',
                        label=f'±{K_MAD:g} × median MAD' if sign == 1 else None)
            if (row, col) == (0, 0):   # legend in the first panel only
                ax.legend(loc='upper right', fontsize=9, frameon=False)
        else:
            print(f'No MAD_median for {ecoregion} (rerun 03_calc_2dhist_diff_from_median_by_ecoregion_batch.py to add it); skipping MAD lines.')

        ax.set_xticks(months_doy)
        ax.set_xticklabels(months_labs)
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(ecoregion_titles[ecoregion])

    fig.supylabel(f'{idx} minus {idx}_median')
    fig.suptitle(f'{idx} difference from median (colorbars: pixels per year)')
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.05, top=0.93, wspace=0.55, hspace=0.35)
    plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Assembles the per-site NDVI/CCI two-axes panel PNGs (produced by the
# 01a-01f QC-comparison scripts) into a single multi-panel summary
# figure: one row per site, NDVI panels in the left column, CCI panels
# in the right column.
#
# Run 01a-01f (or run_all_QC_compare.sh) first so the panel PNGs exist.

#%%
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plot_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/CVMVC_QC_regression_stats/'
panel_dir = os.path.join(plot_dir, 'panels')

# Row order, top to bottom. site_name values match those set in the
# 01a-01f scripts.
site_order = ['DEJU', 'Ca-Obs', 'US-UMB', 'US-Ne3', 'US-NR1', 'OSBS']


n_rows = int(len(site_order)/2)
legend_height = 1.0  # inches reserved for the legend row

# Extra bottom row (spanning both columns) holds the legend image saved by 03a
fig, axes = plt.subplots(n_rows + 1, 2, figsize=(11, 3 * n_rows + legend_height),
                         gridspec_kw={'height_ratios': [1] * n_rows + [legend_height / 3]})

for isite, site_name in enumerate(site_order):
    nobs_panel_file = os.path.join(panel_dir, f'{site_name}_nobs_AODcompare_panel.png')
    ax_nobs = axes[isite // 2, isite % 2]


    ax_nobs.imshow(mpimg.imread(nobs_panel_file))

    for ax in (ax_nobs,):
        ax.axis('off')

gs = axes[0, 0].get_gridspec()
for ax in axes[-1]:
    ax.remove()
ax_legend = fig.add_subplot(gs[-1, :])
ax_legend.imshow(mpimg.imread(os.path.join(panel_dir, 'nobs_AODcompare_legend.png')))
ax_legend.axis('off')

plt.tight_layout()

summary_fig_file = os.path.join(plot_dir, 'QC_AODcompare_nobs_summary_panel_figure.png')
plt.savefig(summary_fig_file, dpi=300, bbox_inches='tight')
print(f'Saved summary panel figure to {summary_fig_file}')

plt.show()

#%%

# Extra bottom row (spanning both columns) holds the legend image saved by 03a
fig, axes = plt.subplots(n_rows + 1, 2, figsize=(11, 3 * n_rows + legend_height),
                         gridspec_kw={'height_ratios': [1] * n_rows + [legend_height / 3]})

for isite, site_name in enumerate(site_order):
    nobs_panel_file = os.path.join(panel_dir, f'{site_name}_nobs_Adjcompare_panel.png')
    ax_nobs = axes[isite // 2, isite % 2]


    ax_nobs.imshow(mpimg.imread(nobs_panel_file))

    for ax in (ax_nobs,):
        ax.axis('off')

gs = axes[0, 0].get_gridspec()
for ax in axes[-1]:
    ax.remove()
ax_legend = fig.add_subplot(gs[-1, :])
ax_legend.imshow(mpimg.imread(os.path.join(panel_dir, 'nobs_Adjcompare_legend.png')))
ax_legend.axis('off')

plt.tight_layout()

summary_fig_file = os.path.join(plot_dir, 'QC_Adjcompare_nobs_summary_panel_figure.png')
plt.savefig(summary_fig_file, dpi=300, bbox_inches='tight')
print(f'Saved summary panel figure to {summary_fig_file}')

plt.show()
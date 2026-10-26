#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Assembles the per-site NDVI/CCI two-axes panel PNGs (produced by the
# 01a-01f QC-comparison scripts) into a single multi-panel summary
# figure: one row per site, NDVI panels in the left column, CCI panels
# in the right column. Sites in wide_sites (long time series, saved at
# double width) instead get one full-width row each for NDVI and CCI.
#
# Run 01a-01f (or run_all_QC_compare.sh) first so the panel PNGs exist.

#%%
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

plot_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/CVMVC_QC_regression_stats/final/'
panel_dir = os.path.join(plot_dir, 'panels')

# Row order, top to bottom. site_name values match those set in the
# 01a-01f scripts.
site_order = ['DEJU', 'Ca-Obs', 'US-UMB', 'US-Ne3', 'US-NR1', 'OSBS']

# Sites whose NDVI and CCI panels each get their own full-width row
wide_sites = {'Ca-Obs'}

# Build row layout: each row is a list of 1 (full width) or 2 (paired) images
rows = []
for site_name in site_order:
    ndvi_img = mpimg.imread(os.path.join(panel_dir, f'{site_name}_NDVI_panel.png'))
    cci_img = mpimg.imread(os.path.join(panel_dir, f'{site_name}_CCI_panel.png'))
    if site_name in wide_sites:
        rows.append([ndvi_img])
        rows.append([cci_img])
    else:
        rows.append([ndvi_img, cci_img])

# Row heights from image aspect ratios, in units of figure width
fig_width = 11
height_ratios = []
for imgs in rows:
    col_frac = 1 / len(imgs)
    height_ratios.append(max(img.shape[0] / img.shape[1] * col_frac for img in imgs))

# Legend image saved by 01ab; drawn in a row spanning both columns at the same inches-per-pixel as the panels
legend_img = mpimg.imread(os.path.join(panel_dir, 'final_compare_legend.png'))
in_per_px = fig_width / max(sum(img.shape[1] for img in imgs) for imgs in rows)
# Reserve additional height for the legend so it is displayed larger relative
# to the panel images.
legend_scale = 1.5
height_ratios.append(legend_img.shape[0] * in_per_px / fig_width * legend_scale)

fig = plt.figure(figsize=(fig_width, fig_width * sum(height_ratios)))
gs = GridSpec(len(height_ratios), 2, figure=fig, height_ratios=height_ratios,
              wspace=0.02, hspace=0.05, left=0, right=1, bottom=0, top=1)

for r, imgs in enumerate(rows):
    if len(imgs) == 1:
        ax = fig.add_subplot(gs[r, :])
        ax.imshow(imgs[0])
        ax.set_anchor('C')
        ax.axis('off')
    else:
        for c, img in enumerate(imgs):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(img)
            ax.set_anchor('E' if c == 0 else 'W')  # push paired images toward center
            ax.axis('off')

ax_legend = fig.add_subplot(gs[-1, :])
ax_legend.imshow(legend_img, aspect='auto')
ax_legend.set_anchor('C')
ax_legend.axis('off')

# plt.show()
plt.savefig(os.path.join(plot_dir, 'final_compare_panels.png'), dpi=300, bbox_inches='tight')
plt.close()

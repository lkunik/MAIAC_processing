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

plot_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/CVMVC_QC_regression_stats/'
panel_dir = os.path.join(plot_dir, 'panels')

# Row order, top to bottom. site_name values match those set in the
# 01a-01f scripts.
site_order = ['DEJU', 'Ca-Obs', 'US-UMB', 'US-Ne3', 'US-NR1', 'OSBS']

# Sites whose NDVI and CCI panels each get their own full-width row
wide_sites = {'Ca-Obs'}

# Build row layout: each row is (images, col_frac), 1 full-width or 2 paired images
rows = []
for site_name in site_order:
    ndvi_img = mpimg.imread(os.path.join(panel_dir, f'{site_name}_NDVI_panel.png'))
    cci_img = mpimg.imread(os.path.join(panel_dir, f'{site_name}_CCI_panel.png'))
    if site_name in wide_sites:
        rows.append(([ndvi_img], 1))
        rows.append(([cci_img], 1))
    else:
        rows.append(([ndvi_img, cci_img], 0.5))

# Legend image saved by 01ab
legend_img = mpimg.imread(os.path.join(panel_dir, 'QCcompare_legend.png'))

# Row heights from image aspect ratios, in units of figure width
fig_width = 11
height_ratios = [max(img.shape[0] / img.shape[1] * col_frac for img in imgs)
                 for imgs, col_frac in rows]

# Legend row (spanning both columns) drawn at the same inches-per-pixel as the panels
in_per_px = fig_width / max(sum(img.shape[1] for img in imgs) for imgs, _ in rows)
height_ratios.append(legend_img.shape[0] * in_per_px / fig_width)

fig = plt.figure(figsize=(fig_width, fig_width * sum(height_ratios)))
gs = GridSpec(len(height_ratios), 2, figure=fig, height_ratios=height_ratios,
              wspace=0.02, hspace=0.05, left=0, right=1, bottom=0, top=1)

for r, (imgs, col_frac) in enumerate(rows):
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
ax_legend.imshow(legend_img)
ax_legend.set_anchor('C')
ax_legend.axis('off')

summary_fig_file = os.path.join(plot_dir, 'QC_compare_summary_panel_figure.png')
plt.savefig(summary_fig_file, dpi=300, bbox_inches='tight')
print(f'Saved summary panel figure to {summary_fig_file}')

plt.show()

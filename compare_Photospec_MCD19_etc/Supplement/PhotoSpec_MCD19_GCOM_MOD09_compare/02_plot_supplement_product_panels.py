#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Assembles the per-site supplement panel PNGs (produced by the 01a-01f
# scripts) into one multi-panel figure per comparison product: one row per
# site, with a standalone legend row at the bottom.
#   MOD09:       NDVI (left) | CCI (right)
#   GCOM:        NDVI (left) | CCI (right)
#   MOD13_MCD43: MOD13 NDVI (left) | MCD43 NDVI (right)
# Sites in wide_sites (long time series, saved at double width) instead get
# one full-width row for each of their two panels.
#
# Run 01a-01f (or 01-run_all_supplement.sh) first so the panel and legend PNGs exist.

#%%
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

plot_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/CVMVC_QC_regression_stats/final/supplement/'
panel_dir = os.path.join(plot_dir, 'panels')

# Top-to-bottom site order (matches panel letters a-l)
site_order = ['DEJU', 'Ca-Obs', 'US-UMB', 'US-NR1', 'US-Ne3', 'OSBS']

# Sites whose two panels each get their own full-width row
wide_sites = {'Ca-Obs'}

fig_width = 11

# Figure name -> (left panel, right panel) as (element, product)
figures = {
    'MOD09': [('NDVI', 'MOD09'), ('CCI', 'MOD09')],
    'GCOM': [('NDVI', 'GCOM'), ('CCI', 'GCOM')],
    'MOD13_MCD43': [('NDVI', 'MOD13'), ('NDVI', 'MCD43')],
}


def build_rows(panel_pair):
    """Each row is (images, col_frac): 1 full-width image or 2 paired images."""
    rows = []
    for site_name in site_order:
        imgs = [mpimg.imread(os.path.join(panel_dir, f'{site_name}_{element}_{prod}_panel.png'))
                for element, prod in panel_pair]
        if site_name in wide_sites:
            rows.extend(([img], 1) for img in imgs)
        else:
            rows.append((imgs, 0.5))
    return rows


def plot_panel_rows(rows, legend_img):
    # Row heights from image aspect ratios, in units of figure width
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
    return fig


# Legend images saved by 01d (DEJU)
for fig_descr, panel_pair in figures.items():
    fig = plot_panel_rows(build_rows(panel_pair),
                          mpimg.imread(os.path.join(panel_dir, f'{fig_descr}_legend.png')))

    summary_fig_file = os.path.join(plot_dir, f'supplement_{fig_descr}_summary_panel_figure.png')
    plt.savefig(summary_fig_file, dpi=300, bbox_inches='tight')
    print(f'Saved summary panel figure to {summary_fig_file}')

    plt.show()

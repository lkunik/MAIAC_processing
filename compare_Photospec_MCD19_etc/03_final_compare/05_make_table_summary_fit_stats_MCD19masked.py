#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
#
# MCD19-masked version of 04_make_table_summary_fit_stats.py: summary tables of the
# final satellite product comparison (MCD19 lm-fill vs MOD09 / MOD13 / MCD43 / GCOM-C,
# each vs tower PhotoSpec / FloX) from the 05*_MCD19masked_batch.py scripts, where the
# comparison products are put on the MCD19 composites and masked wherever MCD19 has no
# valid value. Also writes a table of N (number of paired composites).
#
# Each table has one row per site and index (all sites for NDVI, then all sites
# for CCI), and one column per dataset/scenario. Each cell holds a single
# statistic, so one table image is written per statistic. The best value in each
# row is shown in bold.
#

#########################################
# Import packages
#########################################
#%%
import os
import glob

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


#########################################
# Define Global Filepaths
#########################################

stats_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/output/CVMVC_QC_regression_stats/'

table_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/CVMVC_QC_regression_stats/tables'
os.makedirs(table_dir, exist_ok=True)


#########################################
# Define Global Variables and constants
#########################################

# (site name used in filenames, site label shown in tables)
sites = [
    ('DEJU', 'US-xDJ'),
    ('Ca-Obs', 'Ca-Obs'),
    ('US-UMB', 'US-UMB'),
    ('US-NR1', 'US-NR1'),
    ('US-Ne3', 'US-Ne3'),
    ('OSBS', 'US-xSB'),
]

elements = ['NDVI', 'CCI']

# statistic -> (label shown in the table title, number format, how "best" is chosen)
stats = {
    'R2': ('R$^2$', '{:.2f}', 'max'),
    'Spearman_r': ('Spearman r', '{:.2f}', 'max'),
    'CRMSE': ('CRMSE', '{:.3f}', 'min'),
    'bias': ('Bias', '{:.3f}', 'min_abs'),
    'CCC': ("Lin's CCC", '{:.2f}', 'max'),
    'SD_ratio': ('SD ratio (satellite / tower)', '{:.2f}', 'closest_one'),
    'N': ('N (paired composites)', '{:.0f}', 'none'),
}

# Each column: (column label, csv file pattern relative to stats_dir, key column, key value).
final_file = 'final/MCD19masked/PhotoSpec_v_MCD19_etc_fit_stats_{site}.csv'

tables = {
    'Final_MCD19masked': {
        'title': 'Satellite product comparison (MCD19-masked)',
        'columns': [
            ('MCD19', final_file, 'QC', 'MCD19'),
            ('MOD09', final_file, 'QC', 'MOD09'),
            ('MOD13', final_file, 'QC', 'MOD13'),
            ('MCD43', final_file, 'QC', 'MCD43'),
            ('GCOM-C', final_file, 'QC', 'GCOM'),
        ],
    },
}

# (site label, column label) cells that are reported but left out of the bold "best" ranking
# (footnoted in the paper): GCOM-C at US-NR1 has only 3 paired composites after masking
exclude_from_ranking = {('US-NR1', 'GCOM-C')}

# Draw the table name and statistic above each table image
show_title = True

# Taylor diagrams (one per index) for these tables; they need Pearson_r and SD_ratio in the
# stats csv, which only the 03_final_compare batch scripts write.
# Colors match the product colors in the timeseries panels; markers identify sites.
taylor_tables = ['Final_MCD19masked']
product_colors = {
    'MCD19': '#BF2665',
    'MOD09': '#B288E9',
    'MOD13': '#0DA10A',
    'MCD43': '#B5C230',
    'GCOM-C': '#56B4E9',
}
site_markers = ['o', 's', '^', 'D', 'v', 'P']

# A second, zoomed Taylor diagram per index (*_zoom.png) that only shows points with
# r >= r_min and SD ratio <= sd_cap; points outside that range are listed below the plot.
taylor_zoom = {'r_min': 0, 'sd_cap': 4}


#########################################
# Define Global functions
#########################################

def read_stats_file(pattern, site):
    matches = sorted(glob.glob(os.path.join(stats_dir, pattern.format(site=site))))
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one file for {pattern.format(site=site)}, found {matches}')
    return pd.read_csv(matches[0])


def build_table(columns, stat):
    """Rows: (site, index) with all sites for NDVI then all sites for CCI. Columns: datasets."""
    index = pd.MultiIndex.from_tuples(
        [(site_label, element) for element in elements for _, site_label in sites],
        names=['Site', 'Index'],
    )
    table = pd.DataFrame(index=index, columns=[c[0] for c in columns], dtype=float)

    for col_label, pattern, key_col, key_val in columns:
        for site_name, site_label in sites:
            df = read_stats_file(pattern, site_name)
            df = df[df[key_col] == key_val].set_index('element')
            for element in elements:
                if element in df.index and stat in df.columns:
                    table.loc[(site_label, element), col_label] = df.loc[element, stat]
    if stat == 'N':
        table = table.replace(0, np.nan)  # no pairs (e.g. MOD13/MCD43 CCI) shown as '–' like the other tables
    return table


def best_columns(row, rule):
    """Column labels holding the best value in a row (ties all count)."""
    vals = row.dropna()
    if vals.empty or rule == 'none':
        return []
    if rule == 'max':
        target = vals.max()
    elif rule == 'min':
        target = vals.min()
    elif rule == 'closest_one':
        vals = (vals - 1).abs()
        target = vals.min()
    else:
        vals = vals.abs()
        target = vals.min()
    return list(vals.index[np.isclose(vals.values, target)])


def plot_table(table, stat, title, outfile):
    stat_label, fmt, rule = stats[stat]
    col_labels = ['Site', 'Index'] + [c for c in table.columns]

    cell_text = []
    bold_cells = set()
    for i, ((site_label, element), row) in enumerate(table.iterrows()):
        cell_text.append([site_label, element] + ['–' if pd.isna(v) else fmt.format(v) for v in row.values])
        # pick the best value as displayed, so equal printed values are bolded together
        row_rounded = row.map(lambda v: v if pd.isna(v) else float(fmt.format(v)))
        row_rounded = row_rounded.drop([c for s, c in exclude_from_ranking if s == site_label])
        for col in best_columns(row_rounded, rule):
            bold_cells.add((i + 1, 2 + list(table.columns).index(col)))

    # Column widths (inches) from the longest line of text in each column
    char_width = 0.11
    col_widths = []
    for c, label in enumerate(col_labels):
        texts = label.split('\n') + [row[c] for row in cell_text]
        col_widths.append(char_width * max(len(t) for t in texts) + 0.35)

    n_rows = len(cell_text) + 1
    row_height = 0.27
    fig_width = sum(col_widths)
    fig_height = row_height * (n_rows + 1)  # header row is double height
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    # bbox makes the table fill the axes; cell widths/heights below are relative
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc='center', bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    n_sites = len(sites)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_width(col_widths[c] / fig_width)
        cell.set_height(1 / (n_rows + 1))
        # booktabs-style horizontal rules: top, below header, between NDVI and CCI, bottom
        edges = ''
        if r == 0:
            edges = 'TB'
            cell.set_height(2 / (n_rows + 1))
            cell.set_text_props(fontweight='bold')
        elif r == n_sites + 1:
            edges = 'T'
        if r == n_rows - 1:
            edges += 'B'
        cell.visible_edges = edges if edges else 'open'
        cell.set_linewidth(0.8)
        if (r, c) in bold_cells:
            cell.set_text_props(fontweight='bold')

    if show_title:
        ax.set_title(f'{title}: {stat_label}', fontsize=13, pad=4)

    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_taylor(columns, title, outfile_base, r_min=None, sd_cap=None):
    """
    One Taylor diagram per index. Angle = Pearson r, radius = satellite SD / tower SD,
    so the distance from each point to the tower reference (r = 1, SD ratio = 1) is the
    CRMSE normalized by the tower SD (grey dashed contours).
    If r_min / sd_cap are given, the axes are limited to that range and points outside
    it are left out and listed in a note below the plot.
    """
    r_table = build_table(columns, 'Pearson_r')
    sd_table = build_table(columns, 'SD_ratio')

    for element in elements:
        r_el = r_table.xs(element, level='Index')
        sd_el = sd_table.xs(element, level='Index')
        valid = r_el.notna() & sd_el.notna()
        if not valid.values.any():
            print(f'Taylor diagram {element}: no Pearson_r / SD_ratio values, skipped')
            continue

        # points left out of a zoomed diagram
        shown = valid.copy()
        if r_min is not None:
            shown &= r_el >= r_min
        if sd_cap is not None:
            shown &= sd_el <= sd_cap
        omitted = [f'{col} at {site_label} (r = {r_el.loc[site_label, col]:.2f}, SD ratio = {sd_el.loc[site_label, col]:.2f})'
                   for _, site_label in sites for col in r_el.columns
                   if valid.loc[site_label, col] and not shown.loc[site_label, col]]
        valid = shown

        # use a half circle only if some correlation is negative
        half = (r_el[valid] < 0).values.any()
        theta_max = np.pi if half else np.pi / 2
        if sd_cap is not None:
            sd_max = sd_cap
        else:
            sd_max = max(1.5, np.ceil(1.1 * np.nanmax(sd_el[valid].values) / 0.25) * 0.25)

        fig = plt.figure(figsize=(10, 6) if half else (8, 6.5))
        ax = fig.add_subplot(projection='polar')
        ax.set_thetamin(0)
        ax.set_thetamax(np.degrees(theta_max))
        ax.set_rlim(0, sd_max)
        ax.set_rlabel_position(np.degrees(theta_max))
        ax.tick_params(labelsize=10)
        ax.grid(color='0.85', linewidth=0.6)

        r_ticks = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]
        if half:
            r_ticks = [-1, -0.9, -0.6, -0.4, -0.2] + r_ticks
        ax.set_thetagrids(np.degrees(np.arccos(r_ticks)), labels=[f'{v:g}' for v in r_ticks])

        # normalized CRMSE contours around the reference point
        tt, rr = np.meshgrid(np.linspace(0, theta_max, 300), np.linspace(0, sd_max, 300))
        crmse = np.sqrt(1 + rr**2 - 2 * rr * np.cos(tt))
        crmse_step = 0.25 if sd_max <= 2 else 0.5
        cs = ax.contour(tt, rr, crmse, levels=np.arange(crmse_step, 2 * sd_max, crmse_step),
                        colors='0.6', linewidths=0.7, linestyles='--')
        ax.clabel(cs, fmt='%.2g', fontsize=8)

        # tower reference: SD ratio = 1 arc and the reference point
        t = np.linspace(0, theta_max, 200)
        ax.plot(t, np.ones_like(t), color='black', linewidth=0.8)
        ax.plot(0, 1, marker='*', color='black', markersize=14, clip_on=False, zorder=5)

        for i, (_, site_label) in enumerate(sites):
            for col in r_el.columns:
                if not valid.loc[site_label, col]:
                    continue
                excluded = (site_label, col) in exclude_from_ranking
                color = product_colors.get(col, 'gray')
                ax.plot(np.arccos(r_el.loc[site_label, col]), sd_el.loc[site_label, col],
                        marker=site_markers[i], markersize=9, linestyle='none', zorder=4,
                        markerfacecolor='none' if excluded else color,
                        markeredgecolor=color if excluded else 'black',
                        markeredgewidth=1.5 if excluded else 0.6)

        # axis labels, offset in points from the plot edge (the polar axes box is larger than the plot)
        sd_label = 'SD ratio (satellite / tower)'
        ax.annotate(sd_label, (0, sd_max / 2), textcoords='offset points', xytext=(0, -30),
                    ha='center', va='top', fontsize=11, annotation_clip=False)
        if half:
            ax.annotate('Pearson r', (np.pi / 2, sd_max), textcoords='offset points', xytext=(0, 32),
                        ha='center', va='bottom', fontsize=11, annotation_clip=False)
        else:
            ax.annotate('Pearson r', (np.pi / 4, sd_max), textcoords='offset points', xytext=(45, 45),
                        ha='center', va='center', rotation=-45, fontsize=11, annotation_clip=False)
            ax.annotate(sd_label, (np.pi / 2, sd_max / 2), textcoords='offset points', xytext=(-38, 0),
                        ha='right', va='center', rotation=90, fontsize=11, annotation_clip=False)

        # legends: color = product, marker = site
        prod_handles = [Line2D([], [], marker='o', linestyle='none', markersize=9, markerfacecolor=product_colors.get(c, 'gray'),
                               markeredgecolor='black', markeredgewidth=0.6, label=c)
                        for c in r_el.columns if valid[c].any()]
        prod_handles.append(Line2D([], [], marker='*', linestyle='none', markersize=12, color='black', label='Tower (reference)'))
        site_handles = [Line2D([], [], marker=site_markers[i], linestyle='none', markersize=9, markerfacecolor='0.7',
                               markeredgecolor='black', markeredgewidth=0.6, label=site_label)
                        for i, (_, site_label) in enumerate(sites) if valid.loc[site_label].any()]
        if any(valid.loc[s, c] for s, c in exclude_from_ranking if s in valid.index and c in valid.columns):
            site_handles.append(Line2D([], [], marker='o', linestyle='none', markersize=9, markerfacecolor='none',
                                       markeredgecolor='black', markeredgewidth=1.5, label='Few pairs (footnoted)'))
        leg1 = ax.legend(handles=prod_handles, title='Product', loc='upper left', bbox_to_anchor=(1.05, 1.0),
                         fontsize=10, title_fontsize=10, frameon=False)
        ax.add_artist(leg1)
        leg2 = ax.legend(handles=site_handles, title='Site', loc='upper left', bbox_to_anchor=(1.05, 0.55),
                  fontsize=10, title_fontsize=10, frameon=False)

        if show_title:
            if half:
                ax.annotate(f'{title}: {element}', (np.pi / 2, sd_max), textcoords='offset points',
                            xytext=(0, 55), ha='center', va='bottom', fontsize=13, annotation_clip=False)
            else:
                ax.set_title(f'{title}: {element}', fontsize=13, pad=30)

        if omitted:
            limits = ' or '.join(([f'r < {r_min:g}'] if r_min is not None else []) +
                                 ([f'SD ratio > {sd_cap:g}'] if sd_cap is not None else []))
            ax.text(0, -0.13, f'Not shown ({limits}):\n' + '\n'.join(omitted), transform=ax.transAxes,
                    ha='left', va='top', fontsize=9, color='0.3')

        fig.savefig(f'{outfile_base}_{element}.png', dpi=300, bbox_inches='tight', bbox_extra_artists=[leg1, leg2, *ax.texts])
        plt.close(fig)


#########################################
# Begin main
#########################################
#%%


for table_name, table_info in tables.items():
    for stat in stats:
        table = build_table(table_info['columns'], stat)
        if table.isna().all().all():
            print(f'\n{table_name} - {stat}: not in the stats files, skipped')
            continue
        print(f'\n{table_name} - {stat}')
        print(table.round(3).to_string())

        out_base = os.path.join(table_dir, f'fit_stats_table_{table_name}_{stat}')
        table.rename(columns=lambda c: c.replace('\n', ' ')).to_csv(f'{out_base}.csv')
        plot_table(table, stat, table_info['title'], f'{out_base}.png')

    if table_name in taylor_tables:
        plot_taylor(table_info['columns'], table_info['title'],
                    os.path.join(table_dir, f'taylor_diagram_{table_name}'))
        plot_taylor(table_info['columns'], table_info['title'],
                    os.path.join(table_dir, f'taylor_diagram_{table_name}_zoom'), **taylor_zoom)




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
#
# Summary tables of the MCD19 (etc.) vs tower (PhotoSpec / FloX) fit statistics
# for each stage of the analysis:
#   1. QC         - MCD19 QC options (01_QC_compare/01*.py)
#   2. OutlierFilter - pre-fill vs outlier-filtered (02_gapfill_compare/01*.py)
#   3. GapFill    - outlier-filtered vs lm-fill vs basic-fill (02_gapfill_compare/01*.py)
#   4. Final      - MCD19 lm-fill vs MOD09 / MOD13 / MCD43 / GCOM-C (03_final_compare/01*.py)
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
}

# Each column: (column label, csv file pattern relative to stats_dir, key column, key value).
# '*' in the pattern matches the tower name (PhotoSpec or FloX).
QC_file = 'MCD19_v_*_fit_stats_QC_Compare_{site}.csv'

tables = {
    'QC': {
        'title': 'MCD19 QC options',
        'columns': [
            ('All AOD/\nAll Adj', QC_file, 'QC_descr', 'CloudFree_AllAOD_AllAdj'),
            ('Low AOD/\nAll Adj', QC_file, 'QC_descr', 'CloudFree_LowAOD_AllAdj'),
            ('All AOD/\nClear Adj', QC_file, 'QC_descr', 'CloudFree_AllAOD_ClearAdj'),
            ('Low AOD/\nClear Adj', QC_file, 'QC_descr', 'CloudFree_LowAOD_ClearAdj'),
        ],
    },
    'OutlierFilter': {
        'title': 'MCD19 outlier filter',
        'columns': [
            ('All composites', QC_file, 'QC_descr', 'CloudFree_LowAOD_ClearAdj'),
            ('Outlier-filtered', 'outlier_removal/MCD19_v_*_fit_stats_OutlierFilter_{site}.csv', 'QC_descr', 'CloudFree_LowAOD_ClearAdj'),
        ],
    },
    'GapFill': {
        'title': 'MCD19 gap-filling',
        'columns': [
            ('Without\nGap-fill', 'outlier_removal/MCD19_v_*_fit_stats_OutlierFilter_{site}.csv', 'QC_descr', 'CloudFree_LowAOD_ClearAdj'),
            ('Ecoregion-stratified\nGap-fill', 'lm-fill/MCD19_v_*_fit_stats_lm-fill_{site}.csv', 'QC_descr', 'CloudFree_LowAOD_ClearAdj'),
            ('Simple Climatology\nGap-fill', 'basic-fill/MCD19_v_*_fit_stats_basic-fill_{site}.csv', 'QC_descr', 'CloudFree_LowAOD_ClearAdj'),
        ],
    },
    'Final': {
        'title': 'Satellite product comparison',
        'columns': [
            ('MCD19', 'final/PhotoSpec_v_MCD19_etc_fit_stats_{site}.csv', 'QC', 'MCD19'),
            ('MOD09', 'final/PhotoSpec_v_MCD19_etc_fit_stats_{site}.csv', 'QC', 'MOD09'),
            ('MOD13', 'final/PhotoSpec_v_MCD19_etc_fit_stats_{site}.csv', 'QC', 'MOD13'),
            ('MCD43', 'final/PhotoSpec_v_MCD19_etc_fit_stats_{site}.csv', 'QC', 'MCD43'),
            ('GCOM-C', 'final/PhotoSpec_v_MCD19_etc_fit_stats_{site}.csv', 'QC', 'GCOM'),
        ],
    },
}

# Draw the table name and statistic above each table image
show_title = True


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
                if element in df.index:
                    table.loc[(site_label, element), col_label] = df.loc[element, stat]
    return table


def best_columns(row, rule):
    """Column labels holding the best value in a row (ties all count)."""
    vals = row.dropna()
    if vals.empty:
        return []
    if rule == 'max':
        target = vals.max()
    elif rule == 'min':
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


#########################################
# Begin main
#########################################
#%%


for table_name, table_info in tables.items():
    for stat in stats:
        table = build_table(table_info['columns'], stat)
        print(f'\n{table_name} - {stat}')
        print(table.round(3).to_string())

        out_base = os.path.join(table_dir, f'fit_stats_table_{table_name}_{stat}')
        table.rename(columns=lambda c: c.replace('\n', ' ')).to_csv(f'{out_base}.csv')
        plot_table(table, stat, table_info['title'], f'{out_base}.png')




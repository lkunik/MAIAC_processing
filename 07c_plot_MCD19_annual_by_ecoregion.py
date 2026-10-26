#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Multi-panel (3 x 2) version of 07c_plot_MCD19_annual_by_ecoregion.py:
# plots annual CCI and NDVI (with linear trends) for six ecoregions in a single figure.
#

#########################################
# Load packages
#########################################
#%%
# File system packages
import os  # operating system library
from pathlib import Path
import sys

# runtime packages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# numerical/data management packages
import numpy as np
from scipy import stats
import pymannkendall as mk

# plotting packages
import pickle
from matplotlib import pyplot as plt  # primary plotting module
from matplotlib.ticker import MultipleLocator

#%%
#########################################
# Define Global Filepaths
#########################################

mask_LC_high = True
mask_LC_low = False

# directories
QC_option = 'CloudFree_LowAOD_ClearAdj'

dat_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/summary_data'
plot_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/annual/QCfilt_{QC_option}/'
Path(plot_basedir).mkdir(parents=True, exist_ok=True)


#########################################
# Define Global Variables and constants
#########################################
#%%

# panels in row-major order: (ecoregion file key, panel title)
panels = [
    ('Boreal',                 'Boreal Forests (ENF)'),
    ('Western',                'Western Cordillera (ENF)'),
    ('GREAT_PLAINS',           'Great Plains (Grassland)'),
    ('Eastern',                'Eastern Forests (Mixed/DBF)'),
    ('NORTH_AMERICAN_DESERTS', 'Deserts (Shrubland)'),
    ('Southwest_US_Mexico',    'Temperate Sierras (ENF)'),
]
panel_letters = 'abcdefghijklmnopqrstuvwxyz'

# 'sen': Sen's slope + Mann-Kendall p-value (Hamed & Rao 1998 autocorrelation correction)
# 'ols': ordinary least squares slope + p-value
trend_method = 'sen'

trend_years = np.arange(1999, 2027)
tick_years = [2000, 2005, 2010, 2015, 2020, 2025]

CCI_tick = 0.1
NDVI_tick = 0.2

CCI_color = "#A1993D"
CCI_text_color = "#7A7428"  # darker shade of CCI_color so trend text is legible over the shading
NDVI_color = "#006B30"

trend_fontsize = 16

plt.rcParams.update({
'axes.labelsize': 16,
'axes.titlesize': 19,
'xtick.labelsize': 15,
'ytick.labelsize': 15,
'legend.fontsize': 12,
})


#########################################
# Define Global Functions
#########################################

def load_ecoregion_dict(ecoregion):
    if mask_LC_high:
        input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_annual_dict_LCmask_only.pkl')
    elif mask_LC_low:
        input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_annual_dict_LCmask_low.pkl')
    else:
        input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_annual_dict.pkl')

    with open(input_file, 'rb') as f:
        ecoregion_dat_dict = pickle.load(f)

    print(f'Loaded ecoregion_annual_dict from {input_file}')
    return ecoregion_dat_dict


def get_range(ecoregion_dat_dict, var):
    # min/max of mean ± std for a variable
    vmin = np.nanmin(ecoregion_dat_dict[f'{var}_mean'] - ecoregion_dat_dict[f'{var}_std'])
    vmax = np.nanmax(ecoregion_dat_dict[f'{var}_mean'] + ecoregion_dat_dict[f'{var}_std'])
    return vmin, vmax


def fit_trend(years, vals):
    mask = ~np.isnan(vals)
    if trend_method == 'sen':
        slope, intercept, _, _ = stats.theilslopes(vals[mask], years[mask])
        p_value = mk.hamed_rao_modification_test(vals[mask]).p
    else:
        slope, intercept, _, p_value, _ = stats.linregress(years[mask], vals[mask])
    return slope, intercept, p_value


def format_p(p_value):
    if p_value < 0.001:
        return 'p < 0.001'
    return f'p = {p_value:.3f}'


def format_slope(slope):
    # signed to 4 decimals, but show an exact-looking zero without a sign
    if round(slope, 4) == 0:
        return '0.0000'
    return f'{slope:+.4f}'


def plot_annual_panel(ax, ecoregion_dat_dict, title, max_CCI_range, max_NDVI_range):
    years = ecoregion_dat_dict['years']

    slope_CCI, intercept_CCI, p_value_CCI = fit_trend(years, ecoregion_dat_dict['CCI_mean'])
    trend_CCI = slope_CCI * trend_years + intercept_CCI
    CCI_min, CCI_max = get_range(ecoregion_dat_dict, 'CCI')

    slope_NDVI, intercept_NDVI, p_value_NDVI = fit_trend(years, ecoregion_dat_dict['NDVI_mean'])
    trend_NDVI = slope_NDVI * trend_years + intercept_NDVI
    NDVI_min, NDVI_max = get_range(ecoregion_dat_dict, 'NDVI')

    print(f'{title} (trend_method = {trend_method})')
    print(f'  CCI slope: {slope_CCI:.4f}, intercept: {intercept_CCI:.4f}, p-value: {p_value_CCI:.4g}')
    print(f'  NDVI slope: {slope_NDVI:.4f}, intercept: {intercept_NDVI:.4f}, p-value: {p_value_NDVI:.4g}')

    # Set CCI so it lives in the lower vertical half of the plot
    cci_leftover_range = max_CCI_range - (CCI_max - CCI_min)
    CCI_ymin = CCI_min - cci_leftover_range
    CCI_ymax = CCI_max + max_CCI_range

    # Set NDVI so it lives in the upper vertical half of the plot
    ndvi_leftover_range = max_NDVI_range - (NDVI_max - NDVI_min)
    NDVI_ymin = NDVI_min - max_NDVI_range
    NDVI_ymax = NDVI_max + ndvi_leftover_range

    ax.grid(color = "gray", linestyle = "--", linewidth = 0.6, alpha = 0.3) # add grid lines

    ax.plot(years, ecoregion_dat_dict['CCI_mean'], color=CCI_color, lw = 2, alpha = 0.7)
    # Plot shaded region for mean ± representative std
    ax.fill_between(
        years,
        ecoregion_dat_dict['CCI_mean'] - ecoregion_dat_dict['CCI_std'],
        ecoregion_dat_dict['CCI_mean'] + ecoregion_dat_dict['CCI_std'],
        color=CCI_color, alpha=0.2, label='±1 std'
    )

    ax2 = ax.twinx()

    ax2.plot(years, ecoregion_dat_dict['NDVI_mean'], color=NDVI_color, lw = 2, alpha = 0.7)
    # Plot shaded region for mean ± representative std
    ax2.fill_between(
        years,
        ecoregion_dat_dict['NDVI_mean'] - ecoregion_dat_dict['NDVI_std'],
        ecoregion_dat_dict['NDVI_mean'] + ecoregion_dat_dict['NDVI_std'],
        color=NDVI_color, alpha=0.2, label='±1 std'
    )

    # Add trend lines for CCI and NDVI
    ax.plot(trend_years, trend_CCI, color=CCI_color, lw=2.5, alpha=1, linestyle='--')
    ax2.plot(trend_years, trend_NDVI, color=NDVI_color, lw=2.5, alpha=1, linestyle='--')

    
    # Trend annotations: NDVI at top of plot area, CCI at bottom (on ax2 so they draw above both datasets)
    ax2.text(0.5, 0.95, f'NDVI trend: {format_slope(slope_NDVI)}/yr ({format_p(p_value_NDVI)})', transform=ax2.transAxes,
            ha='center', va='top', color=NDVI_color, fontsize=trend_fontsize, zorder=5)
    ax2.text(0.5, 0.04, f'CCI trend: {format_slope(slope_CCI)}/yr ({format_p(p_value_CCI)})', transform=ax2.transAxes,
            ha='center', va='bottom', color=CCI_text_color, fontsize=trend_fontsize, zorder=5)

    # Define the axis, title labels
    ax.set_xticks(ticks=tick_years)
    ax.set_xticklabels([str(year) for year in tick_years])
    ax.set_xlim(years[0]-0.5, years[-1]+0.5)

    ax.yaxis.set_major_locator(MultipleLocator(CCI_tick))
    ax2.yaxis.set_major_locator(MultipleLocator(NDVI_tick))

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))

    ax.yaxis.label.set_color(CCI_color)
    ax.tick_params(axis='y', colors=CCI_color)
    ax.spines['left'].set_color(CCI_color)

    ax2.yaxis.label.set_color(NDVI_color)
    ax2.tick_params(axis='y', colors=NDVI_color)
    ax2.spines['right'].set_color(NDVI_color)
    ax2.spines['left'].set_color(CCI_color)

    ax.set_ylabel('CCI')
    ax2.set_ylabel('NDVI')
    ax.set_title(title)

    ax.set_ylim(CCI_ymin, CCI_ymax)
    ax2.set_ylim(NDVI_ymin, NDVI_ymax)

    return ax2


#########################################
# Begin Main
#########################################

#%%
### Load all ecoregions and establish the max range of CCI and NDVI across them

ecoregion_dicts = {ecoregion: load_ecoregion_dict(ecoregion) for ecoregion, _ in panels}

CCI_ranges = [get_range(d, 'CCI') for d in ecoregion_dicts.values()]
NDVI_ranges = [get_range(d, 'NDVI') for d in ecoregion_dicts.values()]
max_CCI_range = max(vmax - vmin for vmin, vmax in CCI_ranges)
max_NDVI_range = max(vmax - vmin for vmin, vmax in NDVI_ranges)
print(f'Max CCI range = {max_CCI_range:.2f}')
print(f'Max NDVI range = {max_NDVI_range:.2f}')

#%%
fig, axes = plt.subplots(3, 2, figsize=(14, 9.5))

for i, (ax, (ecoregion, title)) in enumerate(zip(axes.flat, panels)):
    plot_annual_panel(ax, ecoregion_dicts[ecoregion], f'{panel_letters[i]}) {title}', max_CCI_range, max_NDVI_range)

fig.tight_layout(h_pad=1.5, w_pad=2.0)

if mask_LC_high:
    plot_filename = 'ecoregions_annual_multipanel_LCmask.png'
elif mask_LC_low:
    plot_filename = 'ecoregions_annual_multipanel_LCmask_low.png'
else:
    plot_filename = 'ecoregions_annual_multipanel.png'

plt.savefig(os.path.join(plot_basedir, plot_filename), dpi=300, bbox_inches='tight')
print(f'Saved {os.path.join(plot_basedir, plot_filename)}')
# plt.close()
plt.show()

# %%

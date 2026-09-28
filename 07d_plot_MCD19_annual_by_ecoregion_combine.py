#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#

#########################################
# Load packages
#########################################
#%%
import os
import pickle
from pathlib import Path

import numpy as np
from scipy import stats

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

#%%
#########################################
# Define Global Filepaths
#########################################

dat_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/summary_data'

QC_option = 'CloudFree_LowAOD_ClearAdj'
plot_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/annual/QCfilt_{QC_option}/'
Path(plot_basedir).mkdir(parents=True, exist_ok=True)

# ordered mapping: pkl filename stem -> (color, legend label)
ecoregions = {
    'Boreal':                 {'color': '#305A94', 'label': 'Boreal Forests (ENF)'},
    'Western':                {'color': '#005214', 'label': 'Western Cordillera (ENF)'},
    'GREAT_PLAINS':           {'color': '#CCB000', 'label': 'Great Plains (Grassland)'},
    'Eastern':                {'color': '#7A7524', 'label': 'Eastern Forests (Mixed/DBF)'},
    'NORTH_AMERICAN_DESERTS': {'color': '#915E00', 'label': 'Deserts (Shrubland)'},
    'Southwest_US_Mexico':    {'color': '#4E7000', 'label': 'Temperate Sierras (ENF)'},
}

#########################################
# Load data for all ecoregions
#########################################
#%%

trend_years = np.arange(1999, 2027)

for name, info in ecoregions.items():
    input_file = os.path.join(dat_dir, f'{name}_annual_dict_LCmask_only.pkl')
    with open(input_file, 'rb') as f:
        dat_dict = pickle.load(f)
    print(f'Loaded {name} annual_dict from {input_file}')

    info['dat_dict'] = dat_dict

    mask_CCI = ~np.isnan(dat_dict['CCI_mean'])
    slope_CCI, intercept_CCI, _, _, _ = stats.linregress(
        dat_dict['years'][mask_CCI], dat_dict['CCI_mean'][mask_CCI]
    )
    info['trend_CCI'] = slope_CCI * trend_years + intercept_CCI

    mask_NDVI = ~np.isnan(dat_dict['NDVI_mean'])
    slope_NDVI, intercept_NDVI, _, _, _ = stats.linregress(
        dat_dict['years'][mask_NDVI], dat_dict['NDVI_mean'][mask_NDVI]
    )
    info['trend_NDVI'] = slope_NDVI * trend_years + intercept_NDVI

#%%
#########################################
# Plot aesthetics
#########################################

plt.rcParams.update({
    'axes.labelsize': 17,
    'axes.titlesize': 17,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 13,
})

tick_years = [2000, 2005, 2010, 2015, 2020, 2025]

# use the full x-range across all ecoregions in case any differ in coverage
all_years = np.concatenate([info['dat_dict']['years'] for info in ecoregions.values()])
xlim = (all_years.min() - 0.5, all_years.max() + 0.5)

#########################################
# CCI plot: all ecoregions, single axis
#########################################
#%%

fig, ax = plt.subplots(figsize=(6, 3))
ax.grid(color="gray", linestyle="--", linewidth=0.6, alpha=0.3)

for name, info in ecoregions.items():
    dat_dict = info['dat_dict']
    color = info['color']

    ax.plot(dat_dict['years'], dat_dict['CCI_mean'], color=color, lw=2, alpha=0.7, label=info['label'])
    ax.fill_between(
        dat_dict['years'],
        dat_dict['CCI_mean'] - dat_dict['CCI_std'],
        dat_dict['CCI_mean'] + dat_dict['CCI_std'],
        color=color, alpha=0.3
    )
    ax.plot(trend_years, info['trend_CCI'], color=color, lw=2.5, alpha=1, linestyle='--')

ax.set_xticks(ticks=tick_years)
ax.set_xticklabels([str(year) for year in tick_years])
ax.set_xlim(*xlim)

ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
ax.set_ylabel('CCI')

ax.legend(loc='best')

plt.show()

#########################################
# NDVI plot: all ecoregions, single axis
#########################################
#%%

fig, ax = plt.subplots(figsize=(8, 3))
ax.grid(color="gray", linestyle="--", linewidth=0.6, alpha=0.3)

for name, info in ecoregions.items():
    dat_dict = info['dat_dict']
    color = info['color']

    ax.plot(dat_dict['years'], dat_dict['NDVI_mean'], color=color, lw=2, alpha=0.7, label=info['label'])
    ax.fill_between(
        dat_dict['years'],
        dat_dict['NDVI_mean'] - dat_dict['NDVI_std'],
        dat_dict['NDVI_mean'] + dat_dict['NDVI_std'],
        color=color, alpha=0.3
    )
    ax.plot(trend_years, info['trend_NDVI'], color=color, lw=2.5, alpha=1, linestyle='--')

ax.set_xticks(ticks=tick_years)
ax.set_xticklabels([str(year) for year in tick_years])
ax.set_xlim(*xlim)

ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))
ax.set_ylabel('NDVI')

ax.legend(loc='best')

plt.show()

# %%
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
# File system packages
import os  # operating system library
import glob
from pathlib import Path
import copy
import sys
import gc


# runtime packages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Custom packages
utils_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/pyutil'
if utils_dir not in sys.path:
   sys.path.append(utils_dir)
import data_utils
import plot_utils


# numerical/data management packages
import math
import numpy as np
import xarray as xr  # for multi dimensional data
import rioxarray as rxr
import pandas as pd
from rasterio.enums import Resampling
from scipy import stats

# shapefile/geospatial packages
import geopandas as gpd
from shapely.geometry import mapping

# time and date packages
import time
from datetime import datetime as dt  # date time library
from datetime import timedelta

# plotting packages
import pickle
from matplotlib import pyplot as plt  # primary plotting module

#%%
#########################################
# Define Global Filepaths
#########################################


# directories
QC_option = 'CloudFree'

dat_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/summary_data'
plot_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/annual_ts/QCfilt_{QC_option}/'
Path(plot_basedir).mkdir(parents=True, exist_ok=True)

# TerraClimate_pickle_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/TerraClimate/data/summary_data'
# FluxCom_pickle_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/FLUXCOM/FLUXCOM-X/data/summary_data'

########################################
# Define Global Variables and constants
#########################################
#%%

ecoregions_list = ['Temperate', 'Boreal']

#########################################
# Define Global Functions
#########################################



#########################################
# Begin Main
#########################################



# print(ecoregion_l2_gdf['NA_L2NAME'].unique())
analysis_years = np.arange(2000, 2026)
tick_years = [2000, 2005, 2010, 2015, 2020, 2025]



#%%
# mark start time to keep track of elapsed
start_time = time.time()

ENF_regrids = {}

analysis_years = np.arange(2000, 2026)

ecoregions = ['Temperate', 'Boreal']

TC_vars = ['CWD_anomaly', 'PDSI', 'SoilM_anomaly', 'VPD_anomaly', 'PET_anomaly']
TC_labels = {
    'CWD_anomaly': 'CWD anomaly (mm)',
    'PDSI': 'PDSI',
    'SoilM_anomaly': 'SM anomaly (mm)',
    'VPD_anomaly': 'VPD anomaly (kPa)',
    'PET_anomaly': 'PET anomaly (mm)'
}
mask_ENF = True

avg_period = 'JJA'


#######################################################################
### Pass 1 - get ylims
#######################################################################

ax_ylim = [1000, -1000]

for ecoregion in ecoregions:
    print(f'Processing ecoregion: {ecoregion}...')

    if mask_ENF:
        input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_annual_std_anom_dict_ENF_only.pkl')
    else:
        input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_annual_std_anom_dict_nonENF.pkl')

    with open(input_file, 'rb') as f:
        ecoregion_dat_dict = pickle.load(f)


    def plot_trend(ax, years, values, color, trend_years=None, lw=1.8, alpha=0.7):
        mask = ~np.isnan(values)
        if mask.sum() < 2:
            return
        slope, intercept, _, _, _ = stats.linregress(years[mask], values[mask])
        if trend_years is None:
            trend_years = years
        trend = slope * trend_years + intercept
        ax.plot(trend_years, trend, color=color, lw=lw, alpha=alpha, linestyle='--')


    suffix = 'ENF' if mask_ENF else 'nonENF'

    # -------------------------
    # Plot 1: CCI, NDVI, FLUXCOM GPP
    # -------------------------
    CCI_mean_all = np.array(ecoregion_dat_dict['CCI_mean'])
    CCI_mean = CCI_mean_all[np.isfinite(CCI_mean_all)]
    NDVI_mean_all = np.array(ecoregion_dat_dict['NDVI_mean'])
    NDVI_mean = NDVI_mean_all[np.isfinite(NDVI_mean_all)]
    CCI_std = np.array(ecoregion_dat_dict['CCI_std'])
    CCI_std = CCI_std[np.isfinite(CCI_mean_all)]
    NDVI_std = np.array(ecoregion_dat_dict['NDVI_std'])
    NDVI_std = NDVI_std[np.isfinite(NDVI_mean_all)]
    CCI_ymin = np.nanmin(CCI_mean - (np.array(CCI_std)/2))
    CCI_ymax = np.nanmax(CCI_mean + (np.array(CCI_std)/2))
    NDVI_ymin = np.nanmin(NDVI_mean - (np.array(NDVI_std)/2))
    NDVI_ymax = np.nanmax(NDVI_mean + (np.array(NDVI_std)/2))

    ax_ylim[0] = min(ax_ylim[0], CCI_ymin, NDVI_ymin)
    ax_ylim[1] = max(ax_ylim[1], CCI_ymax, NDVI_ymax)

# %%





#######################################################################
### Pass 2 - plot data
#######################################################################


ecoregion = 'Boreal'
for ecoregion in ecoregions:
    print(f'Processing ecoregion: {ecoregion}...')

    if mask_ENF:
        input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_annual_std_anom_dict_ENF_only.pkl')
    else:
        input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_annual_std_anom_dict_nonENF.pkl')

    with open(input_file, 'rb') as f:
        ecoregion_dat_dict = pickle.load(f)


    plt.rcParams.update({
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 13,
    })

    cci_color   = '#b2df8a'
    ndvi_color  = '#33a02c'


    from scipy import stats

    def plot_trend(ax, years, values, color, trend_years=None, lw=1.8, alpha=0.7):
        mask = ~np.isnan(values)
        if mask.sum() < 2:
            return
        slope, intercept, _, _, _ = stats.linregress(years[mask], values[mask])
        if trend_years is None:
            trend_years = years
        trend = slope * trend_years + intercept
        ax.plot(trend_years, trend, color=color, lw=lw, alpha=alpha, linestyle='--')


    suffix = 'ENF' if mask_ENF else 'nonENF'

    # -------------------------
    # Plot 1: CCI, NDVI,
    # -------------------------
    fig1, ax = plt.subplots(figsize=(6, 2.6))
    fig1.subplots_adjust(left=0.2, right=0.85)

    ax.grid(color='gray', linestyle='--', linewidth=1, alpha=0.4)

    ax.spines['left'].set_position(('outward', 0))
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    ax.plot(ecoregion_dat_dict['years'], ecoregion_dat_dict['NDVI_mean'], color=ndvi_color, lw=2)
    ax.plot(ecoregion_dat_dict['years'], ecoregion_dat_dict['CCI_mean'], color=cci_color, lw=2)
    ax.set_ylabel('z', color=ndvi_color)
    ax.set_ylim(ax_ylim)

    plot_trend(ax,  ecoregion_dat_dict['years'], np.array(ecoregion_dat_dict['CCI_mean']),  cci_color)
    plot_trend(ax,  ecoregion_dat_dict['years'], np.array(ecoregion_dat_dict['NDVI_mean']), ndvi_color)

    ax.set_xticks(tick_years)
    ax.set_xticklabels([str(y) for y in tick_years], rotation=45)
    ax.set_xlim(ecoregion_dat_dict['years'][0] - 0.5, ecoregion_dat_dict['years'][-1] + 0.5)

    plot_filename1 = f'{ecoregion.replace(" ", "_")}_CCI_NDVI_annual_std_anom_{suffix}.png'
    # fig1.savefig(os.path.join(plot_basedir, plot_filename1), dpi=300, bbox_inches='tight')
    # plt.close(fig1)
    # print(f'Plot 1 saved to {os.path.join(plot_basedir, plot_filename1)}')
    plt.show()
# %%

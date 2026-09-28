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
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
#%%
#########################################
# Define Global Filepaths
#########################################

ecoregion = "Boreal"
# ecoregion = 'Western'
# ecoregion = "Eastern"
# ecoregion = "Southwest_US_Mexico"
# ecoregion = "GREAT_PLAINS"
# ecoregion = "NORTH_AMERICAN_DESERTS"

mask_LC_high = True
mask_LC_low = False


# directories
QC_option = 'CloudFree_LowAOD_ClearAdj'


plot_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/annual/QCfilt_{QC_option}/'
Path(plot_basedir).mkdir(parents=True, exist_ok=True)



########################################
# Define Global Variables and constants
#########################################
#%%


#########################################
# Define Global Functions
#########################################

def to_year_doy(ds):
   year = ds.time.dt.year
   doy = ds.time.dt.dayofyear

   # assign new coords
   ds = ds.assign_coords(year=("time", year.data), doy=("time", doy.data))

   # reshape the array to (..., "doy", "year")
   return ds.set_index(time=("year", "doy")).unstack("time") 

# vectorize the sqrt function for applying to std deviation on annual means
v_sqrt = np.vectorize(math.sqrt)



#########################################
# Begin Main
#########################################

#%%
# mark start time to keep track of elapsed
start_time = time.time()

### Establish the max range of CCI and NDVI across all the ecoregions


# import pickle
dat_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/summary_data'

CCI_ranges = []
NDVI_ranges = []
for ecoregion in ['Boreal', 'Western', 'GREAT_PLAINS', 'Eastern', 'NORTH_AMERICAN_DESERTS', 'Southwest_US_Mexico']:

    print(f'checking range for {ecoregion}')
    if mask_LC_high:
        input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_annual_dict_LCmask_only.pkl')
    elif mask_LC_low:
        input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_annual_dict_LCmask_low.pkl')
    else:
        input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_annual_dict.pkl')

    with open(input_file, 'rb') as f:
        ecoregion_dat_dict = pickle.load(f)
    
    CCI_min = np.nanmin(ecoregion_dat_dict['CCI_mean'] - ecoregion_dat_dict['CCI_std'])
    CCI_max = np.nanmax(ecoregion_dat_dict['CCI_mean'] + ecoregion_dat_dict['CCI_std'])
    CCI_ranges.append(CCI_max - CCI_min)

    NDVI_min = np.nanmin(ecoregion_dat_dict['NDVI_mean'] - ecoregion_dat_dict['NDVI_std'])
    NDVI_max = np.nanmax(ecoregion_dat_dict['NDVI_mean'] + ecoregion_dat_dict['NDVI_std'])
    NDVI_ranges.append(NDVI_max - NDVI_min)

max_CCI_range = np.max(CCI_ranges)
max_NDVI_range = np.max(NDVI_ranges)
print(f'Max CCI range = {max_CCI_range:.2f}')
print(f'Max NDVI range = {max_NDVI_range:.2f}')


for ecoregion in ['Boreal', 'Western', 'GREAT_PLAINS', 'Eastern', 'NORTH_AMERICAN_DESERTS', 'Southwest_US_Mexico']:

    ### Now import this ecoregion's data specifically
    if mask_LC_high:
        input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_annual_dict_LCmask_only.pkl')
    elif mask_LC_low:
        input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_annual_dict_LCmask_low.pkl')
    else:
        input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_annual_dict.pkl')

    with open(input_file, 'rb') as f:
        ecoregion_dat_dict = pickle.load(f)

    print(f'Loaded ecoregion_annual_dict from {input_file}')


    trend_years = np.arange(1999, 2027)
    mask_CCI = ~np.isnan(ecoregion_dat_dict['CCI_mean'])
    slope_CCI, intercept_CCI, _, _, _ = stats.linregress(ecoregion_dat_dict['years'][mask_CCI], ecoregion_dat_dict['CCI_mean'][mask_CCI])
    trend_CCI = slope_CCI * trend_years + intercept_CCI
    CCI_min = np.nanmin(ecoregion_dat_dict['CCI_mean'] - ecoregion_dat_dict['CCI_std'])
    CCI_max = np.nanmax(ecoregion_dat_dict['CCI_mean'] + ecoregion_dat_dict['CCI_std'])



    mask_NDVI = ~np.isnan(ecoregion_dat_dict['NDVI_mean'])
    slope_NDVI, intercept_NDVI, _, _, _ = stats.linregress(ecoregion_dat_dict['years'][mask_NDVI], ecoregion_dat_dict['NDVI_mean'][mask_NDVI])
    trend_NDVI = slope_NDVI * trend_years + intercept_NDVI

    NDVI_min = np.nanmin(ecoregion_dat_dict['NDVI_mean'] - ecoregion_dat_dict['NDVI_std'])
    NDVI_max = np.nanmax(ecoregion_dat_dict['NDVI_mean'] + ecoregion_dat_dict['NDVI_std'])


    CCI_color = "#A1993D"
    NDVI_color = "#006B30"

    plt.rcParams.update({
        'axes.labelsize': 17,
        'axes.titlesize': 17,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 13,
    })

    print(f'Ecoregion: {ecoregion}')
    print(f'CCI slope: {slope_CCI:.4f}, intercept: {intercept_CCI:.4f}')
    print(f'NDVI slope: {slope_NDVI:.4f}, intercept: {intercept_NDVI:.4f}')

    # Set CCI so it lives in the lower vertical half of the plot
    cci_range = CCI_max - CCI_min
    cci_leftover_range = max_CCI_range - cci_range
    CCI_ymin = CCI_min - cci_leftover_range
    CCI_ymax = CCI_max + max_CCI_range

    # Set NDVI so it lives in the upper vertical half of the plot
    ndvi_range = NDVI_max - NDVI_min
    ndvi_leftover_range = max_NDVI_range - ndvi_range
    NDVI_ymin = NDVI_min - max_NDVI_range
    NDVI_ymax = NDVI_max + ndvi_leftover_range

    tick_years = [2000, 2005, 2010, 2015, 2020, 2025]


    fig, ax = plt.subplots(figsize=(6, 3)) # establish the figure, axes
    ax.grid(color = "gray", linestyle = "--", linewidth = 0.6, alpha = 0.3) # add grid lines
    # in faint lines, plot the individual years of the data

    ax.plot(ecoregion_dat_dict['years'], ecoregion_dat_dict['CCI_mean'], color=CCI_color, lw = 2, alpha = 0.7)
    # Plot shaded region for mean ± representative std
    ax.fill_between(
        ecoregion_dat_dict['years'],
        ecoregion_dat_dict['CCI_mean'] - ecoregion_dat_dict['CCI_std'],
        ecoregion_dat_dict['CCI_mean'] + ecoregion_dat_dict['CCI_std'],
        color=CCI_color, alpha=0.3, label='±1 std'
    )

    ax2 = ax.twinx()

    ax2.plot(ecoregion_dat_dict['years'], ecoregion_dat_dict['NDVI_mean'], color=NDVI_color, lw = 2, alpha = 0.7)
    # Plot shaded region for mean ± representative std
    ax2.fill_between(
        ecoregion_dat_dict['years'],
        ecoregion_dat_dict['NDVI_mean'] - ecoregion_dat_dict['NDVI_std'],
        ecoregion_dat_dict['NDVI_mean'] + ecoregion_dat_dict['NDVI_std'],
        color=NDVI_color, alpha=0.3, label='±1 std'
    )

    # Add trend lines for CCI and NDVI
    ax.plot(trend_years, trend_CCI, color=CCI_color, lw=2.5, alpha=1, linestyle='--')
    ax2.plot(trend_years, trend_NDVI, color=NDVI_color, lw=2.5, alpha=1, linestyle='--')


    ax2.set_ylabel('NDVI')
    ax2.yaxis.label.set_color(NDVI_color)
    ax2.tick_params(axis='y', colors=NDVI_color)
    ax2.spines['right'].set_color(NDVI_color)



    # Define the axis, title labels
    ax.set_xticks(ticks=tick_years)
    ax.set_xticklabels([str(year) for year in tick_years])
    ax.set_xlim(ecoregion_dat_dict['years'][0]-0.5, ecoregion_dat_dict['years'][-1]+0.5)

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax2.yaxis.set_major_locator(MultipleLocator(0.2))

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))

    ax.yaxis.label.set_color(CCI_color)
    ax.tick_params(axis='y', colors=CCI_color)
    ax.spines['left'].set_color(CCI_color)

    ax2.yaxis.label.set_color(NDVI_color)
    ax2.tick_params(axis='y', colors=NDVI_color)
    ax2.spines['right'].set_color(NDVI_color)

    ax.set_ylabel('CCI')
    ax2.set_ylabel('NDVI')

    ax.set_ylim(CCI_ymin, CCI_ymax)
    ax2.set_ylim(NDVI_ymin, NDVI_ymax)

    if mask_LC_high:
        plot_filename = f'{ecoregion.replace(" ", "_")}_annual_LCmask.png'
    elif mask_LC_low:
        plot_filename = f'{ecoregion.replace(" ", "_")}_annual_LCmask_low.png'
    else:
        plot_filename = f'{ecoregion.replace(" ", "_")}_annual.png'

    # plt.savefig(os.path.join(plot_basedir, plot_filename), dpi=300, bbox_inches='tight')
    # plt.close()
    plt.show()

    print(f'CCI ylim range = {(CCI_ymax - CCI_ymin):.2f}')
    print(f'NDVI ylim range = {(NDVI_ymax - NDVI_ymin):.2f}')


    # %%

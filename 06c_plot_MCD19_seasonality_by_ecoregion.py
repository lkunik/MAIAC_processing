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

# ecoregion = 'Western'
# ecoregion = "Boreal"
ecoregion = "Eastern"
# ecoregion = "Southwest_US_Mexico"
# ecoregion = "GREAT_PLAINS"
# ecoregion = "NORTH_AMERICAN_DESERTS"

mask_LC_high = True
mask_LC_low = False


# directories
QC_option = 'CloudFree_LowAOD_ClearAdj'


plot_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/seasonality/QCfilt_{QC_option}/'
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

DOY_composite_starts = np.arange(1, 366, 16)
composite_dates_2020 = [dt(2020, 1, 1) + timedelta(days=int(DOY) - 1) for DOY in DOY_composite_starts]

composite_periods = np.arange(0,23)


#%%

# import pickle
dat_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/summary_data'

if mask_LC_high:
    input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_seasonality_dict_LCmask_only.pkl')
elif mask_LC_low:
    input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_seasonality_dict_LCmask_low.pkl')
else:
    input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_seasonality_dict.pkl')

with open(input_file, 'rb') as f:
    ecoregion_dat_dict = pickle.load(f)

print(f'Loaded ecoregion_seasonality_dict from {input_file}')

#%%

#plotting vars
months_begin = np.array([1, 32, 61, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365])
months_doy = (months_begin[1:] + months_begin[:-1]) / 2
months_labs = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

CCI_color = "#A1993D"
NDVI_color = "#006B30"

plt.rcParams.update({
'axes.labelsize': 20,
'axes.titlesize': 20,
'xtick.labelsize': 17,
'ytick.labelsize': 16,
'legend.fontsize': 13,
})



fig, ax = plt.subplots(figsize=(5.5, 3.5))  # establish the figure, axes
ax.grid(color = "gray", linestyle = "--", linewidth = 0.4, alpha = 0.4) # add grid lines

# if ecoregion in ['TUNDRA', 'TAIGA', 'HUDSON PLAIN', 'BOREAL CORDILLERA','NORTHWESTERN FORESTED MOUNTAINS', 'MARINE WEST COAST FOREST', "NORTHERN FORESTS"]:
# Find first and last non-nan indices
non_nan_indices = np.where(~np.isnan(ecoregion_dat_dict['CCI_mean']))[0]
if len(non_nan_indices) > 0:
    first_non_nan_idx = non_nan_indices[0]
    last_non_nan_idx = non_nan_indices[-1]
    
    # Plot gray shaded regions
    ax.axvspan(-10, DOY_composite_starts[first_non_nan_idx], color='gray', alpha=0.2, hatch='///')
    ax.axvspan(DOY_composite_starts[last_non_nan_idx], 380, color='gray', alpha=0.2, hatch='///')

ax2 = ax.twinx()
# ax2.set_zorder(ax.get_zorder() - 1)
# ax.patch.set_visible(False)
ax2.plot(DOY_composite_starts, ecoregion_dat_dict['NDVI_mean'], color=NDVI_color, lw=2)
ax2.plot(DOY_composite_starts, np.array(ecoregion_dat_dict['NDVI_mean']) - np.array(ecoregion_dat_dict['NDVI_std']), color=NDVI_color, linestyle='--', lw=1.5)
ax2.plot(DOY_composite_starts, np.array(ecoregion_dat_dict['NDVI_mean']) + np.array(ecoregion_dat_dict['NDVI_std']), color=NDVI_color, linestyle='--', lw=1.5)
ax2.set_ylabel('NDVI')
ax2.yaxis.label.set_color('black')
ax2.tick_params(axis='y', colors='black')
ax2.spines['right'].set_color('black')

# in faint lines, plot the individual years of the data

ax.plot(DOY_composite_starts, ecoregion_dat_dict['CCI_mean'], color=CCI_color, lw = 2.5)
ax.plot(DOY_composite_starts, np.array(ecoregion_dat_dict['CCI_mean']) - np.array(ecoregion_dat_dict['CCI_std']), color=CCI_color, linestyle='--', lw=1.5)
ax.plot(DOY_composite_starts, np.array(ecoregion_dat_dict['CCI_mean']) + np.array(ecoregion_dat_dict['CCI_std']), color=CCI_color, linestyle='--', lw=1.5)

# Plot shaded region for mean ± representative std
ax.fill_between(
    DOY_composite_starts,
    np.array(ecoregion_dat_dict['CCI_mean']) - np.array(ecoregion_dat_dict['CCI_std']),
    np.array(ecoregion_dat_dict['CCI_mean']) + np.array(ecoregion_dat_dict['CCI_std']),
    color=CCI_color, alpha=0.35, label='±1 std'
)

# Define the axis, title labels
# set xticks to Month abbreviations
ax.set_xticks(ticks=months_doy)
ax.set_xticklabels(months_labs)
ax.set_xlim(0, 365)
# ax.set_ylim(PRB_min, PRB_max)

ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax2.yaxis.set_major_locator(MultipleLocator(0.1))

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
ax.set_xlim(0, 353)

if mask_LC_high:
    plot_filename = f'{ecoregion.replace(" ", "_")}_seasonality_LCmask.png'
elif mask_LC_low:
    plot_filename = f'{ecoregion.replace(" ", "_")}_seasonality_LCmask_low.png'
else:
    plot_filename = f'{ecoregion.replace(" ", "_")}_seasonality.png'

# plt.savefig(os.path.join(plot_basedir, plot_filename), dpi=300, bbox_inches='tight')
# plt.close()
plt.show()

# %%

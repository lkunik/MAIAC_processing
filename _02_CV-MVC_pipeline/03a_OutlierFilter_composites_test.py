#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Created on Tue Oct 24 2023
#
#

#########################################
# Import packages
#########################################
#%%
# File system packages
import os  # operating system library
import sys
import glob
from pathlib import Path
import copy

# runtime packages
import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=DeprecationWarning)
import gc

# numerical/data management packages
import math
import numpy as np
import xarray as xr  # for multi dimensional data
# import rioxarray as rxr
import pandas as pd
import xesmf as xe
import struct
from rasterio.enums import Resampling

# shapefile/geospatial packages

# time and date packages
import time
from datetime import datetime as dt  # date time library
from datetime import timedelta

# plotting packages
from matplotlib import pyplot as plt  # primary plotting module
from shapely.geometry import mapping
import cftime
import random
from matplotlib.patches import Patch
from matplotlib.dates import MonthLocator, DateFormatter
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter as FFmt  # need this for formatting plot ticks
import cartopy.crs as ccrs  #
from cartopy.io import img_tiles  # cartopy's implementation of webtiles
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature

#########################################
# Define Global Filepaths
#########################################
#%%

#########################################
# Define Global functions
#########################################


####################################
### Set up map parameters
####################################
#%%


#########################################
# Define Global Variables and constants
#########################################

hist_years = np.arange(2000, 2025)
DOY_composite_starts = np.arange(1, 366, 16)
hist_composites = range(len(DOY_composite_starts))  # 0-22


# Site coordinates, date ranges, and save paths
site_info = {
    'US-NR1': {
        'coords': (-105.5464, 40.0329),
        'start': '2017-01-01',
        'end': '2018-12-31',
        'tile': 'h09v04'
    },
    'OSBS': {
        'coords': (-81.993431, 29.689282),
        'start': '2021-01-01',
        'end': '2022-12-31',
        'tile': 'h10v06'
    },
    'Ca-Obs': {
        'coords': (-105.1178, 53.9872),
        'start': '2018-01-01',
        'end': '2022-12-31',
        'tile': 'h11v03'
    },
    'DEJU': {
        'coords': (-145.75136, 63.88112),
        'start': '2019-01-01',
        'end': '2021-12-31',
        'tile': 'h11v02'
    },
    'US-UMB': {
        'coords': (-84.7138, 45.5598),
        'start': '2018-01-01',
        'end': '2019-12-31',
        'tile': 'h12v04'
    },
    'US-Ne3': {
        'coords': (-96.4397, 41.1797),
        'start': '2018-01-01',
        'end': '2019-12-31',
        'tile': 'h10v04'
    }
}
#########################################
# Begin main
#########################################
#%%

# mark start time to keep track of elapsed
start_total = time.time()

# Command line arguments
site = 'US-NR1'
MODIS_tile = 'h09v04'
QC_option = 'CloudFree_LowAOD_ClearAdj' #
composite_year = 2017

dat_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC_option}/{MODIS_tile}/'
climatology_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC_option}/{MODIS_tile}/hist_avg/'
plot_dir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/MAIAC_CV-MVC/1km_curv/hist_avg/QCfilt_{QC_option}/{MODIS_tile}/'
filtered_out_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill_OutlierFilter/QCfilt_{QC_option}/{MODIS_tile}/'
QC_out_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/QC_OutlierFilter/QCfilt_{QC_option}/{MODIS_tile}/'

Path(plot_dir).mkdir(parents=True, exist_ok=True)
Path(QC_out_dir).mkdir(parents=True, exist_ok=True)
Path(filtered_out_dir).mkdir(parents=True, exist_ok=True)

min_valid_count = 3  # minimum number of valid points required to compute mean for a composite period
composite_periods = np.arange(0, 23)  # 0-22

#%%
#########################################
# Step 1: define timesteps/timeperiods (always +/- 2 timesteps padding; missing files become NaN)
# Note each time step is 16 days
#########################################

# names of the variables to read from each climatology file, per index:
# key = index name used throughout this script, 
# value = (median variable name, MAD variable name) in the climatology file
CLIM_VARS = {'NDVI': ('NDVI_median', 'NDVI_MAD'),
            'CCI':  ('CCI_median',  'CCI_MAD')}
K = 3.0 # threshold multiplier: a value is flagged if it is more than K * MAD away from the median
MAD_SCALE = 1.0 # multiplier applied to MAD before thresholding; 1.0 = raw MAD, 1.4826 = MAD scaled to approximate one standard deviation
MAD_FLOOR = 0.0 # minimum allowed (scaled) MAD; 0.0 = no floor, so a pixel with MAD = 0 gets a threshold of 0
MIN_RUN = 3 # minimum number of consecutive same-direction exceedances needed for a run to count as persistent (and be kept)

# list of (year, composite_period) tuples, one per time slot, in chronological order (27 slots total):
steps = ([(composite_year - 1, cp) for cp in (21, 22)]            # slots 0-1: last two composite periods of the previous year (padding)
         + [(composite_year, cp) for cp in composite_periods]     # slots 2-24: all 23 composite periods (0-22) of the target year
         + [(composite_year + 1, cp) for cp in (0, 1)])           # slots 25-26: first two composite periods of the next year (padding)

step_years = np.array([y for y, _ in steps])   # 1D array (length 27): the year of each slot, e.g. [2017, 2017, 2018, ..., 2018, 2019, 2019]
step_cps = np.array([cp for _, cp in steps])   # 1D array (length 27): the composite period (0-22) of each slot, e.g. [21, 22, 0, 1, ..., 22, 0, 1]
is_target = step_years == composite_year       # 1D boolean array (length 27): True for the 23 target-year slots, False for the 4 padding slots
n_t = len(steps)                               # number of time slots (27); used as the size of the time axis for every array in Step 2

def dat_file_for(year, cp):
    # returns the full path of the pre-fill CV-MVC composite file for a given year and composite period
    # (the file may or may not exist; existence is checked in Step 2)
    return os.path.join(dat_dir, f'MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_QCfilt_{QC_option}_{year}-{cp:02d}.nc')

def clim_file_for(cp):
    # returns the full path of the climatology file for a given composite period (no year: the climatology spans all years)
    return os.path.join(climatology_dir, f'MCD19A1.{MODIS_tile}.061_1km_CV-MVC_QCfilt_{QC_option}_hist_avg_{cp:02d}.nc')

# find the first slot whose composite file actually exists, to use as a template for the tile grid:
# the inner generator builds the file path for each slot in order; next() returns the first path for which os.path.exists() is True
# (raises StopIteration if none of the 27 files exist)
first_file = next(f for f in (dat_file_for(y, cp) for y, cp in steps) if os.path.exists(f))

with xr.open_dataset(first_file) as tmpl:   # open the template file; 'with' closes it automatically at the end of the block
    ny, nx = tmpl.sizes['y'], tmpl.sizes['x']   # number of rows (y) and columns (x) in the tile grid (1200 x 1200 for a 1 km MODIS tile)
    lat2d = tmpl['lat'].load()   # 2D DataArray (y, x) of pixel-center latitudes on the sinusoidal grid; .load() reads it into memory before the file closes
    lon2d = tmpl['lon'].load()   # 2D DataArray (y, x) of pixel-center longitudes; same as above

def read2d(ds, var):
    # reads one variable from an open dataset as a 2D numpy array of shape (ny, nx)
    arr = np.squeeze(ds[var].values)   # .values converts to numpy; np.squeeze drops any length-1 dimensions (e.g. a single 'time' dim) so the result is 2D
    assert arr.shape == (ny, nx), f'{var}: unexpected shape {arr.shape}'   # stop with an error if the array isn't on the same grid as the template
    return arr   # 2D numpy array (ny, nx)

#%%
#%%
#########################################
# Step 2: preallocate NaN arrays for the whole tile and fill each time slot from its file
#########################################

# dictionary of 3D numpy arrays, one per variable needed for the outlier test
# every array has shape (n_t, ny, nx) = (27 time slots, rows, columns) and starts as all NaN
# float32 halves memory relative to float64 (~155 MB per array for a 1200 x 1200 tile, ~0.9 GB for all six)
stack = {k: np.full((n_t, ny, nx), np.nan, dtype=np.float32)
         for k in ('NDVI',          # this year's NDVI in each slot (computed from the composite's bands below)
                   'CCI',           # this year's CCI in each slot
                   'NDVI_median',   # historical (2000-2025) median NDVI for that slot's composite period
                   'NDVI_MAD',      # historical MAD of NDVI for that slot's composite period
                   'CCI_median',    # historical median CCI for that slot's composite period
                   'CCI_MAD')}      # historical MAD of CCI for that slot's composite period

#----------------------------------------
# 2a: composite data, one file per slot
#----------------------------------------
for i, (year, cp) in enumerate(steps):   # i = slot index (0-26); year, cp = the year and composite period that slot represents
    f = dat_file_for(year, cp)           # full path of this slot's composite file
    if not os.path.exists(f):            # file missing (e.g. year 1999 or 2026 padding, early 2000, or high-latitude winter periods)
        print(f'Missing data file (slot left NaN): {f}')
        continue                         # skip to the next slot; this slot stays NaN for every pixel, so it keeps its position in the time axis
    with xr.open_dataset(f) as ds:       # open the composite file; closes automatically at the end of the block
        # read the three bands needed for NDVI and CCI as 2D (ny, nx) arrays
        # b1 = red (620-670 nm), b2 = NIR (841-876 nm), b11 = 526-536 nm
        b1, b2, b11 = (read2d(ds, v) for v in ('Sur_refl1', 'Sur_refl2', 'Sur_refl11'))
    with np.errstate(divide='ignore', invalid='ignore'):   # suppress warnings for pixels where the denominator is 0 or a band is NaN (result is inf/NaN)
        stack['NDVI'][i] = (b2 - b1) / (b2 + b1)           # NDVI for every pixel in slot i; pixels with no valid composite observation stay NaN
        stack['CCI'][i] = (b11 - b1) / (b11 + b1)          # CCI for every pixel in slot i

#----------------------------------------
# 2b: climatology, one file per composite period
#----------------------------------------
# the climatology depends only on composite period, not year, so each of the 23 files is read once
# and copied into every slot with that composite period
# (cp 0 and 1 fill both a target slot and a next-year padding slot; cp 21 and 22 fill both a target slot and a previous-year padding slot)
for cp in np.unique(step_cps):           # loop over the unique composite periods present in the 27 slots (0-22)
    fc = clim_file_for(cp)               # full path of this composite period's climatology file
    if not os.path.exists(fc):           # climatology file missing (e.g. a high-latitude winter period that was never computed)
        print(f'Missing climatology file (slots left NaN): {fc}')
        continue                         # median and MAD stay NaN for these slots, so no pixel can be flagged there (Step 3 gives sign 0)
    with xr.open_dataset(fc) as cds:     # open the climatology file
        vals = {}                        # temporary dictionary of 2D (ny, nx) arrays read from this file, keyed like 'stack'
        for idx, (med_var, mad_var) in CLIM_VARS.items():   # idx = 'NDVI' or 'CCI'; med_var / mad_var = variable names in the file
            vals[f'{idx}_median'] = read2d(cds, med_var)    # median of this index across all years for this composite period
            vals[f'{idx}_MAD'] = read2d(cds, mad_var)       # MAD of this index across all years for this composite period
    for i in np.where(step_cps == cp)[0]:   # slot indices whose composite period equals cp (one or two slots)
        for k, v in vals.items():           # k = e.g. 'NDVI_median', v = its 2D array
            stack[k][i] = v                 # copy the 2D climatology field into slot i of the matching 3D array

#%%
#########################################
# Step 3: vectorized functions for signed exceedance and run length (time must be axis 0)
#########################################

def exceedance_sign(x, med, mad):
    # x   = 3D array (n_t, ny, nx): this year's index values in each slot (e.g. stack['NDVI'])
    # med = 3D array (n_t, ny, nx): historical median for each slot's composite period (e.g. stack['NDVI_median'])
    # mad = 3D array (n_t, ny, nx): historical MAD for each slot's composite period (e.g. stack['NDVI_MAD'])
    # returns a 3D int8 array (n_t, ny, nx): +1 = above median + threshold, -1 = below median - threshold, 0 = within threshold or not testable

    thr = K * np.maximum(MAD_SCALE * mad, MAD_FLOOR)   # 3D array: allowed deviation from the median at every pixel and slot
                                                        # MAD_SCALE * mad = (optionally scaled) MAD; np.maximum applies the floor elementwise
                                                        # K * ... = threshold (3 * MAD with the current settings); NaN where mad is NaN
    dev = x - med                                       # 3D array: signed deviation of this year's value from the historical median (NaN if either is NaN)
    sign = np.zeros(x.shape, dtype=np.int8)             # 3D int8 array initialized to 0 everywhere (int8 = 1 byte per value, ~39 MB per tile)
    with np.errstate(invalid='ignore'):                 # suppress "invalid value" warnings from comparing NaN; NaN > thr and NaN < -thr are both False
        sign[dev > thr] = 1                             # set +1 where the value is more than thr above the median
        sign[dev < -thr] = -1                           # set -1 where the value is more than thr below the median
    return sign                                         # any slot with a NaN value, median, or MAD stays 0 (cannot be flagged, breaks runs)


def run_lengths(sign):
    # sign = 3D int8 array (n_t, ny, nx) from exceedance_sign
    # returns a 3D int16 array (n_t, ny, nx): for each flagged slot (+1 or -1), the length of the consecutive
    # same-sign run it belongs to along the time axis; 0 for unflagged slots

    fwd = np.zeros(sign.shape, dtype=np.int16)   # forward count: number of consecutive same-sign slots ending at slot t (including t)
    bwd = np.zeros(sign.shape, dtype=np.int16)   # backward count: number of consecutive same-sign slots starting at slot t (including t)

    # forward pass: walk from the first slot to the last
    fwd[0] = sign[0] != 0                        # first slot: count is 1 where flagged, 0 where not (the boolean is stored as 1/0)
    for t in range(1, sign.shape[0]):            # t = 1 ... n_t-1; each iteration operates on the full 2D (ny, nx) slice at once
        same = (sign[t] != 0) & (sign[t] == sign[t - 1])        # 2D boolean: True where slot t is flagged AND has the same sign as slot t-1
        fwd[t] = np.where(same, fwd[t - 1] + 1, sign[t] != 0)   # where the run continues: previous count + 1
                                                                # otherwise: 1 if slot t starts a new run (flagged), 0 if unflagged

    # backward pass: walk from the last slot to the first (mirror image of the forward pass)
    bwd[-1] = sign[-1] != 0                      # last slot: 1 where flagged, 0 where not
    for t in range(sign.shape[0] - 2, -1, -1):   # t = n_t-2 ... 0
        same = (sign[t] != 0) & (sign[t] == sign[t + 1])        # 2D boolean: True where slot t is flagged AND has the same sign as slot t+1
        bwd[t] = np.where(same, bwd[t + 1] + 1, sign[t] != 0)   # continue the run from the right, or start a new one (1), or 0 if unflagged

    # run length = slots before t in the run (fwd) + slots after t in the run (bwd) - 1 (slot t is counted in both)
    return np.where(sign != 0, fwd + bwd - 1, 0).astype(np.int16)   # 0 for unflagged slots
#%%
#%%
#########################################
# Step 4: classify every slot at every pixel, combine NDVI and CCI, keep the target year
#########################################

flags = {}                    # dictionary of 3D int8 arrays (n_t, ny, nx), one per index ('NDVI', 'CCI'), holding the classification codes below
for idx in CLIM_VARS:         # idx = 'NDVI', then 'CCI' (iterating a dict loops over its keys)
    sign = exceedance_sign(stack[idx],                # 3D: this year's values for this index
                           stack[f'{idx}_median'],    # 3D: historical median for each slot's composite period
                           stack[f'{idx}_MAD'])       # 3D: historical MAD for each slot's composite period
                                                      # -> sign: 3D int8, +1 above / -1 below threshold / 0 otherwise
    rl = run_lengths(sign)                            # 3D int16: length of the same-sign run each flagged slot belongs to (0 if unflagged)
    flag = np.zeros(sign.shape, dtype=np.int8)        # 3D int8 initialized to 0 = within threshold or not testable (missing value/climatology)
    flag[(sign != 0) & (rl < MIN_RUN)] = 1            # 1 = transient outlier: flagged, but its run is shorter than MIN_RUN (1 or 2 slots)
    flag[(sign != 0) & (rl >= MIN_RUN)] = 2           # 2 = persistent anomaly: flagged and part of a run of MIN_RUN or more slots (kept, not removed)
    flags[idx] = flag                                 # store this index's classification

# 3D boolean (n_t, ny, nx): True = observation to remove
# (flags['NDVI'] == 1) | (flags['CCI'] == 1) -> transient outlier in NDVI OR in CCI
# & is_target[:, None, None]                 -> is_target is 1D (n_t,); [:, None, None] reshapes it to (n_t, 1, 1) so it
#                                               broadcasts over (ny, nx), setting every padding slot to False
outlier = ((flags['NDVI'] == 1) | (flags['CCI'] == 1)) & is_target[:, None, None]

# keep the target year only, as an xarray Dataset on the tile grid
tgt = np.where(is_target)[0]              # 1D int array of the target-year slot indices (2 ... 24)
dims = ('composite_period', 'y', 'x')     # dimension names for every variable in the output Dataset
flags_ds = xr.Dataset(
    {'NDVI_flag': (dims, flags['NDVI'][tgt]),   # (23, ny, nx) int8: NDVI classification (0/1/2) for the 23 target-year composites
     'CCI_flag':  (dims, flags['CCI'][tgt]),    # (23, ny, nx) int8: CCI classification (0/1/2)
     'outlier':   (dims, outlier[tgt])},        # (23, ny, nx) bool: True where the observation should be removed
    coords={'composite_period': step_cps[tgt],  # 1D coordinate: composite period 0-22 for each target slot
            'lat': lat2d,                       # 2D (y, x) latitude coordinate from the template file
            'lon': lon2d})                      # 2D (y, x) longitude coordinate from the template file

#%%
#########################################
# Step 5: sanity checks (tile-wide flag rate, site pixel, map)
#########################################

#----------------------------------------
# 5a: percent of valid observations flagged as transient outliers, per composite period
#----------------------------------------
n_valid = np.isfinite(stack['NDVI'][tgt]).sum(axis=(1, 2))   # 1D (23,): number of pixels with a valid (non-NaN) NDVI in each target composite
                                                              # np.isfinite -> 3D boolean; .sum over y and x (axes 1, 2) counts True per slot
pct_flagged = 100 * flags_ds['outlier'].sum(('y', 'x')).values / np.maximum(n_valid, 1)
                                                              # 1D (23,): outlier pixel count per composite / valid pixel count * 100
                                                              # np.maximum(n_valid, 1) avoids division by zero for composites with no valid pixels
print(pd.Series(pct_flagged, index=step_cps[tgt], name='% flagged').round(2))   # print as a table indexed by composite period

#----------------------------------------
# 5b: flags at the site pixel, to compare against the point-version results
#----------------------------------------
lon_pt, lat_pt = site_info[site]['coords']     # site longitude and latitude (degrees)
dist = np.abs(lon2d.values - lon_pt) + np.abs(lat2d.values - lat_pt)   # 2D (ny, nx): summed absolute lat/lon difference from the site
                                                                        # (same distance measure as the old clip_2dcurv_ds_to_point)
y_idx, x_idx = np.unravel_index(np.nanargmin(dist), dist.shape)   # np.nanargmin -> flat index of the smallest distance (ignoring NaN)
                                                                  # np.unravel_index -> converts it to (row, column) indices on the tile
site_check = flags_ds.isel(y=y_idx, x=x_idx)[['NDVI_flag', 'CCI_flag', 'outlier']].to_dataframe()
                                                              # select the site pixel -> 1D over composite_period -> pandas DataFrame
                                                              # (lat/lon come along as extra columns because they are coordinates)
print(site_check[['NDVI_flag', 'CCI_flag', 'outlier']])      # print only the flag columns, one row per composite period

#----------------------------------------
# 5c: map of how many target-year composites were flagged as transient outliers at each pixel
#----------------------------------------
flags_ds['outlier'].sum('composite_period').plot(x='lon', y='lat', cmap='inferno_r')   # 2D (ny, nx) count (0-23), plotted on the curvilinear lat/lon grid
plt.title(f'{MODIS_tile} {composite_year}: # transient outliers')   # title with tile and year
plt.show()                                                            # display interactively


#%%
#########################################
# Step 6: save the annual number of outliers per pixel, same structure as the input composite files
# but with a length-1 'year' dimension in place of 'time'
#########################################

# full path of the annual outlier QC file for this tile and year
QC_OutlierFilter_outfile = os.path.join(QC_out_dir, f'MCD19A1.{MODIS_tile}.061_1km_CV-MVC_QCfilt_{QC_option}_annual_OutlierFilterQC_{composite_year}.nc')

#----------------------------------------
# 6a: per-composite outlier masks for NDVI and CCI as separate DataArrays
#----------------------------------------
# flags_ds['NDVI_flag'] and flags_ds['CCI_flag'] are (composite_period, y, x) int8 with codes
# 0 = within threshold / not testable, 1 = outlier (short run), 2 = persistent anomaly (kept)
# == 1 picks out the outliers for each index; stored as int8 (0 = kept, 1 = removed)
outlier_NDVI = (flags_ds['NDVI_flag'] == 1).astype(np.int8)   # (composite_period, y, x): 1 where NDVI triggered removal
outlier_CCI = (flags_ds['CCI_flag'] == 1).astype(np.int8)     # (composite_period, y, x): 1 where CCI triggered removal

# attributes shared by both variables: the settings that produced the masks
filter_settings = {'K': K,                  # threshold multiplier
                   'MAD_SCALE': MAD_SCALE,  # MAD scaling factor (1.0 = raw MAD)
                   'MAD_FLOOR': MAD_FLOOR,  # minimum MAD applied before thresholding
                   'MIN_RUN': MIN_RUN}      # minimum run length counted as persistent (kept)

for idx, da in (('NDVI', outlier_NDVI), ('CCI', outlier_CCI)):   # idx = index name, da = that index's mask DataArray
    da.name = f'outlier_{idx}'                                    # variable name in the output file
    da.attrs = {'long_name': f'outlier filter mask for {idx} (1 = composite removed because of {idx})',
                'units': '1',
                'flag_values': '0, 1',
                'flag_meanings': 'kept removed',
                'description': (f'1 where {idx} exceeded median +/- K*MAD (historical, per composite period) '
                                f'in a same-direction run shorter than MIN_RUN consecutive composites'),
                **filter_settings}                                # add the shared filter settings to this variable's attributes


#----------------------------------------
# 6b: read the structure of the input composite files from the template file (same file used in Step 1)
#----------------------------------------
with xr.open_dataset(first_file) as tmpl:                    # reopen the template composite file
    tmpl_dims = list(tmpl['Sur_refl1'].dims)                 # dimension order of a data variable in the input files, e.g. ['time', 'y', 'x']
    tmpl_attrs = dict(tmpl.attrs)                            # global attributes of the input file (copied onto the output below)
    tmpl_coords = {c: tmpl[c].load()                         # every coordinate variable in the input file, loaded into memory...
                   for c in tmpl.coords
                   if c != 'time' and 'time' not in tmpl[c].dims}   # ...except 'time' and anything defined along time


#----------------------------------------
# 6c: build the output Dataset
#----------------------------------------
QC_ds = xr.Dataset({'outlier_NDVI': outlier_NDVI,    # Dataset with two data variables (composite_period, y, x);
                    'outlier_CCI': outlier_CCI})     # composite_period, lat and lon coordinates carry over from flags_ds
QC_ds = QC_ds.assign_coords(tmpl_coords)            # attach the input file's non-time coordinates (lat, lon, and any others, e.g. x/y)

# same dimension order as the input files, with 'time' replaced by 'composite_period'
out_dims = ['composite_period' if d == 'time' else d for d in tmpl_dims]   # e.g. ['time', 'y', 'x'] -> ['composite_period', 'y', 'x']
if 'composite_period' not in out_dims:               # if the input files have no 'time' dimension (only a scalar time coordinate)...
    out_dims = ['composite_period'] + out_dims       # ...put 'composite_period' first
QC_ds = QC_ds.transpose(*out_dims)                   # reorder dimensions to match

QC_ds.attrs = tmpl_attrs                             # copy the input file's global attributes
QC_ds.attrs['year'] = int(composite_year)            # record the year as a global attribute (no year coordinate); delete this line if not wanted
QC_ds.attrs['history'] = (f'{dt.now():%Y-%m-%d}: outlier filter QC mask computed from '
                          f'{os.path.basename(first_file)} and neighboring composites')   # note how this file was made

#----------------------------------------
# 6d: save
#----------------------------------------
QC_ds.to_netcdf(path=QC_OutlierFilter_outfile, format="NETCDF4")   # write to NetCDF4, same format as the input files
QC_ds.close()                                                       # close the Dataset
print(f'saved annual outlier QC file to {QC_OutlierFilter_outfile}')






#%%
#----------------------------------------
# 6e: check the saved file
#----------------------------------------


#%%
# maps of the annual number of composites removed at each pixel, NDVI and CCI side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))                  # one row, two panels
for ax, idx in zip(axes, ['NDVI', 'CCI']):                       # left panel = NDVI, right panel = CCI
    QC_ds[f'outlier_{idx}'].sum('composite_period').plot(        # 2D (y, x) count of removed composites (0-23) for this index
        ax=ax, x='lon', y='lat', cmap='inferno_r', vmin=0,       # plot on the curvilinear lat/lon grid; count starts at 0
        cbar_kwargs={'label': '# composites removed'})
    ax.set_title(f'{MODIS_tile} {composite_year}: outlier_{idx}')  # panel title with tile, year and index
plt.tight_layout()
plt.show()


with xr.open_dataset(QC_OutlierFilter_outfile) as chk:   # reopen the file just written
    print(chk)                                            # dimensions, coordinates, variables, and attributes
    with xr.open_dataset(first_file) as tmpl:             # reopen the template for a side-by-side comparison
        print(tmpl)                                       # the output should match this structure, with 'year' in place of 'time'



#########################################
# Step 7: apply the outlier mask to each target-year composite (all bands) and save the filtered files
#########################################

for i in tgt:                                    # i = slot index of each target-year composite (2 ... 24)
    year, cp = steps[i]                          # year (= composite_year) and composite period (0-22) of this slot
    f = dat_file_for(year, cp)                   # full path of the original (pre-filter) composite file for this slot
    if not os.path.exists(f):                    # no composite file for this period (e.g. high-latitude winter): nothing to filter
        print(f'No composite file for {year}-{cp:02d}, skipping: {f}')
        continue                                 # move on to the next composite period

    # full path of the outlier-filtered output file (same naming convention as the previous version of this script)
    # e.g. MCD19A1.h14v03.061_1km_16day_CV-MVC_QCfilt_CloudFree_LowAOD_ClearAdj_OutlierFilter_2025-17.nc
    outfile_filtered = os.path.join(filtered_out_dir, f'MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_QCfilt_{QC_option}_OutlierFilter_{year}-{cp:02d}.nc')

    with xr.open_dataset(f) as ds:               # open the original composite file
        dat_ds = ds.load()                       # read everything into memory so the file can close before we write

    # 2D keep-mask for this slot: True = keep, False = remove (outlier in NDVI or CCI)
    # built from the numpy array with only (y, x) dims so no extra coordinates get attached to the output
    keep_da = xr.DataArray(~outlier[i], dims=('y', 'x'))

    n_removed = int(outlier[i].sum())                               # number of pixels removed in this composite
    n_valid_before = int(np.isfinite(stack['NDVI'][i]).sum())       # number of pixels with valid NDVI before filtering

    for v in dat_ds.data_vars:                   # loop over every data variable in the file (bands, geometry, etc.)
        if ('y' in dat_ds[v].dims) and ('x' in dat_ds[v].dims):    # only per-pixel variables; variables without (y, x) are left untouched
            dat_ds[v] = dat_ds[v].where(keep_da)                    # set removed pixels to NaN (broadcasts over a length-1 time dim if present)
                                                                    # note: .where converts integer variables to float

    dat_ds.attrs['history'] = (f'{dt.now():%Y-%m-%d}: outlier filter applied '
                                f'(K={K}, MAD_SCALE={MAD_SCALE}, MAD_FLOOR={MAD_FLOOR}, MIN_RUN={MIN_RUN}); '
                                f'{n_removed} pixels removed')      # record the filter settings and removal count in the file

    # dat_ds.to_netcdf(path=outfile_filtered, format="NETCDF4")       # write the filtered composite
    # dat_ds.close()

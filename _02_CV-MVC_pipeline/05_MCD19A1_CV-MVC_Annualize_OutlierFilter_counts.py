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
import rasterio
from rasterio.enums import Resampling
import numpy as np
import rasterio
from rasterio.warp import reproject
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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

def td_min(td):
    return td.seconds//60

def td_sec(td):
    return td.seconds%60


####################################
### Set up plotting parameters
####################################

plt.rcParams.update({
   'axes.labelsize': 16,
   'axes.titlesize': 16,
   'xtick.labelsize': 16,
   'ytick.labelsize': 15,
   'legend.fontsize': 13,
})

# use Google Satellite imagery as basemap
tiler = img_tiles.GoogleTiles(style='satellite')
crs = tiler.crs # set crs of map tiler
alpha = 0.8  # transparency 0-1
transform = ccrs.PlateCarree()  # transform specifies the crs that the data is in

tiler_zoom = 7  # define a zoom level of detail
extent = [-125.3, -102.7, 29.7, 50.3]  # [minx, maxx, miny, maxy], bounds for map

### Load state boundaries
fn = shpreader.natural_earth(
   resolution='10m', category='cultural', 
   name='admin_1_states_provinces',
)
reader = shpreader.Reader(fn)
states = [x for x in reader.records() if x.attributes["admin"] == "United States of America"] # get all states in US
states_geom = cfeature.ShapelyFeature([x.geometry for x in states], ccrs.PlateCarree())

# matplotlib map plotting functions
def z_axis_formatter(x, pos, deci=2):
   '''Format ticks to format with deci number of decimals'''

   return f'{x:.{deci}f}'



#########################################
# Define Global Variables and constants
#########################################

analysis_years = np.arange(2000, 2026)
DOY_composite_starts = np.arange(1, 366, 16)
composite_dates_2020 = [dt(2020, 1, 1) + timedelta(days=int(DOY) - 1) for DOY in DOY_composite_starts]

annualize_months_dict = {'MJJ':[5, 6, 7],
                    'JJA':[6, 7, 8],
                    'JAS':[7, 8, 9],
                    'ASO':[8, 9, 10]}

annualize_months_descrs_dict = {'MJJ':'May June July',
                    'JJA':'June July August',
                    'JAS':'July August September',
                    'ASO':'August September October'}

# CHANGED: outlier QC variables to annualize
outlier_vars = ['outlier_NDVI', 'outlier_CCI']


#########################################
# Begin main
#########################################
#%%

def main():

    MODIS_tile = sys.argv[1] #'h09v04'
    QC_option = sys.argv[2] # 'CloudFree_LowAOD_ClearAdj'   # CHANGED: was sys.argv[3] (no months argument anymore)

    # CHANGED: input is the annual outlier QC files on the 1 km sinusoidal grid
    dat_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/QC_OutlierFilter/QCfilt_{QC_option}/'
    # CHANGED: pre-fill composite files, used only to count how many composites exist per year (denominator)
    composite_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC_option}/'

    print(f'Processing tile {MODIS_tile}...')
    dat_dir = os.path.join(dat_basedir, MODIS_tile)
    dat_files = sorted(glob.glob(f'{dat_dir}/MCD19A1*{MODIS_tile}*_annual_OutlierFilterQC_*.nc'))
    composite_dir = os.path.join(composite_basedir, MODIS_tile)

    out_dir = os.path.join(dat_dir, 'annualized')
    outfile = os.path.join(out_dir, f'MCD19A1.{MODIS_tile}.061_1km_annual_CV-MVC_QCfilt_{QC_option}_OutlierFilterQC.nc')
    if os.path.exists(outfile):
        print(f'Output file already exists: {outfile}. Skipping this tile.')
        # sys.exit(0)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # CHANGED: compose a list of years from the QC filenames (one file per year)
    years = []
    for fname in dat_files:

        basename = os.path.basename(fname)
        # Example: 'MCD19A1.h09v04.061_1km_CV-MVC_QCfilt_CloudFree_LowAOD_ClearAdj_annual_OutlierFilterQC_2020.nc'
        year = int(basename.split('_')[-1].split('.')[0])
        years.append(year)
    years = np.array(years)


    # loop through years: fraction of composites flagged as outliers in each year

    dat_xr_annual_arr = []
    n_composites_dict = {}   # number of composites that exist in each year (denominator)

    for year in analysis_years:
        print(f'Processing year {year}...')
        if year not in years:
            print(f'No outlier QC file for {year}. Skipping this year.')
            continue
        f = dat_files[np.where(years == year)[0][0]]

        # denominator: number of pre-fill composite files that exist for this tile/year
        # (missing composites were saved as 0 in the QC file, not NaN, so they can't be identified from the QC file itself)
        composite_files_year = sorted(glob.glob(os.path.join(
            composite_dir, f'MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_QCfilt_{QC_option}_{year}-??.nc')))
        existing_cps = [int(os.path.basename(cf).split('-')[-1].split('.')[0]) for cf in composite_files_year]
        n_composites = len(existing_cps)
        n_composites_dict[year] = n_composites
        print(f'{n_composites} composites exist for {year}')
        if n_composites == 0:
            print(f'No composite files for {year}. Skipping this year.')
            continue

        with xr.open_dataset(f) as ds:
            dat_xr_slice = ds[outlier_vars].load()

        # sanity check: composite periods with no composite file should have no flagged outliers
        missing_cps = [cp for cp in dat_xr_slice['composite_period'].values if cp not in existing_cps]
        if len(missing_cps) > 0:
            n_flagged_missing = int(dat_xr_slice.sel(composite_period=missing_cps).to_array().sum())
            if n_flagged_missing > 0:
                print(f'WARNING: {n_flagged_missing} outlier flags found in composite periods with no composite file ({missing_cps})')

        # number of composites flagged in this year / number of composites that exist in this year
        dat_xr_ann_frac = (dat_xr_slice.sum(dim='composite_period') / n_composites).astype(np.float32)
        dat_xr_ann_frac = dat_xr_ann_frac.expand_dims(year = [year])
        dat_xr_annual_arr.append(dat_xr_ann_frac)

    dat_xr_annual = xr.concat(dat_xr_annual_arr, dim='year')
    del dat_xr_annual_arr
    gc.collect()

    # CHANGED: average across years so the output has only spatial dimensions
    dat_xr_annual = dat_xr_annual.mean(dim='year')
    dat_xr_annual = dat_xr_annual.rename({v: f'{v}_frac' for v in outlier_vars})

    # variable attributes
    with xr.open_dataset(dat_files[0]) as ds0:
        filter_settings = {k: ds0['outlier_CCI'].attrs[k] for k in ('K', 'MAD_SCALE', 'MAD_FLOOR', 'MIN_RUN')}
    for idx in ('NDVI', 'CCI'):
        dat_xr_annual[f'outlier_{idx}_frac'].attrs = {
            'long_name': f'mean annual fraction of 16-day composites removed as {idx} outliers',
            'units': '1',
            'description': (f'For each year, number of composites flagged as {idx} outliers divided by the number of '
                            f'composites that exist in that year; then averaged across years. '
                            f'Missing composites count as not flagged.'),
            **filter_settings}

    # define global attributes
    years_used = sorted(n_composites_dict.keys())
    dat_xr_annual.attrs = {}
    dat_xr_annual.attrs['title'] = 'MODIS MCD19A1.061 MAIAC 16-day CV-MVC composites - outlier filter QC summary: mean annual fraction of composites removed as NDVI or CCI outliers'
    dat_xr_annual.attrs['QC_option'] = QC_option
    dat_xr_annual.attrs['MODIS_tile'] = MODIS_tile
    dat_xr_annual.attrs['grid'] = 'MODIS 1 km sinusoidal tile grid (y, x); lat and lon are 2D pixel-center coordinates'
    dat_xr_annual.attrs['years_averaged'] = f'{years_used[0]}-{years_used[-1]}'
    dat_xr_annual.attrs['composites_per_year'] = ', '.join(f'{y}: {n_composites_dict[y]}' for y in years_used)
    dat_xr_annual.attrs['outlier_filter'] = (f'Composite removed where NDVI or CCI exceeded the historical (per composite period) '
                                             f'median +/- K*MAD in a same-direction run shorter than MIN_RUN consecutive composites '
                                             f'(K={filter_settings["K"]}, MAD_SCALE={filter_settings["MAD_SCALE"]}, '
                                             f'MAD_FLOOR={filter_settings["MAD_FLOOR"]}, MIN_RUN={filter_settings["MIN_RUN"]})')
    dat_xr_annual.attrs['source_files'] = f'{dat_dir}/*_annual_OutlierFilterQC_YYYY.nc'
    dat_xr_annual.attrs['Citation'] = 'Lyapustin, A., Wang, Y. (2022). MODIS/Terra+Aqua Land Surface BRF Daily L2G Global 500m and 1km SIN Grid V061 [Data set]. NASA EOSDIS Land Processes Distributed Active Archive Center. Accessed from https://doi.org/10.5067/MODIS/MCD19A1.061'
    dat_xr_annual.attrs['Processed_by'] = 'Lewis Kunik, lewis.kunik@utah.edu'
    dat_xr_annual.attrs['Date_created'] = str(dt.today().date())

    # Finally, save to netcdf
    dat_xr_annual.to_netcdf(path=outfile, format="NETCDF4")
    dat_xr_annual.close()

    del dat_xr_annual
    gc.collect()

    print(f'tile {MODIS_tile} done')

if __name__ == "__main__":
    main()
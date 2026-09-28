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


#########################################
# Define Global functions
#########################################

def load_MCD19_files_to_xr(xr_files_list, crs = 4326, coords_xy = True):
   xr_arr = []
   for ii, xr_file in enumerate(xr_files_list):
      print(f'ii = {ii}: loading file {xr_file}')
      if not os.path.exists(xr_file):
         print(f'File not found: {xr_file}. Skipping this file.')
         continue
      xr_raw = xr.open_dataset(xr_file, decode_coords=all)
      if coords_xy:
         if 'lon' in xr_raw.dims:
            xr_raw = xr_raw.rename({'lon':'x'})
         if 'lat' in xr_raw.dims:
            xr_raw = xr_raw.rename({'lat':'y'})
         xr_raw = xr_raw.sortby(['x','y'])
      else:
         if 'x' in xr_raw.dims:
            xr_raw = xr_raw.rename({'x':'lon'})
         if 'y' in xr_raw.dims:
            xr_raw = xr_raw.rename({'y':'lat'})
         xr_raw = xr_raw.sortby(['lon','lat'])
         
      if 'time' not in xr_raw.dims:
         if 'time' in xr_raw.variables:
            xr_raw = xr_raw.expand_dims(time = [xr_raw.time.values])
         else:
            xr_raw = xr_raw.expand_dims(time = [ii])
         xr_arr.append(xr_raw)
      else:
         xr_arr.append(xr_raw)
   print('concatenating xarray objects...')
   return xr.concat(xr_arr, dim = 'time').rio.write_crs(crs)


#########################################
# Begin main
#########################################
#%%

def main():

    MODIS_tile = sys.argv[1] #'h12v04'
    annualize_months_abbr = sys.argv[2] # 'JJA'
    QC_option = sys.argv[3] # 'CloudFree'


    dat_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/01d_rect/QCfilt_{QC_option}/'

    print(f'Processing tile {MODIS_tile}...')
    dat_dir = os.path.join(dat_basedir, MODIS_tile)
    dat_files = sorted(glob.glob(f'{dat_dir}/MCD19A1*{MODIS_tile}*.nc'))

    out_dir = os.path.join(dat_dir, 'annualized')
    outfile = os.path.join(out_dir, f'MCD19A1.{MODIS_tile}.061_01d_annual_{annualize_months_abbr}_CV-MVC_QCfilt_{QC_option}.nc')
    if os.path.exists(outfile):
        print(f'Output file already exists: {outfile}. Skipping this tile.')
        # sys.exit(0)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Compose a list of dates from the MCD19 filenames
    dates = []
    for fname in dat_files:

        basename = os.path.basename(fname)
        # Example: 'MCD19A1_2001-05.nc'
        parts = basename.split('_')[-2].split('-')
        year = int(parts[0])
        composite_idx = int(parts[1])
        # NN is 01-based index into DOY_composite_starts
        doy = DOY_composite_starts[composite_idx]

        # Convert year and day-of-year to a proper date
        date_dt = dt(year, 1, 1) + timedelta(days=int(doy) - 1)
        date = np.datetime64(date_dt.date())
        dates.append(date)
    dates = np.array(dates)

    # Create a mask for dates in months May (5) through October (10)
    months_to_keep = [5, 6, 7, 8, 9, 10]
    date_months = np.array([d.astype('datetime64[M]').astype(int) % 12 + 1 for d in dates])
    date_mask = np.isin(date_months, months_to_keep)


    # loop through years and annualize summer data

    annualize_months_descr = annualize_months_descrs_dict[annualize_months_abbr]


    print(f'Annualizing months: {annualize_months_abbr}...')

    

    annualize_months = annualize_months_dict[annualize_months_abbr]
    dat_xr_annual_arr = []

    for year in analysis_years:
        print(f'Processing year {year}...')
        files_year = []
        for f, d in zip(dat_files, dates):
            if d.astype('datetime64[Y]').astype(int) + 1970 == year and (d.astype('datetime64[M]').astype(int) % 12 + 1) in annualize_months:
                files_year.append(f)
        MCD19_files_year = files_year

        dat_xr_slice = load_MCD19_files_to_xr(MCD19_files_year)

        dat_xr_ann_mean = dat_xr_slice.mean(dim='time', skipna=True)
        dat_xr_ann_mean = dat_xr_ann_mean.expand_dims(year = [year])
        dat_xr_annual_arr.append(dat_xr_ann_mean)

    dat_xr_annual = xr.concat(dat_xr_annual_arr, dim='year')
    del dat_xr_annual_arr
    gc.collect()

    # define global attributes
    dat_xr_annual.attrs['title'] = 'MODIS MCD19A1.061 MAIAC Vegetation Indices calculated from surface reflectance - 16-day composite using Constrained View angle - Maximum Value Composite (CV-MVC) algorithm - ANNUAL MEAN taken over 3-month window'
    dat_xr_annual.attrs['averaging_window'] = annualize_months_descr
    dat_xr_annual.attrs['Comment1'] = 'Original files downloaded from https://e4ftl01.cr.usgs.gov/MOTA/MCD19A1.061/'
    dat_xr_annual.attrs['Comment2'] = 'hdf files converted to netcdf4 using HDF-EOS conversion tool (https://www.hdfeos.org/software/h4toh5.php)'
    dat_xr_annual.attrs['Comment3'] = 'gap filled using historical fill scaled with canopy-elevation regression'
    dat_xr_annual.attrs['Regridding'] = 'Regridded from MODIS sinusoidal 1km grid to 0.01° rectilinear grid using xESMF bilinear regridder'
    dat_xr_annual.attrs['Spatial_extent_descr'] = 'Western United States'
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

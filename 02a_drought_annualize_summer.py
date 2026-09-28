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

QC_option = 'CloudAOD'
MCD19_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/01d_rect/QCfilt_{QC_option}/WUS_combined/'
MCD19_files = sorted(glob.glob(os.path.join(MCD19_dir, '*.nc')))

nClimGrid_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/nClimGrid/05d'
nClimGrid_SPEI_01_file = os.path.join(nClimGrid_dir, 'nClimGrid_SPEI_01month_05d_WUS_monthly_2000-2024.nc')
nClimGrid_SPEI_12_file = os.path.join(nClimGrid_dir, 'nClimGrid_SPEI_12month_05d_WUS_monthly_2000-2024.nc')
nClimGrid_SPEI_36_file = os.path.join(nClimGrid_dir, 'nClimGrid_SPEI_36month_05d_WUS_monthly_2000-2024.nc')

GRIDMET_PDSI_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/GRIDMET/PDSI/05d'
GRIDMET_PDSI_files = sorted(glob.glob(os.path.join(GRIDMET_PDSI_dir, '*.nc')))

GRIDMET_VPD_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/GRIDMET/VPD/05d'
GRIDMET_VPD_files = sorted(glob.glob(os.path.join(GRIDMET_VPD_dir, '*.nc')))


TerraClimate_CWD_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/TerraClimate/CWD/05d'
TerraClimate_CWD_files = sorted(glob.glob(os.path.join(TerraClimate_CWD_dir, '*.nc')))

TerraClimate_SoilM_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/TerraClimate/SoilM/05d'
TerraClimate_SoilM_files = sorted(glob.glob(os.path.join(TerraClimate_SoilM_dir, '*.nc')))

TerraClimate_PET_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/TerraClimate/PET/05d'
TerraClimate_PET_files = sorted(glob.glob(os.path.join(TerraClimate_PET_dir, '*.nc')))


nClimGrid_SPEI_out_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/nClimGrid/SPEI/05d/annualized/'
GRIDMET_VPD_out_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/GRIDMET/VPD/05d/annualized/'
GRIDMET_PDSI_out_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/GRIDMET/PDSI/05d/annualized/'
TerraClimate_CWD_out_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/TerraClimate/CWD/05d/annualized/'
TerraClimate_SoilM_out_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/TerraClimate/SoilM/05d/annualized/'
TerraClimate_PET_out_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/TerraClimate/PET/05d/annualized/'

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

analysis_years = np.arange(2000, 2025)
DOY_composite_starts = np.arange(1, 366, 16)
composite_dates_2020 = [dt(2020, 1, 1) + timedelta(days=int(DOY) - 1) for DOY in DOY_composite_starts]

annualize_months_dict = {'MJJ':[5, 6, 7],
                    'JJA':[6, 7, 8],
                    'JAS':[7, 8, 9],
                    'ASO':[8, 9, 10]}

annualize_months_abbrs = ['MJJ', 'JJA', 'JAS', 'ASO']
annualize_months_descrs = ['May June July', 'June July August', 'July August September', 'August September October']

#########################################
# Begin main
#########################################
#%%

# Compose a list of dates from the MCD19 filenames
dates = []
for fname in MCD19_files:
    basename = os.path.basename(fname)
    # Example: 'MCD19A1_2001-05.nc'
    parts = basename.split('_')[-1].replace('.nc', '').split('-')
    year = int(parts[0])
    composite_idx = int(parts[1])
    # NN is 01-based index into DOY_composite_starts
    doy = DOY_composite_starts[composite_idx]

    # Convert year and day-of-year to a proper date
    date_dt = dt(year, 1, 1) + timedelta(days=int(doy) - 1)
    date = np.datetime64(date_dt.date())
    dates.append(date)
dates = np.array(dates)


# Load nClimGrid SPEI-01
nClimGrid_SPEI_01_xr = xr.open_dataset(nClimGrid_SPEI_01_file, decode_coords='all').rename({'lon':'x','lat':'y'})['SPEI_pearson']
nClimGrid_SPEI_01_xr = nClimGrid_SPEI_01_xr.sortby(['y', 'x'], ascending=True)
nClimGrid_SPEI_01_xr.rio.write_crs(4326, inplace=True)
nClimGrid_SPEI_01_xr_interp = nClimGrid_SPEI_01_xr.interp(time=dates, method='linear')
nClimGrid_SPEI_01_xr_interp['year'] = nClimGrid_SPEI_01_xr_interp['time.year']

# Load nClimGrid SPEI-12
nClimGrid_SPEI_12_xr = xr.open_dataset(nClimGrid_SPEI_12_file, decode_coords='all').rename({'lon':'x','lat':'y'})['SPEI_pearson']
nClimGrid_SPEI_12_xr = nClimGrid_SPEI_12_xr.sortby(['y', 'x'], ascending=True)
nClimGrid_SPEI_12_xr.rio.write_crs(4326, inplace=True)
nClimGrid_SPEI_12_xr_interp = nClimGrid_SPEI_12_xr.interp(time=dates, method='linear')
nClimGrid_SPEI_12_xr_interp['year'] = nClimGrid_SPEI_12_xr_interp['time.year']

# Load nClimGrid SPEI-36
nClimGrid_SPEI_36_xr = xr.open_dataset(nClimGrid_SPEI_36_file, decode_coords='all').rename({'lon':'x','lat':'y'})['SPEI_pearson']
nClimGrid_SPEI_36_xr = nClimGrid_SPEI_36_xr.sortby(['y', 'x'], ascending=True)
nClimGrid_SPEI_36_xr.rio.write_crs(4326, inplace=True)
nClimGrid_SPEI_36_xr_interp = nClimGrid_SPEI_36_xr.interp(time=dates, method='linear')
nClimGrid_SPEI_36_xr_interp['year'] = nClimGrid_SPEI_36_xr_interp['time.year']

# Load GRIDMET PDSI
PDSI_xr_raw = data_utils.load_files_to_xr(GRIDMET_PDSI_files)
PDSI_xr = PDSI_xr_raw['pdsi']
PDSI_xr = PDSI_xr.sortby(['y', 'x'], ascending=True)
PDSI_xr.rio.write_crs(4326, inplace=True)

# Interpolate PDSI_xr_regrid to match NDVI_xr time dimension
PDSI_xr_interp = PDSI_xr.interp(time=dates, method='linear')
# Add year coordinate for grouping
PDSI_xr_interp['year'] = PDSI_xr_interp['time.year']


#%%

# Filter TerraClimate_CWD_files for files matching *_{yyyy}.nc for each year in analysis_years
GRIDMET_VPD_files_filtered = []
TerraClimate_CWD_files_filtered = []
TerraClimate_SoilM_files_filtered = []
TerraClimate_PET_files_filtered = []
for yyyy in analysis_years:
    pattern = f'*_{yyyy}.nc'
    VPD_matched_files = [f for f in GRIDMET_VPD_files if Path(f).match(pattern)]
    GRIDMET_VPD_files_filtered.extend(VPD_matched_files)
    CWD_matched_files = [f for f in TerraClimate_CWD_files if Path(f).match(pattern)]
    TerraClimate_CWD_files_filtered.extend(CWD_matched_files)
    SoilM_matched_files = [f for f in TerraClimate_SoilM_files if Path(f).match(pattern)]
    TerraClimate_SoilM_files_filtered.extend(SoilM_matched_files)
    PET_matched_files = [f for f in TerraClimate_PET_files if Path(f).match(pattern)]
    TerraClimate_PET_files_filtered.extend(PET_matched_files)

# Load GRIDMET VPD
VPD_xr_raw = data_utils.load_files_to_xr(GRIDMET_VPD_files_filtered)
VPD_xr = VPD_xr_raw['mean_vapor_pressure_deficit']
VPD_xr = VPD_xr.sortby(['y', 'x'], ascending=True)
VPD_xr.rio.write_crs(4326, inplace=True)

# Interpolate VPD_xr to match NDVI_xr time dimension
VPD_xr_interp = VPD_xr.interp(time=dates, method='linear')
# Add year coordinate for grouping
VPD_xr_interp['year'] = VPD_xr_interp['time.year']


# Load TerraClimate CWD
CWD_xr_raw = data_utils.load_files_to_xr(TerraClimate_CWD_files_filtered)
CWD_xr = CWD_xr_raw['def']
CWD_xr = CWD_xr.sortby(['y', 'x'], ascending=True)
CWD_xr.rio.write_crs(4326, inplace=True)

# Interpolate CWD_xr to match NDVI_xr time dimension
CWD_xr_interp = CWD_xr.interp(time=dates, method='linear')
# Add year coordinate for grouping
CWD_xr_interp['year'] = CWD_xr_interp['time.year']



# Load TerraClimate SoilM
SoilM_xr_raw = data_utils.load_files_to_xr(TerraClimate_SoilM_files_filtered)
SoilM_xr = SoilM_xr_raw['soil']
SoilM_xr = SoilM_xr.sortby(['y', 'x'], ascending=True)
SoilM_xr.rio.write_crs(4326, inplace=True)
# Interpolate SoilM_xr to match NDVI_xr time dimension
SoilM_xr_interp = SoilM_xr.interp(time=dates, method='linear')
# Add year coordinate for grouping
SoilM_xr_interp['year'] = SoilM_xr_interp['time.year']


# Load TerraClimate PET
PET_xr_raw = data_utils.load_files_to_xr(TerraClimate_PET_files_filtered)
PET_xr = PET_xr_raw['pet']
PET_xr = PET_xr.sortby(['y', 'x'], ascending=True)
PET_xr.rio.write_crs(4326, inplace=True)
# Interpolate PET_xr to match NDVI_xr time dimension
PET_xr_interp = PET_xr.interp(time=dates, method='linear')
# Add year coordinate for grouping
PET_xr_interp['year'] = PET_xr_interp['time.year']



#####################################
### Perform annual averaging
#####################################


#%%

### Annualize nClimGrid SPEI
Path(nClimGrid_SPEI_out_dir).mkdir(parents=True, exist_ok=True)
for ii, annualize_months_abbr in enumerate(annualize_months_abbrs):
    print(f'Annualizing months: {annualize_months_abbr}...')

    outfile = os.path.join(nClimGrid_SPEI_out_dir, f'nClimGrid_SPEI_01month_05d_annual_{annualize_months_abbr}.nc')

    annualize_months = annualize_months_dict[annualize_months_abbr]
    SPEI_months_mask = np.isin(pd.to_datetime(nClimGrid_SPEI_01_xr_interp['time'].values).month, annualize_months)
    SPEI_xr_interp_mask = nClimGrid_SPEI_01_xr_interp.sel(time=SPEI_months_mask)
    SPEI_xr_annual_mean = SPEI_xr_interp_mask.groupby('year').mean(skipna=True)  # group by year and take mean

    # define global attributes
    SPEI_xr_annual_mean.attrs['averaging_window'] = annualize_months_descrs[ii]
    SPEI_xr_annual_mean.attrs['Processed_by'] = 'Lewis Kunik, lewis.kunik@utah.edu'
    SPEI_xr_annual_mean.attrs['Date_created'] = str(dt.today().date())

    # Finally, save to netcdf
    SPEI_xr_annual_mean.to_netcdf(path=outfile, format="NETCDF4")
    SPEI_xr_annual_mean.close()

print('done annualizing nClimGrid SPEI-01')

#%%
for ii, annualize_months_abbr in enumerate(annualize_months_abbrs):
    print(f'Annualizing months: {annualize_months_abbr}...')

    outfile = os.path.join(nClimGrid_SPEI_out_dir, f'nClimGrid_SPEI_12month_05d_annual_{annualize_months_abbr}.nc')

    annualize_months = annualize_months_dict[annualize_months_abbr]
    SPEI_months_mask = np.isin(pd.to_datetime(nClimGrid_SPEI_12_xr_interp['time'].values).month, annualize_months)
    SPEI_xr_interp_mask = nClimGrid_SPEI_12_xr_interp.sel(time=SPEI_months_mask)
    SPEI_xr_annual_mean = SPEI_xr_interp_mask.groupby('year').mean(skipna=True)  # group by year and take mean

    # define global attributes
    SPEI_xr_annual_mean.attrs['averaging_window'] = annualize_months_descrs[ii]
    SPEI_xr_annual_mean.attrs['Processed_by'] = 'Lewis Kunik, lewis.kunik@utah.edu'
    SPEI_xr_annual_mean.attrs['Date_created'] = str(dt.today().date())

    # Finally, save to netcdf
    SPEI_xr_annual_mean.to_netcdf(path=outfile, format="NETCDF4")
    SPEI_xr_annual_mean.close()

print('done annualizing nClimGrid SPEI-12')

#%%
for ii, annualize_months_abbr in enumerate(annualize_months_abbrs):
    print(f'Annualizing months: {annualize_months_abbr}...')

    outfile = os.path.join(nClimGrid_SPEI_out_dir, f'nClimGrid_SPEI_36month_05d_annual_{annualize_months_abbr}.nc')

    annualize_months = annualize_months_dict[annualize_months_abbr]
    SPEI_months_mask = np.isin(pd.to_datetime(nClimGrid_SPEI_36_xr_interp['time'].values).month, annualize_months)
    SPEI_xr_interp_mask = nClimGrid_SPEI_36_xr_interp.sel(time=SPEI_months_mask)
    SPEI_xr_annual_mean = SPEI_xr_interp_mask.groupby('year').mean(skipna=True)  # group by year and take mean

    # define global attributes
    SPEI_xr_annual_mean.attrs['averaging_window'] = annualize_months_descrs[ii]
    SPEI_xr_annual_mean.attrs['Processed_by'] = 'Lewis Kunik, lewis.kunik@utah.edu'
    SPEI_xr_annual_mean.attrs['Date_created'] = str(dt.today().date())

    # Finally, save to netcdf
    SPEI_xr_annual_mean.to_netcdf(path=outfile, format="NETCDF4")
    SPEI_xr_annual_mean.close()

print('done annualizing nClimGrid SPEI-36')



#%%

### Annualize GRIDMET PDSI
Path(GRIDMET_PDSI_out_dir).mkdir(parents=True, exist_ok=True)
for ii, annualize_months_abbr in enumerate(annualize_months_abbrs):
    print(f'Annualizing months: {annualize_months_abbr}...')

    outfile = os.path.join(GRIDMET_PDSI_out_dir, f'pdsi_05d_annual_{annualize_months_abbr}.nc')

    annualize_months = annualize_months_dict[annualize_months_abbr]
    PDSI_months_mask = np.isin(pd.to_datetime(PDSI_xr_interp['time'].values).month, annualize_months)
    PDSI_xr_interp_mask = PDSI_xr_interp.sel(time=PDSI_months_mask)
    PDSI_xr_annual_mean = PDSI_xr_interp_mask.groupby('year').mean(skipna=True)  # group by year and take mean


    # define global attributes
    PDSI_xr_annual_mean.attrs['averaging_window'] = annualize_months_descrs[ii]
    PDSI_xr_annual_mean.attrs['Processed_by'] = 'Lewis Kunik, lewis.kunik@utah.edu'
    PDSI_xr_annual_mean.attrs['Date_created'] = str(dt.today().date())

    # Finally, save to netcdf
    PDSI_xr_annual_mean.to_netcdf(path=outfile, format="NETCDF4")
    PDSI_xr_annual_mean.close()

print('done annualizing GRIDMET PDSI')


#%%
### Annualize GRIDMET VPD
Path(GRIDMET_VPD_out_dir).mkdir(parents=True, exist_ok=True)
for ii, annualize_months_abbr in enumerate(annualize_months_abbrs):
    print(f'Annualizing months: {annualize_months_abbr}...')

    outfile = os.path.join(GRIDMET_VPD_out_dir, f'vpd_05d_annual_{annualize_months_abbr}.nc')

    annualize_months = annualize_months_dict[annualize_months_abbr]
    VPD_months_mask = np.isin(pd.to_datetime(VPD_xr_interp['time'].values).month, annualize_months)
    VPD_xr_interp_mask = VPD_xr_interp.sel(time=VPD_months_mask)
    VPD_xr_annual_mean = VPD_xr_interp_mask.groupby('year').mean(skipna=True)  # group by year and take mean


    # define global attributes
    VPD_xr_annual_mean.attrs['averaging_window'] = annualize_months_descrs[ii]
    VPD_xr_annual_mean.attrs['Processed_by'] = 'Lewis Kunik, lewis.kunik@utah.edu'
    VPD_xr_annual_mean.attrs['Date_created'] = str(dt.today().date())

    # Finally, save to netcdf
    VPD_xr_annual_mean.to_netcdf(path=outfile, format="NETCDF4")
    VPD_xr_annual_mean.close()

print('done annualizing GRIDMET VPD')


# %%

### Annualize TerraClimate CWD
Path(TerraClimate_CWD_out_dir).mkdir(parents=True, exist_ok=True)
for ii, annualize_months_abbr in enumerate(annualize_months_abbrs):
    print(f'Annualizing months: {annualize_months_abbr}...')

    outfile = os.path.join(TerraClimate_CWD_out_dir, f'TerraClimate_def_05d_annual_{annualize_months_abbr}.nc')

    annualize_months = annualize_months_dict[annualize_months_abbr]
    CWD_months_mask = np.isin(pd.to_datetime(CWD_xr_interp['time'].values).month, annualize_months)
    CWD_xr_interp_mask = CWD_xr_interp.sel(time=CWD_months_mask)
    CWD_xr_annual_mean = CWD_xr_interp_mask.groupby('year').mean(skipna=True)  # group by year and take mean


    # define global attributes
    CWD_xr_annual_mean.attrs['averaging_window'] = annualize_months_descrs[ii]
    CWD_xr_annual_mean.attrs['Processed_by'] = 'Lewis Kunik, lewis.kunik@utah.edu'
    CWD_xr_annual_mean.attrs['Date_created'] = str(dt.today().date())

    # Finally, save to netcdf
    CWD_xr_annual_mean.to_netcdf(path=outfile, format="NETCDF4")
    CWD_xr_annual_mean.close()

print('done annualizing TerraClimate CWD')
# %%
### Annualize TerraClimate SoilM
Path(TerraClimate_SoilM_out_dir).mkdir(parents=True, exist_ok=True)
for ii, annualize_months_abbr in enumerate(annualize_months_abbrs):
    print(f'Annualizing months: {annualize_months_abbr}...')

    outfile = os.path.join(TerraClimate_SoilM_out_dir, f'TerraClimate_soil_05d_annual_{annualize_months_abbr}.nc')

    annualize_months = annualize_months_dict[annualize_months_abbr]
    SoilM_months_mask = np.isin(pd.to_datetime(SoilM_xr_interp['time'].values).month, annualize_months)
    SoilM_xr_interp_mask = SoilM_xr_interp.sel(time=SoilM_months_mask)
    SoilM_xr_annual_mean = SoilM_xr_interp_mask.groupby('year').mean(skipna=True)  # group by year and take mean


    # define global attributes
    SoilM_xr_annual_mean.attrs['averaging_window'] = annualize_months_descrs[ii]
    SoilM_xr_annual_mean.attrs['Processed_by'] = 'Lewis Kunik, lewis.kunik@utah.edu'
    SoilM_xr_annual_mean.attrs['Date_created'] = str(dt.today().date())

    # Finally, save to netcdf
    SoilM_xr_annual_mean.to_netcdf(path=outfile, format="NETCDF4")
    SoilM_xr_annual_mean.close()

print('done annualizing TerraClimate SoilM')



# %%
### Annualize TerraClimate PET
Path(TerraClimate_PET_out_dir).mkdir(parents=True, exist_ok=True)
for ii, annualize_months_abbr in enumerate(annualize_months_abbrs):
    print(f'Annualizing months: {annualize_months_abbr}...')

    outfile = os.path.join(TerraClimate_PET_out_dir, f'TerraClimate_pet_05d_annual_{annualize_months_abbr}.nc')

    annualize_months = annualize_months_dict[annualize_months_abbr]
    PET_months_mask = np.isin(pd.to_datetime(PET_xr_interp['time'].values).month, annualize_months)
    PET_xr_interp_mask = PET_xr_interp.sel(time=PET_months_mask)
    PET_xr_annual_mean = PET_xr_interp_mask.groupby('year').mean(skipna=True)  # group by year and take mean


    # define global attributes
    PET_xr_annual_mean.attrs['averaging_window'] = annualize_months_descrs[ii]
    PET_xr_annual_mean.attrs['Processed_by'] = 'Lewis Kunik, lewis.kunik@utah.edu'
    PET_xr_annual_mean.attrs['Date_created'] = str(dt.today().date())

    # Finally, save to netcdf
    PET_xr_annual_mean.to_netcdf(path=outfile, format="NETCDF4")
    PET_xr_annual_mean.close()

print('done annualizing TerraClimate PET')
# %%
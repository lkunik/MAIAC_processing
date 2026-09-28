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
# stats packages
from scipy import stats

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

NLCD_TCC_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/NLCD/TreeCover/2011'
NLCD_TCC_file = os.path.join(NLCD_TCC_dir, 'NLCD_TCC_01d_WUS_2011.nc')

NLCD_Landcover_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/NLCD/LandCover/regrid'
NLCD_ENF_mask_file = os.path.join(NLCD_Landcover_dir, 'NLCD2008_evergreen_wus_01d.nc')

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

plot_dir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/drought/scatter_16day/QCfilt_{QC_option}/'
Path(plot_dir).mkdir(parents=True, exist_ok=True)

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

TCC_thresh = 80
ENF_fraction_thresh = 0.9
# avg_window_strs = ['MJJ', 'JJA', 'JAS', 'ASO']
avg_window_str = 'JAS'

analysis_years = np.arange(2000, 2025)
DOY_composite_starts = np.arange(1, 366, 16)
composite_dates_2020 = [dt(2020, 1, 1) + timedelta(days=int(DOY) - 1) for DOY in DOY_composite_starts]

annualize_months_dict = {'MJJ':[5, 6, 7],
                    'JJA':[6, 7, 8],
                    'JAS':[7, 8, 9],
                    'ASO':[8, 9, 10]}
#########################################
# Begin main
#########################################
#%%

NLCD_TCC_xr = xr.open_dataset(NLCD_TCC_file, decode_coords = 'all').squeeze('band')['tree_canopy_cover']
NLCD_ENF_fraction_xr = xr.open_dataset(NLCD_ENF_mask_file, decode_coords = 'all')['fraction_ENF']

# plt.figure(figsize=(8, 5))
# plt.hist(NLCD_TCC_xr.values.flatten(), bins=30, color='skyblue', edgecolor='black')
# plt.xlabel('Tree Canopy Cover (%)')
# plt.ylabel('Frequency')
# plt.title('Histogram of NLCD Tree Canopy Cover')
# plt.show()

NLCD_forest_mask = xr.ones_like(NLCD_ENF_fraction_xr)
NLCD_forest_mask = NLCD_forest_mask.where(NLCD_ENF_fraction_xr >= ENF_fraction_thresh, other=np.nan)
NLCD_forest_mask = NLCD_forest_mask.where(NLCD_TCC_xr >= TCC_thresh, other=np.nan)


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

# Create a mask for dates in months May (5) through October (10)
months_to_keep = annualize_months_dict[avg_window_str]
date_months = np.array([d.astype('datetime64[M]').astype(int) % 12 + 1 for d in dates])
date_mask = np.isin(date_months, months_to_keep)

MCD19_files_summer = np.array(MCD19_files)[date_mask]

MCD19_xr_arr = []

MCD19_file = MCD19_files_summer[0]
for MCD19_file in MCD19_files_summer:
    print(f'Loading MCD19 file: {MCD19_file}')
    dat_xr_raw = xr.open_dataset(MCD19_file, decode_coords = 'all').rename({'lon': 'x', 'lat': 'y'}).sortby(['x', 'y'], ascending=True)
    dat_xr_raw.rio.write_crs(4326, inplace=True)
    timestamp = pd.to_datetime(dat_xr_raw.time.values.item()).strftime('%Y-%m-%d')
    dat_xr_raw = dat_xr_raw.expand_dims(time = [timestamp])
    dat_xr_clip = dat_xr_raw.where(~np.isnan(NLCD_forest_mask), drop = False)
    del dat_xr_raw; gc.collect() # remove large variables from env to clear up memory
    MCD19_xr_arr.append(dat_xr_clip)
    del dat_xr_clip; gc.collect() # remove large variables from env to clear up memory    

MCD19_xr = xr.concat(MCD19_xr_arr, dim='time')
del MCD19_xr_arr; gc.collect() # remove large variables from env to clear up memory

#%%


# Load nClimGrid SPEI-01
nClimGrid_SPEI_01_xr = xr.open_dataset(nClimGrid_SPEI_01_file, decode_coords='all').rename({'lon':'x','lat':'y'})['SPEI_pearson']
nClimGrid_SPEI_01_xr = nClimGrid_SPEI_01_xr.sortby(['y', 'x'], ascending=True)
nClimGrid_SPEI_01_xr.rio.write_crs(4326, inplace=True)
nClimGrid_SPEI_01_xr_interp = nClimGrid_SPEI_01_xr.interp(time=dates, method='linear')
del nClimGrid_SPEI_01_xr; gc.collect() # remove large variables from env to clear up memory
nClimGrid_SPEI_01_xr_timeMask = nClimGrid_SPEI_01_xr_interp.isel(time=date_mask)
del nClimGrid_SPEI_01_xr_interp; gc.collect() # remove large variables from env to clear up memory
SPEI_01_xr = nClimGrid_SPEI_01_xr_timeMask.rio.reproject_match(MCD19_xr, resampling = Resampling.bilinear, nodata = np.nan) # bilinear regridding with rasterio
del nClimGrid_SPEI_01_xr_timeMask; gc.collect() # remove large variables from env to clear up memory


# Load nClimGrid SPEI-12
nClimGrid_SPEI_12_xr = xr.open_dataset(nClimGrid_SPEI_12_file, decode_coords='all').rename({'lon':'x','lat':'y'})['SPEI_pearson']
nClimGrid_SPEI_12_xr = nClimGrid_SPEI_12_xr.sortby(['y', 'x'], ascending=True)
nClimGrid_SPEI_12_xr.rio.write_crs(4326, inplace=True)
nClimGrid_SPEI_12_xr_interp = nClimGrid_SPEI_12_xr.interp(time=dates, method='linear')
del nClimGrid_SPEI_12_xr; gc.collect() # remove large variables from env to clear up memory
nClimGrid_SPEI_12_xr_timeMask = nClimGrid_SPEI_12_xr_interp.isel(time=date_mask)
del nClimGrid_SPEI_12_xr_interp; gc.collect() # remove large variables from env to clear up memory
SPEI_12_xr = nClimGrid_SPEI_12_xr_timeMask.rio.reproject_match(MCD19_xr, resampling = Resampling.bilinear, nodata = np.nan) # bilinear regridding with rasterio
del nClimGrid_SPEI_12_xr_timeMask; gc.collect() # remove large variables from env to clear up memory

# # Load nClimGrid SPEI-36
# nClimGrid_SPEI_36_xr = xr.open_dataset(nClimGrid_SPEI_36_file, decode_coords='all').rename({'lon':'x','lat':'y'})['SPEI_pearson']
# nClimGrid_SPEI_36_xr = nClimGrid_SPEI_36_xr.sortby(['y', 'x'], ascending=True)
# nClimGrid_SPEI_36_xr.rio.write_crs(4326, inplace=True)
# nClimGrid_SPEI_36_xr_interp = nClimGrid_SPEI_36_xr.interp(time=dates, method='linear')
# nClimGrid_SPEI_36_xr_timeMask = nClimGrid_SPEI_36_xr_interp.isel(time=date_mask)
# SPEI_36_xr = nClimGrid_SPEI_36_xr_timeMask.rio.reproject_match(MCD19_xr, resampling = Resampling.bilinear, nodata = np.nan) # bilinear regridding with rasterio


# Load GRIDMET PDSI
PDSI_xr_raw = data_utils.load_files_to_xr(GRIDMET_PDSI_files)
PDSI_xr_preproc = PDSI_xr_raw['pdsi']
PDSI_xr_preproc = PDSI_xr_preproc.sortby(['y', 'x'], ascending=True)
PDSI_xr_preproc.rio.write_crs(4326, inplace=True)
del PDSI_xr_raw; gc.collect() # remove large variables from env to clear up memory
# Interpolate PDSI_xr_regrid to match NDVI_xr time dimension
PDSI_xr_interp = PDSI_xr_preproc.interp(time=dates, method='linear')
del PDSI_xr_preproc; gc.collect() # remove large variables from env to clear up memory
PDSI_xr_timeMask = PDSI_xr_interp.isel(time=date_mask)
del PDSI_xr_interp; gc.collect() # remove large variables from env to clear up memory
PDSI_xr = PDSI_xr_timeMask.rio.reproject_match(MCD19_xr, resampling = Resampling.bilinear, nodata = np.nan) # bilinear regridding with rasterio
del PDSI_xr_timeMask; gc.collect() # remove large variables from env to clear up memory



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
VPD_xr_preproc = VPD_xr_raw['mean_vapor_pressure_deficit']
VPD_xr_preproc = VPD_xr_preproc.sortby(['y', 'x'], ascending=True)
VPD_xr_preproc.rio.write_crs(4326, inplace=True)
del VPD_xr_raw; gc.collect() # remove large variables from env to clear up memory
# Interpolate VPD_xr to match NDVI_xr time dimension
VPD_xr_interp = VPD_xr_preproc.interp(time=dates, method='linear')
del VPD_xr_preproc; gc.collect() # remove large variables from env to clear up memory
VPD_xr_timeMask = VPD_xr_interp.isel(time=date_mask)
del VPD_xr_interp; gc.collect() # remove large variables from env to clear up memory
VPD_xr = VPD_xr_timeMask.rio.reproject_match(MCD19_xr, resampling = Resampling.bilinear, nodata = np.nan) # bilinear regridding with rasterio
del VPD_xr_timeMask; gc.collect() # remove large variables from env to clear up memory

# Load TerraClimate CWD
CWD_xr_raw = data_utils.load_files_to_xr(TerraClimate_CWD_files_filtered)
CWD_xr_preproc = CWD_xr_raw['def']
CWD_xr_preproc = CWD_xr_preproc.sortby(['y', 'x'], ascending=True)
CWD_xr_preproc.rio.write_crs(4326, inplace=True)
del CWD_xr_raw; gc.collect() # remove large variables from env to clear up memory
# Interpolate CWD_xr to match NDVI_xr time dimension
CWD_xr_interp = CWD_xr_preproc.interp(time=dates, method='linear')
del CWD_xr_preproc; gc.collect() # remove large variables from env to clear up memory
CWD_xr_timeMask = CWD_xr_interp.isel(time=date_mask)
del CWD_xr_interp; gc.collect() # remove large variables from env to clear up memory
CWD_xr = CWD_xr_timeMask.rio.reproject_match(MCD19_xr, resampling = Resampling.bilinear, nodata = np.nan) # bilinear regridding with rasterio
del CWD_xr_timeMask; gc.collect() # remove large variables from env to clear up memory

# Load TerraClimate SoilM
SoilM_xr_raw = data_utils.load_files_to_xr(TerraClimate_SoilM_files_filtered)
SoilM_xr_preproc = SoilM_xr_raw['soil']
SoilM_xr_preproc = SoilM_xr_preproc.sortby(['y', 'x'], ascending=True)
SoilM_xr_preproc.rio.write_crs(4326, inplace=True)
del SoilM_xr_raw; gc.collect() # remove large variables from env to clear up memory
# Interpolate SoilM_xr to match NDVI_xr time dimension
SoilM_xr_interp = SoilM_xr_preproc.interp(time=dates, method='linear')
del SoilM_xr_preproc; gc.collect() # remove large variables from env to clear up memory
SoilM_xr_timeMask = SoilM_xr_interp.isel(time=date_mask)
del SoilM_xr_interp; gc.collect() # remove large variables from env to clear up memory
SoilM_xr = SoilM_xr_timeMask.rio.reproject_match(MCD19_xr, resampling = Resampling.bilinear, nodata = np.nan) # bilinear regridding with rasterio
del SoilM_xr_timeMask; gc.collect() # remove large variables from env to clear up memory

# Load TerraClimate PET
PET_xr_raw = data_utils.load_files_to_xr(TerraClimate_PET_files_filtered)
PET_xr_preproc = PET_xr_raw['pet']
PET_xr_preproc = PET_xr_preproc.sortby(['y', 'x'], ascending=True)
PET_xr_preproc.rio.write_crs(4326, inplace=True)
del PET_xr_raw; gc.collect() # remove large variables from env to clear up memory

# Interpolate PET_xr to match NDVI_xr time dimension
PET_xr_interp = PET_xr_preproc.interp(time=dates, method='linear')
del PET_xr_preproc; gc.collect() # remove large variables from env to clear up memory
PET_xr_timeMask = PET_xr_interp.isel(time=date_mask)
del PET_xr_interp; gc.collect() # remove large variables from env to clear up memory
PET_xr = PET_xr_timeMask.rio.reproject_match(MCD19_xr, resampling = Resampling.bilinear, nodata = np.nan) # bilinear regridding with rasterio
del PET_xr_timeMask; gc.collect() # remove large variables from env to clear up memory

# %%
### mask out non-forest areas
MCD19_xr_mask = MCD19_xr
SPEI_01_xr_mask = SPEI_01_xr.where(~np.isnan(NLCD_forest_mask), drop = False)
SPEI_12_xr_mask = SPEI_12_xr.where(~np.isnan(NLCD_forest_mask), drop = False)
# SPEI_36_xr_mask = SPEI_36_xr.where(~np.isnan(NLCD_forest_mask), drop = False)
PDSI_xr_mask = PDSI_xr.where(~np.isnan(NLCD_forest_mask), drop = False)
VPD_xr_mask = VPD_xr.where(~np.isnan(NLCD_forest_mask), drop = False)
CWD_xr_mask = CWD_xr.where(~np.isnan(NLCD_forest_mask), drop = False)
SoilM_xr_mask = SoilM_xr.where(~np.isnan(NLCD_forest_mask), drop = False)
PET_xr_mask = PET_xr.where(~np.isnan(NLCD_forest_mask), drop = False)

#%%

bins_2dhist = 100

drought_xr_arr = [SPEI_01_xr_mask, SPEI_12_xr_mask, PDSI_xr_mask, VPD_xr_mask, CWD_xr_mask, SoilM_xr_mask, PET_xr_mask]
drought_names = ['SPEI_01', 'SPEI_12', 'PDSI', 'VPD', 'CWD', 'SoilM', 'PET']
drought_labels = ['SPEI 1-month', 'SPEI 12-month', 'PDSI', 'VPD (kPa)', 'CWD (mm)', 'Soil Moisture (mm)', 'PET (mm)']

for drought_xr, drought_name, drought_label in zip(drought_xr_arr, drought_names, drought_labels):

    plot_file = os.path.join(plot_dir, f'MCD19_VIs_16day_vs_{drought_name}_scatter_{avg_window_str}.png')
    # if os.path.exists(plot_file):
    #     print(f'Plot file {plot_file} already exists, skipping...')
    #     continue
    
    print(f'Correlating VIs vs. {drought_name} for avg window {avg_window_str}...')


    # CCI vs drought
    drought_flat = drought_xr.values.flatten()
    cci_flat = MCD19_xr_mask['CCI'].values.flatten()
    mask_cci = ~np.isnan(drought_flat) & ~np.isnan(cci_flat)
    drought_cci = drought_flat[mask_cci].reshape(-1, 1)
    cci_clean = cci_flat[mask_cci]

    reg_cci = LinearRegression().fit(drought_cci, cci_clean)
    cci_pred = reg_cci.predict(drought_cci)
    r2_cci = r2_score(cci_clean, cci_pred)
    spearman_cci, p_value = stats.spearmanr(drought_cci.flatten(), cci_clean)

    # 2D Histogram Plotting with 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)

    h1 = axes[0, 0].hist2d(drought_cci.flatten(), cci_clean, bins=bins_2dhist, cmap='bone_r')
    axes[0, 0].set_xlabel(drought_label)
    axes[0, 0].set_ylabel('CCI')
    axes[0, 0].set_title(f'{drought_name} vs. CCI\nR²={r2_cci:.3f}\nSpearman r={spearman_cci:.3f}')
    axes[0, 0].grid(True)
    axes[0, 0].set_ylim(-0.5, 0.5)
    axes[0, 0].plot(drought_cci.flatten(), cci_pred, color='red', linewidth=2, alpha=0.8)
    cb2 = plt.colorbar(h1[3], ax=axes[0, 0], fraction=0.035, pad=0.04)
    cb2.ax.tick_params(labelsize=12)
    cb2.set_label('Counts', rotation=0, labelpad=10, fontsize=14, y=1.1,x=0.01, ha='center')

    # delete memory-intensive variables
    del cci_flat, mask_cci, drought_cci, cci_clean, cci_pred
    gc.collect()

    # NDVI vs drought
    ndvi_flat = MCD19_xr_mask['NDVI'].values.flatten()
    mask_ndvi = ~np.isnan(drought_flat) & ~np.isnan(ndvi_flat)
    drought_ndvi = drought_flat[mask_ndvi].reshape(-1, 1)
    ndvi_clean = ndvi_flat[mask_ndvi]

    reg_ndvi = LinearRegression().fit(drought_ndvi, ndvi_clean)
    ndvi_pred = reg_ndvi.predict(drought_ndvi)
    r2_ndvi = r2_score(ndvi_clean, ndvi_pred)
    spearman_ndvi, p_value = stats.spearmanr(drought_ndvi.flatten(), ndvi_clean)


    h2 = axes[0, 1].hist2d(drought_ndvi.flatten(), ndvi_clean, bins=bins_2dhist, cmap='bone_r')
    axes[0, 1].set_xlabel(drought_label)
    axes[0, 1].set_ylabel('NDVI')
    axes[0, 1].set_title(f'{drought_name} vs. NDVI\nR²={r2_ndvi:.3f}\nSpearman r={spearman_ndvi:.3f}')
    axes[0, 1].grid(True)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].plot(drought_ndvi.flatten(), ndvi_pred, color='red', linewidth=2, alpha=0.8)
    cb1 = plt.colorbar(h2[3], ax=axes[0, 1], fraction=0.035, pad=0.04)
    cb1.ax.tick_params(labelsize=12)
    cb1.set_label('Counts', rotation=0, labelpad=10, fontsize=14, y=1.1,x=0.01, ha='center')

    # delete memory-intensive variables
    del drought_ndvi, ndvi_flat, mask_ndvi, ndvi_clean, ndvi_pred
    gc.collect()

    # NDMI vs drought
    ndmi_flat = MCD19_xr_mask['NDMI'].values.flatten()
    mask_ndmi = ~np.isnan(drought_flat) & ~np.isnan(ndmi_flat)
    drought_ndmi = drought_flat[mask_ndmi].reshape(-1, 1)
    ndmi_clean = ndmi_flat[mask_ndmi]

    reg_ndmi = LinearRegression().fit(drought_ndmi, ndmi_clean)
    ndmi_pred = reg_ndmi.predict(drought_ndmi)
    r2_ndmi = r2_score(ndmi_clean, ndmi_pred)
    spearman_ndmi, p_value = stats.spearmanr(drought_ndmi.flatten(), ndmi_clean)

    h3 = axes[1, 0].hist2d(drought_ndmi.flatten(), ndmi_clean, bins=bins_2dhist, cmap='bone_r')
    axes[1, 0].set_xlabel(drought_label)
    axes[1, 0].set_ylabel('NDMI')
    axes[1, 0].set_title(f'{drought_name} vs. NDMI\nR²={r2_ndmi:.3f}\nSpearman r={spearman_ndmi:.3f}')
    axes[1, 0].grid(True)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].plot(drought_ndmi.flatten(), ndmi_pred, color='red', linewidth=2, alpha=0.8)
    cb3 = plt.colorbar(h3[3], ax=axes[1, 0], fraction=0.035, pad=0.04)
    cb3.ax.tick_params(labelsize=12)
    cb3.set_label('Counts', rotation=0, labelpad=10, fontsize=14, y=1.1,x=0.01, ha='center')

    # delete memory-intensive variables
    del drought_ndmi, ndmi_flat, mask_ndmi, ndmi_clean, ndmi_pred
    gc.collect()

    # NIRv vs drought
    nirv_flat = MCD19_xr_mask['NIRv'].values.flatten()
    mask_nirv = ~np.isnan(drought_flat) & ~np.isnan(nirv_flat)
    drought_nirv = drought_flat[mask_nirv].reshape(-1, 1)
    nirv_clean = nirv_flat[mask_nirv]

    reg_nirv = LinearRegression().fit(drought_nirv, nirv_clean)
    nirv_pred = reg_nirv.predict(drought_nirv)
    r2_nirv = r2_score(nirv_clean, nirv_pred)
    spearman_nirv, p_value = stats.spearmanr(drought_nirv.flatten(), nirv_clean)

    h4 = axes[1, 1].hist2d(drought_nirv.flatten(), nirv_clean, bins=bins_2dhist, cmap='bone_r')
    axes[1, 1].set_xlabel(drought_label)
    axes[1, 1].set_ylabel('NIRv')
    axes[1, 1].set_title(f'{drought_name} vs. NIRv\nR²={r2_nirv:.3f}\nSpearman r={spearman_nirv:.3f}')
    axes[1, 1].grid(True)
    axes[1, 1].set_ylim(0, 0.3)
    axes[1, 1].plot(drought_nirv.flatten(), nirv_pred, color='red', linewidth=2, alpha=0.8)
    cb4 = plt.colorbar(h4[3], ax=axes[1, 1], fraction=0.035, pad=0.04)
    cb4.ax.tick_params(labelsize=12)
    cb4.set_label('Counts', rotation=0, labelpad=10, fontsize=14, y=1.1,x=0.01, ha='center')

    # delete memory-intensive variables
    del drought_nirv, nirv_flat, mask_nirv, nirv_clean, nirv_pred
    gc.collect()

    fig.suptitle(f'MCD19A1 VI-drought comparisons: ENF fraction > {ENF_fraction_thresh}, TCC > {TCC_thresh}%', fontsize=16, y=1.02)
    plt.tight_layout()

    plt.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)
# %%

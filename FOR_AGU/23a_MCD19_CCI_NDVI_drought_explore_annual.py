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

QC_option = 'Cloud'
MCD19_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/01d_rect/QCfilt_{QC_option}/WUS_combined/annualized/'

NLCD_TCC_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/NLCD/TreeCover/2011'
NLCD_TCC_file = os.path.join(NLCD_TCC_dir, 'NLCD_TCC_01d_WUS_2011.nc')

NLCD_Landcover_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/NLCD/LandCover/regrid'
NLCD_ENF_mask_file = os.path.join(NLCD_Landcover_dir, 'NLCD2008_evergreen_wus_01d.nc')

nClimGrid_SPEI_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/nClimGrid/SPEI/05d/annualized/'
GRIDMET_VPD_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/GRIDMET/VPD/05d/annualized/'
GRIDMET_PDSI_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/GRIDMET/PDSI/05d/annualized/'
TerraClimate_CWD_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/TerraClimate/CWD/05d/annualized/'
TerraClimate_SoilM_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/TerraClimate/SoilM/05d/annualized/'
TerraClimate_PET_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/TerraClimate/PET/05d/annualized/'


plot_dir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/drought/scatter_annual/QCfilt_{QC_option}/'
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
avg_window_strs = ['MJJ', 'JJA', 'JAS', 'ASO']

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

#%%
NLCD_forest_mask = xr.ones_like(NLCD_ENF_fraction_xr)
NLCD_forest_mask = NLCD_forest_mask.where(NLCD_ENF_fraction_xr >= ENF_fraction_thresh, other=np.nan)
NLCD_forest_mask = NLCD_forest_mask.where(NLCD_TCC_xr >= TCC_thresh, other=np.nan)

# plt.figure(figsize=(10, 6))
# NLCD_forest_mask.plot(cmap='Greens', add_colorbar=True)
# plt.title(f'NLCD Forest Mask (ENF fraction > {ENF_fraction_thresh}, TCC > {TCC_thresh}%)')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.tight_layout()
# plt.show()

print(f'Number of forest pixels: {np.sum(~np.isnan(NLCD_forest_mask.values))}')


plt.rcParams.update({
'axes.labelsize': 18,
'axes.titlesize': 16,
'xtick.labelsize': 16,
'ytick.labelsize': 15,
'legend.fontsize': 13,
})

bins_2dhist = 100

#%%

# for avg_window_str in avg_window_strs:
avg_window_str = 'JAS'

MCD19_file = glob.glob(f'{MCD19_dir}/MCD19A1.061_01d_annual_{avg_window_str}_CV-MVC_QCfilt_{QC_option}_WUS.nc')[0]
MCD19_xr = xr.open_dataset(MCD19_file, decode_coords = 'all')

SPEI_01_file = glob.glob(f'{nClimGrid_SPEI_dir}/*01month_*_{avg_window_str}.nc')[0]
SPEI_01_xr_raw = xr.open_dataset(SPEI_01_file, decode_coords = 'all')['SPEI_pearson']

SPEI_12_file = glob.glob(f'{nClimGrid_SPEI_dir}/*12month_*_{avg_window_str}.nc')[0]
SPEI_12_xr_raw = xr.open_dataset(SPEI_12_file, decode_coords = 'all')['SPEI_pearson']

SPEI_36_file = glob.glob(f'{nClimGrid_SPEI_dir}/*36month_*_{avg_window_str}.nc')[0]
SPEI_36_xr_raw = xr.open_dataset(SPEI_36_file, decode_coords = 'all')['SPEI_pearson']

PDSI_file = glob.glob(f'{GRIDMET_PDSI_dir}/*_{avg_window_str}.nc')[0]
PDSI_xr_raw = xr.open_dataset(PDSI_file, decode_coords = 'all')['pdsi']

VPD_file = glob.glob(f'{GRIDMET_VPD_dir}/*_{avg_window_str}.nc')[0]
VPD_xr_raw = xr.open_dataset(VPD_file, decode_coords = 'all')['mean_vapor_pressure_deficit']

CWD_file = glob.glob(f'{TerraClimate_CWD_dir}/*_{avg_window_str}.nc')[0]
CWD_xr_raw = xr.open_dataset(CWD_file, decode_coords = 'all')['def']

SoilM_file = glob.glob(f'{TerraClimate_SoilM_dir}/*_{avg_window_str}.nc')[0]
SoilM_xr_raw = xr.open_dataset(SoilM_file, decode_coords = 'all')['soil']

PET_file = glob.glob(f'{TerraClimate_PET_dir}/*_{avg_window_str}.nc')[0]
PET_xr_raw = xr.open_dataset(PET_file, decode_coords = 'all')['pet']
#%%
# Regrid the 0.05° drought indices to match 0.01° MCD19 grid using rasterio

SPEI_01_xr = SPEI_01_xr_raw.rio.reproject_match(MCD19_xr, resampling = Resampling.bilinear, nodata = np.nan) # bilinear regridding with rasterio
SPEI_12_xr = SPEI_12_xr_raw.rio.reproject_match(MCD19_xr, resampling = Resampling.bilinear, nodata = np.nan) # bilinear regridding with rasterio
SPEI_36_xr = SPEI_36_xr_raw.rio.reproject_match(MCD19_xr, resampling = Resampling.bilinear, nodata = np.nan) # bilinear regridding with
PDSI_xr = PDSI_xr_raw.rio.reproject_match(MCD19_xr, resampling = Resampling.bilinear, nodata = np.nan) # bilinear regridding with rasterio
VPD_xr = VPD_xr_raw.rio.reproject_match(MCD19_xr, resampling = Resampling.bilinear, nodata = np.nan) # bilinear regridding with rasterio
CWD_xr = CWD_xr_raw.rio.reproject_match(MCD19_xr, resampling = Resampling.bilinear, nodata = np.nan) # bilinear regridding with rasterio
SoilM_xr = SoilM_xr_raw.rio.reproject_match(MCD19_xr, resampling = Resampling.bilinear, nodata = np.nan) # bilinear regridding with rasterio
PET_xr = PET_xr_raw.rio.reproject_match(MCD19_xr, resampling = Resampling.bilinear, nodata = np.nan) # bilinear regridding with rasterio

### mask out non-forest areas
MCD19_xr_mask = MCD19_xr.where(~np.isnan(NLCD_forest_mask), drop = False)
SPEI_01_xr_mask = SPEI_01_xr.where(~np.isnan(NLCD_forest_mask), drop = False)
SPEI_12_xr_mask = SPEI_12_xr.where(~np.isnan(NLCD_forest_mask), drop = False)
SPEI_36_xr_mask = SPEI_36_xr.where(~np.isnan(NLCD_forest_mask), drop = False)
PDSI_xr_mask = PDSI_xr.where(~np.isnan(NLCD_forest_mask), drop = False)
VPD_xr_mask = VPD_xr.where(~np.isnan(NLCD_forest_mask), drop = False)
CWD_xr_mask = CWD_xr.where(~np.isnan(NLCD_forest_mask), drop = False)
SoilM_xr_mask = SoilM_xr.where(~np.isnan(NLCD_forest_mask), drop = False)
PET_xr_mask = PET_xr.where(~np.isnan(NLCD_forest_mask), drop = False)

#%%

drought_xr_arr = [SPEI_01_xr_mask, SPEI_12_xr_mask, SPEI_36_xr_mask, PDSI_xr_mask, VPD_xr_mask, CWD_xr_mask, SoilM_xr_mask, PET_xr_mask]
drought_names = ['SPEI_01', 'SPEI_12', 'SPEI_36', 'PDSI', 'VPD', 'CWD', 'SoilM', 'PET']
drought_labels = ['SPEI 1-month', 'SPEI 12-month', 'SPEI 36-month', 'PDSI', 'VPD (kPa)', 'CWD (mm)', 'Soil Moisture ()', 'PET (mm)']


drought_xr = SPEI_01_xr_mask
drought_name = 'SPEI_01'
drought_label = 'SPEI 1-month'

for drought_xr, drought_name, drought_label in zip(drought_xr_arr, drought_names, drought_labels):

    plot_file = os.path.join(plot_dir, f'MCD19_VIs_annual_vs_{drought_name}_scatter_{avg_window_str}.png')
    # if os.path.exists(plot_file):
    #     print(f'Plot file {plot_file} already exists, skipping...')
    #     continue
    
    print(f'Correlating VIs vs. {drought_name} for avg window {avg_window_str}...')
    # NDVI vs drought
    drought_flat = drought_xr.values.flatten()
    ndvi_flat = MCD19_xr_mask['NDVI'].values.flatten()
    mask_ndvi = ~np.isnan(drought_flat) & ~np.isnan(ndvi_flat)
    drought_ndvi = drought_flat[mask_ndvi].reshape(-1, 1)
    ndvi_clean = ndvi_flat[mask_ndvi]

    reg_ndvi = LinearRegression().fit(drought_ndvi, ndvi_clean)
    ndvi_pred = reg_ndvi.predict(drought_ndvi)
    r2_ndvi = r2_score(ndvi_clean, ndvi_pred)
    spearman_ndvi, p_value = stats.spearmanr(drought_ndvi.flatten(), ndvi_clean)


    # CCI vs drought
    cci_flat = MCD19_xr_mask['CCI'].values.flatten()
    mask_cci = ~np.isnan(drought_flat) & ~np.isnan(cci_flat)
    drought_cci = drought_flat[mask_cci].reshape(-1, 1)
    cci_clean = cci_flat[mask_cci]

    reg_cci = LinearRegression().fit(drought_cci, cci_clean)
    cci_pred = reg_cci.predict(drought_cci)
    r2_cci = r2_score(cci_clean, cci_pred)
    spearman_cci, p_value = stats.spearmanr(drought_cci.flatten(), cci_clean)

    # NDMI vs drought
    ndmi_flat = MCD19_xr_mask['NDMI'].values.flatten()
    mask_ndmi = ~np.isnan(drought_flat) & ~np.isnan(ndmi_flat)
    drought_ndmi = drought_flat[mask_ndmi].reshape(-1, 1)
    ndmi_clean = ndmi_flat[mask_ndmi]

    reg_ndmi = LinearRegression().fit(drought_ndmi, ndmi_clean)
    ndmi_pred = reg_ndmi.predict(drought_ndmi)
    r2_ndmi = r2_score(ndmi_clean, ndmi_pred)
    spearman_ndmi, p_value = stats.spearmanr(drought_ndmi.flatten(), ndmi_clean)

    # NIRv vs drought
    nirv_flat = MCD19_xr_mask['NIRv'].values.flatten()
    mask_nirv = ~np.isnan(drought_flat) & ~np.isnan(nirv_flat)
    drought_nirv = drought_flat[mask_nirv].reshape(-1, 1)
    nirv_clean = nirv_flat[mask_nirv]

    reg_nirv = LinearRegression().fit(drought_nirv, nirv_clean)
    nirv_pred = reg_nirv.predict(drought_nirv)
    r2_nirv = r2_score(nirv_clean, nirv_pred)
    spearman_nirv, p_value = stats.spearmanr(drought_nirv.flatten(), nirv_clean)

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

    fig.suptitle(f'MCD19A1 VI-drought comparisons: ENF fraction > {ENF_fraction_thresh}, TCC > {TCC_thresh}%', fontsize=16, y=1.02)
    plt.tight_layout()

    plt.savefig(plot_file, bbox_inches='tight')
    plt.close(fig)
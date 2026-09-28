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

# shapefile/geospatial packages
import geopandas as gpd
from shapely.geometry import mapping

# time and date packages
import time
from datetime import datetime as dt  # date time library
from datetime import timedelta

# plotting packages
import matplotlib
import pyreadr
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt  # primary plotting module

#%%

#########################################
# Define Global Filepaths
#########################################

# directories
QC_option = 'CloudAOD'
MCD19_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/01d_rect/QCfilt_{QC_option}/WUS_combined/'
all_MCD19_files = sorted(glob.glob(os.path.join(MCD19_dir, '*.nc')))

polygon_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/disturbance/data/flux_tower_polygons/'
plot_dir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/drought/flux_tower_scatter/QCfilt_{QC_option}/'
Path(plot_dir).mkdir(parents=True, exist_ok=True)
# eddy covariance data
EC_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/eddy_covariance/data/'

# nClimGrid_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/nClimGrid/05d'
# nClimGrid_SPEI_01_file = os.path.join(nClimGrid_dir, 'nClimGrid_SPEI_01month_05d_WUS_monthly_2000-2024.nc')
# nClimGrid_SPEI_12_file = os.path.join(nClimGrid_dir, 'nClimGrid_SPEI_12month_05d_WUS_monthly_2000-2024.nc')
# nClimGrid_SPEI_36_file = os.path.join(nClimGrid_dir, 'nClimGrid_SPEI_36month_05d_WUS_monthly_2000-2024.nc')

# GRIDMET_PDSI_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/GRIDMET/PDSI/05d'
# GRIDMET_PDSI_files = sorted(glob.glob(os.path.join(GRIDMET_PDSI_dir, '*.nc')))


########################################
# Define Global Variables and constants
#########################################
#%%

### Unit conversion:
secs_per_day = 3600*24 # number of seconds in a day
kg_per_g = 0.001 # conversion factor from grams to kilograms
umol_to_mol = 1e-6 # conversion factor from micromoles to moles
gC_per_mol = 12.01 # grams of carbon per mole of CO2
m2_to_ha = 1e-4 # conversion factor from m^2 to hectares

umol_per_m2_per_s_to_gC_per_m2_per_day = (secs_per_day * umol_to_mol * gC_per_mol) # conversion factor from umol/m^2/s to gC/ha/16d

flux_tower_lon = -119.2566
flux_tower_lat = 37.0311

analysis_years = np.arange(2009, 2018)

VI_vars = ['NDVI', 'CCI', 'NDMI', 'NIRv']
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

# Compose a list of dates from the MCD19 filenames
dates = []
for fname in all_MCD19_files:
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

# Filter dates for those matching analysis_years
dates_filtered_idx = [i for i, d in enumerate(dates) if d.astype('datetime64[Y]').astype(int) + 1970 in analysis_years]
dates = dates[dates_filtered_idx]
MCD19_files = [all_MCD19_files[i] for i in dates_filtered_idx]

# Load MCD19_VIs at the flux tower location
MCD19_xr_arr = []

# MCD19_file = MCD19_files[0]
for ii, MCD19_file in enumerate(MCD19_files):
    print(f'Loading MCD19 file: {MCD19_file}')
    dat_xr_raw = xr.open_dataset(MCD19_file, decode_coords = 'all')[VI_vars]
    dat_xr_raw.rio.write_crs(4326, inplace=True)
    # timestamp = pd.to_datetime(dat_xr_raw.time.values.item().strftime('%Y-%m-%d'))
    timestamp = pd.to_datetime(dates[ii])
    dat_xr_raw = dat_xr_raw.expand_dims(time = [timestamp])
    dat_xr_point = dat_xr_raw.sel(lat=flux_tower_lat, lon=flux_tower_lon, method='nearest')
    MCD19_xr_arr.append(dat_xr_point)

for i in range(len(MCD19_xr_arr)):
    ds = MCD19_xr_arr[i]
    # Drop VZA_rank if it's a variable
    if 'VZA_rank' in ds.variables:
        ds = ds.drop_vars('VZA_rank')
    # Drop VZA_rank if it's a coordinate
    if 'VZA_rank' in ds.coords:
        ds = ds.drop_vars('VZA_rank')
    MCD19_xr_arr[i] = ds

MCD19_xr = xr.concat(MCD19_xr_arr, dim='time')

# MCD19_xr_monthly = MCD19_xr.resample(time='1M').mean()



#%%

# Read the US-CZ2 RDS file
rds_file = os.path.join(EC_dir, 'AMF_US-CZ2_flux_part_s2010-2016_df.rds')
result = pyreadr.read_r(rds_file)

# Extract the dataframe
df = result[None]


# # Display the column names in the dataframe
for col in df.columns:
   print(col)

# Convert posix_time to datetime
# df['datetime'] = pd.to_datetime(df['posix_time'], unit='s')

# # Set datetime as the index
# df.set_index('datetime', inplace=True)
# Convert posix_time to numpy datetime
df['datetime'] = pd.to_datetime(df['posix_time'], unit='s')
df.set_index('datetime', inplace=True)

# Filter the dataframe for specific columns
df = df[["posix_time", "NEE_U50_f", "GPP_U50_f", "Reco_U50"]]

# Resample to 16-day intervals using mean interpolation
df_resampled = df.resample('16D').mean()

# # Interpolate the flux tower dataframe to match MCD19_xr.time
# df_resampled = df.reindex(pd.to_datetime(MCD19_xr.time.values), method=None)
# df_resampled.index.name = 'datetime'

# Reset the index to get datetime back as a column
df_resampled.reset_index(inplace=True)

# Convert fluxes from umol/m2/s to gC/ha/16d
df_resampled["NEE_U50_f"] = df_resampled["NEE_U50_f"] * umol_per_m2_per_s_to_gC_per_m2_per_day
df_resampled["Reco_U50"] = df_resampled["Reco_U50"] * umol_per_m2_per_s_to_gC_per_m2_per_day
df_resampled["GPP_U50_f"] = df_resampled["GPP_U50_f"] * umol_per_m2_per_s_to_gC_per_m2_per_day

# Display the resampled dataframe
print(df_resampled)


#%%
# Plot GPP_U50_f vs. time
print('plotting GPP_U50_f vs. time')

fig, ax = plt.subplots(figsize=(10, 3))  # establish the figure, axes
ax.grid(color="gray", linestyle="--", linewidth=1, alpha=0.5)  # add grid lines

# plot the GPP_U50_f field as time series
ax.plot(df_resampled['datetime'], df_resampled['GPP_U50_f'], color='black', label='GPP_U50_f', zorder=4)

# Define the axis, title labels
ax.set(title='GPP_U50_f vs. Time', ylabel='GPP_U50_f', xlabel='Time (UTC)')
ax.set_xlim(df_resampled['datetime'].min(), df_resampled['datetime'].max())

# add legend
fig.legend(loc="upper left", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

plt.show()


#%%

#########################################
# Plot MCD19 time series
#########################################

# Interpolate MCD19_xr to match the datetime column of df_resampled
MCD19_xr_interp = MCD19_xr.interp(time=pd.to_datetime(df_resampled['datetime'].values))


# Plot the interpolated monthly time series
fig, ax = plt.subplots(figsize=(10, 3)) # establish the figure, axes
ax.grid(color = "gray", linestyle = "--", linewidth = 1, alpha = 0.5) # add grid lines

# plot the desired variable as time series
ax.plot(MCD19_xr_interp.time, MCD19_xr_interp['NDVI'], color='black', 
      label = 'NDVI', zorder=4)

# Define the axis, title labels
ax.set(title=f'MCD19A1 monthly NDVI at CZ2', ylabel='NDVI', xlabel='Time (UTC)')
ax.set_xlim(dt(2010, 1, 1), dt(2016, 12, 31))

# add legend including the shaded error region
fig.legend(loc="upper left", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
plt.show()


# Plot the interpolated monthly time series
fig, ax = plt.subplots(figsize=(10, 3)) # establish the figure, axes
ax.grid(color = "gray", linestyle = "--", linewidth = 1, alpha = 0.5) # add grid lines

# plot the desired variable as time series
ax.plot(MCD19_xr_interp.time, MCD19_xr_interp['CCI'], color='black', 
      label = 'CCI', zorder=4)

# Define the axis, title labels
ax.set(title=f'MCD19A1 monthly CCI at CZ2', ylabel='CCI', xlabel='Time (UTC)')
ax.set_xlim(dt(2010, 1, 1), dt(2016, 12, 31))

# add legend including the shaded error region
fig.legend(loc="upper left", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
plt.show()

#%%


plt.rcParams.update({
'axes.labelsize': 18,
'axes.titlesize': 16,
'xtick.labelsize': 16,
'ytick.labelsize': 15,
'legend.fontsize': 13,
})


# Scatterplot of NDVI vs. GPP_U50_f with fit line and correlation stats


# Prepare data
x = MCD19_xr_interp['NDVI'].values
y = df_resampled['GPP_U50_f'].values

# Remove NaNs
mask = ~np.isnan(x) & ~np.isnan(y)
x_clean = x[mask].reshape(-1, 1)
y_clean = y[mask]

# Fit linear regression
reg = LinearRegression().fit(x_clean, y_clean)
y_pred = reg.predict(x_clean)
r2 = reg.score(x_clean, y_clean)

# Spearman correlation
spearman_r, _ = spearmanr(x_clean.flatten(), y_clean)

# Plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x_clean, y_clean, color='blue', alpha=0.6, label='Data')
ax.plot(x_clean, y_pred, color='red', label='Fit line')
ax.set_xlabel('MCD19 NDVI')
ax.set_ylabel('US-CZ2 GPP (gC/m2/day)')
ax.set_title(f'NDVI vs. US-CZ2 GPP\n$R^2$={r2:.2f}, Spearman r={spearman_r:.2f}')
ax.grid(True)
plt.show()



#%%
### CCI vs GPP

# Prepare data
CCI_vals = MCD19_xr_interp['CCI'].values
GPP_vals = df_resampled['GPP_U50_f'].values

# Remove NaNs
mask = ~np.isnan(CCI_vals) & ~np.isnan(GPP_vals)
x_clean = CCI_vals[mask].reshape(-1, 1)
y_clean = GPP_vals[mask]


# Fit linear regression
reg_CCI = LinearRegression().fit(x_clean, y_clean)
GPP_pred = reg_CCI.predict(x_clean)
r2 = reg_CCI.score(x_clean, y_clean)

# Spearman correlation
spearman_r, _ = spearmanr(x_clean.flatten(), y_clean)


# Plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x_clean, y_clean, color='blue', alpha=0.6, label='Data')
ax.plot(x_clean, GPP_pred, color='red', label='Fit line')
ax.set_xlabel('MCD19 CCI')
ax.set_ylabel('US-CZ2 GPP (gC/m2/day)')
ax.set_title(f'CCI vs. US-CZ2 GPP\n$R^2$={r2:.2f}, Spearman r={spearman_r:.2f}')
ax.grid(True)
plt.show()


#%%


plt.rcParams.update({
'axes.labelsize': 18,
'axes.titlesize': 16,
'xtick.labelsize': 16,
'ytick.labelsize': 17,
'legend.fontsize': 13,
})


# Prepare data for all VIs
VI_labels = ['CCI', 'NDVI', 'NDMI', 'NIRv']
VI_titles = ['CCI vs. GPP', 'NDVI vs. GPP', 'NDMI vs. GPP', 'NIRv vs. GPP']
VI_ylims = [None, None, None, None]  # Set to None or custom limits if desired

i = 2
vi = 'NDMI'

# for i, vi in enumerate(VI_labels):
x = MCD19_xr_interp[vi].values
y = df_resampled['GPP_U50_f'].values
mask = ~np.isnan(x) & ~np.isnan(y)
x_clean = x[mask].reshape(-1, 1)
y_clean = y[mask]

# Linear regression
reg = LinearRegression().fit(x_clean, y_clean)
y_pred = reg.predict(x_clean)
r2 = reg.score(x_clean, y_clean)
spearman_r, _ = spearmanr(x_clean.flatten(), y_clean)
print(f'{VI_titles[i]}\n$R^2$={r2:.2f}, Spearman r={spearman_r:.2f}')

fig, ax = plt.subplots(figsize=(4, 3.5))  # establish the figure, axes
ax.scatter(x_clean, y_clean, color='blue', alpha=0.6, label='Data')
ax.plot(x_clean, y_pred, color='red', label='Fit line')
ax.set_xlabel(f'{vi}')
ax.set_ylabel('GPP\n(gC/m2/day)')
ax.set_title('')
ax.grid(True)
if VI_ylims[i]:
    ax.set_ylim(VI_ylims[i])

# ax.set_xticks([0.09, 0.12, 0.15])

plt.tight_layout()
plt.show()

# save to file
# plotfile = f'scatter_US-CZ2_GPP_vs_MCD19A1_VIs_2010-2016.png'
# plt.savefig(os.path.join(plot_dir, plotfile), format='png', bbox_inches="tight")
# plt.close('all')  # close the plot


# %%

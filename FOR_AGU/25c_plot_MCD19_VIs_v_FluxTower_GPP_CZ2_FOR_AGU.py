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
import sys
from pathlib import Path
import copy

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
plot_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/drought/flux_tower_ts/QCfilt_{QC_option}/'
plot_dir = os.path.join(plot_basedir, 'CZ2/')
Path(plot_dir).mkdir(parents=True, exist_ok=True)
# eddy covariance data
EC_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/eddy_covariance/data/'

nClimGrid_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/nClimGrid/05d'
nClimGrid_SPEI_01_file = os.path.join(nClimGrid_dir, 'nClimGrid_SPEI_01month_05d_WUS_monthly_2000-2024.nc')
nClimGrid_SPEI_12_file = os.path.join(nClimGrid_dir, 'nClimGrid_SPEI_12month_05d_WUS_monthly_2000-2024.nc')
nClimGrid_SPEI_36_file = os.path.join(nClimGrid_dir, 'nClimGrid_SPEI_36month_05d_WUS_monthly_2000-2024.nc')

GRIDMET_PDSI_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/GRIDMET/PDSI/05d'
GRIDMET_PDSI_files = sorted(glob.glob(os.path.join(GRIDMET_PDSI_dir, '*.nc')))


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

MCD19_xr_monthly = MCD19_xr.resample(time='1M').mean()

#%%


# Load nClimGrid SPEI-01
nClimGrid_SPEI_01_xr = xr.open_dataset(nClimGrid_SPEI_01_file, decode_coords='all')['SPEI_pearson']
nClimGrid_SPEI_01_xr.rio.write_crs(4326, inplace=True)
nClimGrid_SPEI_01_xr_point = nClimGrid_SPEI_01_xr.sel(lat=flux_tower_lat, lon=flux_tower_lon, method='nearest')
nClimGrid_SPEI_01_xr_interp = nClimGrid_SPEI_01_xr_point.interp(time=MCD19_xr.time, method='linear')
SPEI_01_xr = nClimGrid_SPEI_01_xr_interp.sel(time=slice(dt(analysis_years[0],1,1), dt(analysis_years[-1],12,31)))

# Load nClimGrid SPEI-12
nClimGrid_SPEI_12_xr = xr.open_dataset(nClimGrid_SPEI_12_file, decode_coords='all')['SPEI_pearson']
nClimGrid_SPEI_12_xr.rio.write_crs(4326, inplace=True)
nClimGrid_SPEI_12_xr_point = nClimGrid_SPEI_12_xr.sel(lat=flux_tower_lat, lon=flux_tower_lon, method='nearest')
nClimGrid_SPEI_12_xr_interp = nClimGrid_SPEI_12_xr_point.interp(time=MCD19_xr.time, method='linear')
SPEI_12_xr = nClimGrid_SPEI_12_xr_interp.sel(time=slice(dt(analysis_years[0],1,1), dt(analysis_years[-1],12,31)))

# Load GRIDMET PDSI
PDSI_xr_raw = data_utils.load_files_to_xr(GRIDMET_PDSI_files)
PDSI_xr_raw.rio.write_crs(4326, inplace=True)
PDSI_xr_point = PDSI_xr_raw.sel(y=flux_tower_lat, x=flux_tower_lon, method='nearest')['pdsi']
PDSI_xr_interp = PDSI_xr_point.interp(time=MCD19_xr.time, method='linear') # Interpolate PDSI_xr_regrid to match MCD19_xr time dimension
PDSI_xr = PDSI_xr_interp.sel(time=slice(dt(analysis_years[0],1,1), dt(analysis_years[-1],12,31)))


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

# Resample to 8-day intervals using mean interpolation
df_resampled = df.resample('16D').mean()

# Reset the index to get datetime back as a column
df_resampled.reset_index(inplace=True)

# Convert fluxes from umol/m2/s to gC/ha/16d
df_resampled["NEE_U50_f"] = df_resampled["NEE_U50_f"] * umol_per_m2_per_s_to_gC_per_m2_per_day
df_resampled["Reco_U50"] = df_resampled["Reco_U50"] * umol_per_m2_per_s_to_gC_per_m2_per_day
df_resampled["GPP_U50_f"] = df_resampled["GPP_U50_f"] * umol_per_m2_per_s_to_gC_per_m2_per_day

# Display the resampled dataframe
print(df_resampled)


#%%
# # Plot GPP_U50_f vs. time
# print('plotting GPP_U50_f vs. time')

# fig, ax = plt.subplots(figsize=(10, 3))  # establish the figure, axes
# ax.grid(color="gray", linestyle="--", linewidth=1, alpha=0.5)  # add grid lines

# # plot the GPP_U50_f field as time series
# ax.plot(df_resampled['datetime'], df_resampled['GPP_U50_f'], color='black', label='GPP_U50_f', zorder=4)

# # Define the axis, title labels
# ax.set(title='GPP_U50_f vs. Time', ylabel='GPP_U50_f', xlabel='Time (UTC)')
# ax.set_xlim(df_resampled['datetime'].min(), df_resampled['datetime'].max())

# # add legend
# fig.legend(loc="upper left", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)

# plt.show()


#%%

#########################################
# Plot MCD19 time series
#########################################


# Plot the interpolated monthly time series
fig, ax = plt.subplots(figsize=(10, 3)) # establish the figure, axes
ax.grid(color = "gray", linestyle = "--", linewidth = 1, alpha = 0.5) # add grid lines

# plot the desired variable as time series
ax.plot(MCD19_xr_monthly.time, MCD19_xr_monthly['NDVI'], color='black', 
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
ax.plot(MCD19_xr_monthly.time, MCD19_xr_monthly['CCI'], color='black', 
      label = 'CCI', zorder=4)

# Define the axis, title labels
ax.set(title=f'MCD19A1 monthly CCI at CZ2', ylabel='CCI', xlabel='Time (UTC)')
ax.set_xlim(dt(2010, 1, 1), dt(2016, 12, 31))

# add legend including the shaded error region
fig.legend(loc="upper left", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
plt.show()



#%%
timing_label = '16day'
MCD19_plot_xr = MCD19_xr

plot_var = 'NIRv'

# for plot_var in VI_vars:
# Plot the interpolated monthly time series with GPP_U50_f on a second y-axis
print(f'plotting monthly timeseries of MCD19A1 {plot_var} with GPP_U50_f')

fig, ax1 = plt.subplots(figsize=(7, 3))  # establish the figure, axes
ax1.grid(color="gray", linestyle="--", linewidth=1, alpha=0.5)  # add grid lines

# Add shaded regions
ax1.axvspan(dt(2011, 6, 1), dt(2013, 8, 1), color='#9e9e9e', alpha=0.6)
ax1.axvspan(dt(2013, 8, 1), dt(2015, 8, 1), color='#474747', alpha=0.6)
# ax1.axvspan(dt(2015, 8, 1), dt(2020, 8, 1), facecolor='none',  alpha = 0.6,
#             hatch='//', edgecolor='#474747', linewidth=5)
ax1.axvspan(dt(2015, 8, 1), dt(2017, 12, 31), color='#715d47', alpha=0.6)

# plot the desired variable as time series
ax1.plot(df_resampled['datetime'], df_resampled['GPP_U50_f'], color='black', 
        label='GPP', zorder=4, linewidth = 4)
ax1.set_ylabel('Flux Tower GPP\n' + r'(gC m$^{-2}$ day$^{-1}$)', fontsize=16)

# Define the axis, title labels
# ax1.set(title=None, xlabel=None)
ax1.set_xlim(dt(2008, 10, 1), dt(2017, 12, 31))

# Get the current y-limits
ymin, ymax = ax1.get_ylim()
yspan = ymax - ymin
yspan_spacer = 0.4
ax1.set_ylim(ymin, ymax + (yspan * yspan_spacer))# Set the y-limits to be X% higher
# ax1.set_ylim(ymin - (yspan * yspan_spacer), ymax)# Set the y-limits to be X% higher

ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=16)
for label in ax1.get_xticklabels():
    label.set_rotation(45)

# Create a second y-axis
ax2 = ax1.twinx()
ax2.plot(MCD19_plot_xr.time, MCD19_plot_xr[plot_var], color='green', 
        label=plot_var, zorder=5, linewidth = 4)

# Reset ylimits to add extra space for label
ymin, ymax = ax2.get_ylim()
yspan = ymax - ymin
ax2.set_ylim(ymin, ymax + (yspan * yspan_spacer)) # Set the y-limits to be X% higher
# ax2.set_ylim(ymin - (yspan * yspan_spacer), ymax)# Set the y-limits to be X% higher

ax2.tick_params(axis='y', labelsize=16)

ax2.set_ylabel(plot_var, color='green', fontsize=17)
ax2.tick_params(axis='y', labelcolor='green')

# save to file
plotfile = f'ts_US-CZ2_GPP_U50_f_vs_{timing_label}_MCD19A1_{plot_var}.png'

plt.show()





# %%

timing_label = '16day'
MCD19_plot_xr = MCD19_xr
drought_plot_xr = PDSI_xr
drought_var_label = 'PDSI'

plot_var = 'CCI'

# for plot_var in VI_vars:
# Plot the interpolated monthly time series with GPP_U50_f on a second y-axis
print(f'plotting monthly timeseries of MCD19A1 {plot_var} with GPP_U50_f')

fig, ax1 = plt.subplots(figsize=(7, 3))  # establish the figure, axes
ax1.grid(color="gray", linestyle="--", linewidth=1, alpha=0.5)  # add grid lines

# Add shaded regions
ax1.axvspan(dt(2011, 6, 1), dt(2013, 8, 1), color='#9e9e9e', alpha=0.4)
ax1.axvspan(dt(2013, 8, 1), dt(2015, 8, 1), color='#474747', alpha=0.4)
# ax1.axvspan(dt(2015, 8, 1), dt(2020, 8, 1), facecolor='none',  alpha = 0.6,
#             hatch='//', edgecolor='#474747', linewidth=5)
ax1.axvspan(dt(2015, 8, 1), dt(2017, 12, 31), color='#715d47', alpha=0.4)

# plot the desired variable as time series
ax1.plot(df_resampled['datetime'], df_resampled['GPP_U50_f'], color='black', 
        label='GPP', zorder=4, linewidth = 4)
ax1.set_ylabel('Flux Tower GPP\n' + r'(gC m$^{-2}$ day$^{-1}$)', fontsize=16)

# Define the axis, title labels
# ax1.set(title=None, xlabel=None)
ax1.set_xlim(dt(2008, 10, 1), dt(2017, 12, 31))

# Get the current y-limits
ymin, ymax = ax1.get_ylim()
yspan = ymax - ymin
yspan_spacer = 0.4
ax1.set_ylim(ymin, ymax + (yspan * yspan_spacer))# Set the y-limits to be X% higher
# ax1.set_ylim(ymin - (yspan * yspan_spacer), ymax)# Set the y-limits to be X% higher

ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=16)
for label in ax1.get_xticklabels():
    label.set_rotation(45)

# Create a second y-axis
ax2 = ax1.twinx()
ax2.plot(drought_plot_xr.time, drought_plot_xr.values, color='orange', label=drought_var_label, zorder=3, linewidth=3)

# Reset ylimits to add extra space for label
ymin, ymax = ax2.get_ylim()
yspan = ymax - ymin
ax2.set_ylim(ymin, ymax + (yspan * yspan_spacer)) # Set the y-limits to be X% higher
# ax2.set_ylim(ymin - (yspan * yspan_spacer), ymax)# Set the y-limits to be X% higher

ax2.tick_params(axis='y', labelsize=16)

ax2.set_ylabel(drought_var_label, color='orange', fontsize=17)
ax2.tick_params(axis='y', labelcolor='orange')

# add legend
fig.legend(loc="lower left", bbox_to_anchor=(0, 0), bbox_transform=ax1.transAxes, fontsize=16, ncol=2)
fig.legend(loc="lower left", bbox_to_anchor=(0, 0), bbox_transform=ax1.transAxes, fontsize = 16)

# # Add text "Drying phase" at the top of the plot grid area centered on date May 1, 2012
# ax2.text(dt(2012, 7, 1), ymax + (yspan * yspan_spacer * 0.5), 'Early Drought', 
#     horizontalalignment='center', fontsize=15, color='black')
# ax2.text(dt(2014, 8, 1), ymax + (yspan * yspan_spacer * 0.5), 'Late Drought', 
#     horizontalalignment='center', fontsize=15, color='black')
# ax2.text(dt(2016, 5, 1), ymax + (yspan * yspan_spacer * 0.5), 'Mortality', 
#     horizontalalignment='center', fontsize=15, color='black')


# save to file
plotfile = f'ts_US-CZ2_GPP_U50_f_vs_{timing_label}_MCD19A1_{plot_var}.png'

# plt.savefig(os.path.join(plot_dir, plotfile), format='png', bbox_inches="tight")
# plt.close('all')  # close the plot

plt.show()





#%%
timing_label = '16day'
MCD19_plot_xr = MCD19_xr
drought_plot_xr = PDSI_xr
drought_var_label = 'PDSI'

plot_var = 'CCI'


# for plot_var in VI_vars:
# Plot the interpolated monthly time series with GPP_U50_f on a second y-axis
print(f'plotting monthly timeseries of MCD19A1 {plot_var} with GPP_U50_f')

fig, ax1 = plt.subplots(figsize=(7, 3))  # establish the figure, axes
ax1.grid(color="gray", linestyle="--", linewidth=1, alpha=0.5)  # add grid lines

# Add shaded regions
ax1.axvspan(dt(2011, 6, 1), dt(2013, 8, 1), color='#9e9e9e', alpha=0.4)
ax1.axvspan(dt(2013, 8, 1), dt(2015, 8, 1), color='#474747', alpha=0.4)
# ax1.axvspan(dt(2015, 8, 1), dt(2020, 8, 1), facecolor='none',  alpha = 0.6,
#             hatch='//', edgecolor='#474747', linewidth=5)
ax1.axvspan(dt(2015, 8, 1), dt(2017, 12, 31), color='#715d47', alpha=0.4)

# plot the desired variable as time series
ax1.plot(df_resampled['datetime'], df_resampled['GPP_U50_f'], color='black', 
        label='GPP', zorder=4, linewidth = 4)
ax1.set_ylabel('Flux Tower GPP\n' + r'(gC m$^{-2}$ day$^{-1}$)', fontsize=16)

# Define the axis, title labels

# ax1.set(title=None, xlabel=None)
ax1.set_xlim(dt(2008, 10, 1), dt(2017, 12, 31))

# Get the current y-limits
ymin, ymax = ax1.get_ylim()
yspan = ymax - ymin
yspan_spacer = 0.4
ax1.set_ylim(ymin, ymax + (yspan * yspan_spacer))# Set the y-limits to be X% higher
# ax1.set_ylim(ymin - (yspan * yspan_spacer), ymax)# Set the y-limits to be X% higher

ax1.tick_params(axis='x', labelsize=15)
ax1.tick_params(axis='y', labelsize=16)
for label in ax1.get_xticklabels():
    label.set_rotation(45)

# Create a second y-axis
ax2 = ax1.twinx()
ax2.plot(drought_plot_xr.time, drought_plot_xr.values, color='orange', label=drought_var_label, zorder=3, linewidth=3)
ax2.set_ylabel(drought_var_label, color='orange', fontsize=17)

# Reset ylimits to add extra space for label
ymin, ymax = ax2.get_ylim()
yspan = ymax - ymin
ax2.set_ylim(ymin, ymax + (yspan * yspan_spacer)) # Set the y-limits to be X% higher
# ax2.set_ylim(ymin - (yspan * yspan_spacer), ymax)# Set the y-limits to be X% higher

ax2.tick_params(axis='y', labelsize=16)

ax2.set_ylabel(drought_var_label, color='orange', fontsize=17)
ax2.tick_params(axis='y', labelcolor='orange')

# Add a third y-axis for SPEI_01_xr
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # move the third axis outward
ax3.plot(MCD19_plot_xr.time, MCD19_plot_xr[plot_var], color='green', 
        label=plot_var, zorder=5, linewidth = 4)
ax3.set_ylabel(plot_var, color='green', fontsize=17)
ax3.tick_params(axis='y', labelcolor='green', labelsize=16)

# save to file
plotfile = f'ts_US-CZ2_GPP_U50_f_vs_{timing_label}_MCD19A1_{plot_var}_with_{drought_var_label}.png'

# plt.savefig(os.path.join(plot_dir, plotfile), format='png', bbox_inches="tight")
# plt.close('all')  # close the plot

plt.show()





# %%

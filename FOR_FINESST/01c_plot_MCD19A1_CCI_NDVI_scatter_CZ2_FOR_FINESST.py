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

# runtime packages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt  # primary plotting module
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

#%%
#########################################
# Define Global Filepaths
#########################################

# directories
dat_basedir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/'
polygon_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/disturbance/data/flux_tower_polygons/'
plot_basedir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/disturbance/plots/'
snow_basedir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/Snowcover/output/'

# eddy covariance data
EC_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/eddy_covariance/data/'

# specify string to match for files using glob.glob (unix style pattern matching)
# If dataset is contained in a single file, just put the file basename
file_pattern = '*.nc'

########################################
# Define Global Variables and constants
#########################################
#%%
### Variables specific to the dataset 
plot_var = 'CCI' # this is the name of the variable to be accessed in the dataset

### set subfolder and filename 
subfolder_name = 'CCI' # this is the name of the subfolder the plots will be saved in
plot_fileID = 'MCD19A1_CCI_01d_16day'
  # filename identifier for the dataset and variable

# Plot labels
plotTitle = 'MODIS MCD19A1 Chlorophyll/Carotenoid Index (CCI)\n(0.01°, 16-day composite)'

### Plotting options:
plot_filetype = 'pdf' # can be 'pdf', 'png', or 'jpeg'

### Unit conversion:
secs_per_day = 3600*24 # number of seconds in a day
kg_per_g = 0.001 # conversion factor from grams to kilograms
umol_to_mol = 1e-6 # conversion factor from micromoles to moles
gC_per_mol = 12.01 # grams of carbon per mole of CO2
m2_to_ha = 1e-4 # conversion factor from m^2 to hectares

umol_per_m2_per_s_to_gC_per_m2_per_day = (secs_per_day * umol_to_mol * gC_per_mol) # conversion factor from umol/m^2/s to gC/ha/16d

#########################################
# Define Global Functions
#########################################

def check_dat_files_recursive(polygon_ID, basedir, pattern):
   # Polygon ID is comprised of 3 parts - state, subregion, and area name. Separated by '_' underscore
   ID_split_arr = polygon_ID.split('_')
   state = ID_split_arr[0]
   subregion = ID_split_arr[1]
   area_name = ID_split_arr[2]

   # Look for appropriate files first in the basedir/[state]/[subregion]/[area_name]/ dir,
   # continue on up to parent dir if dir does not exist or is empty
   if os.path.isdir(os.path.join(basedir, state)): # if state dir exists, check subregion
      if os.path.isdir(os.path.join(basedir, state, subregion)): # if subregion dir exists, check area_name
         if os.path.isdir(os.path.join(basedir, state, subregion, area_name)): # if area_name dir exists...
            # list any files matching the desired pattern here
            dat_files = sorted(glob.glob(os.path.join(basedir, state, subregion, area_name, pattern)))
            if len(dat_files) > 0:
               return dat_files
         # if no matching files in the deepest (area_name) dir, go up to the parent and search there
         dat_files = sorted(glob.glob(os.path.join(basedir, state, subregion, pattern)))
         if len(dat_files) > 0:
            return dat_files
      # if no matching files in the subregion dir, go up to the parent and search there
      dat_files = sorted(glob.glob(os.path.join(basedir, state, pattern)))
      if len(dat_files) > 0:
         return dat_files
   # Finally, if no matching files in any dir, search in the basedir
   dat_files = sorted(glob.glob(os.path.join(basedir, pattern)))
   if len(dat_files) > 0:
      return dat_files
   else:
      raise Exception(f'No files matching {pattern} found in {basedir} or in {state}, {state}/{subregion}, or {state}/{subregion}/{area_name} sub-directories')

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

#########################################
### Begin with standard argument handling
#########################################

#%%
# mark start time to keep track of elapsed
start_time = time.time()

### get command line args 

# 0th arg is the script name
# 1st arg is the polygon filepath
polygon_filepath = os.path.join(polygon_dir, 'CA_Sierra_SOAP.shp')
polygon_file = os.path.basename(polygon_filepath)

# 2nd and 3rd args are True/False flags indicating whether to remove winter periods and snowcover, respectively
mask_winter = False #TODO: make this default to False unless explicitly requested by args
mask_snow = True #TODO: make this default to False unless explicitly requested by args

# check that this is a '.shp' file, if not raise an error
polygon_ext = polygon_file[-4:]
if not polygon_ext == '.shp':
   raise Exception("Invalid filetype provided - must provide '.shp' file")

# the polygon ID is the file basename minus the extension, which is 4 characters including the '.'
polygon_ID = polygon_file[:-4]

# use pre-defined file search function to find relevant files
dat_files = check_dat_files_recursive(polygon_ID, dat_basedir, file_pattern)


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

# save to file
plotfile = f'GPP_U50_f_timeseries.{plot_filetype}'
# plt.savefig(os.path.join(plot_dir, plotfile), format=plot_filetype, bbox_inches="tight")
# plt.close('all')  # close the plot

plt.show()


#%%
# Load polygon file
print(f'\nFile={os.path.basename(polygon_filepath)}')
polygon_gdf = gpd.read_file(polygon_filepath)
polygon_geom = polygon_gdf.geometry.apply(mapping)



############################
### Load satellite data
############################

#%%
dat_rxr_arr = [] # set empty array

for dat_file in dat_files:
   
   dat_rxr = rxr.open_rasterio(dat_file)
   dat_rxr.rio.write_crs(4326, inplace=True)
   # dat_rxr = dat_rxr.rename({'lon':'x', 'lat':'y'})

   # # convert the time coordinates to datetime date (see here: https://stackoverflow.com/questions/55786995/converting-cftime-datetimejulian-to-datetime)
   datetimeindex = dat_rxr.indexes['time'].to_datetimeindex()
   dat_rxr['time'] = datetimeindex

   # if there is at least 1 full grid cell contained in the polygon, use that
   try:
      dat_rxr_clip = dat_rxr.rio.clip(polygon_geom, crs=4326)
   except Exception:
      #otherwise, use all the grid cells that the polygon touches, and take the mean
      try:
         dat_rxr_clip = dat_rxr.rio.clip(polygon_geom, crs=4326, all_touched = True)

      except Exception:
         raise Exception(f'Disturbance perimeter out of bounds for polygon file {polygon_file}. Exiting...')

   num_clipcells = np.count_nonzero(~np.isnan(dat_rxr_clip[plot_var].mean(dim = 'time')))

   print(f'number of grid cells used in clipping = {num_clipcells}')

   dat_da = dat_rxr_clip.mean(dim = ('x', 'y'), skipna=True) # take spatial average
   dat_rxr_arr.append(dat_da)

# after looping through all files, combine all at the time dimension
dat_ds = xr.concat(dat_rxr_arr, dim = 'time') # concatenate along time axis

# Select only the timestamps from dat_ds that are within the range of df_resampled
dat_ds_trim = dat_ds.sel(time=slice(df_resampled['datetime'].min(), df_resampled['datetime'].max()))

# Resample dat_ds to the same timestamps as in df_resampled
dat_ds_resampled = dat_ds.interp(time=df_resampled['datetime'])
dat_ds_trim_resampled = dat_ds_trim.interp(time=df_resampled['datetime'])

#%%

# Extract data for months 7-11
dat_ds_resampled_filtered = dat_ds_resampled.sel(time=dat_ds_resampled['time.month'].isin([7, 8, 9, 10, 11]))
# Extract data for months 7-11 from df_resampled
df_resampled_filtered = df_resampled[df_resampled['datetime'].dt.month.isin([7, 8, 9, 10, 11])]

# Merge df_resampled_filtered with dat_ds_resampled_filtered into one dataset
merged_dataset = df_resampled_filtered.copy()
merged_dataset['MCD19_CCI'] = dat_ds_resampled_filtered['CCI'].values
merged_dataset['MCD19_NDVI'] = dat_ds_resampled_filtered['NDVI'].values


# Locate the maximum value of MCD19_CCI and its corresponding timestamp
max_CCI_value = merged_dataset['MCD19_CCI'].max()
max_CCI_timestamp = merged_dataset.loc[merged_dataset['MCD19_CCI'].idxmax(), 'datetime']

print(f'Maximum MCD19_CCI value: {max_CCI_value}')
print(f'Timestamp of maximum MCD19_CCI value: {max_CCI_timestamp}')


# Display the merged dataset
print(merged_dataset)

# Set all values in merged_dataset to NaN where MCD19_NDVI is less than 0.5
merged_dataset = merged_dataset[merged_dataset['MCD19_NDVI'] >= 0.5]
merged_dataset = merged_dataset[merged_dataset['MCD19_CCI'] < 0.1]
#########################################
# Plot scatter: GPP vs CCI and GPP vs. NDVI
#########################################

#%%
# Calculate R^2 value

# model_CCI = LinearRegression().fit(merged_dataset['GPP_U50_f'].values.reshape(-1, 1), merged_dataset['MCD19_CCI'])
# y_pred_CCI = model_CCI.predict(merged_dataset['GPP_U50_f'].values.reshape(-1, 1))
# r2_CCI = r2_score(merged_dataset['MCD19_CCI'], y_pred_CCI)
r_CCI = pearsonr(merged_dataset['GPP_U50_f'], merged_dataset['MCD19_CCI']).statistic
r2_CCI = r_CCI ** 2

# Plot scatter: GPP vs CCI
plt.figure(figsize=(3, 3))
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.scatter(merged_dataset['GPP_U50_f'], merged_dataset['MCD19_CCI'], color='black', alpha=0.8)
plt.title(f'R^2 = {r2_CCI:.2f}')
plt.gca().set_facecolor('lightgrey')
plt.ylabel('MODIS CCI', fontsize=12)
plt.xlabel(r'GPP (gC m$^{-2}$ day$^{-1}$)', fontsize=12)
plt.show()



# %%
# Calculate R^2 value
r_NDVI = pearsonr(merged_dataset['GPP_U50_f'], merged_dataset['MCD19_NDVI']).statistic
r2_NDVI = r_NDVI ** 2

# Plot scatter: GPP vs NDVI
plt.figure(figsize=(3, 3))
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.scatter(merged_dataset['GPP_U50_f'], merged_dataset['MCD19_NDVI'], color='black', alpha=0.8)
plt.gca().set_facecolor('lightgrey')
plt.title(f'R^2 = {r2_NDVI:.2f}')
plt.ylabel('MODIS NDVI', fontsize=12)
plt.xlabel(r'GPP (gC m$^{-2}$ day$^{-1}$)', fontsize=12)
plt.show()

# %%

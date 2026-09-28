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
import re


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
import geopandas as gpd
import statsmodels.formula.api as smf

# shapefile/geospatial packages

# time and date packages
import time
from datetime import datetime as dt  # date time library
from datetime import timedelta

# plotting packages
from matplotlib import pyplot as plt  # primary plotting module
from shapely.geometry import mapping
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

EPA_ecoregion_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/L2/NAm_shp_FOR_MAIAC_PROC/'
EPA_ecoregion_L2_file = os.path.join(EPA_ecoregion_dir, 'NA_CEC_Eco_Level2_withOther.shp')
EPA_ecoregion_regrid_mask_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/L2/for_MAIAC_proc/NAm/'


# EPA_ecoregion_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/L3/'
# EPA_ecoregion_L3_file = os.path.join(EPA_ecoregion_dir, 'NAm_cec_eco_l3.shp')
# EPA_ecoregion_regrid_mask_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/L3/for_MAIAC_proc/NAm/'

LC_basedir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/for_MAIAC_proc/'
forest_frac_dir = os.path.join(LC_basedir, 'forest_fraction/')
crop_frac_dir = os.path.join(LC_basedir, 'crop_fraction/')
otherveg_frac_dir = os.path.join(LC_basedir, 'other_vegetation_fraction/')

elev_regrid_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/USGS/GTOPO30/for_MAIAC_proc/'

#########################################
# Define Global functions
#########################################

def floor(value, decimal_digits):
  scale = (10**decimal_digits)
  return  np.floor(value * scale)/scale

def ceil(value, decimal_digits):
  scale = (10**decimal_digits)
  return  np.ceil(value * scale)/scale


def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))

binary_v = np.vectorize(binary)

def get_bits(str, start_bit, end_bit):
   str_split = [*str][start_bit:end_bit+1]
   return ''.join(str_split)

get_bits_v = np.vectorize(get_bits)

tanh_v = np.vectorize(np.tanh)

arccos_v = np.vectorize(np.arccos)

def rad_to_deg(rad):
   return rad*180/math.pi

def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta

def td_min(td):
    return td.seconds//60

def td_sec(td):
    return td.seconds%60


def parse_timestamps_to_datetime_from_global_attrs(orbit_time_stamp):
    timestamps_raw = orbit_time_stamp.strip().split()
    orbit_times = []
    for timestamp in timestamps_raw:
        year = int(timestamp[0:4])
        julian_day = int(timestamp[4:7])
        hour = int(timestamp[7:9])
        minute = int(timestamp[9:11])
        dt_obj = dt(year, 1, 1) + timedelta(days=julian_day - 1, hours=hour, minutes=minute)
        orbit_times.append(dt_obj)
    # Convert to numpy.datetime64 array
    np_times = np.array(orbit_times, dtype='datetime64[m]')
    return np_times


def print_timestamp_satellite_info_from_global_attrs(orbit_time_stamp):
    timestamps_raw = orbit_time_stamp.strip().split()
    # Parse each timestamp
    orbit_times = []
    for timestamp in timestamps_raw:
        year = int(timestamp[0:4])
        julian_day = int(timestamp[4:7])
        hour = int(timestamp[7:9])
        minute = int(timestamp[9:11])
        if timestamp[11] == 'T':
            satellite = 'Terra'
        else:
            satellite = 'Aqua'

        dt_obj = dt(year, 1, 1) + timedelta(days=julian_day - 1, hours=hour, minutes=minute)
        orbit_times.append({'datetime': dt_obj, 'satellite': satellite})

    for i, orbit_info in enumerate(orbit_times):
        print(f"Orbit {i}: {orbit_info['datetime']} - Satellite: {orbit_info['satellite']}")



####################################
### Set up map parameters
####################################
#%%
# use Google Satellite imagery as basemap
tiler = img_tiles.GoogleTiles(style='satellite')
crs = tiler.crs # set crs of map tiler
alpha = 0.8  # transparency 0-1
transform = ccrs.PlateCarree()  # transform specifies the crs that the data is in

### Load state boundaries
fn = shpreader.natural_earth(
   resolution='10m', category='cultural', 
   name='admin_1_states_provinces',
)

reader = shpreader.Reader(fn)
states = [x for x in reader.records() if x.attributes["admin"] == "United States of America"] # get all states in US
states_geom = cfeature.ShapelyFeature([x.geometry for x in states], ccrs.PlateCarree())

# ### Create a custom colormap
# plot_colors = ["#ffffb2", "#fd8d3c", "#f03b20", "#810f7c", "#21013B"]
# cmap = mcolors.ListedColormap(plot_colors)
# cmaplist = [cmap(i) for i in range(cmap.N)] # extract all colors from the cmap

# # create the new map
# cmap = mcolors.LinearSegmentedColormap.from_list(
#     'Custom cmap', cmaplist, cmap.N)

tiler_zoom = 5  # define a zoom level of detail

def get_tile_plot_extent(dat_xr_curv):
    lat_min = float(dat_xr_curv['lat'].min())
    lat_max = float(dat_xr_curv['lat'].max())
    lon_min = float(dat_xr_curv['lon'].min())
    lon_max = float(dat_xr_curv['lon'].max())
    plot_extent =[lon_min, lon_max, lat_min, lat_max]

    # special case if some lon values cross the 180 meridian, then lon_max will be 
    # large because it extends into Russia (lon ~ +100 to +180)
    if ((lon_min < -175) & (lon_max > 100)):
        lon_neg = dat_xr_curv.lon.values.copy()
        lon_neg[lon_neg > 0] = np.nan
        plot_extent = [-179.99, np.nanmax(lon_neg), lat_min, lat_max]

    return plot_extent


def test_plot(dat_da, plot_title, zlab, plot_extent = None, plot_filename = None, plot_min = None, plot_max = None):
    if plot_extent is None:
        plot_extent = get_tile_plot_extent(dat_da)
    if plot_min is None:
        plot_min = float(np.nanmin(dat_da))
    if plot_max is None:
        plot_max = float(np.nanmax(dat_da))
    fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={'projection': crs}) # establish the figure, axes
    ax.set_extent(plot_extent, crs=ccrs.PlateCarree())  # PlateCarree is default lat/lon    
    ax.add_image(tiler, tiler_zoom, interpolation='none') # Add base image
    ax.add_feature(states_geom, facecolor="none", edgecolor="black", linewidth=1) # add state boundaries to the map
    dat_da.plot(ax=ax, x='lon', y='lat', transform=transform, 
                    alpha=alpha, cmap='inferno_r',
                    vmin=plot_min, vmax=plot_max,
                    cbar_kwargs={'orientation': 'vertical',
                                            'pad': 0.05,
                                            'label': zlab,
                                            'shrink': 0.75})

    plt.title(plot_title, fontsize=18)
    plt.xlabel('lon', fontsize=15)
    plt.ylabel('lat', fontsize=14)

    if plot_filename is not None:
        if not os.path.exists(os.path.dirname(plot_filename)):
            Path(os.path.dirname(plot_filename)).mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_filename, format = "png", bbox_inches = "tight")
        plt.close('all')
    else:
        plt.show()



#########################################
# Define Global Variables and constants
#########################################


composite_years = np.arange(2000, 2025)
DOY_composite_starts = np.arange(1, 366, 16)
composite_periods = range(len(DOY_composite_starts))  # 0-22

refl_vars = ['Sur_refl1', 'Sur_refl2', 'Sur_refl4', 'Sur_refl6', 'Sur_refl7', 'Sur_refl11']
#########################################
# Begin main
#########################################
#%%
def main():

    try:
        # mark start time to keep track of elapsed
        start_total = time.time()

        MODIS_tile = sys.argv[1] #'h10v04'
        composite_year = int(sys.argv[2]) # command line argument
        composite_period = int(sys.argv[3]) # command line argument
        QC_option = sys.argv[4] # command line argument - options are 'CloudFree', 'CloudAOD', 'Cloud', 'AOD', or 'None'

        dat_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC_option}/{MODIS_tile}/'
        MCD19A1_files = sorted(glob.glob(os.path.join(dat_dir, f'MCD19A1.{MODIS_tile}.*.nc')))

        hist_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC_option}/{MODIS_tile}/hist_avg'
        MCD19A1_hist_files = sorted(glob.glob(os.path.join(hist_dir, f'MCD19A1.{MODIS_tile}.*.nc')))

        scale_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/gapfill_scale/QCfilt_{QC_option}/'

        out_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/lm-fill/QCfilt_{QC_option}/{MODIS_tile}'
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        outfile = os.path.join(out_dir, f'MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_lmFill_QCfilt_{QC_option}_{composite_year}-{composite_period:02d}.nc')
        if os.path.exists(outfile):
            print(f'Output file {outfile} already exists, exiting...')
            sys.exit(0)

        # Skip processing if 'v01' is in MODIS_tile and composite_period is in [0, 1, 2, 3, 19, 20, 21, 22]
        if 'v01' in MODIS_tile and composite_period in [0, 1, 2, 3, 19, 20, 21, 22]:
            print(f"Skipping composite_period {composite_period} for MODIS_tile {MODIS_tile} (high latitude winter, no data).")
            sys.exit(0)
            
        # Skip processing if 'v02' is in MODIS_tile and composite_period is in [0, 1, 20, 21, 22]
        if 'v02' in MODIS_tile and composite_period in [0, 1, 20, 21, 22]:
            print(f"Skipping composite_period {composite_period} for MODIS_tile {MODIS_tile} (high latitude winter, no data).")
            sys.exit(0)


        hist_files = [f for f in MCD19A1_hist_files if f.endswith(f'{composite_period:02d}.nc')]
        if len(hist_files) == 0:
            print(f'Historical average period does not exist for tile {MODIS_tile} period {composite_period}')
            if composite_period in [0, 1, 20, 21, 22]:
                print(f"Exit with error code 0 is permissible for composite_period {composite_period} (winter month, possible no data)")
                sys.exit(0)
            else:
                print(f"Exiting with error code due to missing historical average file for composite_period {composite_period}.")
                sys.exit(1)
        else:
            hist_file = hist_files[0]
        hist_ds = xr.open_dataset(hist_file)[refl_vars]

        hist_ds_nan_placeholder = xr.DataArray(
            np.full(hist_ds[refl_vars[0]].shape, np.nan),
            coords={dim: hist_ds[refl_vars[0]].coords[dim] for dim in hist_ds[refl_vars[0]].dims},
            dims=hist_ds[refl_vars[0]].dims)

        forest_frac_file = os.path.join(forest_frac_dir, f'forest_frac_1km_curv_{MODIS_tile}.nc')
        crop_frac_file = os.path.join(crop_frac_dir, f'crop_frac_1km_curv_{MODIS_tile}.nc')
        otherveg_frac_file = os.path.join(otherveg_frac_dir, f'otherveg_frac_1km_curv_{MODIS_tile}.nc')
        elev_regrid_file = os.path.join(elev_regrid_dir, f'GTOPO30_DEM_1km_curv_{MODIS_tile}.nc') 


        # Load in Land cover fraction datasets, pre-gridded to 1km curvilinear grid
        if not os.path.exists(forest_frac_file):
            print(f'Forest fraction file {forest_frac_file} does not exist, creating blank file ')
            forest_frac_xr_curv = hist_ds_nan_placeholder
        else:
            forest_frac_xr_curv = xr.open_dataset(forest_frac_file)['__xarray_dataarray_variable__']

        if not os.path.exists(crop_frac_file):
            print(f'Crop fraction file {crop_frac_file} does not exist, creating blank file ')
            crop_frac_xr_curv = hist_ds_nan_placeholder
        else:
            crop_frac_xr_curv = xr.open_dataset(crop_frac_file)['__xarray_dataarray_variable__']

        if not os.path.exists(otherveg_frac_file):
            print(f'Other vegetation fraction file {otherveg_frac_file} does not exist, creating blank file ')
            otherveg_frac_xr_curv = hist_ds_nan_placeholder
        else:
            otherveg_frac_xr_curv = xr.open_dataset(otherveg_frac_file)['__xarray_dataarray_variable__']

        # Load in Elevation dataset and regrid to 1km curvilinear grid
        if not os.path.exists(elev_regrid_file):
            elev_xr_curv = hist_ds_nan_placeholder
        else:
            elev_xr_curv = xr.open_dataset(elev_regrid_file)['__xarray_dataarray_variable__']

        ecoregion_L2_gdf = gpd.read_file(EPA_ecoregion_L2_file)

        NAm_ecoregions = np.unique(ecoregion_L2_gdf.NA_L2NAME)


        # Pattern to match: tile_scale_factors_ecoregion_year-period.csv
        pattern = rf'{MODIS_tile}_scale_factors_(.+?)_\d{{4}}-\d{{2}}\.csv'

        # Get unique ecoregion names for this tile
        ecoregion_names_thisTile = set()
        for filepath in Path(scale_dir).glob(f'{MODIS_tile}_scale_factors_*.csv'):
            match = re.search(pattern, filepath.name)
            if match:
                ecoregion_names_thisTile.add(match.group(1))

        NAm_ecoregion_codes_for_tile = sorted(ecoregion_names_thisTile)
        print(f"Found {len(NAm_ecoregion_codes_for_tile)} unique ecoregions for tile {MODIS_tile}")



        # Find the file that matches the current composite_year and composite_period
        dat_files = [f for f in MCD19A1_files if f'{composite_year}-{composite_period:02d}.nc' in f]
        if len(dat_files) == 0:
            print(f'No data file found for year {composite_year}, period {composite_period}, exiting with error code...')
            sys.exit(0)

        print(f'Gap-filling with climatology - composite year: {composite_year}, time period {composite_period}')
        dat_file = dat_files[0]
        dat_ds_temp = xr.open_dataset(dat_file)
        if 'time' in dat_ds_temp.dims:
            dat_ds_allvars = dat_ds_temp.squeeze('time')
        else:
            dat_ds_allvars = dat_ds_temp
        dat_ds_gapfill = copy.deepcopy(dat_ds_allvars)
        dat_ds = dat_ds_allvars[refl_vars]
        dat_date = pd.to_datetime(dat_ds_allvars.time.values)

        # represent the scaling from historical to this year's data as a ratio
        scale_ds_raw = dat_ds / hist_ds

        if len(NAm_ecoregion_codes_for_tile) > 0:
            # Loop through the ecoregions and create masks
            for NAm_ecoregion_code in NAm_ecoregion_codes_for_tile:

                scale_files = sorted(glob.glob(os.path.join(scale_dir, f'*scale_factors_{NAm_ecoregion_code}_{composite_year}-{composite_period:02d}.csv')))
                if len(scale_files) == 0:
                    continue

                all_scale_dfs = pd.DataFrame()
                for scale_file in scale_files:
                    scale_df_part = pd.read_csv(scale_file).dropna(how='any')
                    if scale_df_part.empty:
                        continue
                    all_scale_dfs = pd.concat([all_scale_dfs, scale_df_part], ignore_index=True)

                if all_scale_dfs.empty:
                    continue

                #"EPA_L2_ecoregion_mask_1km_curv_h09v04_Arizona_New_Mexico_Mountains.nc"
                ecoregion_mask_file = os.path.join(EPA_ecoregion_regrid_mask_dir, f'EPA_L2_ecoregion_mask_1km_curv_{MODIS_tile}_{NAm_ecoregion_code}.nc')
                if not os.path.exists(ecoregion_mask_file):
                    print(f'Ecoregion mask file {ecoregion_mask_file} does not exist, skipping...')
                    continue

                print(f'Ecoregion masking for {NAm_ecoregion_code}')
                xr_ecoregion_mask_curv = xr.open_dataset(ecoregion_mask_file)['__xarray_dataarray_variable__']

                dat_ds_ecoregion_clip = dat_ds.where(xr_ecoregion_mask_curv == 1)
                forest_frac_ecoregion_clip = forest_frac_xr_curv.where(xr_ecoregion_mask_curv == 1)
                crop_frac_ecoregion_clip = crop_frac_xr_curv.where(xr_ecoregion_mask_curv == 1)
                otherveg_frac_ecoregion_clip = otherveg_frac_xr_curv.where(xr_ecoregion_mask_curv == 1)
                elev_ecoregion_clip = elev_xr_curv.where(xr_ecoregion_mask_curv == 1)

                # Flatten the data arrays and create a mask for valid (non-NaN) entries
                valid_mask = np.full_like(dat_ds_ecoregion_clip[refl_vars[0]], False, dtype=bool)
                for var in refl_vars:

                    # create model
                    formula = f'{var} ~ forest_fraction + crop_fraction + otherveg_fraction + elevation'
                    model = smf.ols(formula, data=all_scale_dfs)
                    fitted_model = model.fit()

                    # get the missing data for this variable
                    missing_mask = np.isnan(dat_ds_ecoregion_clip[var])
                    missing_elev = elev_ecoregion_clip.where(missing_mask)
                    missing_forest_frac = forest_frac_ecoregion_clip.where(missing_mask)
                    missing_crop_frac = crop_frac_ecoregion_clip.where(missing_mask)
                    missing_otherveg_frac = otherveg_frac_ecoregion_clip.where(missing_mask)

                    # Create prediction dataframe from flattened valid points
                    valid_pred_mask = ~np.isnan(missing_elev) & ~np.isnan(missing_forest_frac) & ~np.isnan(missing_crop_frac) & ~np.isnan(missing_otherveg_frac)
                    pred_df = pd.DataFrame({
                        'forest_fraction': missing_forest_frac.values[valid_pred_mask],
                        'crop_fraction': missing_crop_frac.values[valid_pred_mask],
                        'otherveg_fraction': missing_otherveg_frac.values[valid_pred_mask],
                        'elevation': missing_elev.values[valid_pred_mask]
                    })
                    predicted_scales = fitted_model.predict(pred_df)
                    predicted_scales[predicted_scales <= 0] = np.nan
                    # Create a scale array with predicted values in the right positions
                    predicted_scale_grid = np.full_like(missing_elev.values, np.nan)
                    predicted_scale_grid[valid_pred_mask] = predicted_scales
                    
                    # Create xarray DataArray with proper coordinates
                    predicted_scale_da = xr.DataArray(
                        predicted_scale_grid,
                        coords=missing_elev.coords,
                        dims=missing_elev.dims
                    )
                    
                    # Fill gaps: multiply predicted scales by historical data
                    gapfilled_values = predicted_scale_da * hist_ds[var]
                    
                    # Progressively fill gaps - only fills where dat_ds_gapfill is currently NaN
                    dat_ds_gapfill[var] = dat_ds_gapfill[var].fillna(gapfilled_values)
        else:
            for var in refl_vars:
                dat_ds_gapfill[var] = dat_ds_gapfill[var].fillna(hist_ds[var])

        # define global attributes
        dat_ds_gapfill.attrs['title'] = 'MODIS MCD19A1.061 MAIAC Vegetation Indices calculated from surface reflectance - 16-day composite using Constrained View angle - Maximum Value Composite (CV-MVC) algorithm'
        dat_ds_gapfill.attrs['date_string'] = str(dat_date.date())
        dat_ds_gapfill.attrs['Comment1'] = 'Original files downloaded from https://e4ftl01.cr.usgs.gov/MOTA/MCD19A1.061/'
        dat_ds_gapfill.attrs['Comment2'] = 'hdf files converted to netcdf4 using HDF-EOS conversion tool (https://www.hdfeos.org/software/h4toh5.php)'
        dat_ds_gapfill.attrs['Comment3'] = 'gap filled using historical fill scaled with canopy-elevation regression'
        dat_ds_gapfill.attrs['Citation'] = 'Lyapustin, A., Wang, Y. (2022). MODIS/Terra+Aqua Land Surface BRF Daily L2G Global 500m and 1km SIN Grid V061 [Data set]. NASA EOSDIS Land Processes Distributed Active Archive Center. Accessed from https://doi.org/10.5067/MODIS/MCD19A1.061'
        dat_ds_gapfill.attrs['Processed_by'] = 'Lewis Kunik, lewis.kunik@utah.edu'
        dat_ds_gapfill.attrs['Date_created'] = str(dt.today().date())

        # Finally, save to netcdf
        dat_ds_gapfill.to_netcdf(path=outfile, format="NETCDF4")
        dat_ds_gapfill.close()

        print(f'Wrote gap-filled data to {outfile}')

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)  
    
if __name__ == '__main__':
   main()
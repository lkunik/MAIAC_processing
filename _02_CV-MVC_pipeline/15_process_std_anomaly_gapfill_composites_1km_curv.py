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

# Custom packages
utils_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/pyutil'
if utils_dir not in sys.path:
    sys.path.append(utils_dir)
from MODIS_tile_corners import MODIS_tilecorners_df

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
# use Google Satellite imagery as basemap
tiler = img_tiles.GoogleTiles(style='satellite')
crs = tiler.crs # set crs of map tiler
alpha = 0.5  # transparency 0-1
transform = ccrs.PlateCarree()  # transform specifies the crs that the data is in
plot_extent = [-128, -106, 39, 51]  # [minx, maxx, miny, maxy], bounds for western US

### Load state boundaries
fn = shpreader.natural_earth(
   resolution='10m', category='cultural', 
   name='admin_1_states_provinces',
)

reader = shpreader.Reader(fn)
states = [x for x in reader.records() if x.attributes["admin"] == "United States of America"] # get all states in US
states_geom = cfeature.ShapelyFeature([x.geometry for x in states], ccrs.PlateCarree())



tiler_zoom = 5  # define a zoom level of detail


tile_plot_extents = {
    'h08v04': [-125, -120, 39, 45],
    'h08v05': [-125, -106, 39, 41],
    'h09v04': [-128, -106, 39, 51],
    'h09v05': [-120, -106, 29, 41],
    'h10v04': [-125, -100, 39, 51],
    # Add more tiles and their extents as needed
}


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

plot_years = [2000, 2005, 2010, 2015, 2020, 2025]
plot_period = 12 # 16-day period to plot
#########################################
# Begin main
#########################################

def main():

    try:
        MODIS_tile = sys.argv[1] #'h10v04'
        composite_year = int(sys.argv[2]) # command line argument
        composite_period = int(sys.argv[3]) # command line argument
        QC_option = sys.argv[4] # command line argument - options are 'CloudFree', 'CloudAOD', 'Cloud', 'AOD', or 'None'

        dat_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/lm-fill/QCfilt_{QC_option}/{MODIS_tile}/'
        hist_avg_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/lm-fill/QCfilt_{QC_option}/{MODIS_tile}/hist_avg/'
        out_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/std_anom/QCfilt_{QC_option}/{MODIS_tile}'
        plot_dir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/MAIAC_CV-MVC/1km_curv/std_anomaly/QCfilt_{QC_option}/{MODIS_tile}/'
        domain_scale_factor_dir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/std_anom/QCfilt_{QC_option}/domain_scale_factors/'

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(plot_dir).mkdir(parents=True, exist_ok=True)

        # Skip processing if 'v01' is in MODIS_tile and composite_period is in [0, 1, 2, 3, 19, 20, 21, 22]
        if 'v01' in MODIS_tile and composite_period in [0, 1, 2, 3, 19, 20, 21, 22]:
            print(f"Skipping composite_period {composite_period} for MODIS_tile {MODIS_tile} (high latitude winter, no data).")
            sys.exit(0)
            
        # Skip processing if 'v02' is in MODIS_tile and composite_period is in [0, 1, 20, 21, 22]
        if 'v02' in MODIS_tile and composite_period in [0, 1, 20, 21, 22]:
            print(f"Skipping composite_period {composite_period} for MODIS_tile {MODIS_tile} (high latitude winter, no data).")
            sys.exit(0)
            
        # Sample filename: MCD19A1.h12v04.061_1km_16day_CV-MVC_lmFill_QCfilt_CloudFree_2012-19.nc
        MCD19A1_file = os.path.join(dat_dir, f'MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_lmFill_QCfilt_{QC_option}_{str(composite_year)}-{composite_period:02d}.nc')
        if not os.path.exists(MCD19A1_file):
            print(f"No MCD19A1 file found for tile {MODIS_tile}, composite_year {composite_year}, composite_period {composite_period}, QC_option {QC_option}, exiting with error code...")
            sys.exit(0)

        # sample filename: MCD19A1.h12v04.061_1km_CV-MVC_lmFill_QCfilt_CloudFree_hist_avg_10.nc
        hist_avg_file = os.path.join(hist_avg_dir, f'MCD19A1.{MODIS_tile}.061_1km_CV-MVC_lmFill_QCfilt_{QC_option}_hist_avg_{composite_period:02d}.nc')
        if not os.path.exists(hist_avg_file):
            print(f"No historical average file found for tile {MODIS_tile}, composite_year {composite_year}, composite_period {composite_period}, QC_option {QC_option}, exiting with error code...")
            sys.exit(0)
            
        out_file = os.path.join(out_dir, f'MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_stdAnom_QCfilt_{QC_option}_{composite_year}-{composite_period:02d}.nc')
        if os.path.exists(out_file):
            print(f"Output file already exists, exiting...")
            # sys.exit(0)



        all_maxmin_files = glob.glob(os.path.join(domain_scale_factor_dir, f'*.csv'))

        maxmin_data = {}
        for csv_file in all_maxmin_files:
            df = pd.read_csv(csv_file, index_col=0)
            for var in df.index:
                if var not in maxmin_data:
                    maxmin_data[var] = {'min': [], 'max': []}
                maxmin_data[var]['min'].append(df.loc[var, 'min']) # Note - this is actually 0.01 quantile, not the true min, to avoid outliers dominating the scaling
                maxmin_data[var]['max'].append(df.loc[var, 'max']) # Note - this is actually 0.99 quantile, not the true max, to avoid outliers dominating the scaling
        
        domain_scale_factors = {var: {'min': np.nanmin(maxmin_data[var]['min']), 
                           'max': np.nanmax(maxmin_data[var]['max'])} 
                       for var in maxmin_data}

        # Load in file and compute VIs
        try:
            dat_ds_1km_curv = xr.open_dataset(MCD19A1_file)
        except OSError as e:
            print(f"Skipping corrupt file {MCD19A1_file}: {e}")
            sys.exit(0)
            
        dat_ds_1km_curv = dat_ds_1km_curv.assign(CCI=(dat_ds_1km_curv["Sur_refl11"] - dat_ds_1km_curv["Sur_refl1"])/
                    (dat_ds_1km_curv["Sur_refl11"] + dat_ds_1km_curv["Sur_refl1"]))
        dat_ds_1km_curv = dat_ds_1km_curv.assign(NDMI=(dat_ds_1km_curv["Sur_refl6"] - dat_ds_1km_curv["Sur_refl1"])/
                    (dat_ds_1km_curv["Sur_refl6"] + dat_ds_1km_curv["Sur_refl1"]))
        dat_ds_1km_curv = dat_ds_1km_curv.assign(NDVI=(dat_ds_1km_curv["Sur_refl2"] - dat_ds_1km_curv["Sur_refl1"])/
                    (dat_ds_1km_curv["Sur_refl2"] + dat_ds_1km_curv["Sur_refl1"]))
        dat_ds_1km_curv = dat_ds_1km_curv.assign(NDSI=(dat_ds_1km_curv["Sur_refl4"] - dat_ds_1km_curv["Sur_refl6"])/
                    (dat_ds_1km_curv["Sur_refl4"] + dat_ds_1km_curv["Sur_refl6"]))
        dat_ds_1km_curv = dat_ds_1km_curv.assign(NIRv=(dat_ds_1km_curv["Sur_refl2"] - dat_ds_1km_curv["Sur_refl1"])*dat_ds_1km_curv["Sur_refl2"]/
                    (dat_ds_1km_curv["Sur_refl2"] + dat_ds_1km_curv["Sur_refl1"]))

        hist_avg_ds_1km_curv = xr.open_dataset(hist_avg_file)

        dat_ds_1km_curv_scaled = dat_ds_1km_curv.copy()
        hist_avg_ds_1km_curv_scaled = hist_avg_ds_1km_curv.copy()
        for dat_var in ['CCI', 'NDVI', 'NIRv', 'NDMI', 'NDSI']:
            dat_ds_1km_curv_scaled[dat_var] = (dat_ds_1km_curv[dat_var] - domain_scale_factors[dat_var]['min']) / (domain_scale_factors[dat_var]['max'] - domain_scale_factors[dat_var]['min'])
            hist_avg_ds_1km_curv_scaled[dat_var] = (hist_avg_ds_1km_curv[dat_var] - domain_scale_factors[dat_var]['min']) / (domain_scale_factors[dat_var]['max'] - domain_scale_factors[dat_var]['min'])
            hist_avg_ds_1km_curv_scaled[f'{dat_var}_std'] = hist_avg_ds_1km_curv[f'{dat_var}_std'] / (domain_scale_factors[dat_var]['max'] - domain_scale_factors[dat_var]['min'])

        dat_ds_CCI_std_anom = (dat_ds_1km_curv_scaled['CCI'] - hist_avg_ds_1km_curv_scaled['CCI']) / hist_avg_ds_1km_curv_scaled['CCI_std']
        dat_ds_NDVI_std_anom = (dat_ds_1km_curv_scaled['NDVI'] - hist_avg_ds_1km_curv_scaled['NDVI']) / hist_avg_ds_1km_curv_scaled['NDVI_std']
        dat_ds_NIRv_std_anom = (dat_ds_1km_curv_scaled['NIRv'] - hist_avg_ds_1km_curv_scaled['NIRv']) / hist_avg_ds_1km_curv_scaled['NIRv_std']
        dat_ds_NDMI_std_anom = (dat_ds_1km_curv_scaled['NDMI'] - hist_avg_ds_1km_curv_scaled['NDMI']) / hist_avg_ds_1km_curv_scaled['NDMI_std']
        dat_ds_NDSI_std_anom = (dat_ds_1km_curv_scaled['NDSI'] - hist_avg_ds_1km_curv_scaled['NDSI']) / hist_avg_ds_1km_curv_scaled['NDSI_std']
        dat_ds_std_anom = xr.Dataset({
            'CCI_std_anomaly': dat_ds_CCI_std_anom,
            'NDVI_std_anomaly': dat_ds_NDVI_std_anom,
            'NIRv_std_anomaly': dat_ds_NIRv_std_anom,
            'NDMI_std_anomaly': dat_ds_NDMI_std_anom,
            'NDSI_std_anomaly': dat_ds_NDSI_std_anom,
        })


        # Finally, save to netcdf
        dat_ds_std_anom.to_netcdf(path=out_file, format="NETCDF4")


        del dat_ds_CCI_std_anom, dat_ds_NDVI_std_anom, dat_ds_NIRv_std_anom, dat_ds_NDMI_std_anom, dat_ds_NDSI_std_anom
        gc.collect()

        ### Plot a sample to verify that regridding worked
        if (int(composite_year) in plot_years) and (int(composite_period) == plot_period):

            tile_plot_extent = [MODIS_tilecorners_df.loc[MODIS_tile].lon_min, MODIS_tilecorners_df.loc[MODIS_tile].lon_max, MODIS_tilecorners_df.loc[MODIS_tile].lat_min, MODIS_tilecorners_df.loc[MODIS_tile].lat_max]
            # plot CCI std anomaly (curvilinear grid)
            plot_file = os.path.join(plot_dir, f'MCD19A1_{MODIS_tile}_1km_curv_CCI_std_anom_{composite_year}-{composite_period}.png')
            test_plot(dat_ds_std_anom['CCI_std_anomaly'],plot_title=f'CCI Std Anomaly, {composite_year}-{composite_period}', 
                    zlab='z', plot_filename=plot_file, plot_extent=tile_plot_extent)
            
            # plot NDVI std anomaly (curvilinear grid)
            plot_file = os.path.join(plot_dir, f'MCD19A1_{MODIS_tile}_1km_curv_NDVI_std_anom_{composite_year}-{composite_period}.png')
            test_plot(dat_ds_std_anom['NDVI_std_anomaly'],plot_title=f'NDVI Std Anomaly, {composite_year}-{composite_period}', 
                    zlab='z', plot_filename=plot_file, plot_extent=tile_plot_extent)

            # plot NDMI std anomaly (curvilinear grid)
            plot_file = os.path.join(plot_dir, f'MCD19A1_{MODIS_tile}_1km_curv_NDMI_std_anom_{composite_year}-{composite_period}.png')
            test_plot(dat_ds_std_anom['NDMI_std_anomaly'],plot_title=f'NDMI Std Anomaly, {composite_year}-{composite_period}', 
                    zlab='z', plot_filename=plot_file, plot_extent=tile_plot_extent)

            # plot NIRv std anomaly (curvilinear grid)
            plot_file = os.path.join(plot_dir, f'MCD19A1_{MODIS_tile}_1km_curv_NIRv_std_anom_{composite_year}-{composite_period}.png')
            test_plot(dat_ds_std_anom['NIRv_std_anomaly'],plot_title=f'NIRv Std Anomaly, {composite_year}-{composite_period}', 
                    zlab='z', plot_filename=plot_file, plot_extent=tile_plot_extent)

        # Clean up
        dat_ds_std_anom.close()
        gc.collect()

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)  
    
if __name__ == "__main__":
    main()

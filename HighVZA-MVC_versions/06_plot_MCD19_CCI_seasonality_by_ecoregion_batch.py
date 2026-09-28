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
import gc

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
from rasterio.enums import Resampling

# shapefile/geospatial packages
import geopandas as gpd
from shapely.geometry import mapping

# time and date packages
import time
from datetime import datetime as dt  # date time library
from datetime import timedelta

# plotting packages
import pickle
from matplotlib import pyplot as plt  # primary plotting module

#%%
#########################################
# Define Global Filepaths
#########################################
mask_ENF_high = False
mask_ENF_low = False

pixel_presence_threshold = 5  # threshold for % of possible pixels that must be present in a composite period to include it in the analysis

# directories
QC_option = 'CloudFree_LowAOD_ClearAdj'
MCD19_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/HighVZA-MVC/01d_rect/QCfilt_{QC_option}/'

EPA_ecoregion_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/'
EPA_ecoregion_L1_file = os.path.join(EPA_ecoregion_dir, 'L1', 'NA_CEC_Eco_Level1_EPSG4326_regrid.shp')
EPA_ecoregion_L2_file = os.path.join(EPA_ecoregion_dir, 'L2/NAm_shp_FOR_MAIAC_PROC', 'NA_CEC_Eco_Level2_withOther.shp')
EPA_ecoregion_L3_file = os.path.join(EPA_ecoregion_dir, 'L3/NAm_shp_FOR_MAIAC_PROC', 'NAm_cec_eco_l3_v2.shp')

plot_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/seasonality_HighVZA-MVC/QCfilt_{QC_option}/'
Path(plot_basedir).mkdir(parents=True, exist_ok=True)
ENF_mask_file = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/needleleaf_evergreen_closed_forest_fraction_0.01deg_north_america.nc'
ENF_threshold_high = 0.8
ENF_threshold_low = 0.2

# eddy covariance data


########################################
# Define Global Variables and constants
#########################################
#%%

ecoregions_l1_list = ['TUNDRA',
                      'TAIGA', 
                      'NORTHERN FORESTS',
                      'HUDSON PLAIN',
                      'NORTHWESTERN FORESTED MOUNTAINS',
                      'MARINE WEST COAST FOREST', 
                      "EASTERN TEMPERATE FORESTS", 
                      "GREAT PLAINS", 
                      "NORTH AMERICAN DESERTS",
                      "MEDITERRANEAN CALIFORNIA",
                      "SOUTHERN SEMIARID HIGHLANDS",
                      "TEMPERATE SIERRAS",
                      "TROPICAL WET FORESTS",
                      "TROPICAL DRY FORESTS"]

ecoregions_l2_list = ['SOUTH CENTRAL SEMIARID PRAIRIES', 
                      'WESTERN CORDILLERA', 
                      'BOREAL CORDILLERA', 
                      "SOFTWOOD SHIELD", 
                      "SOUTHEASTERN USA PLAINS", 
                      "WARM DESERTS"]

ecoregions_l1_tiles_list={
    'TUNDRA': ['h11v02', 'h12v02', 'h13v02', 'h14v02', 'h15v02', 'h14v03','h12v01', 'h13v01', 'h14v01', 'h15v01'],
    'TAIGA': ['h10v02', 'h11v02', 'h12v02', 'h11v03', 'h12v03', 'h13v03', 'h14v03'],
    'NORTHERN FORESTS': ['h11v03', 'h12v03', 'h13v03', 'h14v03', 'h11v04', 'h12v04', 'h13v04', 'h14v04'],
    'HUDSON PLAIN': ['h12v03', 'h13v03'],
    'NORTHWESTERN FORESTED MOUNTAINS': ['h11v02', 'h12v02', 'h10v03', 'h11v03', 'h08v04', 'h09v04', 'h10v04', 'h08v05', 'h09v05'],
    'MARINE WEST COAST FOREST': ['h10v02', 'h09v03', 'h10v03', 'h11v03', 'h09v04', 'h08v04','h08v05'],
    "EASTERN TEMPERATE FORESTS": ['h11v04', 'h12v04', 'h13v04', 'h10v05', 'h11v05', 'h12v05', 'h10v06'],
    "GREAT PLAINS": ['h11v03', 'h10v04', 'h11v04', 'h09v05', 'h10v05', 'h09v06'],
    "NORTH AMERICAN DESERTS": ['h08v04', 'h09v04', 'h10v04', 'h08v05', 'h09v05', 'h07v06', 'h08v06'],
    "MEDITERRANEAN CALIFORNIA": ['h08v04', 'h08v05'],
    "SOUTHERN SEMIARID HIGHLANDS": ['h08v05', 'h08v06'],
    "TEMPERATE SIERRAS": ['h08v05', 'h09v05', 'h08v06', 'h08v07', 'h09v07'],
    "TROPICAL WET FORESTS": ['h08v06', 'h09v06', 'h10v06', 'h08v07', 'h09v07'],
    "TROPICAL DRY FORESTS": ['h07v06', 'h08v06', 'h09v06', 'h08v07', 'h09v07']
}

ecoregions_l2_tiles_list={
    'SOUTH CENTRAL SEMIARID PRAIRIES': ['h10v04', 'h10v05', 'h09v05'],
    'WESTERN CORDILLERA': ['h10v03', 'h11v03', 'h09v04', 'h10v04', 'h08v04', 'h08v05', 'h09v05'],
    'BOREAL CORDILLERA': ['h10v02', 'h11v02', 'h10v03', 'h11v03'],
    "SOFTWOOD SHIELD": ['h12v04', 'h13v03', 'h13v04', 'h12v03'],
    "SOUTHEASTERN USA PLAINS": ['h10v05', 'h11v05'],
    "WARM DESERTS": ['h08v05', 'h08v06', 'h09v05', 'h07v06']
}

all_tiles_list = sorted(set(tile for tiles in ecoregions_l2_tiles_list.values() for tile in tiles))
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

DOY_composite_starts = np.arange(1, 366, 16)
composite_dates_2020 = [dt(2020, 1, 1) + timedelta(days=int(DOY) - 1) for DOY in DOY_composite_starts]

composite_periods = np.arange(0,23)
#########################################
# Begin Main
#########################################

### L1 Colors
tundra_color = "#BFDDF2"
taiga_color = "#5173B1"
northern_forests_color = '#44AA99'
hudson_plain_color = "#88AFEE"
northwestern_forested_mountains_color = "#006713"
marine_west_coast_forest_color = "#4492AA"
eastern_temperate_forests_color = "#98CE68"
great_plains_color = "#A59918"
north_american_deserts_color = "#BFB55C"
mediterranean_california_color = "#B1BE77"
southern_semiarid_highlands_color = "#BCB781"
temperate_sierras_color = "#99BD8A"
tropical_wet_forests_color = "#891574"
tropical_dry_forests_color = "#992424"

### L2 Colors
warm_desert_color = '#CC6677'
western_cordillera_color = '#117733'
softwood_shield_color = '#44AA99'
boreal_cordillera_color = '#88CCEE'
southeastern_usa_plains_color = '#332288'
south_central_semiarid_prairies_color = '#825F09'


def main():


    ecoregion = f'{sys.argv[1].replace("_", " ")}' #SOUTH_CENTRAL_SEMIARID_PRAIRIES
    QC_option = sys.argv[2] # 'CloudFree'


    ecoregion_dat_dict = {}

    plot_color = {
        'SOUTH CENTRAL SEMIARID PRAIRIES': south_central_semiarid_prairies_color,
        'WESTERN CORDILLERA': western_cordillera_color,
        'BOREAL CORDILLERA': boreal_cordillera_color,
        "SOFTWOOD SHIELD": softwood_shield_color,
        "SOUTHEASTERN USA PLAINS": southeastern_usa_plains_color,
        "WARM DESERTS": warm_desert_color,
        "TUNDRA": tundra_color,
        'TAIGA': taiga_color,
        'NORTHERN FORESTS': northern_forests_color,
        'HUDSON PLAIN': hudson_plain_color,
        'NORTHWESTERN FORESTED MOUNTAINS': northwestern_forested_mountains_color,
        'MARINE WEST COAST FOREST': marine_west_coast_forest_color,
        "EASTERN TEMPERATE FORESTS": eastern_temperate_forests_color,
        "GREAT PLAINS": great_plains_color,
        "NORTH AMERICAN DESERTS": north_american_deserts_color,
        "MEDITERRANEAN CALIFORNIA": mediterranean_california_color,
        "SOUTHERN SEMIARID HIGHLANDS": southern_semiarid_highlands_color,
        "TEMPERATE SIERRAS": temperate_sierras_color,
        "TROPICAL WET FORESTS": tropical_wet_forests_color,
        "TROPICAL DRY FORESTS": tropical_dry_forests_color
    }[ecoregion]

    print(f'Processing ecoregion: {ecoregion}...')
    

    if ecoregion in ecoregions_l1_list:
        ecoregions_gdf = gpd.read_file(EPA_ecoregion_L1_file)
        # print(ecoregion_l2_gdf['NA_L2NAME'].unique())
        ecoregion_gdf = ecoregions_gdf[ecoregions_gdf.NA_L1NAME == ecoregion]
        tiles_to_process = ecoregions_l1_tiles_list[ecoregion]
    elif ecoregion in ecoregions_l2_list:
        ecoregions_gdf = gpd.read_file(EPA_ecoregion_L2_file)
        # print(ecoregion_l2_gdf['NA_L2NAME'].unique())
        ecoregion_gdf = ecoregions_gdf[ecoregions_gdf.NA_L2NAME == ecoregion]
        tiles_to_process = ecoregions_l2_tiles_list[ecoregion]
    else:
        raise ValueError(f'Ecoregion {ecoregion} not found in either L1 or L2 ecoregion shapefiles. Please check the name and try again.')
    
    ecoregion_gdf_geom = ecoregion_gdf.geometry

    print("Loading ENF mask...")
    ENF_xr = xr.open_dataset(ENF_mask_file, decode_coords='all')['fraction_ENF'].squeeze()

    ENF_regrids = {}

    CCI_mean_arr = []
    CCI_std_arr = []
    NDVI_mean_arr = []
    NDVI_std_arr = []
    npixels_arr = []
    # composite_period = composite_periods[0]
    for composite_period in composite_periods:

        print(f'  Processing composite period: {composite_period}...')

        CCI_mean_pd_arr = []
        CCI_std_pd_arr = []
        NDVI_mean_pd_arr = []
        NDVI_std_pd_arr = []
        npixels_pd_arr = []
        # MODIS_tile = tiles_to_process[0]
        for MODIS_tile in tiles_to_process:

            if ('v01' in MODIS_tile) and (composite_period in [0, 1, 2, 3, 4, 18, 19, 20, 21, 22]):
                print(f'    Skipping tile {MODIS_tile} for composite period {composite_period} due to lack of data...')
                continue

            if ('v02' in MODIS_tile) and (composite_period in [0, 1, 2, 3, 19, 20, 21, 22]):
                print(f'    Skipping tile {MODIS_tile} for composite period {composite_period} due to lack of data...')
                continue

            MCD19_tile_dir = os.path.join(MCD19_basedir, MODIS_tile, 'hist_avg')

            #MCD19A1.h10v04.061_01d_16day_HighVZA-MVC_QCfilt_CloudFree_LowAOD_ClearAdj_2025-05_rect.nc
            MCD19_tile_file = os.path.join(MCD19_tile_dir, f'MCD19A1.{MODIS_tile}.061_01d_HighVZA-MVC_QCfilt_{QC_option}_hist_avg_{composite_period:02d}.nc')
            
            if not os.path.exists(MCD19_tile_file):
                print(f'    File {MCD19_tile_file} does not exist, skipping tile {MODIS_tile} for composite period {composite_period}...')
                continue

            MCD19_xr_tile = xr.open_dataset(MCD19_tile_file, decode_coords = 'all').rename({'lat': 'y', 'lon': 'x'})

            # Assume that both mask_ENF_high and mask_ENF_low cannot be True at the same time. If both are False, then no masking is applied.
            if mask_ENF_high and mask_ENF_low:
                raise ValueError("Both mask_ENF_high and mask_ENF_low cannot be True at the same time. Please set one of them to False.")

            # If masking for either high or low ENF, perform filtering here
            if mask_ENF_high | mask_ENF_low:
                # Regrid ENF mask to this tile's grid if doesn't already exist in the dictionary
                if MODIS_tile not in ENF_regrids:
                    print("  Regridding ENF mask to tile grid...")
                    ENF_regrids[MODIS_tile] = ENF_xr.rio.write_crs("EPSG:4326").rio.reproject_match(MCD19_xr_tile, resampling=Resampling.bilinear, nodata=np.nan)

                
                ENF_regrid = ENF_regrids[MODIS_tile]

                if mask_ENF_high:
                    MCD19_xr_tile = MCD19_xr_tile.where(ENF_regrid > ENF_threshold_high)
                
                if mask_ENF_low:
                    MCD19_xr_tile = MCD19_xr_tile.where(ENF_regrid < ENF_threshold_low)

            
            # Now continue, assuming appropriate land cover masking has been applied if necessary
            MCD19_xr_clip = MCD19_xr_tile.rio.clip(ecoregion_gdf_geom.apply(mapping), drop=False)
            if MCD19_xr_clip['CCI'].isnull().all():
                print(f'    No valid data in clipped region for tile {MODIS_tile}, composite period {composite_period}')
                continue
            MCD19_xr_composite_mean = MCD19_xr_clip.mean(dim=['x', 'y'], skipna=True)
            MCD19_xr_composite_std = MCD19_xr_clip.std(dim=['x', 'y'], skipna=True)
            MCD19_xr_CCI_spacetime_err = np.sqrt((MCD19_xr_composite_std['CCI']**2) + MCD19_xr_composite_mean['CCI_std']**2)
            MCD19_xr_NDVI_spacetime_err = np.sqrt((MCD19_xr_composite_std['NDVI']**2) + MCD19_xr_composite_mean['NDVI_std']**2)
            MCD19_xr_npixels = MCD19_xr_clip['CCI'].count(dim=['x', 'y'])

            CCI_mean_pd_arr.append(MCD19_xr_composite_mean['CCI'].values)
            CCI_std_pd_arr.append(MCD19_xr_CCI_spacetime_err.values)
            NDVI_mean_pd_arr.append(MCD19_xr_composite_mean['NDVI'].values)
            NDVI_std_pd_arr.append(MCD19_xr_NDVI_spacetime_err.values)
            npixels_pd_arr.append(MCD19_xr_npixels.values)
        
        all_pixels = np.sum(npixels_pd_arr)
        npixels_arr.append(all_pixels)

        if all_pixels == 0 | np.isnan(all_pixels):
            print(f'Warning: No valid pixels found for ecoregion {ecoregion} and composite period {composite_period}. Skipping this period.')
            CCI_mean_arr.append(np.nan)
            CCI_std_arr.append(np.nan)
            NDVI_mean_arr.append(np.nan)
            NDVI_std_arr.append(np.nan)
            continue

        weights = np.array(npixels_pd_arr) / all_pixels

        CCI_mean_pd = np.average(CCI_mean_pd_arr, weights=weights)
        CCI_std_pd = np.average(CCI_std_pd_arr, weights=weights)
        NDVI_mean_pd = np.average(NDVI_mean_pd_arr, weights=weights)
        NDVI_std_pd = np.average(NDVI_std_pd_arr, weights=weights)


        CCI_mean_arr.append(CCI_mean_pd)
        CCI_std_arr.append(CCI_std_pd)
        NDVI_mean_arr.append(NDVI_mean_pd)
        NDVI_std_arr.append(NDVI_std_pd)

        del CCI_mean_pd_arr, CCI_std_pd_arr, NDVI_mean_pd_arr, NDVI_std_pd_arr, npixels_pd_arr
        gc.collect()    

    del ENF_regrids
    gc.collect()

    npixels_max = np.max(npixels_arr)

    ecoregion_dat_dict = {
        'CCI_mean': CCI_mean_arr,
        'CCI_std': CCI_std_arr,
        'NDVI_mean': NDVI_mean_arr,
        'NDVI_std': NDVI_std_arr,
        'pixels_perc': npixels_arr / npixels_max if npixels_max > 0 else np.zeros_like(npixels_arr),  
    }

    # Exclude composite periods with insufficient pixel coverage from the summary data.
    low_presence = np.asarray(ecoregion_dat_dict['pixels_perc']) < (pixel_presence_threshold/100)
    for variable in ['CCI_mean', 'CCI_std', 'NDVI_mean', 'NDVI_std']:
        values = np.asarray(ecoregion_dat_dict[variable], dtype=float)
        values[low_presence] = np.nan
        ecoregion_dat_dict[variable] = values

    # import pickle
    output_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/summary_data'
    Path(output_dir).mkdir(parents=True, exist_ok=True)


    if mask_ENF_high:
        output_file = os.path.join(output_dir, f'{ecoregion.replace(" ", "_")}_seasonality_HighVZA-MVC_dict_ENF_only.pkl')
    elif mask_ENF_low:
        output_file = os.path.join(output_dir, f'{ecoregion.replace(" ", "_")}_seasonality_HighVZA-MVC_dict_nonENF.pkl')
    else:
        output_file = os.path.join(output_dir, f'{ecoregion.replace(" ", "_")}_seasonality_HighVZA-MVC_dict.pkl')

    with open(output_file, 'wb') as f:
        pickle.dump(ecoregion_dat_dict, f)

    print(f'Saved ecoregion_seasonality_dict to {output_file}')

    #plotting vars
    months_begin = np.array([1, 32, 61, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365])
    months_doy = (months_begin[1:] + months_begin[:-1]) / 2
    months_labs = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    plt.rcParams.update({
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 17,
    'legend.fontsize': 13,
    })


    fig, ax = plt.subplots(figsize=(4.5, 4)) # establish the figure, axes
    ax.grid(color = "gray", linestyle = "--", linewidth = 0.5, alpha = 0.3) # add grid lines

    # if ecoregion in ['TUNDRA', 'TAIGA', 'HUDSON PLAIN', 'BOREAL CORDILLERA','NORTHWESTERN FORESTED MOUNTAINS', 'MARINE WEST COAST FOREST', "NORTHERN FORESTS"]:
    # Find first and last non-nan indices
    non_nan_indices = np.where(~np.isnan(ecoregion_dat_dict['CCI_mean']))[0]
    if len(non_nan_indices) > 0:
        first_non_nan_idx = non_nan_indices[0]
        last_non_nan_idx = non_nan_indices[-1]
        
        # Plot gray shaded regions
        ax.axvspan(-10, DOY_composite_starts[first_non_nan_idx], color='gray', alpha=0.2)
        ax.axvspan(DOY_composite_starts[last_non_nan_idx], 380, color='gray', alpha=0.2)

    ax2 = ax.twinx()
    # ax2.set_zorder(ax.get_zorder() - 1)
    # ax.patch.set_visible(False)
    ax2.plot(DOY_composite_starts, ecoregion_dat_dict['NDVI_mean'], color='black', lw=2)
    ax2.plot(DOY_composite_starts, ecoregion_dat_dict['NDVI_mean'] - ecoregion_dat_dict['NDVI_std'], color='black', linestyle='--', lw=1.5)
    ax2.plot(DOY_composite_starts, ecoregion_dat_dict['NDVI_mean'] + ecoregion_dat_dict['NDVI_std'], color='black', linestyle='--', lw=1.5)
    ax2.set_ylabel('NDVI')
    # in faint lines, plot the individual years of the data

    ax.plot(DOY_composite_starts, ecoregion_dat_dict['CCI_mean'], color=plot_color, lw = 2.5)
    ax.plot(DOY_composite_starts, ecoregion_dat_dict['CCI_mean'] - ecoregion_dat_dict['CCI_std'], color=plot_color, linestyle='--', lw=1.5)
    ax.plot(DOY_composite_starts, ecoregion_dat_dict['CCI_mean'] + ecoregion_dat_dict['CCI_std'], color=plot_color, linestyle='--', lw=1.5)

    # Plot shaded region for mean ± representative std
    ax.fill_between(
        DOY_composite_starts,
        ecoregion_dat_dict['CCI_mean'] - ecoregion_dat_dict['CCI_std'],
        ecoregion_dat_dict['CCI_mean'] + ecoregion_dat_dict['CCI_std'],
        color=plot_color, alpha=0.5, label='±1 std'
    )

    # Define the axis, title labels
    # set xticks to Month abbreviations
    ax.set_xticks(ticks=months_doy)
    ax.set_xticklabels(months_labs)
    ax.set_xlim(0, 365)
    # ax.set_ylim(PRB_min, PRB_max)

    ax.yaxis.label.set_color(plot_color)
    ax.tick_params(axis='y', colors=plot_color)
    ax.spines['left'].set_color(plot_color)

    ax.set_ylabel('CCI')

    if mask_ENF_high:
        plot_filename = f'{ecoregion.replace(" ", "_")}_seasonality_HighVZA-MVC_ENF.png'
    elif mask_ENF_low:
        plot_filename = f'{ecoregion.replace(" ", "_")}_seasonality_HighVZA-MVC_nonENF.png'
    else:
        plot_filename = f'{ecoregion.replace(" ", "_")}_seasonality_HighVZA-MVC.png'
    
    plt.savefig(os.path.join(plot_basedir, plot_filename), dpi=300, bbox_inches='tight')
    plt.close()
    # plt.show()
    
if __name__ == "__main__":
    main()
# %%


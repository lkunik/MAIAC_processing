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
import pickle
from matplotlib import pyplot as plt  # primary plotting module

#%%
#########################################
# Define Global Filepaths
#########################################

# directories
QC_option = 'CloudFree'
avg_period = 'JJA'

dat_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/summary_data'

plot_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/annual_ts/QCfilt_{QC_option}/'
Path(plot_basedir).mkdir(parents=True, exist_ok=True)

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

#%%

analysis_years = np.arange(2000, 2026)
tick_years = [2000, 2005, 2010, 2015, 2020, 2025]

ecoregions_super_list = sorted(set(ecoregions_l1_list + ['WESTERN CORDILLERA', 'BOREAL CORDILLERA']))
for ecoregion in ecoregions_super_list:


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


    # for ecoregion in ecoregions_l2_list:
    print(f'Processing ecoregion: {ecoregion}...')
    
    input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_annual_dict.pkl')

    if not os.path.exists(input_file):
        print(f'Input file not found for {ecoregion}: {input_file}')
        continue

    with open(input_file, 'rb') as f:
        ecoregion_dat_dict = pickle.load(f)

    print(f'Loaded ecoregion_annual_dict from {input_file}')

    plt.rcParams.update({
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 17,
    'legend.fontsize': 13,
    })

    CCI_mean_arr = ecoregion_dat_dict['CCI_mean']
    CCI_std_arr = ecoregion_dat_dict['CCI_std']
    NDVI_mean_arr = ecoregion_dat_dict['NDVI_mean']
    NDVI_std_arr = ecoregion_dat_dict['NDVI_std']

    fig, ax = plt.subplots(figsize=(5.5, 2.6)) # establish the figure, axes
    # ax.grid(color = "gray", linestyle = "--", linewidth = 1, alpha = 0.5) # add grid lines
    # in faint lines, plot the individual years of the data

    ax2 = ax.twinx()
    # ax2.set_zorder(ax.get_zorder() - 1)
    # ax.patch.set_visible(False)
    ax2.plot(analysis_years, NDVI_mean_arr, color='black', lw=2)
    ax2.plot(analysis_years, np.array(NDVI_mean_arr) - np.array(NDVI_std_arr), color='black', linestyle='--', lw=1.5)
    ax2.plot(analysis_years, np.array(NDVI_mean_arr) + np.array(NDVI_std_arr), color='black', linestyle='--', lw=1.5)
    ax2.set_ylabel('NDVI')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))


    ax.plot(analysis_years, CCI_mean_arr, color=plot_color, lw = 2.5)
    ax.plot(analysis_years, np.array(CCI_mean_arr) - np.array(CCI_std_arr), color=plot_color, linestyle='--', lw=1.5)
    ax.plot(analysis_years, np.array(CCI_mean_arr) + np.array(CCI_std_arr), color=plot_color, linestyle='--', lw=1.5)

    # Plot shaded region for mean ± representative std
    ax.fill_between(
        analysis_years,
        np.array(CCI_mean_arr) - np.array(CCI_std_arr),
        np.array(CCI_mean_arr) + np.array(CCI_std_arr),
        color=plot_color, alpha=0.2, label='±1 std'
    )

    # Define the axis, title labels
    # set xticks to Month abbreviations
    ax.set_xticks(ticks=tick_years)
    ax.set_xticklabels([str(year) for year in tick_years], rotation=45)
    ax.set_xlim(analysis_years[0]-0.5, analysis_years[-1]+0.5)

    ax.yaxis.label.set_color(plot_color)
    ax.tick_params(axis='y', colors=plot_color)
    ax.spines['left'].set_color(plot_color)
    ax.set_ylabel('CCI')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))


    plot_filename = f'{ecoregion.replace(" ", "_")}_annual.png'
    plt.savefig(os.path.join(plot_basedir, plot_filename), dpi=300, bbox_inches='tight')
    plt.close()
    # plt.show()


# %%

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
from scipy import stats

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

dat_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/summary_data'
plot_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/annual_ts/QCfilt_{QC_option}/'
Path(plot_basedir).mkdir(parents=True, exist_ok=True)

# TerraClimate_pickle_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/TerraClimate/data/summary_data'
# FluxCom_pickle_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/FLUXCOM/FLUXCOM-X/data/summary_data'

########################################
# Define Global Variables and constants
#########################################
#%%

ecoregions_list = ['Temperate', 'Boreal']

#########################################
# Define Global Functions
#########################################



#########################################
# Begin Main
#########################################



# print(ecoregion_l2_gdf['NA_L2NAME'].unique())
analysis_years = np.arange(2000, 2026)
tick_years = [2000, 2005, 2010, 2015, 2020, 2025]



#%%
# mark start time to keep track of elapsed
start_time = time.time()

ENF_regrids = {}

analysis_years = np.arange(2000, 2026)

ecoregions = ['Temperate', 'Boreal']

TC_vars = ['CWD_anomaly', 'PDSI', 'SoilM_anomaly', 'VPD_anomaly', 'PET_anomaly']
TC_labels = {
    'CWD_anomaly': 'CWD anomaly (mm)',
    'PDSI': 'PDSI',
    'SoilM_anomaly': 'SM anomaly (mm)',
    'VPD_anomaly': 'VPD anomaly (kPa)',
    'PET_anomaly': 'PET anomaly (mm)'
}
mask_ENF = True

avg_period = 'JJA'


#######################################################################
### Pass 1 - get ylims
#######################################################################

ax_ylim = [1000, -1000]
ax2_ylim = [1000, -1000]
ax3_ylim = [1000, -1000]
vpd_ylim = [1000, -1000]
pdsi_ylim = [1000, -1000]

for ecoregion in ecoregions:
    print(f'Processing ecoregion: {ecoregion}...')

    if mask_ENF:
        input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_annual_dict_ENF_only.pkl')
        FLUXCOM_input_file = os.path.join(dat_dir, f'FLUXCOM_GPP_{ecoregion.replace(" ", "_")}_annual_dict_ENF_only.pkl')
        TC_VPD_file = os.path.join(dat_dir, f'TerraClimate_VPD_anomaly_{ecoregion.replace(" ", "_")}_annual_dict_ENF_only.pkl')
        TC_PDSI_file = os.path.join(dat_dir, f'TerraClimate_PDSI_{ecoregion.replace(" ", "_")}_annual_dict_ENF_only.pkl')
    else:
        input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_annual_dict_nonENF.pkl')
        FLUXCOM_input_file = os.path.join(dat_dir, f'FLUXCOM_GPP_{ecoregion.replace(" ", "_")}_annual_dict_nonENF.pkl')
        TC_VPD_file = os.path.join(dat_dir, f'TerraClimate_VPD_anomaly_{ecoregion.replace(" ", "_")}_annual_dict_nonENF.pkl')
        TC_PDSI_file = os.path.join(dat_dir, f'TerraClimate_PDSI_{ecoregion.replace(" ", "_")}_annual_dict_nonENF.pkl')

    with open(input_file, 'rb') as f:
        ecoregion_dat_dict = pickle.load(f)
    # with open(FLUXCOM_input_file, 'rb') as f:
    #     FLUXCOM_ecoregion_dat_dict = pickle.load(f)
    # with open(TC_VPD_file, 'rb') as f:
    #     TC_VPD_dict = pickle.load(f)
    # with open(TC_PDSI_file, 'rb') as f:
    #     TC_PDSI_dict = pickle.load(f)

    

    def plot_trend(ax, years, values, color, trend_years=None, lw=1.8, alpha=0.7):
        mask = ~np.isnan(values)
        if mask.sum() < 2:
            return
        slope, intercept, _, _, _ = stats.linregress(years[mask], values[mask])
        if trend_years is None:
            trend_years = years
        trend = slope * trend_years + intercept
        ax.plot(trend_years, trend, color=color, lw=lw, alpha=alpha, linestyle='--')


    suffix = 'ENF' if mask_ENF else 'nonENF'

    # -------------------------
    # Plot 1: CCI, NDVI, FLUXCOM GPP
    # -------------------------
    # FC_years = np.array(FLUXCOM_ecoregion_dat_dict['years'])
    # FC_mean  = np.array(FLUXCOM_ecoregion_dat_dict['GPP_mean'])

    ax1_ylim_scale = 0.75
    ax2_ylim_scale = 0.5
    ax3_ylim_scale = 0.25

    ax_ymin = np.nanmin(np.array(ecoregion_dat_dict['Refl1_mean']) - (np.array(ecoregion_dat_dict['Refl1_std'])/2))
    ax_ymax = np.nanmax(np.array(ecoregion_dat_dict['Refl1_mean']) + (np.array(ecoregion_dat_dict['Refl1_std'])/2))
    ax_yrange = ax_ymax - ax_ymin
    ax_ymin_adj = ax_ymin - (ax_yrange * (ax1_ylim_scale - 0.5))
    ax_ymax_adj = ax_ymax - (ax_yrange * (ax1_ylim_scale - 0.5))
    ax_ylim_primer = [min(ax_ymin_adj, np.nanmin(ecoregion_dat_dict['Refl1_mean'])),
            max(ax_ymax_adj, np.nanmax(ecoregion_dat_dict['Refl1_mean']))]

    ax_ylim[0] = min(ax_ylim[0], ax_ylim_primer[0])
    ax_ylim[1] = max(ax_ylim[1], ax_ylim_primer[1])

    ax2_ymin = np.nanmin(np.array(ecoregion_dat_dict['Refl2_mean']) - (np.array(ecoregion_dat_dict['Refl2_std'])/2))
    ax2_ymax = np.nanmax(np.array(ecoregion_dat_dict['Refl2_mean']) + (np.array(ecoregion_dat_dict['Refl2_std'])/2))
    ax2_yrange = ax2_ymax - ax2_ymin
    ax2_ymin_adj = ax2_ymin - (ax2_yrange * (ax2_ylim_scale - 0.5))
    ax2_ymax_adj = ax2_ymax - (ax2_yrange * (ax2_ylim_scale - 0.5))
    ax2_ylim_primer = [min([ax2_ymin_adj, np.nanmin(ecoregion_dat_dict['Refl2_mean'])]), 
                max([ax2_ymax_adj, np.nanmax(ecoregion_dat_dict['Refl2_mean'])])]

    ax2_ylim[0] = min(ax2_ylim[0], ax2_ylim_primer[0])
    ax2_ylim[1] = max(ax2_ylim[1], ax2_ylim_primer[1])


    ax3_ymin = np.nanmin(np.array(ecoregion_dat_dict['Refl11_mean']) - (np.array(ecoregion_dat_dict['Refl11_std'])/2))
    ax3_ymax = np.nanmax(np.array(ecoregion_dat_dict['Refl11_mean']) + (np.array(ecoregion_dat_dict['Refl11_std'])/2))
    ax3_yrange = ax3_ymax - ax3_ymin
    ax3_ymin_adj = ax3_ymin - (ax3_yrange * (ax3_ylim_scale - 0.5))
    ax3_ymax_adj = ax3_ymax - (ax3_yrange * (ax3_ylim_scale - 0.5))
    ax3_ylim_primer = [min([ax3_ymin_adj, np.nanmin(ecoregion_dat_dict['Refl11_mean'])]), 
                max([ax3_ymax_adj, np.nanmax(ecoregion_dat_dict['Refl11_mean'])])]

    ax3_ylim[0] = min(ax3_ylim[0], ax3_ylim_primer[0])
    ax3_ylim[1] = max(ax3_ylim[1], ax3_ylim_primer[1])


# %%


ax_ylim[1] = 0.036

ax3_ylim[0] = 0.023
#######################################################################
### Pass 2 - plot data
#######################################################################



for ecoregion in ecoregions:
    print(f'Processing ecoregion: {ecoregion}...')

    if mask_ENF:
        input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_annual_dict_ENF_only.pkl')
        FLUXCOM_input_file = os.path.join(dat_dir, f'FLUXCOM_GPP_{ecoregion.replace(" ", "_")}_annual_dict_ENF_only.pkl')
        TC_VPD_file = os.path.join(dat_dir, f'TerraClimate_VPD_anomaly_{ecoregion.replace(" ", "_")}_annual_dict_ENF_only.pkl')
        TC_PDSI_file = os.path.join(dat_dir, f'TerraClimate_PDSI_{ecoregion.replace(" ", "_")}_annual_dict_ENF_only.pkl')
    else:
        input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_annual_dict_nonENF.pkl')
        FLUXCOM_input_file = os.path.join(dat_dir, f'FLUXCOM_GPP_{ecoregion.replace(" ", "_")}_annual_dict_nonENF.pkl')
        TC_VPD_file = os.path.join(dat_dir, f'TerraClimate_VPD_anomaly_{ecoregion.replace(" ", "_")}_annual_dict_nonENF.pkl')
        TC_PDSI_file = os.path.join(dat_dir, f'TerraClimate_PDSI_{ecoregion.replace(" ", "_")}_annual_dict_nonENF.pkl')

    with open(input_file, 'rb') as f:
        ecoregion_dat_dict = pickle.load(f)
    # with open(FLUXCOM_input_file, 'rb') as f:
    #     FLUXCOM_ecoregion_dat_dict = pickle.load(f)
    # with open(TC_VPD_file, 'rb') as f:
    #     TC_VPD_dict = pickle.load(f)
    # with open(TC_PDSI_file, 'rb') as f:
    #     TC_PDSI_dict = pickle.load(f)

    plt.rcParams.update({
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 13,
    })

    # cci_color   = '#b2df8a'
    # ndvi_color  = '#33a02c'
    # gpp_color   = 'black'
    # vpd_color   = '#E4C217'
    # pdsi_color  = '#1f78b4'
    Refl1_color  = '#1f78b4'
    Refl2_color  = '#E4C217'
    Refl11_color = '#33a02c'

    from scipy import stats

    def plot_trend(ax, years, values, color, trend_years=None, lw=1.8, alpha=0.7):
        mask = ~np.isnan(values)
        if mask.sum() < 2:
            return
        slope, intercept, _, _, _ = stats.linregress(years[mask], values[mask])
        print(f'Trend slope: {slope:.2e} per year')
        if trend_years is None:
            trend_years = years
        trend = slope * trend_years + intercept
        ax.plot(trend_years, trend, color=color, lw=lw, alpha=alpha, linestyle='--')


    suffix = 'ENF' if mask_ENF else 'nonENF'

    # -------------------------
    # Plot 1: CCI, NDVI, FLUXCOM GPP
    # -------------------------
    # FC_years = np.array(FLUXCOM_ecoregion_dat_dict['years'])
    # FC_mean  = np.array(FLUXCOM_ecoregion_dat_dict['GPP_mean'])

    fig1, ax = plt.subplots(figsize=(6, 2.6))
    fig1.subplots_adjust(left=0.2, right=0.85)

    ax.grid(color='gray', linestyle='--', linewidth=1, alpha=0.4)

    # NDVI on the far left
    ax2 = ax.twinx()
    ax2.yaxis.set_label_position('left')
    ax2.yaxis.tick_left()
    ax2.spines['left'].set_position(('outward', 80))
    ax2.spines['right'].set_visible(False)

    # GPP on the right
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 0))

    ax.spines['left'].set_position(('outward', 0))
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    ax2.plot(analysis_years, ecoregion_dat_dict['Refl2_mean'], color=Refl2_color, lw=2)
    ax2.set_ylabel('Refl2', color=Refl2_color)
    ax2.set_ylim(ax2_ylim)
    ax2.tick_params(axis='y', colors=Refl2_color)
    ax2.spines['left'].set_color(Refl2_color)

    ax.plot(analysis_years, ecoregion_dat_dict['Refl11_mean'], color=Refl11_color, lw=2.5)
    ax.set_ylabel('Refl11', color=Refl11_color)
    ax.set_ylim(ax_ylim)
    ax.yaxis.label.set_color(Refl11_color)
    ax.tick_params(axis='y', colors=Refl11_color)
    ax.spines['left'].set_color(Refl11_color)

    ax3.plot(analysis_years, ecoregion_dat_dict['Refl1_mean'], color=Refl1_color, lw=2.5)
    ax3.set_ylabel('Refl1', color=Refl1_color)
    ax3.set_ylim(ax3_ylim)
    ax3.tick_params(axis='y', colors=Refl1_color)
    ax3.spines['right'].set_color(Refl1_color)

    print('plotting trend for Refl11...')
    plot_trend(ax,  analysis_years, np.array(ecoregion_dat_dict['Refl11_mean']),  Refl11_color)

    print('plotting trend for Refl2...')
    plot_trend(ax2, analysis_years, np.array(ecoregion_dat_dict['Refl2_mean']), Refl2_color)

    print('plotting trend for Refl1...')
    plot_trend(ax3, analysis_years, np.array(ecoregion_dat_dict['Refl1_mean']), Refl1_color)

    ax.set_xticks(tick_years)
    ax.set_xticklabels([str(y) for y in tick_years], rotation=45)
    ax.set_xlim(analysis_years[0] - 0.5, analysis_years[-1] + 0.5)

    plot_filename1 = f'{ecoregion.replace(" ", "_")}_Refls_annual_{suffix}.png'
    fig1.savefig(os.path.join(plot_basedir, plot_filename1), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f'Plot 1 saved to {os.path.join(plot_basedir, plot_filename1)}')

# %%

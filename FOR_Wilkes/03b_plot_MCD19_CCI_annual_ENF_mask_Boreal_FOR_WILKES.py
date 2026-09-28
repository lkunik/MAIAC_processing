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
# from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter

#%%
#########################################
# Define Global Filepaths
#########################################


# directories
QC_option = 'CloudFree'

dat_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/summary_data'
plot_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/FOR_Wilkes/'
Path(plot_basedir).mkdir(parents=True, exist_ok=True)

# TerraClimate_pickle_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/TerraClimate/data/summary_data'
# FluxCom_pickle_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/FLUXCOM/FLUXCOM-X/data/summary_data'

########################################
# Define Global Variables and constants
#########################################
#%%

ecoregions_list = ['Boreal']

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

ecoregions = ['Boreal']

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
    with open(FLUXCOM_input_file, 'rb') as f:
        FLUXCOM_ecoregion_dat_dict = pickle.load(f)
    with open(TC_VPD_file, 'rb') as f:
        TC_VPD_dict = pickle.load(f)
    with open(TC_PDSI_file, 'rb') as f:
        TC_PDSI_dict = pickle.load(f)

    

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
    FC_years = np.array(FLUXCOM_ecoregion_dat_dict['years'])
    FC_mean  = np.array(FLUXCOM_ecoregion_dat_dict['GPP_mean'])

    ax1_ylim_scale = 0.75
    ax2_ylim_scale = 0.5
    ax3_ylim_scale = 0.25

    ax_ymin = np.nanmin(np.array(ecoregion_dat_dict['CCI_mean']) - (np.array(ecoregion_dat_dict['CCI_std'])/2))
    ax_ymax = np.nanmax(np.array(ecoregion_dat_dict['CCI_mean']) + (np.array(ecoregion_dat_dict['CCI_std'])/2))
    ax_yrange = ax_ymax - ax_ymin
    ax_ymin_adj = ax_ymin - (ax_yrange * (ax1_ylim_scale - 0.5))
    ax_ymax_adj = ax_ymax - (ax_yrange * (ax1_ylim_scale - 0.5))
    ax_ylim_primer = [min(ax_ymin_adj, np.nanmin(ecoregion_dat_dict['CCI_mean'])),
            max(ax_ymax_adj, np.nanmax(ecoregion_dat_dict['CCI_mean']))]

    ax_ylim[0] = min(ax_ylim[0], ax_ylim_primer[0])
    ax_ylim[1] = max(ax_ylim[1], ax_ylim_primer[1])

    ax2_ymin = np.nanmin(np.array(ecoregion_dat_dict['NDVI_mean']) - (np.array(ecoregion_dat_dict['NDVI_std'])/2))
    ax2_ymax = np.nanmax(np.array(ecoregion_dat_dict['NDVI_mean']) + (np.array(ecoregion_dat_dict['NDVI_std'])/2))
    ax2_yrange = ax2_ymax - ax2_ymin
    ax2_ymin_adj = ax2_ymin - (ax2_yrange * (ax2_ylim_scale - 0.5))
    ax2_ymax_adj = ax2_ymax - (ax2_yrange * (ax2_ylim_scale - 0.5))
    ax2_ylim_primer = [min([ax2_ymin_adj, np.nanmin(ecoregion_dat_dict['NDVI_mean'])]), 
                max([ax2_ymax_adj, np.nanmax(ecoregion_dat_dict['NDVI_mean'])])]

    ax2_ylim[0] = min(ax2_ylim[0], ax2_ylim_primer[0])
    ax2_ylim[1] = max(ax2_ylim[1], ax2_ylim_primer[1])


    ax3_ymin = np.nanmin(np.array(FLUXCOM_ecoregion_dat_dict['GPP_mean']) - (np.array(FLUXCOM_ecoregion_dat_dict['GPP_std'])/2))
    ax3_ymax = np.nanmax(np.array(FLUXCOM_ecoregion_dat_dict['GPP_mean']) + (np.array(FLUXCOM_ecoregion_dat_dict['GPP_std'])/2))
    ax3_yrange = ax3_ymax - ax3_ymin
    ax3_ymin_adj = ax3_ymin - (ax3_yrange * (ax3_ylim_scale - 0.5))
    ax3_ymax_adj = ax3_ymax - (ax3_yrange * (ax3_ylim_scale - 0.5))
    ax3_ylim_primer = [min([ax3_ymin_adj, np.nanmin(FLUXCOM_ecoregion_dat_dict['GPP_mean'])]), 
                max([ax3_ymax_adj, np.nanmax(FLUXCOM_ecoregion_dat_dict['GPP_mean'])])]

    ax3_ylim[0] = min(ax3_ylim[0], ax3_ylim_primer[0])
    ax3_ylim[1] = max(ax3_ylim[1], ax3_ylim_primer[1])


    # -------------------------
    # Plot 2: VPD anomaly (left) and PDSI (right)
    # -------------------------
    TC_VPD_years  = np.array(TC_VPD_dict['years'])
    TC_VPD_mean   = np.array(TC_VPD_dict['VPD_anomaly_mean'])
    TC_PDSI_years = np.array(TC_PDSI_dict['years'])
    TC_PDSI_mean  = np.array(TC_PDSI_dict['PDSI_mean'])

    ax1_ylim_scale = 0.75
    ax2_ylim_scale = 0.25

    ax_ymin = np.nanmin(np.array(TC_VPD_mean) - (np.array(TC_VPD_dict['VPD_anomaly_std'])/2))
    ax_ymax = np.nanmax(np.array(TC_VPD_mean) + (np.array(TC_VPD_dict['VPD_anomaly_std'])/2))
    ax_yrange = ax_ymax - ax_ymin
    ax_ymin_adj = ax_ymin - (ax_yrange * (ax1_ylim_scale - 0.5))
    ax_ymax_adj = ax_ymax - (ax_yrange * (ax1_ylim_scale - 0.5))
    vpd_ylim_primer = [min(ax_ymin_adj, np.nanmin(TC_VPD_mean)),
            max(ax_ymax_adj, np.nanmax(TC_VPD_mean))]

    vpd_ylim[0] = min(vpd_ylim[0], vpd_ylim_primer[0])
    vpd_ylim[1] = max(vpd_ylim[1], vpd_ylim_primer[1])

    ax2_ymin = np.nanmin(np.array(TC_PDSI_mean) - (np.array(TC_PDSI_dict['PDSI_std'])/2))
    ax2_ymax = np.nanmax(np.array(TC_PDSI_mean) + (np.array(TC_PDSI_dict['PDSI_std'])/2))
    ax2_yrange = ax2_ymax - ax2_ymin
    ax2_ymin_adj = ax2_ymin - (ax2_yrange * (ax2_ylim_scale - 0.5))
    ax2_ymax_adj = ax2_ymax - (ax2_yrange * (ax2_ylim_scale - 0.5))
    pdsi_ylim_primer = [min([ax2_ymin_adj, np.nanmin(TC_PDSI_mean)]), 
                max([ax2_ymax_adj, np.nanmax(TC_PDSI_mean)])]

    pdsi_ylim[0] = min(pdsi_ylim[0], pdsi_ylim_primer[0])
    pdsi_ylim[1] = max(pdsi_ylim[1], pdsi_ylim_primer[1])


# %%





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
    with open(FLUXCOM_input_file, 'rb') as f:
        FLUXCOM_ecoregion_dat_dict = pickle.load(f)
    with open(TC_VPD_file, 'rb') as f:
        TC_VPD_dict = pickle.load(f)
    with open(TC_PDSI_file, 'rb') as f:
        TC_PDSI_dict = pickle.load(f)

    plt.rcParams.update({
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 13,
    })

    plt.rcParams['font.sans-serif'] = 'Liberation Sans'
    plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.size'] = 14
    cci_color   = '#BD8F0D'
    ndvi_color  = '#1A915C'
    gpp_color   = 'black'
    vpd_color   = '#E4C217'
    pdsi_color  = '#1f78b4'

    from scipy import stats

    # Create separate formatters for each axis
    cci_formatter = ScalarFormatter(useMathText=True)
    cci_formatter.set_powerlimits((-2, 2))

    # ndvi_formatter = ScalarFormatter(useMathText=True)
    # ndvi_formatter.set_powerlimits((-2, 2))
    ndvi_formatter = ScalarFormatter(useMathText=True)
    ndvi_formatter.set_scientific(True)  # Force scientific notation
    ndvi_formatter.set_powerlimits((0, 0))  # This ensures it always uses scientific notation


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
    FC_years = np.array(FLUXCOM_ecoregion_dat_dict['years'])
    FC_mean  = np.array(FLUXCOM_ecoregion_dat_dict['GPP_mean'])

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

    ax2.plot(analysis_years, ecoregion_dat_dict['NDVI_mean'], color=ndvi_color, lw=2)
    ax2.set_ylabel('NDVI', color=ndvi_color)
    ax2.set_ylim(ax2_ylim)
    ax2.tick_params(axis='y', colors=ndvi_color)
    ax2.spines['left'].set_color(ndvi_color)
    ax2.yaxis.set_major_formatter(ndvi_formatter)

    ax.plot(analysis_years, ecoregion_dat_dict['CCI_mean'], color=cci_color, lw=2.5)
    ax.set_ylabel('CCI', color=cci_color)
    ax.set_ylim(ax_ylim)
    ax.yaxis.label.set_color(cci_color)
    ax.tick_params(axis='y', colors=cci_color)
    ax.spines['left'].set_color(cci_color)
    ax.yaxis.set_major_formatter(cci_formatter)

    ax3.plot(FC_years, FC_mean, color=gpp_color, lw=2)
    ax3.set_ylabel('Modeled GPP\n(gC/m²/day)', color=gpp_color)
    ax3.set_ylim(ax3_ylim)
    ax3.tick_params(axis='y', colors=gpp_color)
    ax3.spines['right'].set_color(gpp_color)

    plot_trend(ax,  analysis_years, np.array(ecoregion_dat_dict['CCI_mean']),  cci_color)
    plot_trend(ax2, analysis_years, np.array(ecoregion_dat_dict['NDVI_mean']), ndvi_color)
    plot_trend(ax3, FC_years, FC_mean, gpp_color, trend_years = analysis_years)

    ax.set_xticks(tick_years)
    ax.set_xticklabels([str(y) for y in tick_years], rotation=45)
    ax.set_xlim(analysis_years[0] - 0.5, analysis_years[-1] + 0.5)

    ax2.yaxis.set_major_formatter(ndvi_formatter)
    ax2.yaxis.get_offset_text().set_visible(True)
    ax2.yaxis.get_offset_text().set_x(-0.35)  # Adjust x position as needed

    plot_filename1 = f'{ecoregion.replace(" ", "_")}_CCI_NDVI_GPP_annual_{suffix}.png'
    fig1.savefig(os.path.join(plot_basedir, plot_filename1), dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig1)
    print(f'Plot 1 saved to {os.path.join(plot_basedir, plot_filename1)}')

    # -------------------------
    # Plot 2: VPD anomaly (left) and PDSI (right)
    # -------------------------
    TC_VPD_years  = np.array(TC_VPD_dict['years'])
    TC_VPD_mean   = np.array(TC_VPD_dict['VPD_anomaly_mean'])
    TC_PDSI_years = np.array(TC_PDSI_dict['years'])
    TC_PDSI_mean  = np.array(TC_PDSI_dict['PDSI_mean'])

    fig2, ax_vpd = plt.subplots(figsize=(5.5, 2.6))
    fig2.subplots_adjust(left=0.15, right=0.85)

    ax_vpd.grid(color='gray', linestyle='--', linewidth=1, alpha=0.4)

    ax_pdsi = ax_vpd.twinx()
    ax_pdsi.spines['right'].set_position(('outward', 0))

    ax_vpd.plot(TC_VPD_years, TC_VPD_mean, color=vpd_color, lw=2)
    ax_vpd.set_ylabel('VPD anomaly (kPa)', color=vpd_color)
    ax_vpd.set_ylim(vpd_ylim)
    ax_vpd.tick_params(axis='y', colors=vpd_color)
    ax_vpd.spines['left'].set_color(vpd_color)
    ax_vpd.yaxis.label.set_color(vpd_color)
    ax_vpd.yaxis.set_major_formatter(cci_formatter)

    ax_pdsi.plot(TC_PDSI_years, TC_PDSI_mean, color=pdsi_color, lw=2)
    ax_pdsi.set_ylabel('PDSI', color=pdsi_color)
    ax_pdsi.set_ylim(pdsi_ylim)
    ax_pdsi.tick_params(axis='y', colors=pdsi_color)
    ax_pdsi.spines['right'].set_color(pdsi_color)
    ax_pdsi.yaxis.set_major_formatter(cci_formatter)

    plot_trend(ax_vpd,  TC_VPD_years,  TC_VPD_mean,  vpd_color)
    plot_trend(ax_pdsi, TC_PDSI_years, TC_PDSI_mean, pdsi_color)

    ax_vpd.set_xticks(tick_years)
    ax_vpd.set_xticklabels([str(y) for y in tick_years], rotation=45)
    ax_vpd.set_xlim(analysis_years[0] - 0.5, analysis_years[-1] + 0.5)

    plot_filename2 = f'{ecoregion.replace(" ", "_")}_VPD_PDSI_annual_{suffix}.png'
    fig2.savefig(os.path.join(plot_basedir, plot_filename2), dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig2)
    print(f'Plot 2 saved to {os.path.join(plot_basedir, plot_filename2)}')

# %%

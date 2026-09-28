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
import os
import sys
from tkinter import font
from pygam import f
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from rasterio.enums import Resampling
import geopandas as gpd


# plotting packages
import matplotlib
from matplotlib import pyplot as plt  # primary plotting module
import matplotlib.colors as mcolors

# Custom packages
utils_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/pyutil'
if utils_dir not in sys.path:
    sys.path.append(utils_dir)
from MODIS_tile_corners import MODIS_tilecorners_df

import data_utils

# plotting packages
from matplotlib import pyplot as plt  # primary plotting module
from shapely.geometry import mapping
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter as FFmt  # need this for formatting plot ticks
import cartopy.crs as ccrs  #
from cartopy.io import img_tiles  # cartopy's implementation of webtiles
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature


#%%
#########################################
# Define Global Filepaths
#########################################

dat_basedir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/MODIS/MCD19A1_analysis/by_ecoregion/summary_data/'
plot_dir = Path('/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/lm_summary/slope')
plot_dir.mkdir(parents=True, exist_ok=True)

# For plotting x,y bounds (consistent with other maps made for the chapter)
ESA_CCI_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/'


EPA_ecoregion_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/'
EPA_ecoregion_L3_file = os.path.join(EPA_ecoregion_dir, 'L3/NAm_shp_FOR_MAIAC_PROC', 'NAm_cec_eco_l3_v2.shp')

########################################
# Define Global Variables and constants
#########################################


ecoregions_l3_list = [
'Thompson-Okanogan Plateau',
# 'Central Basin and Range',
'Colorado Plateaus',
'Arizona/New Mexico Mountains',
'Sierra Madre Occidental with Conifer, Oak, and Mixed Forests',
'Interior Forested Lowlands and Uplands',
'Interior Bottomlands',
'Yukon Flats',
'Ogilvie Mountains',
'Mackenzie and Selwyn Mountains',
'Peel River and Nahanni Plateaus',
'Great Bear Plains',
'Hay and Slave River Lowlands',
'Kazan River and Selwyn Lake Uplands',
'La Grande Hills and New Quebec Central Plateau',
'Smallwood Uplands',
'Ungava Bay Basin and George Plateau',
'Coppermine River and Tazin Lake Uplands',
'Hudson Bay and James Bay Lowlands',
'Athabasca Plain and Churchill River Upland',
'Lake Nipigon and Lac Seul Upland',
'Central Laurentians and Mecatina Plateau',
'Hayes River Upland and Big Trout Lake',
'Abitibi Plains and Riviere Rupert Plateau',
'Algonquin/Southern Laurentians',
'Northern Appalachian and Atlantic Maritime Highlands',
'Mid-Boreal Uplands and Peace-Wabaska Lowlands',
'Clear Hills and Western Alberta Upland',
'Mid-Boreal Lowland and Interlake Plain',
'Interior Highlands and Klondike Plateau',
'Copper Plateau',
'Watson Highlands',
'Yukon-Stikine Highlands/Boreal Mountains and Plateaus',
'Skeena-Omineca-Central Canadian Rocky Mountains',
'Middle Rockies',
'Klamath Mountains',
'Sierra Nevada',
'Wasatch and Uinta Mountains',
'Southern Rockies',
'Idaho Batholith',
'Chilcotin Ranges and Fraser Plateau',
'Columbia Mountains/Northern Rockies',
'Canadian Rockies',
'North Cascades',
'Blue Mountains',
'Coastal Western Hemlock-Sitka Spruce Forests',
'Pacific and Nass Ranges',
'Coast Range',
'Southeastern Plains',
'South Central Plains',
'Southern Coastal Plain',
'Cascades',
'Eastern Cascades Slopes and Foothills'
]

#########################################
# Define Global Functions
#########################################



####################################
### Set up map parameters
####################################

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



# Load Canadian provinces and Mexican states
fn_can = shpreader.natural_earth(
    resolution='10m', category='cultural', 
    name='admin_1_states_provinces',
)
reader_can = shpreader.Reader(fn_can)
canada = [x for x in reader_can.records() if x.attributes["admin"] == "Canada"]
mexico = [x for x in reader_can.records() if x.attributes["admin"] == "Mexico"]

canada_geom = cfeature.ShapelyFeature([x.geometry for x in canada], ccrs.PlateCarree())
mexico_geom = cfeature.ShapelyFeature([x.geometry for x in mexico], ccrs.PlateCarree())

tiler_zoom = 5  # define a zoom level of detail




#########################################
# Begin Main
#########################################

#%%



# Get the ESA CCI forest cover data for plotting x,y bounds consistent with other maps made for the chapter
ESA_CCI_file = os.path.join(ESA_CCI_dir, f'forest_fraction_0.01deg_north_america.nc')
ESA_CCI_xr = xr.open_dataset(ESA_CCI_file, decode_coords = 'all').rename({'x': 'lon', 'y': 'lat'}).sortby(['lon', 'lat'], ascending = True)['fraction_forest']
ESA_CCI_xr.rio.write_crs('EPSG:4326', inplace=True)
plot_extent = [ESA_CCI_xr.lon.min().values, ESA_CCI_xr.lon.max().values, ESA_CCI_xr.lat.min().values, ESA_CCI_xr.lat.max().values]


ecoregions_gdf = gpd.read_file(EPA_ecoregion_L3_file)

climate_names = ['CWD', 'PDSI', 'PET', 'SoilM', 'VPD', 'SPEI01', 'SPEI12']
clim_colorscale_reverse_arr = [False, True, False, True, False, True, True] # set to True if Positive value is wetter, set to False if positive value is drier
clim_units = ['mm', 'PDSI', 'mm', 'mm', 'kPa', 'SPEI-01', 'SPEI-12']
# climate_names = ['SPEI01', 'SPEI12']
# clim_colorscale_reverse_arr = [True, True] # set to True if Positive value is wetter, set to False if positive value is drier
# clim_units = ['SPEI-01', 'SPEI-12']
veg_names = ['CCI_std_anomaly', 'NDMI_std_anomaly', 'NIRv_std_anomaly', 'NDVI_std_anomaly']
timing_abbrev = 'JJA'

#%%
for ii, climate_name in enumerate(climate_names):

    clim_colorscale_reverse = clim_colorscale_reverse_arr[ii]

    for veg_name in veg_names:

        print(f"Plotting map: {climate_name} vs {veg_name}...")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initial pass to get min/max slope values for color scaling
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        vmin = 0
        vmax = 0

        for ecoregion in ecoregions_l3_list:

            ecoregion_filefmt = ecoregion.replace(' ', '_').replace("/", "~").replace(",", "") #Arizona/New Mexico Mountains

            if climate_name in ['SPEI01', 'SPEI12']:
                dat_file = os.path.join(dat_basedir, f'lm_SPEIbase_{climate_name}_vs_{veg_name}_{timing_abbrev}_{ecoregion_filefmt}.csv')
            else:
                dat_file = os.path.join(dat_basedir, f'lm_TerraClimate_{climate_name}_vs_{veg_name}_{timing_abbrev}_{ecoregion_filefmt}.csv')

            if not os.path.exists(dat_file):
                continue

            df = pd.read_csv(dat_file)

            slope = df['slope'].values[0]
            slope_pval = df['lm_p_value'].values[0]
            spearman_r = df['spearman_r'].values[0]
            spearman_pval = df['spearman_p_value'].values[0]

            vmin = np.nanmin([vmin, -np.abs(slope)])
            vmax = np.nanmax([vmax, np.abs(slope)])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Now loop through again to plot, using the vmin/vmax for consistent color scaling across maps
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={'projection': crs}) # establish the figure, axes
        ax.set_extent(plot_extent, crs=ccrs.PlateCarree())  # PlateCarree is default lat/lon    
        ax.add_image(tiler, tiler_zoom, interpolation='none', alpha=0.6) # Add base image
        # ax.add_feature(states_geom, facecolor="none", edgecolor="black", linewidth=1) # add state boundaries to the map


        for ecoregion in ecoregions_l3_list:

            ecoregion_filefmt = ecoregion.replace(' ', '_').replace("/", "~").replace(",", "") #Arizona/New Mexico Mountains
            if climate_name in ['SPEI01', 'SPEI12']:
                dat_file = os.path.join(dat_basedir, f'lm_SPEIbase_{climate_name}_vs_{veg_name}_{timing_abbrev}_{ecoregion_filefmt}.csv')
            else:
                dat_file = os.path.join(dat_basedir, f'lm_TerraClimate_{climate_name}_vs_{veg_name}_{timing_abbrev}_{ecoregion_filefmt}.csv')

            if not os.path.exists(dat_file):
                continue

            # From the script that saved the data......
            # summary_df = pd.DataFrame({
            #     'ecoregion': [ecoregion_withSpaceSlash],
            #     'climate_variable': [climate_name],
            #     'vegetation_index': [veg_name],
            #     'slope': [slope],
            #     'r_squared': [r2],
            #     'lm_p_value': [lm_p_val],
            #     'spearman_r': [spearman_r],
            #     'spearman_p_value': [spearman_p_val]
            # })
            df = pd.read_csv(dat_file)

            # print(ecoregion_l2_gdf['NA_L2NAME'].unique())
            ecoregion_gdf = ecoregions_gdf[ecoregions_gdf.NA_L3NAME == ecoregion]
            ecoregion_geom = ecoregion_gdf.geometry


            slope = df['slope'].values[0]
            slope_pval = df['lm_p_value'].values[0]
            spearman_r = df['spearman_r'].values[0]
            spearman_pval = df['spearman_p_value'].values[0]

            print(f"{climate_name} vs {veg_name} - Ecoregion: {ecoregion}, Slope p-value: {slope_pval:.4f}")
            # print(f"  Ecoregion: {ecoregion}, spearman p-value: {spearman_pval:.4f}")

            ### Determine color from slope value
            ### use spectral_r colormap

            if clim_colorscale_reverse:
                cmap = plt.cm.Spectral_r
            else:
                cmap = plt.cm.Spectral

            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            plot_color = cmap(norm(slope))

            # Add ecoregion to map
            ax.add_geometries(ecoregion_geom, crs=transform, facecolor=plot_color, edgecolor='none', alpha=0.9)
            ax.add_geometries(ecoregion_geom, crs=transform, facecolor='none', edgecolor=plot_color, linewidth=1.2)


        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        

        # Add colorbar in lower left corner within the plot area
        # Add white background box behind the colorbar
        cax = ax.inset_axes([0.02, 0.02, 0.03, 0.3])  # [x, y, width, height] in axes coordinates
        # Draw white rectangle slightly larger than the colorbar axes
        white_box = ax.inset_axes([0.005, 0.005, 0.15, 0.34]) # [x, y, width, height] in axes coordinates
        white_box.set_facecolor('white')
        white_box.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        white_box.set_zorder(2)
        cax.set_zorder(3)
        cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
        cbar.set_label(f'Slope (σ/{clim_units[ii]})', rotation=270, labelpad=20, fontsize=15)
        cbar.ax.tick_params(labelsize=15)

        outfile = os.path.join(plot_dir, f'map_lm_slope_{veg_name}_v_{climate_name}_{timing_abbrev}.png')
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close()




#%%

# %%

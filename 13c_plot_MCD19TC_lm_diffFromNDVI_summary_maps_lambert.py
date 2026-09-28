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
import rasterio

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
plot_dir = Path('/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/lm_summary/slope_diff_from_NDVI')
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



plot_zbounds = np.arange(0, 120, 20) # define the bins and normalize
znorm = mcolors.BoundaryNorm(plot_zbounds, 10)
tiler_zoom = 7

# Load the natural earth raster file
NE_raster_file = "/uufs/chpc.utah.edu/common/home/lin-group19/ltk/pyutil/NaturalEarth/NE2_50M_SR_W/NE2_50M_SR_W.tif"
NE_raster = rasterio.open(NE_raster_file)
vector_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/pyutil/NaturalEarth'
coastlines_file = os.path.join(vector_dir, 'ne_110m_coastline.shp')
lakes_file = os.path.join(vector_dir, 'ne_110m_lakes.shp')
rivers_file = os.path.join(vector_dir, 'ne_110m_rivers_lake_centerlines.shp')
countries_file = os.path.join(vector_dir, 'ne_110m_admin_0_countries.shp')
lambert_conformal_proj4 = '+proj=lcc +lon_0=-100 +lat_0=40 +lat_1=33 +lat_2=45 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'

lakes_gdf = gpd.read_file(lakes_file)
# gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='#2b8dd6', edgecolor='#14629c', linewidth=1, alpha=0.7)


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
EPA_ecoregion_temperate_file = os.path.join(EPA_ecoregion_dir, 'L3/NAm_cec_eco_l3_Temperate.shp')
EPA_ecoregion_boreal_file = os.path.join(EPA_ecoregion_dir, 'L3/NAm_cec_eco_l3_Boreal_combined.shp')

ecoregions_temperate_gdf = gpd.read_file(EPA_ecoregion_temperate_file)
ecoregions_boreal_gdf = gpd.read_file(EPA_ecoregion_boreal_file)

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

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initial pass to get min/max slope values for color scaling
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    vmin = 0
    vmax = 0

    for ecoregion in ecoregions_l3_list:

        ecoregion_filefmt = ecoregion.replace(' ', '_').replace("/", "~").replace(",", "") #Arizona/New Mexico Mountains

        if climate_name in ['SPEI01', 'SPEI12']:
            CCI_dat_file = os.path.join(dat_basedir, f'lm_SPEIbase_{climate_name}_vs_CCI_std_anomaly_{timing_abbrev}_{ecoregion_filefmt}.csv')
            NDVI_dat_file = os.path.join(dat_basedir, f'lm_SPEIbase_{climate_name}_vs_NDVI_std_anomaly_{timing_abbrev}_{ecoregion_filefmt}.csv')
        else:
            CCI_dat_file = os.path.join(dat_basedir, f'lm_TerraClimate_{climate_name}_vs_CCI_std_anomaly_{timing_abbrev}_{ecoregion_filefmt}.csv')
            NDVI_dat_file = os.path.join(dat_basedir, f'lm_TerraClimate_{climate_name}_vs_NDVI_std_anomaly_{timing_abbrev}_{ecoregion_filefmt}.csv')

        if not os.path.exists(CCI_dat_file) or not os.path.exists(NDVI_dat_file):
            continue

        df_CCI = pd.read_csv(CCI_dat_file)
        df_NDVI = pd.read_csv(NDVI_dat_file)

        CCI_slope = df_CCI['slope'].values[0]
        NDVI_slope = df_NDVI['slope'].values[0]
        diff_slope = CCI_slope - NDVI_slope

        vmin = np.nanmin([vmin, -np.abs(diff_slope)])
        vmax = np.nanmax([vmax, np.abs(diff_slope)])
        # vmin_spearman = np.nanmin([vmin_spearman, diff_spearman_r])
        # vmax_spearman = np.nanmax([vmax_spearman, diff_spearman_r])


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Now loop through again to plot, using the vmin/vmax for consistent color scaling across maps
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Set up Lambert figure
    projection = ccrs.LambertConformal(central_longitude=-100, central_latitude=40)
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projection})
    ax.set_extent([-170, -50, 9, 72], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')
    lakes_gdf.plot(ax=ax, facecolor='#2b8dd6', edgecolor='#14629c', linewidth=1, alpha=0.7)

    cmap = plt.cm.Spectral_r if clim_colorscale_reverse else plt.cm.Spectral
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    temperate_plot_gdf = gpd.GeoDataFrame(columns=['geometry'], crs=ecoregions_gdf.crs)
    boreal_plot_gdf = gpd.GeoDataFrame(columns=['geometry'], crs=ecoregions_gdf.crs)

    for ecoregion in ecoregions_l3_list:
        ecoregion_filefmt = ecoregion.replace(' ', '_').replace("/", "~").replace(",", "")

        if climate_name in ['SPEI01', 'SPEI12']:
            CCI_dat_file = os.path.join(dat_basedir, f'lm_SPEIbase_{climate_name}_vs_CCI_std_anomaly_{timing_abbrev}_{ecoregion_filefmt}.csv')
            NDVI_dat_file = os.path.join(dat_basedir, f'lm_SPEIbase_{climate_name}_vs_NDVI_std_anomaly_{timing_abbrev}_{ecoregion_filefmt}.csv')
        else:
            CCI_dat_file = os.path.join(dat_basedir, f'lm_TerraClimate_{climate_name}_vs_CCI_std_anomaly_{timing_abbrev}_{ecoregion_filefmt}.csv')
            NDVI_dat_file = os.path.join(dat_basedir, f'lm_TerraClimate_{climate_name}_vs_NDVI_std_anomaly_{timing_abbrev}_{ecoregion_filefmt}.csv')

        if not os.path.exists(CCI_dat_file) or not os.path.exists(NDVI_dat_file):
            continue

        df_CCI = pd.read_csv(CCI_dat_file)
        df_NDVI = pd.read_csv(NDVI_dat_file)

        CCI_slope = df_CCI['slope'].values[0]
        NDVI_slope = df_NDVI['slope'].values[0]
        diff_slope = CCI_slope - NDVI_slope

        ### Determine color from slope value
        ### use spectral_r colormap

        if clim_colorscale_reverse:
            cmap = plt.cm.Spectral_r
        else:
            cmap = plt.cm.Spectral

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        plot_color = cmap(norm(diff_slope))

        ecoregion_geom_gdf = ecoregions_gdf[ecoregions_gdf.NA_L3NAME == ecoregion]

        # Accumulate for dissolve
        temperate_extra = ['Pacific and Nass Ranges', 'Algonquin/Southern Laurentians', 'Central Laurentians and Mecatina Plateau']
        if ecoregion in ecoregions_temperate_gdf.NA_L3NAME.values or ecoregion in temperate_extra:
            temperate_plot_gdf = pd.concat([temperate_plot_gdf, ecoregion_geom_gdf[['geometry']]], ignore_index=True)
        elif ecoregion in ecoregions_boreal_gdf.NA_L3NAME.values:
            boreal_plot_gdf = pd.concat([boreal_plot_gdf, ecoregion_geom_gdf[['geometry']]], ignore_index=True)

        ax.add_geometries(ecoregion_geom_gdf.geometry, crs=ccrs.PlateCarree(), facecolor=plot_color, edgecolor='none', alpha=0.9)
        ax.add_geometries(ecoregion_geom_gdf.geometry, crs=ccrs.PlateCarree(), facecolor='none', edgecolor=plot_color, linewidth=1.2)

    # Overlay dissolved region boundaries
    ax.add_geometries(temperate_plot_gdf.dissolve().geometry, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='#573C1F', linewidth=0.8)
    ax.add_geometries(boreal_plot_gdf.dissolve().geometry, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='#024C14', linewidth=0.8)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    white_box = ax.inset_axes([0.005, 0.005, 0.15, 0.34])
    white_box.set_facecolor('white')
    white_box.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    white_box.set_zorder(2)
    cax = ax.inset_axes([0.02, 0.02, 0.03, 0.3])
    cax.set_zorder(3)
    cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label(f'CCI slope - NDVI slope (σ/{clim_units[ii]})', rotation=270, labelpad=20, fontsize=14)
    cbar.ax.tick_params(labelsize=15)

    outfile = os.path.join(plot_dir, f'map_lm_CCI_NDVI_v_{climate_name}_slope_diff_{timing_abbrev}.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()


# # %%

# import cartopy.crs as ccrs  # spatial plotting library 
# from cartopy.io import img_tiles  # cartopy's implementation of webtiles
# import cartopy.io.shapereader as shpreader
# import cartopy.feature as cfeature
# from matplotlib.ticker import FuncFormatter as FFmt  # need this for formatting plot ticks
# import matplotlib.colors as mcolors
# import rasterio

# EPA_ecoregion_temperate_file = os.path.join(EPA_ecoregion_dir, 'L3/NAm_cec_eco_l3_Temperate.shp')
# EPA_ecoregion_boreal_file = os.path.join(EPA_ecoregion_dir, 'L3/NAm_cec_eco_l3_Boreal_combined.shp')

# ecoregions_temperate_gdf = gpd.read_file(EPA_ecoregion_temperate_file)
# ecoregions_boreal_gdf = gpd.read_file(EPA_ecoregion_boreal_file)

# #%%


# #%%
# # Load ecoregions
# ecoregions_gdf = gpd.read_file(EPA_ecoregion_L3_file)

# # Set up figure with Lambert projection
# projection = ccrs.LambertConformal(central_longitude=-100, central_latitude=40)
# fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projection})
# extent = [-170, -50, 9, 72]
# ax.set_extent(extent, crs=ccrs.PlateCarree())

# # Base map features
# ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
# ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
# ax.add_feature(cfeature.COASTLINE, edgecolor='black')

# # Lakes
# gdf = gpd.read_file(lakes_file)
# gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='#2b8dd6', edgecolor='#14629c', linewidth=1, alpha=0.7)

# # Pick your climate/veg combo
# climate_name = 'SPEI01'
# veg_name = 'CCI_std_anomaly'
# timing_abbrev = 'JJA'
# clim_colorscale_reverse = True
# clim_units_label = 'SPEI-01'

# # Initialize blank GeoDataFrames for temperate and boreal ecoregions
# temperate_plot_gdf = gpd.GeoDataFrame(columns=['geometry', 'slope'], crs=ecoregions_gdf.crs)
# boreal_plot_gdf = gpd.GeoDataFrame(columns=['geometry', 'slope'], crs=ecoregions_gdf.crs)
# # First pass to get vmin/vmax
# vmin, vmax = 0, 0
# for ecoregion in ecoregions_l3_list:


#     ecoregion_filefmt = ecoregion.replace(' ', '_').replace("/", "~").replace(",", "")
#     dat_file = os.path.join(dat_basedir, f'lm_SPEIbase_{climate_name}_vs_{veg_name}_{timing_abbrev}_{ecoregion_filefmt}.csv')
#     if not os.path.exists(dat_file):
#         continue
#     df = pd.read_csv(dat_file)
#     slope = df['slope'].values[0]
#     vmin = np.nanmin([vmin, -np.abs(slope)])
#     vmax = np.nanmax([vmax, np.abs(slope)])


# cmap = plt.cm.Spectral_r if clim_colorscale_reverse else plt.cm.Spectral
# norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# # Second pass to plot
# for ecoregion in ecoregions_l3_list:
#     ecoregion_filefmt = ecoregion.replace(' ', '_').replace("/", "~").replace(",", "")
#     dat_file = os.path.join(dat_basedir, f'lm_SPEIbase_{climate_name}_vs_{veg_name}_{timing_abbrev}_{ecoregion_filefmt}.csv')
#     if not os.path.exists(dat_file):
#         continue
#     df = pd.read_csv(dat_file)
#     slope = df['slope'].values[0]
#     plot_color = cmap(norm(slope))
#     if (ecoregion in ecoregions_temperate_gdf.NA_L3NAME.values) or (ecoregion in ['Pacific and Nass Ranges', 'Algonquin/Southern Laurentians', 'Central Laurentians and Mecatina Plateau']):
#         ecoregion_gdf = ecoregions_gdf[ecoregions_gdf.NA_L3NAME == ecoregion]
#         temperate_plot_gdf = pd.concat([temperate_plot_gdf, ecoregion_gdf[['geometry']]], ignore_index=True)
#     elif ecoregion in ecoregions_boreal_gdf.NA_L3NAME.values:
#         ecoregion_gdf = ecoregions_gdf[ecoregions_gdf.NA_L3NAME == ecoregion]
#         boreal_plot_gdf = pd.concat([boreal_plot_gdf, ecoregion_gdf[['geometry']]], ignore_index=True)

#     ecoregion_geom = ecoregions_gdf[ecoregions_gdf.NA_L3NAME == ecoregion].geometry
#     ax.add_geometries(ecoregion_geom, crs=ccrs.PlateCarree(), facecolor=plot_color, edgecolor='none', alpha=0.9)
#     ax.add_geometries(ecoregion_geom, crs=ccrs.PlateCarree(), facecolor='none', edgecolor=plot_color, linewidth=1.2)

# temperate_plot_gdf_combined = temperate_plot_gdf.dissolve()  # combine all geometries into one
# boreal_plot_gdf_combined = boreal_plot_gdf.dissolve()

# temperate_geom = temperate_plot_gdf_combined.geometry
# ax.add_geometries(temperate_geom, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='#573C1F', linewidth=0.8)

# boreal_geom = boreal_plot_gdf_combined.geometry
# ax.add_geometries(boreal_geom, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='#024C14', linewidth=0.8)

# # Colorbar
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# white_box = ax.inset_axes([0.005, 0.005, 0.15, 0.34])
# white_box.set_facecolor('white')
# white_box.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
# white_box.set_zorder(2)
# cax = ax.inset_axes([0.02, 0.02, 0.03, 0.3])
# cax.set_zorder(3)
# cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
# cbar.set_label(f'Slope (σ/{clim_units_label})', rotation=270, labelpad=20, fontsize=15)
# cbar.ax.tick_params(labelsize=15)

# plt.show()
# # %%

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
import rasterio

from matplotlib.colors import LinearSegmentedColormap
from rasterio.features import geometry_mask
import regionmask

#########################################
# Define Global Filepaths
#########################################

plot_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/ESA_CCI/plots/'
EPA_ecoregion_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/'
EPA_ecoregion_L1_file = os.path.join(EPA_ecoregion_dir, 'L1', 'NA_CEC_Eco_Level1_EPSG4326_regrid.shp')
EPA_ecoregion_L2_file = os.path.join(EPA_ecoregion_dir, 'L2/NAm_shp_FOR_MAIAC_PROC', 'NA_CEC_Eco_Level2_withOther.shp')
EPA_ecoregion_L3_file = os.path.join(EPA_ecoregion_dir, 'L3/NAm_shp_FOR_MAIAC_PROC', 'NAm_cec_eco_l3_v2.shp')
EPA_ecoregion_temperate_file = os.path.join(EPA_ecoregion_dir, 'L3/NAm_cec_eco_l3_Temperate.shp')
EPA_ecoregion_boreal_file = os.path.join(EPA_ecoregion_dir, 'L3/NAm_cec_eco_l3_Boreal_combined.shp')
EPA_ecoregion_L3_file = os.path.join(EPA_ecoregion_dir, 'L3/NAm_shp_FOR_MAIAC_PROC', 'NAm_cec_eco_l3_v2.shp')

# Processed data files
AMF_sites_shp_file = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/eddy_covariance/data/shp/AMF_sites_forCCI.shp'

photospec_site_coords = {
   'US-NR1': (-105.5464, 40.0329), 
   'Ca-Obs': (-105.1178, 53.9872),
   'DEJU': (-145.75136, 63.88112),
   'OSBS': (-81.993431, 29.689282)
}


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


# --- 2. Truncate Greens colormap so vmax=1 hits ~60% of the scale ---
def truncate_cmap(cmap_name, minval=0.0, maxval=0.6, n=256):
    base = plt.cm.get_cmap(cmap_name)
    colors = base(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(
        f'trunc_{cmap_name}', colors, N=n
    )

greens_trunc = truncate_cmap('YlGn', minval=0.0, maxval=0.8)
gist_earth_trunc = truncate_cmap('gist_earth_r', minval=0.5, maxval=1.0)

#%%


# Load AMF sites shapefile
AMF_sites_gdf = gpd.read_file(AMF_sites_shp_file)

# Extract coordinates from the shapefile
coords = [(point.y, point.x) for point in AMF_sites_gdf.geometry]


ENF_openclosed_xr = xr.open_dataset('/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/needleleaf_evergreen_open_closed_forest_fraction_0.01deg_north_america.nc')
ENF_closed_xr = xr.open_dataset('/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/needleleaf_evergreen_closed_forest_fraction_0.01deg_north_america.nc')

# Load ecoregion shapefiles
ecoregions_l1_gdf = gpd.read_file(EPA_ecoregion_L1_file)
ecoregions_l2_gdf = gpd.read_file(EPA_ecoregion_L2_file)
ecoregions_l3_gdf = gpd.read_file(EPA_ecoregion_L3_file)



land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
mask = land.mask(ENF_openclosed_xr.x, ENF_openclosed_xr.y)  # returns NaN outside land, 0 inside
ENF_openclosed_xr_masked = ENF_openclosed_xr.where(mask == 0)

mask = land.mask(ENF_closed_xr.x, ENF_closed_xr.y)  # returns NaN outside land, 0 inside
ENF_closed_xr_masked = ENF_closed_xr.where(mask == 0)
ENF_closed_xr_masked = ENF_closed_xr_masked.where(ENF_closed_xr_masked.fraction_ENF >= 0.01)

# set lower latitude cutoff
ENF_openclosed_xr_masked = ENF_openclosed_xr_masked.where(ENF_openclosed_xr_masked.y >= 16)
ENF_closed_xr_masked = ENF_closed_xr_masked.where(ENF_closed_xr_masked.y >= 16)

# set lower latitude cutoff
ENF_openclosed_xr_masked_binary = ENF_openclosed_xr_masked.where(ENF_openclosed_xr_masked.fraction_ENF >= 0.99)


ecoregions_L3_all_gdf = gpd.read_file(EPA_ecoregion_L3_file)
ecoregions_temperate_gdf = gpd.read_file(EPA_ecoregion_temperate_file)
ecoregions_boreal_gdf = gpd.read_file(EPA_ecoregion_boreal_file)

# ecoregion_temperate_gdf = ecoregions_temperate_gdf.dissolve()
# ecoregion_boreal_gdf = ecoregions_boreal_gdf.dissolve()

temperate_plot_gdf = gpd.GeoDataFrame(columns=['geometry'], crs=ecoregions_L3_all_gdf.crs)
boreal_plot_gdf = gpd.GeoDataFrame(columns=['geometry'], crs=ecoregions_L3_all_gdf.crs)
for ecoregion in ecoregions_l3_list:
        # Accumulate for dissolve
        ecoregion_geom_gdf = ecoregions_L3_all_gdf[ecoregions_L3_all_gdf.NA_L3NAME == ecoregion]
        temperate_extra = ['Pacific and Nass Ranges', 'Algonquin/Southern Laurentians', 'Central Laurentians and Mecatina Plateau']
        if ecoregion in ecoregions_temperate_gdf.NA_L3NAME.values or ecoregion in temperate_extra:
            temperate_plot_gdf = pd.concat([temperate_plot_gdf, ecoregion_geom_gdf[['geometry']]], ignore_index=True)
        elif ecoregion in ecoregions_boreal_gdf.NA_L3NAME.values:
            boreal_plot_gdf = pd.concat([boreal_plot_gdf, ecoregion_geom_gdf[['geometry']]], ignore_index=True)


# dat_ds = dat_ds.where(dat_ds['lon'] > -60 & dat_ds['lat'] > 58, drop = False)
ENF_closed_xr_masked = ENF_closed_xr_masked.where((ENF_closed_xr_masked['x'] < -59) | (ENF_closed_xr_masked['y'] < 58), drop = False)
ENF_openclosed_xr_masked = ENF_openclosed_xr_masked.where((ENF_openclosed_xr_masked['x'] < -59) | (ENF_openclosed_xr_masked['y'] < 58), drop = False)

# %%

plot_temperate_boreal_bounds = False

# Set up Lambert figure
projection = ccrs.LambertConformal(central_longitude=-100, central_latitude=40)
fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': projection})
ax.set_extent([-170, -50, 9, 72], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
lakes_gdf.plot(ax=ax, facecolor='#2b8dd6', edgecolor='#14629c', linewidth=1, alpha=0.7)

im = ax.pcolormesh(ENF_openclosed_xr_masked.x, ENF_openclosed_xr_masked.y, ENF_openclosed_xr_masked.fraction_ENF,
                   transform=ccrs.PlateCarree(),
                   cmap=greens_trunc, vmin=0, vmax=1)

im2 = ax.pcolormesh(ENF_closed_xr_masked.x, ENF_closed_xr_masked.y, ENF_closed_xr_masked.fraction_ENF,
                   transform=ccrs.PlateCarree(),
                   cmap=gist_earth_trunc, vmin=0, vmax=1)


ax.coastlines(resolution='50m', linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')


# Plot KML points on the map
for coord in coords:
   ax.plot(coord[1], coord[0], '^', markersize=14, markerfacecolor='yellow', markeredgecolor='black', markeredgewidth=1, transform=ccrs.PlateCarree())

# # Plot photospec site coordinates with green star markers
# for site, (lon, lat) in photospec_site_coords.items():
#    ax.plot(lon+0.2, lat+0.1, marker='*', color='#FFA947', markeredgecolor='black', markersize=20, markeredgewidth=1, transform=ccrs.PlateCarree())

# if plot_temperate_boreal_bounds:
#     # Overlay dissolved region boundaries
#     ax.add_geometries(temperate_plot_gdf.dissolve().geometry, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='#ECAB0D', linewidth=1.3)
#     ax.add_geometries(boreal_plot_gdf.dissolve().geometry, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='#258279', linewidth=1.3)

# cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, fraction=0.046)
# cbar2 = plt.colorbar(im2, ax=ax, orientation='vertical', pad=0.1, fraction=0.046)
# cbar.set_label('Evergreen Needleleaf Forest Fraction (Open)', fontsize=11)
# cbar2.set_label('Evergreen Needleleaf Forest Fraction (Closed)', fontsize=11)

ax.set_title('', fontsize=13)

if plot_temperate_boreal_bounds:
    plot_file = os.path.join(plot_dir, 'ENF_fraction_NAm_Lambert_with_AMF_TempBorealBounds.png')
else:
    plot_file = os.path.join(plot_dir, 'ENF_fraction_NAm_Lambert_with_AMF.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()
# ds.close()
# print("Plot saved successfully")



#%%


# taiga_color = "#5173B1"
# northern_forests_color = '#44AA99'
# western_cordillera_color = '#117733'
# boreal_cordillera_color = '#88CCEE'


# # Set up Lambert figure
# projection = ccrs.LambertConformal(central_longitude=-100, central_latitude=40)
# fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': projection})
# ax.set_extent([-170, -50, 9, 72], crs=ccrs.PlateCarree())
# ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
# ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
# ax.add_feature(cfeature.COASTLINE, edgecolor='black')
# lakes_gdf.plot(ax=ax, facecolor='#2b8dd6', edgecolor='#14629c', linewidth=1, alpha=0.7)

# im = ax.pcolormesh(ENF_openclosed_xr_masked_binary.x, ENF_openclosed_xr_masked_binary.y, ENF_openclosed_xr_masked_binary.fraction_ENF,
#                    transform=ccrs.PlateCarree(),
#                    cmap='Greens', vmin=0, vmax=1)


# ax.coastlines(resolution='50m', linewidth=0.5)
# ax.add_feature(cfeature.BORDERS, linewidth=0.5)
# ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')

# ecoregion_gdf_geom = ecoregions_l1_gdf[ecoregions_l1_gdf.NA_L1NAME == 'TAIGA']
# ecoregion_gdf_geom.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor=taiga_color, edgecolor=taiga_color, linewidth=1.5, alpha = 0.5)

# ecoregion_gdf_geom = ecoregions_l1_gdf[ecoregions_l1_gdf.NA_L1NAME == 'NORTHERN FORESTS']
# ecoregion_gdf_geom.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor=northern_forests_color, edgecolor=northern_forests_color, linewidth=1.5, alpha = 0.5)

# ecoregion_gdf_geom = ecoregions_l2_gdf[ecoregions_l2_gdf.NA_L2NAME == 'WESTERN CORDILLERA']
# ecoregion_gdf_geom.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor=western_cordillera_color, edgecolor=western_cordillera_color, linewidth=1.5, alpha = 0.5)

# ecoregion_gdf_geom = ecoregions_l2_gdf[ecoregions_l2_gdf.NA_L2NAME == 'BOREAL CORDILLERA']
# ecoregion_gdf_geom.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor=boreal_cordillera_color, edgecolor=boreal_cordillera_color, linewidth=1.5, alpha = 0.5)

# ax.set_title('', fontsize=13)


# plot_file = os.path.join(plot_dir, 'ENF_fraction_NAm_Lambert_with_L1L2.png')
# plt.savefig(plot_file, dpi=300, bbox_inches='tight')
# plt.close()
# # ds.close()
# # print("Plot saved successfully")


# #%%

# # Set up Lambert figure
# projection = ccrs.LambertConformal(central_longitude=-100, central_latitude=40)
# fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': projection})
# ax.set_extent([-170, -50, 9, 72], crs=ccrs.PlateCarree())
# ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
# ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.5)
# ax.add_feature(cfeature.COASTLINE, edgecolor='black')
# lakes_gdf.plot(ax=ax, facecolor='#2b8dd6', edgecolor='#14629c', linewidth=1, alpha=0.7)

# im = ax.pcolormesh(ENF_openclosed_xr_masked_binary.x, ENF_openclosed_xr_masked_binary.y, ENF_openclosed_xr_masked_binary.fraction_ENF,
#                    transform=ccrs.PlateCarree(),
#                    cmap='Greens', vmin=0, vmax=1)


# ax.coastlines(resolution='50m', linewidth=0.5)
# ax.add_feature(cfeature.BORDERS, linewidth=0.5)
# ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray')

# # Overlay dissolved region boundaries
# ax.add_geometries(temperate_plot_gdf.dissolve().geometry, crs=ccrs.PlateCarree(), facecolor='#E1BE6A', edgecolor='#ECAB0D', linewidth=1.3, alpha = 0.5)
# ax.add_geometries(boreal_plot_gdf.dissolve().geometry, crs=ccrs.PlateCarree(), facecolor='#40B0A6', edgecolor='#258279', linewidth=1.3, alpha = 0.5)

# ax.set_title('', fontsize=13)


# plot_file = os.path.join(plot_dir, 'ENF_fraction_NAm_Lambert_with_TemperateBoreal.png')
# plt.savefig(plot_file, dpi=300, bbox_inches='tight')
# plt.close()
# # ds.close()
# # print("Plot saved successfully")

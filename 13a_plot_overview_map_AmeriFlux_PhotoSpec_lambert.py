#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
#

#%%
#########################################
# Load packages
#########################################

import os  # operating system library
import sys
from pathlib import Path

# Custom packages
utils_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/pyutil'
if utils_dir not in sys.path:
   sys.path.append(utils_dir)
import data_utils
import plot_utils
import MODIS_tile_corners as MOD_ref


# runtime packages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# numerical/data management packages
import numpy as np
import xarray as xr  # for multi dimensional data
import geopandas as gpd
# time and date packages
import time

# plotting packages
from matplotlib import pyplot as plt  # primary plotting module
import cartopy.crs as ccrs  # spatial plotting library 
from cartopy.io import img_tiles  # cartopy's implementation of webtiles
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
from matplotlib.ticker import FuncFormatter as FFmt  # need this for formatting plot ticks
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from shapely.geometry import mapping
from shapely.geometry import Polygon
import rasterio

#########################################
# Define Global Filepaths
#########################################
#%%

NLCD_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/NLCD/LandCover/regrid'
nlcd_file = os.path.join(NLCD_dir, 'NLCD2021_evergreen_wus_0083d.nc')

plot_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/North_America'
# Processed data files
AMF_sites_shp_file = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/eddy_covariance/data/shp/AMF_sites_forCCI.shp'


EPA_ecoregion_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/'
EPA_ecoregion_L1_file = os.path.join(EPA_ecoregion_dir, 'L1', 'NA_CEC_Eco_Level1_EPSG4326_regrid.shp')
EPA_ecoregion_L2_file = os.path.join(EPA_ecoregion_dir, 'L2/NAm_shp_FOR_MAIAC_PROC', 'NA_CEC_Eco_Level2_withOther.shp')
EPA_ecoregion_L3_file = os.path.join(EPA_ecoregion_dir, 'L3/NAm_shp_FOR_MAIAC_PROC', 'NAm_cec_eco_l3_v2.shp')
EPA_ecoregion_temperate_file = os.path.join(EPA_ecoregion_dir, 'L3/NAm_cec_eco_l3_Temperate.shp')
EPA_ecoregion_boreal_file = os.path.join(EPA_ecoregion_dir, 'L3/NAm_cec_eco_l3_Boreal_combined.shp')

########################################
# Define Global Variables and constants
#########################################

proc_years = np.array(range(2021, 2024))
fire_years = np.array([2020])
# plot_colors = ["#ffffb2", "#fd8d3c", "#f03b20", "#bd0026", "#810f7c", "#4B0981"]
plot_colors = ["#ffffb2", "#fd8d3c", "#f03b20", "#810f7c", "#21013B"]
burnarea_sumcheck = 0.1

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
### Setup map parameters
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

### Create a custom colormap
cmap = mcolors.ListedColormap(plot_colors)
cmaplist = [cmap(i) for i in range(cmap.N)] # extract all colors from the cmap

# # create the new map
# cmap = mcolors.LinearSegmentedColormap.from_list(
#     'Custom cmap', cmaplist, cmap.N)

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

#########################################
# Define Global Functions
#########################################

# matplotlib map plotting functions
def z_axis_formatter(x, pos, deci=0):
   '''Format ticks to format with deci number of decimals'''

   return f'{x:.{deci}f}'


#########################################
# Begin Main
#########################################

#%%
# mark start time to keep track of elapsed
start_time = time.time()

# # load evergreen forest data
# ENF_xr = xr.open_dataset(nlcd_file).sortby(['y', 'x'])
# ENF_xr_clip = ENF_xr.sel(x = slice(lon_bb_min, lon_bb_max), y = slice(lat_bb_min, lat_bb_max))


# Load AMF sites shapefile
AMF_sites_gdf = gpd.read_file(AMF_sites_shp_file)

# Extract coordinates from the shapefile
coords = [(point.y, point.x) for point in AMF_sites_gdf.geometry]


# Define the Lambert Conformal projection
lambert_proj = ccrs.LambertConformal(
   central_longitude=-100, central_latitude=45, standard_parallels=(30, 60)
)

# Load ecoregion shapefiles
ecoregions_l1_gdf = gpd.read_file(EPA_ecoregion_L1_file)
ecoregions_l2_gdf = gpd.read_file(EPA_ecoregion_L2_file)
ecoregions_l3_gdf = gpd.read_file(EPA_ecoregion_L3_file)


ecoregions_temperate_gdf = gpd.read_file(EPA_ecoregion_temperate_file)
ecoregions_boreal_gdf = gpd.read_file(EPA_ecoregion_boreal_file)

ecoregion_temperate_gdf = ecoregions_temperate_gdf.dissolve()
ecoregion_boreal_gdf = ecoregions_boreal_gdf.dissolve()


# %%

# # Define the Transverse Mercator projection
# tm_proj = ccrs.TransverseMercator(
#     central_longitude=-100,  # Central longitude of the projection
#     central_latitude=45  # Central latitude (default is the equator)
# )


# # use Google Satellite imagery as basemap
# tiler = img_tiles.GoogleTiles(style='satellite')
# # Create the figure and axes with the Lambert projection
# fig, ax = plt.subplots(
#     subplot_kw={"projection": tm_proj}, figsize=(12, 8)
# )

# # Set the extent (longitude, latitude) in degrees
# ax.set_extent([-167, -50, 25, 71], crs=ccrs.PlateCarree())

# # Add Google tiles
# ax.add_image(tiler, 6)  # Adjust zoom level as needed

# # Add gridlines and other map features
# ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

# # Optionally add coastlines, borders, etc.
# ax.coastlines()
# # ax.add_feature(ccrs.feature.BORDERS, linestyle=':')

# # Show the plot
# plt.show()



#%%

#################################################################
### Blank grid with L3 ecoregions, flux tower sites, PhotoSpec sites
#################################################################

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# import rasterio
from cartopy.feature import ShapelyFeature
from shapely.geometry import Polygon

# Set up the map with a projection
projection = ccrs.LambertConformal(central_longitude=-100, central_latitude=40)
fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': projection})

# Set map extent
# extent = [-170, -50, 19, 72]  # Adjusted to show all of Alaska and up to 71 degrees north
extent = [-170, -50, 9, 72]  # Adjusted to show all of Alaska and up to 71 degrees north

# Crop the map to 25% from the left
ax.set_extent(extent, crs=ccrs.PlateCarree())

# Add Natural Earth features
ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha = 0.7)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha = 0.5)
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
# ax.add_feature(cfeature.BORDERS, linestyle='--')

# # Plot the shapefile (Vector)
# shapefile_path = rivers_file # Replace with your shapefile path
# gdf = gpd.read_file(shapefile_path)
# gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=1)

# Plot the shapefile (Vector)
shapefile_path = lakes_file # Replace with your shapefile path
gdf = gpd.read_file(shapefile_path)
gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='#2b8dd6', edgecolor='#14629c', linewidth=1, alpha = 0.7)

# ecoregions_l1_gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='none', edgecolor='#193819', linewidth=1.2, alpha = 0.7)
# ecoregions_l2_gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='none', edgecolor='#519C1F', linewidth=0.8, alpha = 0.6)
# ecoregions_l3_gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='none', edgecolor='#AFC95F', linewidth=0.4, alpha = 0.4)


for ecoregion in ecoregions_l3_list:
    ecoregion_gdf = ecoregions_l3_gdf[ecoregions_l3_gdf.NA_L3NAME == ecoregion]
    ecoregion_gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='#519C1F', edgecolor='#345B1A', linewidth=0.8, alpha = 0.3)

# Plot KML points on the map
for coord in coords:
   ax.plot(coord[1], coord[0], '^', markersize=14, markerfacecolor='yellow', markeredgecolor='black', markeredgewidth=1, transform=ccrs.PlateCarree())

# Plot photospec site coordinates with green star markers
for site, (lon, lat) in photospec_site_coords.items():
   ax.plot(lon+0.2, lat+0.1, marker='*', color='#FFA947', markeredgecolor='black', markersize=20, markeredgewidth=1, transform=ccrs.PlateCarree())


ax.set_title("")
# plt.show()
# # Save the plot to a PNG file
png_path = os.path.join(plot_dir, 'NA_AMF_PhotoSpec_sites_w_L3ecoregions.png')
fig.savefig(png_path, format='png', bbox_inches='tight')
plt.close(fig)

#%%


# ### Blank zoom in on select ecoregions for the legend

# fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': projection})

# # Set map extent
# extent = [-125, -110, 40, 50]  # Adjusted to show all of Alaska and up to 71 degrees north
# ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha = 0.7)

# # Crop the map to 25% from the left
# ax.set_extent(extent, crs=ccrs.PlateCarree())

# # ecoregion_gdf = ecoregions_l3_gdf[ecoregions_l3_gdf.NA_L3NAME == 'Hay and Slave River Lowlands']
# # ecoregion_gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='#40B0A6', edgecolor='#258279', linewidth=1.5, alpha = 0.5)

# ecoregion_gdf = ecoregions_l3_gdf[ecoregions_l3_gdf.NA_L3NAME == 'Idaho Batholith']
# ecoregion_gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='#519C1F', edgecolor='#345B1A', linewidth=1.5, alpha = 0.5)


# ax.set_title("")
# plt.show()
# plt.close(fig)

#%%

#################################################################
### Blank grid with L3 ecoregions, Boreal vs. Temperate
#################################################################


fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': projection})

# Set map extent
# extent = [-170, -50, 19, 72]  # Adjusted to show all of Alaska and up to 71 degrees north
extent = [-170, -50, 9, 72]  # Adjusted to show all of Alaska and up to 71 degrees north

# Crop the map to 25% from the left
ax.set_extent(extent, crs=ccrs.PlateCarree())

# Add Natural Earth features
ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha = 0.7)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha = 0.5)
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
# ax.add_feature(cfeature.BORDERS, linestyle='--')

# # Plot the shapefile (Vector)
# shapefile_path = rivers_file # Replace with your shapefile path
# gdf = gpd.read_file(shapefile_path)
# gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=1)

# Plot the shapefile (Vector)
shapefile_path = lakes_file # Replace with your shapefile path
gdf = gpd.read_file(shapefile_path)
gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='#2b8dd6', edgecolor='#14629c', linewidth=1, alpha = 0.7)


for ecoregion in ecoregions_l3_list:
    if ecoregion in ecoregions_temperate_gdf.NA_L3NAME.values:
        ecoregion_gdf = ecoregions_l3_gdf[ecoregions_l3_gdf.NA_L3NAME == ecoregion]
        ecoregion_gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='#E1BE6A', edgecolor='#ECAB0D', linewidth=1.5, alpha = 0.5)
    elif ecoregion in ecoregions_boreal_gdf.NA_L3NAME.values:
        ecoregion_gdf = ecoregions_l3_gdf[ecoregions_l3_gdf.NA_L3NAME == ecoregion]
        ecoregion_gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='#40B0A6', edgecolor='#258279', linewidth=1.5, alpha = 0.5)

# ecoregion_temperate_gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='#E1BE6A', edgecolor='#ECAB0D', linewidth=1.5, alpha = 0.5)
# ecoregion_boreal_gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='#40B0A6', edgecolor='#258279', linewidth=1.5, alpha = 0.5)

ax.set_title("")
# plt.show()
# # Save the plot to a Png file
png_path = os.path.join(plot_dir, 'NA_Boreal_Temperate.png')
fig.savefig(png_path, format='png', bbox_inches='tight')
plt.close(fig)

#%%



# fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': projection})

# # Set map extent
# extent = [-125, -110, 40, 50]  # Adjusted to show all of Alaska and up to 71 degrees north
# ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha = 0.7)

# # Crop the map to 25% from the left
# ax.set_extent(extent, crs=ccrs.PlateCarree())

# # ecoregion_gdf = ecoregions_l3_gdf[ecoregions_l3_gdf.NA_L3NAME == 'Hay and Slave River Lowlands']
# # ecoregion_gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='#40B0A6', edgecolor='#258279', linewidth=1.5, alpha = 0.5)

# ecoregion_gdf = ecoregions_l3_gdf[ecoregions_l3_gdf.NA_L3NAME == 'Idaho Batholith']
# ecoregion_gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='#E1BE6A', edgecolor='#ECAB0D', linewidth=1.5, alpha = 0.5)


# ax.set_title("")
# plt.show()


#%%



taiga_color = "#5173B1"
northern_forests_color = '#44AA99'
western_cordillera_color = '#117733'
boreal_cordillera_color = '#88CCEE'


fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': projection})

# Set map extent
# extent = [-170, -50, 19, 72]  # Adjusted to show all of Alaska and up to 71 degrees north
extent = [-170, -50, 9, 72]  # Adjusted to show all of Alaska and up to 71 degrees north

# Crop the map to 25% from the left
ax.set_extent(extent, crs=ccrs.PlateCarree())

# Add Natural Earth features
ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha = 0.7)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha = 0.5)
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
# ax.add_feature(cfeature.BORDERS, linestyle='--')

# # Plot the shapefile (Vector)
# shapefile_path = rivers_file # Replace with your shapefile path
# gdf = gpd.read_file(shapefile_path)
# gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=1)

# Plot the shapefile (Vector)
shapefile_path = lakes_file # Replace with your shapefile path
gdf = gpd.read_file(shapefile_path)
gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='#2b8dd6', edgecolor='#14629c', linewidth=1, alpha = 0.7)


ecoregion_gdf_geom = ecoregions_l1_gdf[ecoregions_l1_gdf.NA_L1NAME == 'TAIGA']
ecoregion_gdf_geom.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor=taiga_color, edgecolor=taiga_color, linewidth=1.5, alpha = 0.5)

ecoregion_gdf_geom = ecoregions_l1_gdf[ecoregions_l1_gdf.NA_L1NAME == 'NORTHERN FORESTS']
ecoregion_gdf_geom.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor=northern_forests_color, edgecolor=northern_forests_color, linewidth=1.5, alpha = 0.5)

ecoregion_gdf_geom = ecoregions_l2_gdf[ecoregions_l2_gdf.NA_L2NAME == 'WESTERN CORDILLERA']
ecoregion_gdf_geom.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor=western_cordillera_color, edgecolor=western_cordillera_color, linewidth=1.5, alpha = 0.5)

ecoregion_gdf_geom = ecoregions_l2_gdf[ecoregions_l2_gdf.NA_L2NAME == 'BOREAL CORDILLERA']
ecoregion_gdf_geom.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor=boreal_cordillera_color, edgecolor=boreal_cordillera_color, linewidth=1.5, alpha = 0.5)

# ecoregion_temperate_gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='#40B0A6', edgecolor='#258279', linewidth=1.5, alpha = 0.5)
# ecoregion_boreal_gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='#E1BE6A', edgecolor='#ECAB0D', linewidth=1.5, alpha = 0.5)

ax.set_title("")
# plt.show()
# # Save the plot to a png file
png_path = os.path.join(plot_dir, 'NA_ENF_L1_L2.png')
fig.savefig(png_path, format='png', bbox_inches='tight')
plt.close(fig)

# #%%
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# # import rasterio
# from cartopy.feature import ShapelyFeature
# from shapely.geometry import Polygon

# # Set up the map with a projection
# projection = ccrs.LambertConformal(central_longitude=-100, central_latitude=40)
# fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'projection': projection})

# # Set map extent
# # extent = [-170, -50, 19, 72]  # Adjusted to show all of Alaska and up to 71 degrees north
# extent = [-170, -50, 9, 72]  # Adjusted to show all of Alaska and up to 71 degrees north

# # Crop the map to 25% from the left
# ax.set_extent(extent, crs=ccrs.PlateCarree())

# # Add Natural Earth features
# ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha = 0.7)
# ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha = 0.5)
# ax.add_feature(cfeature.COASTLINE, edgecolor='black')
# # ax.add_feature(cfeature.BORDERS, linestyle='--')

# # # Plot the shapefile (Vector)
# # shapefile_path = rivers_file # Replace with your shapefile path
# # gdf = gpd.read_file(shapefile_path)
# # gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=1)

# # Plot the shapefile (Vector)
# shapefile_path = lakes_file # Replace with your shapefile path
# gdf = gpd.read_file(shapefile_path)
# gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='#2b8dd6', edgecolor='#14629c', linewidth=1, alpha = 0.7)


# # Plot KML points on the map
# for coord in coords:
#    ax.plot(coord[1], coord[0], '^', markersize=14, markerfacecolor='yellow', markeredgecolor='black', markeredgewidth=1, transform=ccrs.PlateCarree())

# # Plot photospec site coordinates with green star markers
# for site, (lon, lat) in photospec_site_coords.items():
#    ax.plot(lon+0.2, lat+0.1, marker='*', color='#FFA947', markeredgecolor='black', markersize=20, markeredgewidth=1, transform=ccrs.PlateCarree())


# ax.set_title("")
# plt.show()
# # # Save the plot to a PNG file
# # png_path = os.path.join(plot_dir, 'NA_AMF_sites_wModisTiles.png')
# # fig.savefig(png_path, format='png', bbox_inches='tight')









# %%

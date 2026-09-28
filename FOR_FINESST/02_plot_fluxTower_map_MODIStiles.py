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

out_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/FOR_FINESST'
# Processed data files
AMF_sites_shp_file = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/eddy_covariance/data/shp/AMF_sites_forCCI.shp'

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

def create_polygon_with_intermediary_points(lon_min, lon_max, lat_min, lat_max, step=0.05):
   # Create lists of points along the perimeter
   lons = np.arange(lon_min, lon_max, step).tolist() + [lon_max] * int((lat_max - lat_min) / step) + np.arange(lon_max, lon_min, -step).tolist() + [lon_min] * int((lat_max - lat_min) / step)
   lats = [lat_min] * int((lon_max - lon_min) / step) + np.arange(lat_min, lat_max, step).tolist() + [lat_max] * int((lon_max - lon_min) / step) + np.arange(lat_max, lat_min, -step).tolist()

   # Combine the lists into a list of coordinate tuples
   coords = list(zip(lons, lats))
   
   # Create and return the polygon
   return Polygon(coords)

# Define domain bounding box

lon_h08v05_min = -130.5407
lon_h08v05_max = -103.9134
lat_h08v05_min = 30.0000
lat_h08v05_max = 40.0000

lon_h08v06_min = -115.4701
lon_h08v06_max = -95.7671
lat_h08v06_min = 20.0000
lat_h08v06_max = 30.0000

lon_h08v07_min = -106.4178
lon_h08v07_max = -91.3799
lat_h08v07_min = 10.0000
lat_h08v07_max = 20.0000

lon_h09v04_min = -140.0151
lon_h09v04_max = -104.4217
lat_h09v04_min = 40.0000
lat_h09v04_max = 50.0000

lon_h09v06_min = -103.9230
lon_h09v06_max = -85.1254
lat_h09v06_min = 20.0000
lat_h09v06_max = 30.0000

lon_h09v07_min = -95.7760
lon_h09v07_max = -81.2257
lat_h09v07_min = 10.0000
lat_h09v07_max = 20.0000

lon_h10v05_min = -104.4326
lon_h10v05_max = -80.8194
lat_h10v05_min = 30.0000
lat_h10v05_max = 40.0000

lon_h10v06_min = -92.3760
lon_h10v06_max = -74.4836
lat_h10v06_min = 20.0000
lat_h10v06_max = 30.0000

lon_h11v04_min = -108.9007   
lon_h11v04_max = -78.3136
lat_h11v04_min = 40.0000
lat_h11v04_max = 50.0000

lon_h11v03_min = -140.0
lon_h11v03_max = -93.3305
lat_h11v03_min = 50.0000
lat_h11v03_max = 60.0000

lon_h12v02_min = -175.4283
lon_h12v02_max = -99.9833
lat_h12v02_min = 60.0000
lat_h12v02_max = 70.0000

lon_h12v03_min = -120.0000
lon_h12v03_max = -77.7732
lat_h12v03_min = 50.0000
lat_h12v03_max = 60.0000

lon_h11v05_min = -91.3785    
lon_h11v05_max = -69.2724
lat_h11v05_min = 30.0000
lat_h11v05_max = 40.0000

lon_h12v04_min = -93.3434    
lon_h12v04_max = -65.2595
lat_h12v04_min = 40.0000
lat_h12v04_max = 50.0000

lon_h13v03_min = -100.0000  
lon_h13v03_max = -62.2160
lat_h13v03_min = 50.0000
lat_h13v03_max = 60.0000

# Create polygons for each bounding box
polygons = [
    create_polygon_with_intermediary_points(lon_h08v05_min, lon_h08v05_max, lat_h08v05_min, lat_h08v05_max),
    create_polygon_with_intermediary_points(lon_h08v06_min, lon_h08v06_max, lat_h08v06_min, lat_h08v06_max),
    create_polygon_with_intermediary_points(lon_h08v07_min, lon_h08v07_max, lat_h08v07_min, lat_h08v07_max),
    create_polygon_with_intermediary_points(lon_h10v06_min, lon_h10v06_max, lat_h10v06_min, lat_h10v06_max),
    create_polygon_with_intermediary_points(lon_h09v04_min, lon_h09v04_max, lat_h09v04_min, lat_h09v04_max),
    # create_polygon_with_intermediary_points(lon_h09v06_min, lon_h09v06_max, lat_h09v06_min, lat_h09v06_max),
    create_polygon_with_intermediary_points(lon_h09v07_min, lon_h09v07_max, lat_h09v07_min, lat_h09v07_max),
    create_polygon_with_intermediary_points(lon_h11v03_min, lon_h11v03_max, lat_h11v03_min, lat_h11v03_max),
    create_polygon_with_intermediary_points(lon_h12v02_min, lon_h12v02_max, lat_h12v02_min, lat_h12v02_max),
    create_polygon_with_intermediary_points(lon_h10v05_min, lon_h10v05_max, lat_h10v05_min, lat_h10v05_max),
    create_polygon_with_intermediary_points(lon_h11v04_min, lon_h11v04_max, lat_h11v04_min, lat_h11v04_max),
    create_polygon_with_intermediary_points(lon_h11v05_min, lon_h11v05_max, lat_h11v05_min, lat_h11v05_max),
    create_polygon_with_intermediary_points(lon_h13v03_min, lon_h13v03_max, lat_h13v03_min, lat_h13v03_max),
    # create_polygon_with_intermediary_points(lon_h12v03_min, lon_h12v03_max, lat_h12v03_min, lat_h12v03_max),
    create_polygon_with_intermediary_points(lon_h12v04_min, lon_h12v04_max, lat_h12v04_min, lat_h12v04_max)
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
print(AMF_sites_gdf.columns)
# Extract coordinates from the shapefile
coords = [(point.y, point.x) for point in AMF_sites_gdf.geometry]

for idx, row in AMF_sites_gdf.iterrows():
    print(f"{row['Name']}: ({round(row.geometry.x, 4)}, {round(row.geometry.y, 4)})")

# Define the Lambert Conformal projection
lambert_proj = ccrs.LambertConformal(
   central_longitude=-100, central_latitude=45, standard_parallels=(30, 60)
)


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



#######################################################
#### Test TEST TEST TEST reprojection of tiles to get lambert to work
#######################################################

#%%
# Load the raster file
NE_raster_file = "/uufs/chpc.utah.edu/common/home/lin-group19/ltk/pyutil/NaturalEarth/NE2_50M_SR_W/NE2_50M_SR_W.tif"
NE_raster = rasterio.open(NE_raster_file)
vector_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/pyutil/NaturalEarth'
coastlines_file = os.path.join(vector_dir, 'ne_110m_coastline.shp')
lakes_file = os.path.join(vector_dir, 'ne_110m_lakes.shp')
rivers_file = os.path.join(vector_dir, 'ne_110m_rivers_lake_centerlines.shp')
countries_file = os.path.join(vector_dir, 'ne_110m_admin_0_countries.shp')
lambert_conformal_proj4 = '+proj=lcc +lon_0=-100 +lat_0=40 +lat_1=33 +lat_2=45 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# import rasterio
from cartopy.feature import ShapelyFeature
from shapely.geometry import Polygon

# Set up the map with a projection
projection = ccrs.LambertConformal(central_longitude=-100, central_latitude=40)
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projection})

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


# Plot the bounding boxes
for shply_polygon in polygons:
   feature = ShapelyFeature([shply_polygon], ccrs.PlateCarree(), edgecolor='none', facecolor='red', alpha = 0.05)
   ax.add_feature(feature)

   feature = ShapelyFeature([shply_polygon], ccrs.PlateCarree(), edgecolor='darkred', facecolor='none')
   ax.add_feature(feature)



# Plot KML points on the map
for coord in coords:
   ax.plot(coord[1], coord[0], '^', markersize=6, markerfacecolor='yellow', markeredgecolor='black', markeredgewidth=0.7, transform=ccrs.PlateCarree())

# Plot photospec site coordinates with green star markers
for site, (lon, lat) in photospec_site_coords.items():
   ax.plot(lon+1, lat+0.5, marker='*', color='green', markeredgecolor='black', markersize=12, markeredgewidth=0.7, transform=ccrs.PlateCarree())


ax.set_title("")
# plt.show()
# Save the plot to a PDF file
pdf_path = os.path.join(out_dir, 'NA_AMF_sites_wModisTiles.pdf')
fig.savefig(pdf_path, format='pdf', bbox_inches='tight')

#%%











































######################################################
##### END TEST 
######################################################

# %%

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM

# Create an OSM tiling object
osm_tiles = OSM()

# Set up the map with a projection
projection = ccrs.LambertConformal(central_longitude=-100, central_latitude=40)
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projection})

# Set map extent
extent = [-130, -60, 20, 55]  # Example: North America
ax.set_extent(extent, crs=ccrs.PlateCarree())

# Add OSM tiles
ax.add_image(osm_tiles, 6)  # Zoom level: 6

# Add coastlines or other features if needed
ax.coastlines()
ax.set_title("Map with OpenStreetMap Tiles", fontsize=14)
plt.show()

# %%


# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import rasterio
from rasterio.plot import show


# Set up the map
projection = ccrs.LambertConformal(central_longitude=-100, central_latitude=40)
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projection})

# Plot the raster as a background
show(NE_raster, ax=ax, transform=ccrs.PlateCarree())

# Add features (optional)
ax.coastlines()
ax.set_title("Map with Static Raster Background", fontsize=14)

plt.show()



# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import rasterio
from rasterio.plot import show
from rasterio.warp import reproject, Resampling

# Set up the Lambert Conformal projection
projection = ccrs.LambertConformal(central_longitude=-100, central_latitude=40)

# Create a plot
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projection})

# Set extent for North America
ax.set_extent([-170, -60, 15, 85], crs=ccrs.PlateCarree())

# # Add natural features like coastlines, borders, and lakes
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.BORDERS, linestyle=':')
# ax.add_feature(cfeature.LAND, edgecolor='black')
# ax.add_feature(cfeature.LAKES, edgecolor='black')

# Plot the raster (GeoTIFF) after reprojection
raster_path = NE_raster_file  # Replace with your GeoTIFF path
with rasterio.open(raster_path) as src:
    # Get the affine transform and CRS for reprojecting
    src_crs = src.crs
    src_transform = src.transform

    # Create a temporary raster for the reprojected image
    data = src.read(1)  # Read the first band

    # Reproject the data to Lambert Conformal
    lambert_conformal_proj4 = '+proj=lcc +lon_0=-100 +lat_0=40 +lat_1=33 +lat_2=45 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
    target_crs = lambert_conformal_proj4  # Use the Proj4 string for Lambert Conformal
    target_transform, width, height = rasterio.warp.calculate_default_transform(
        src_crs, target_crs, src.width, src.height, *src.bounds)
    
    # Create an empty array for the reprojected data
    reprojected_data = np.empty((height, width), dtype=np.float32)
    
    # Reproject the data
    reproject(
        data, reprojected_data,
        src_crs=src_crs, src_transform=src_transform,
        dst_crs=target_crs, dst_transform=target_transform,
        resampling=Resampling.nearest
    )

    # Plot the reprojected raster on the map
    ax.imshow(reprojected_data, origin='upper', transform=projection, extent=src.bounds, interpolation='nearest')

# Plot the shapefile (Vector)
shapefile_path = coastlines_file # Replace with your shapefile path
gdf = gpd.read_file(shapefile_path)
gdf.to_crs(lambert_conformal_proj4).plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=1)

# Show the plot
plt.title("North America Map with Lambert Conformal Projection")
plt.show()

# %%

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
import dask as da
import geopandas as gpd

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

EPA_ecoregion_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/'
EPA_ecoregion_L1_file = os.path.join(EPA_ecoregion_dir, 'L1', 'NA_CEC_Eco_Level1_EPSG4326_regrid.shp')
EPA_ecoregion_L2_file = os.path.join(EPA_ecoregion_dir, 'L2/NAm_shp_FOR_MAIAC_PROC', 'NA_CEC_Eco_Level2_withOther.shp')
EPA_ecoregion_L3_file = os.path.join(EPA_ecoregion_dir, 'L3/NAm_shp_FOR_MAIAC_PROC', 'NAm_cec_eco_l3_v2.shp')

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
                      "SOUTHERN SEMI-ARID HIGHLANDS",
                      "TEMPERATE SIERRAS",
                      "TROPICAL WET FORESTS",
                      "TROPICAL DRY FORESTS"]

ecoregions_l2_list = ['SOUTH CENTRAL SEMIARID PRAIRIES', 
                      'WESTERN CORDILLERA', 
                      'BOREAL CORDILLERA', 
                      "SOFTWOOD SHIELD", 
                      "SOUTHEASTERN USA PLAINS", 
                      "WARM DESERTS"]

#########################################
# Define Global functions
#########################################

def td_min(td):
    return td.seconds//60

def td_sec(td):
    return td.seconds%60



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
# Define Global Variables and constants
#########################################

#########################################
# Begin main
#########################################
#%%
# def main():

# mark start time to keep track of elapsed
start_total = time.time()


plot_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/North_America/'

ESA_CCI_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/'

# Sample 0.01° 1D Rectilinear file for North America
ESA_CCI_file = os.path.join(ESA_CCI_dir, f'forest_fraction_0.01deg_north_america.nc')
ESA_CCI_xr = xr.open_dataset(ESA_CCI_file, decode_coords = 'all').rename({'x': 'lon', 'y': 'lat'}).sortby(['lon', 'lat'], ascending = True)['fraction_forest']
ESA_CCI_xr.rio.write_crs('EPSG:4326', inplace=True)

ecoregion_l1_gdf = gpd.read_file(EPA_ecoregion_L1_file)
ecoregion_l2_gdf = gpd.read_file(EPA_ecoregion_L2_file)

plot_extent = [ESA_CCI_xr.lon.min().values, ESA_CCI_xr.lon.max().values, ESA_CCI_xr.lat.min().values, ESA_CCI_xr.lat.max().values]

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

fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={'projection': crs}) # establish the figure, axes
ax.set_extent(plot_extent, crs=ccrs.PlateCarree())  # PlateCarree is default lat/lon    
ax.add_image(tiler, tiler_zoom, interpolation='none') # Add base image
# ax.add_feature(states_geom, facecolor="none", edgecolor="black", linewidth=1) # add state boundaries to the map


################################################################
### Level 1 Ecoregions
################################################################

# # Add TUNDRA
# ecoregion_gdf_geom = ecoregion_l1_gdf[ecoregion_l1_gdf.NA_L1NAME == 'TUNDRA'].geometry
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor=tundra_color, edgecolor='none', alpha=0.5)
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor='none', edgecolor=tundra_color, linewidth=1.2)

# Add TAIGA
ecoregion_gdf_geom = ecoregion_l1_gdf[ecoregion_l1_gdf.NA_L1NAME == 'TAIGA'].geometry
ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor=taiga_color, edgecolor='none', alpha=0.5)
ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor='none', edgecolor=taiga_color, linewidth=1.2)

# # Add HUDSON PLAIN
# ecoregion_gdf_geom = ecoregion_l1_gdf[ecoregion_l1_gdf.NA_L1NAME == 'HUDSON PLAIN'].geometry
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor=hudson_plain_color, edgecolor='none', alpha=0.5)
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor='none', edgecolor=hudson_plain_color, linewidth=1.2)

# # Add NORTHWESTERN FORESTED MOUNTAINS
# ecoregion_gdf_geom = ecoregion_l1_gdf[ecoregion_l1_gdf.NA_L1NAME == 'NORTHWESTERN FORESTED MOUNTAINS'].geometry
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor=northwestern_forested_mountains_color, edgecolor='none', alpha=0.5)
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor='none', edgecolor=northwestern_forested_mountains_color, linewidth=1.2)

# Add NORTHERN FORESTS
ecoregion_gdf_geom = ecoregion_l1_gdf[ecoregion_l1_gdf.NA_L1NAME == 'NORTHERN FORESTS'].geometry
ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor=northern_forests_color, edgecolor='none', alpha=0.5)
ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor='none', edgecolor=northern_forests_color, linewidth=1.2)

# # ADD MARINE WEST COAST FOREST
# ecoregion_gdf_geom = ecoregion_l1_gdf[ecoregion_l1_gdf.NA_L1NAME == 'MARINE WEST COAST FOREST'].geometry
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor=marine_west_coast_forest_color, edgecolor='none', alpha=0.5)
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor='none', edgecolor=marine_west_coast_forest_color, linewidth=1.2)

# # Add EASTERN TEMPERATE FORESTS
# ecoregion_gdf_geom = ecoregion_l1_gdf[ecoregion_l1_gdf.NA_L1NAME == 'EASTERN TEMPERATE FORESTS'].geometry
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor=eastern_temperate_forests_color, edgecolor='none', alpha=0.5)
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor='none', edgecolor=eastern_temperate_forests_color, linewidth=1.2)

# # Add GREAT PLAINS
# ecoregion_gdf_geom = ecoregion_l1_gdf[ecoregion_l1_gdf.NA_L1NAME == 'GREAT PLAINS'].geometry
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor=great_plains_color, edgecolor='none', alpha=0.5)
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor='none', edgecolor=great_plains_color, linewidth=1.2)

# # Add NORTH AMERICAN DESERTS
# ecoregion_gdf_geom = ecoregion_l1_gdf[ecoregion_l1_gdf.NA_L1NAME == 'NORTH AMERICAN DESERTS'].geometry
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor=north_american_deserts_color, edgecolor='none', alpha=0.5)
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor='none', edgecolor=north_american_deserts_color, linewidth=1.2)

# # Add MEDITERRANEAN CALIFORNIA
# ecoregion_gdf_geom = ecoregion_l1_gdf[ecoregion_l1_gdf.NA_L1NAME == 'MEDITERRANEAN CALIFORNIA'].geometry
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor=mediterranean_california_color, edgecolor='none', alpha=0.5)
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor='none', edgecolor=mediterranean_california_color, linewidth=1.2)

# # Add SOUTHERN SEMI-ARID HIGHLANDS
# ecoregion_gdf_geom = ecoregion_l1_gdf[ecoregion_l1_gdf.NA_L1NAME == 'SOUTHERN SEMIARID HIGHLANDS'].geometry
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor=southern_semiarid_highlands_color, edgecolor='none', alpha=0.5)
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor='none', edgecolor=southern_semiarid_highlands_color, linewidth=1.2)

# # Add TEMPERATE SIERRAS
# ecoregion_gdf_geom = ecoregion_l1_gdf[ecoregion_l1_gdf.NA_L1NAME == 'TEMPERATE SIERRAS'].geometry
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor=temperate_sierras_color, edgecolor='none', alpha=0.5)
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor='none', edgecolor=temperate_sierras_color, linewidth=1.2)

# # Add TROPICAL DRY FORESTS
# ecoregion_gdf_geom = ecoregion_l1_gdf[ecoregion_l1_gdf.NA_L1NAME == 'TROPICAL DRY FORESTS'].geometry
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor=tropical_dry_forests_color, edgecolor='none', alpha=0.5)
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor='none', edgecolor=tropical_dry_forests_color, linewidth=1.2)

# # Add TROPICAL WET FORESTS
# ecoregion_gdf_geom = ecoregion_l1_gdf[ecoregion_l1_gdf.NA_L1NAME == 'TROPICAL WET FORESTS'].geometry
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor=tropical_wet_forests_color, edgecolor='none', alpha=0.5)
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor='none', edgecolor=tropical_wet_forests_color, linewidth=1.2)


################################################################
### Level 2 Ecoregions
################################################################

# # Add SOUTH CENTRAL SEMIARID PRAIRIES
# ecoregion_gdf_geom = ecoregion_l2_gdf[ecoregion_l2_gdf.NA_L2NAME == 'SOUTH CENTRAL SEMIARID PRAIRIES'].geometry
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor=south_central_semiarid_prairies_color, edgecolor='none', alpha=0.5)
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor='none', edgecolor=south_central_semiarid_prairies_color, linewidth=2)

# # Add WARM DESERTS
# ecoregion_gdf_geom = ecoregion_l2_gdf[ecoregion_l2_gdf.NA_L2NAME == 'WARM DESERTS'].geometry
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor=warm_desert_color, edgecolor='none', alpha=0.5)
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor='none', edgecolor=warm_desert_color, linewidth=2)

# Add WESTERN CORDILLERA
ecoregion_gdf_geom = ecoregion_l2_gdf[ecoregion_l2_gdf.NA_L2NAME == 'WESTERN CORDILLERA'].geometry
ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor=western_cordillera_color, edgecolor='none', alpha=0.5)
ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor='none', edgecolor=western_cordillera_color, linewidth=2)

# # Add SOFTWOOD SHIELD
# ecoregion_gdf_geom = ecoregion_l2_gdf[ecoregion_l2_gdf.NA_L2NAME == 'SOFTWOOD SHIELD'].geometry
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor=softwood_shield_color, edgecolor='none', alpha=0.5)
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor='none', edgecolor=softwood_shield_color, linewidth=2)

# Add BOREAL CORDILLERA
ecoregion_gdf_geom = ecoregion_l2_gdf[ecoregion_l2_gdf.NA_L2NAME == 'BOREAL CORDILLERA'].geometry
ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor=boreal_cordillera_color, edgecolor='none', alpha=0.5)
ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor='none', edgecolor=boreal_cordillera_color, linewidth=2)

# # Add SOUTHEASTERN USA PLAINS
# ecoregion_gdf_geom = ecoregion_l2_gdf[ecoregion_l2_gdf.NA_L2NAME == 'SOUTHEASTERN USA PLAINS'].geometry
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor=southeastern_usa_plains_color, edgecolor='none', alpha=0.5)
# ax.add_geometries(ecoregion_gdf_geom, crs=transform, facecolor='none', edgecolor=southeastern_usa_plains_color, linewidth=2)


# outfile = os.path.join(plot_dir, f'NorthAmerica_major_ecoregions.png')
outfile = os.path.join(plot_dir, f'NorthAmerica_major_ecoregions_Boreal.png')
plt.savefig(outfile, dpi=150, bbox_inches='tight')
plt.close()

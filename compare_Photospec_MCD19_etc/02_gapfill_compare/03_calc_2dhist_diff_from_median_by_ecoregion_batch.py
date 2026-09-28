#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# For one ecoregion: accumulate fine-resolution histograms of the difference from the historical median
# (CCI - CCI_median, NDVI - NDVI_median) for every composite period, split into outlier-flagged vs. kept,
# pooled over all tiles/pixels in the ecoregion (with land cover masking) and all years 2001-2024.
# Input: per-tile, per-year diff_from_median files from 06_MCD19A1_CV-MVC_diff_from_median.py (1 km sinusoidal grid)
# Output: one small .npz per ecoregion in summary_data, used by the heatmap plotting script.
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
import shapely  # CHANGED: vectorized point-in-polygon test on the curvilinear grid (requires shapely >= 2.0)
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
mask_LC_high = True
mask_LC_low = False

# directories


EPA_ecoregion_dir = '/uufs/chpc.utah.edu/common/home/lin-group23/ltk/EPA/ecoregions/'
EPA_ecoregion_L1_file = os.path.join(EPA_ecoregion_dir, 'L1', 'NA_CEC_Eco_Level1_EPSG4326_regrid.shp')
EPA_ecoregion_Eastern_file = os.path.join(EPA_ecoregion_dir, 'L3', 'Eastern_Forests.shp')
EPA_ecoregion_boreal_file = os.path.join(EPA_ecoregion_dir, 'L3', 'NAm_cec_eco_l3_Boreal_combined.shp')
EPA_ecoregion_WUS_file = os.path.join(EPA_ecoregion_dir, 'L3', 'NAm_cec_eco_l3_Temperate_ENF.shp')
EPA_ecoregion_US_Mex_file = os.path.join(EPA_ecoregion_dir, 'L2', 'na_cec_eco_l2_ENF_US_Mex.shp')


ESA_CCI_LC_mask_dir = "/uufs/chpc.utah.edu/common/home/lin-group23/ltk/ESA_CCI/LandCover/"
ESA_CCI_mixed_forest_mask_file = os.path.join(ESA_CCI_LC_mask_dir, "mixed_forest_fraction_0.01deg_north_america.nc")
ESA_CCI_ENF_openclosed_mask_file = os.path.join(ESA_CCI_LC_mask_dir, "needleleaf_evergreen_open_closed_forest_fraction_0.01deg_north_america.nc")
ESA_CCI_ENF_closed_mask_file = os.path.join(ESA_CCI_LC_mask_dir, "needleleaf_evergreen_closed_forest_fraction_0.01deg_north_america.nc")
ESA_CCI_Grass_mask_file = os.path.join(ESA_CCI_LC_mask_dir, "grassland_fraction_0.01deg_north_america.nc")
ESA_CCI_Shrub_mask_file = os.path.join(ESA_CCI_LC_mask_dir, "shrubland_fraction_0.01deg_north_america.nc")



########################################
# Define Global Variables and constants
#########################################
#%%

ecoregions_list = ['Boreal', 'Eastern', 'Western', 'Southwest US Mexico', 'GREAT PLAINS', 'NORTH AMERICAN DESERTS']
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



ecoregions_tiles_list={
    'Boreal': ['h10v02', 'h10v03', 'h11v02', 'h11v03', 'h12v02', 'h12v03', 'h12v04', 'h13v02', 'h13v03', 'h13v04', 'h14v03'],
    'Eastern': ['h10v05',  'h11v04', 'h11v05', 'h12v04', 'h13v04'], #['h09v05', 'h10v05', 'h10v06', 'h11v04', 'h11v05', 'h12v04', 'h12v05', 'h13v04'],
    'Western': ['h08v04', 'h08v05', 'h09v04', 'h09v05', 'h10v03', 'h10v04'],
    'Southwest US Mexico': ['h08v06', 'h08v05', 'h09v05'],
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

# CHANGED: settings for the difference-from-median histograms
analysis_years = np.arange(2001, 2025)      # 2000 and 2025 excluded
indices = ['CCI', 'NDVI']                   # axis 0 of the counts array
categories = ['flagged', 'kept']            # axis 1 of the counts array (flagged = outlier_{idx} == 1, kept = outlier_{idx} == 0)
fine_edges = np.round(np.arange(-2.0, 2.0 + 0.00025, 0.0005), 6)   # fine bin edges for the difference from median (0.0005 wide)
n_fine = len(fine_edges) + 1                # number of slots per histogram: underflow (< -2) + fine bins + overflow (>= 2)
                                            # slot k (1..len(fine_edges)-1) covers [fine_edges[k-1], fine_edges[k])


#########################################
# Define Global Functions
#########################################

def dissolve_ecoregion_shapefile(ecoregions_gdf, buffer_dist = 0.001):
  # buffer_dist is in degrees, tune based on your gap size

    ecoregions_gdf['geometry'] = (
        ecoregions_gdf.geometry
        .to_crs(epsg=4326)
        .buffer(buffer_dist)
        .buffer(-buffer_dist)
    )
    return ecoregions_gdf.dissolve()


# CHANGED: masks on the 1 km curvilinear tile grid (rio.clip / rio.reproject_match don't apply to integer y/x dims with 2D lat/lon)
def ecoregion_mask_for_tile(geom, lat2d, lon2d):
    # geom: single (Multi)Polygon in lon/lat (EPSG:4326), already prepared with shapely.prepare
    # returns a 2D boolean (y, x) array: True where the pixel center falls inside the ecoregion
    lat = lat2d.values
    lon = lon2d.values
    minx, miny, maxx, maxy = geom.bounds
    inbox = (np.isfinite(lat) & np.isfinite(lon) &
             (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy))   # only test pixels inside the bounding box
    mask = np.zeros(lat.shape, dtype=bool)
    if inbox.any():
        mask[inbox] = shapely.contains_xy(geom, lon[inbox], lat[inbox])
    return mask


def regrid_LC_to_tile(LC_xr, lat2d, lon2d):
    # bilinear interpolation of the 0.01 deg land cover fraction grid to the 2D pixel-center lat/lon of the tile
    # returns a 2D (y, x) numpy array; NaN where the tile falls outside the land cover grid or lat/lon is NaN
    if 'x' in LC_xr.dims:
        LC_xr = LC_xr.rename({'x': 'lon', 'y': 'lat'})
    LC_xr = LC_xr.sortby(['lat', 'lon'])
    lat_vals = lat2d.values
    lon_vals = lon2d.values
    LC_sub = LC_xr.sel(lat=slice(np.nanmin(lat_vals) - 0.05, np.nanmax(lat_vals) + 0.05))
    if not ((np.nanmin(lon_vals) < -175) & (np.nanmax(lon_vals) > 100)):   # skip the lon subset for tiles crossing the 180 meridian
        LC_sub = LC_sub.sel(lon=slice(np.nanmin(lon_vals) - 0.05, np.nanmax(lon_vals) + 0.05))
    LC_sub = LC_sub.load()
    LC_tile = LC_sub.interp(lat=xr.DataArray(lat_vals, dims=('y', 'x')),
                            lon=xr.DataArray(lon_vals, dims=('y', 'x')),
                            method='linear')
    return LC_tile.values


composite_periods = np.arange(0,23)
#########################################
# Begin Main
#########################################

def main():

    start_total = time.time()

    ecoregion = f'{sys.argv[1].replace("_", " ")}' #SOUTH_CENTRAL_SEMIARID_PRAIRIES
    QC_option = sys.argv[2] # 'CloudFree_LowAOD_ClearAdj'

    if len(sys.argv) > 3:
        LC_threshold_high = float(sys.argv[3])
    else:
        LC_threshold_high = 0.8

    if len(sys.argv) > 4:
        LC_threshold_low = float(sys.argv[4])
    else:
        LC_threshold_low = 0.2

    # CHANGED: input is the per-tile diff_from_median files on the 1 km sinusoidal grid
    QC_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/QC_OutlierFilter/QCfilt_{QC_option}/'
    # historical climatology (median / MAD per composite period), used for the ecoregion median MAD
    prefill_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/pre-fill/QCfilt_{QC_option}/'

    print(f'Processing ecoregion: {ecoregion}...')

    tiles_to_process = ecoregions_tiles_list[ecoregion]

    ecoregions_Eastern_gdf = gpd.read_file(EPA_ecoregion_Eastern_file)
    ecoregions_boreal_gdf = gpd.read_file(EPA_ecoregion_boreal_file)
    ecoregions_WUS_gdf = gpd.read_file(EPA_ecoregion_WUS_file)
    ecoregions_US_Mex_gdf = gpd.read_file(EPA_ecoregion_US_Mex_file)

       


    if ecoregion in ecoregions_l1_list:
        ecoregions_gdf = gpd.read_file(EPA_ecoregion_L1_file)
        # print(ecoregion_l2_gdf['NA_L2NAME'].unique())
        ecoregions_gdf_this_eco = ecoregions_gdf[ecoregions_gdf.NA_L1NAME == ecoregion]
        ecoregion_gdf = ecoregions_gdf_this_eco[ecoregions_gdf_this_eco['Shape_Area'] > 1e+9] # filter out small areas
        tiles_to_process = ecoregions_tiles_list[ecoregion]  # FIXED: was ecoregions_l1_tiles_list (undefined)
    elif ecoregion == 'Boreal':
        ecoregions_gdf = ecoregions_boreal_gdf[ecoregions_boreal_gdf['Shape_Area'] > 1e+9]
        ecoregion_gdf = dissolve_ecoregion_shapefile(ecoregions_gdf)
    elif ecoregion == 'Eastern':
        ecoregions_gdf = ecoregions_Eastern_gdf[ecoregions_Eastern_gdf['Shape_Area'] > 1e+9]
        ecoregion_gdf = dissolve_ecoregion_shapefile(ecoregions_gdf)
    elif ecoregion == 'Western':
        ecoregions_gdf = ecoregions_WUS_gdf[ecoregions_WUS_gdf['Shape_Area'] > 1e+9]
        ecoregion_gdf = dissolve_ecoregion_shapefile(ecoregions_gdf)
    elif ecoregion == 'Southwest US Mexico':  # FIXED: was 'Southwest_US_Mexico', which never matched after underscores were replaced
        ecoregions_gdf = ecoregions_US_Mex_gdf[ecoregions_US_Mex_gdf['Shape_Area'] > 1e+9]
        ecoregion_gdf = dissolve_ecoregion_shapefile(ecoregions_gdf)

    ecoregion_gdf_geom = ecoregion_gdf.geometry


    n_parts = len(list(ecoregion_gdf_geom.iloc[0].geoms)) if ecoregion_gdf_geom.iloc[0].geom_type == 'MultiPolygon' else 1
    n_verts = sum(len(p.exterior.coords) + sum(len(i.coords) for i in p.interiors) 
                for p in (ecoregion_gdf_geom.iloc[0].geoms if n_parts > 1 else [ecoregion_gdf_geom.iloc[0]]))
    if n_verts > 1e+6:
        print(f'vertices before simplify: {n_verts}, simplifying geometry...')
        ecoregion_gdf_geom = ecoregion_gdf_geom.simplify(0.005, preserve_topology=True)

        n_verts_simplified = sum(len(p.exterior.coords) + sum(len(i.coords) for i in p.interiors) 
                                for p in (ecoregion_gdf_geom.iloc[0].geoms 
                                            if ecoregion_gdf_geom.iloc[0].geom_type == 'MultiPolygon' 
                                            else [ecoregion_gdf_geom.iloc[0]]))
        print(f'vertices after simplify: {n_verts_simplified}')

    # CHANGED: single geometry for point-in-polygon tests (L1 ecoregions have multiple rows)
    ecoregion_geom = ecoregion_gdf_geom.union_all() if hasattr(ecoregion_gdf_geom, 'union_all') else ecoregion_gdf_geom.unary_union
    shapely.prepare(ecoregion_geom)


    print("Loading LC masks...")

    ENF_openclosed_xr = xr.open_dataset(ESA_CCI_ENF_openclosed_mask_file, decode_coords='all')['fraction_ENF'].squeeze()
    ENF_closed_xr = xr.open_dataset(ESA_CCI_ENF_closed_mask_file, decode_coords='all')['fraction_ENF'].squeeze()
    Grass_xr = xr.open_dataset(ESA_CCI_Grass_mask_file, decode_coords='all')['fraction_GRA'].squeeze()
    Shrub_xr = xr.open_dataset(ESA_CCI_Shrub_mask_file, decode_coords='all')['fraction_shrubland'].squeeze()
    Mixed_forest_xr = xr.open_dataset(ESA_CCI_mixed_forest_mask_file, decode_coords='all')['fraction_mixed_forest'].squeeze()

    if ecoregion == "Boreal":
        LC_xr = ENF_openclosed_xr
    elif ecoregion == "Eastern":
        LC_xr = Mixed_forest_xr
    elif ecoregion == "Western":
        LC_xr = ENF_openclosed_xr
    elif ecoregion == "Southwest US Mexico":
        LC_xr = ENF_openclosed_xr
    elif ecoregion == "GREAT PLAINS":
        LC_xr = Grass_xr
    elif ecoregion == "NORTH AMERICAN DESERTS":
        LC_xr = Shrub_xr

    # Assume that both mask_LC_high and mask_LC_low cannot be True at the same time. If both are False, then no masking is applied.
    if mask_LC_high and mask_LC_low:
        raise ValueError("Both mask_LC_high and mask_LC_low cannot be True at the same time. Please set one of them to False.")


    #########################################
    # CHANGED: accumulate fine-resolution histograms over tiles and years
    #########################################

    # counts[index, category, composite_period, slot]: number of pixel-years in each fine bin
    counts = np.zeros((len(indices), len(categories), len(composite_periods), n_fine), dtype=np.int64)
    tiles_used = []
    n_region_pixels = {}   # number of pixels in the ecoregion + LC mask, per tile
    # MAD values of every pixel in the ecoregion + LC mask, per index and composite period (lists of 1D arrays, one per tile)
    MAD_values = {idx: [[] for _ in composite_periods] for idx in indices}

    for MODIS_tile in tiles_to_process:

        tile_dir = os.path.join(QC_basedir, MODIS_tile, 'diff_from_median')
        tile_files = {year: os.path.join(tile_dir, f'MCD19A1.{MODIS_tile}.061_1km_CV-MVC_QCfilt_{QC_option}_diff_from_median_{year}.nc')
                      for year in analysis_years}
        existing_years = [year for year, f in tile_files.items() if os.path.exists(f)]
        missing_years = [year for year in analysis_years if year not in existing_years]
        if len(missing_years) > 0:
            print(f'  Tile {MODIS_tile}: missing diff_from_median files for years {missing_years}')
        if len(existing_years) == 0:
            print(f'  No diff_from_median files for tile {MODIS_tile}, skipping...')
            continue

        # tile grid (same for every year) and the combined ecoregion + land cover mask, computed once per tile
        with xr.open_dataset(tile_files[existing_years[0]]) as ds0:
            lat2d = ds0['lat'].load()
            lon2d = ds0['lon'].load()

        print(f'  Tile {MODIS_tile}: building ecoregion mask...')
        region_mask = ecoregion_mask_for_tile(ecoregion_geom, lat2d, lon2d)

        # If masking for either high or low land cover, perform filtering here
        if mask_LC_high | mask_LC_low:
            print(f'  Tile {MODIS_tile}: regridding land cover mask to tile grid...')
            LC_tile = regrid_LC_to_tile(LC_xr, lat2d, lon2d)
            with np.errstate(invalid='ignore'):   # NaN land cover fraction -> comparison is False -> pixel excluded
                if mask_LC_high:
                    region_mask &= (LC_tile > LC_threshold_high)
                if mask_LC_low:
                    region_mask &= (LC_tile < LC_threshold_low)

        n_region_pixels[MODIS_tile] = int(region_mask.sum())
        print(f'  Tile {MODIS_tile}: {n_region_pixels[MODIS_tile]} pixels in ecoregion + LC mask')
        if n_region_pixels[MODIS_tile] == 0:
            continue
        tiles_used.append(MODIS_tile)

        # historical MAD (per composite period) for the pixels in the ecoregion + LC mask
        climatology_dir = os.path.join(prefill_basedir, MODIS_tile, 'hist_avg')
        for i_cp, cp in enumerate(composite_periods):
            fc = os.path.join(climatology_dir, f'MCD19A1.{MODIS_tile}.061_1km_CV-MVC_QCfilt_{QC_option}_hist_avg_{cp:02d}.nc')
            if not os.path.exists(fc):
                print(f'    Missing climatology file (no MAD for this period): {fc}')
                continue
            with xr.open_dataset(fc) as cds:
                for idx in indices:
                    mad = np.squeeze(cds[f'{idx}_MAD'].values)
                    assert mad.shape == region_mask.shape, f'{idx}_MAD: unexpected shape {mad.shape}'
                    mad = mad[region_mask]
                    MAD_values[idx][i_cp].append(mad[np.isfinite(mad)].astype(np.float32))

        for year in existing_years:
            print(f'    Tile {MODIS_tile}, year {year}...')
            with xr.open_dataset(tile_files[year]) as ds:
                assert np.array_equal(ds['composite_period'].values, composite_periods), 'unexpected composite_period coordinate'
                for i_idx, idx in enumerate(indices):
                    diff_vals = ds[f'{idx}_diff'].values        # (composite_period, y, x)
                    flag_vals = ds[f'outlier_{idx}'].values     # (composite_period, y, x)
                    for i_cp in range(len(composite_periods)):
                        d = diff_vals[i_cp][region_mask]       # 1D: differences for pixels in the ecoregion + LC mask
                        fl = flag_vals[i_cp][region_mask]
                        ok = np.isfinite(d)
                        slot = np.searchsorted(fine_edges, d[ok], side='right')   # 0 = underflow, len(fine_edges) = overflow
                        is_flagged = fl[ok] == 1
                        counts[i_idx, 0, i_cp] += np.bincount(slot[is_flagged], minlength=n_fine)
                        counts[i_idx, 1, i_cp] += np.bincount(slot[~is_flagged], minlength=n_fine)
            gc.collect()


    # median (across all pixels in the ecoregion + LC mask) of the historical MAD, per index and composite period
    MAD_median = np.full((len(indices), len(composite_periods)), np.nan)
    MAD_n = np.zeros((len(indices), len(composite_periods)), dtype=np.int64)
    for i_idx, idx in enumerate(indices):
        for i_cp in range(len(composite_periods)):
            if len(MAD_values[idx][i_cp]) == 0:
                continue
            vals = np.concatenate(MAD_values[idx][i_cp])
            MAD_n[i_idx, i_cp] = vals.size
            if vals.size > 0:
                MAD_median[i_idx, i_cp] = np.median(vals)
    del MAD_values
    gc.collect()

    # import pickle
    output_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/summary_data'
    Path(output_dir).mkdir(parents=True, exist_ok=True)


    if mask_LC_high:
        output_file = os.path.join(output_dir, f'{ecoregion.replace(" ", "_")}_diff_from_median_hist_LCmask_only.npz')
    elif mask_LC_low:
        output_file = os.path.join(output_dir, f'{ecoregion.replace(" ", "_")}_diff_from_median_hist_LCmask_low.npz')
    else:
        output_file = os.path.join(output_dir, f'{ecoregion.replace(" ", "_")}_diff_from_median_hist.npz')

    np.savez(output_file,
             counts=counts,
             fine_edges=fine_edges,
             MAD_median=MAD_median,   # (index, composite_period): median historical MAD across ecoregion + LC mask pixels
             MAD_n=MAD_n,             # (index, composite_period): number of pixels with a valid MAD
             indices=np.array(indices),
             categories=np.array(categories),
             composite_periods=composite_periods,
             years=analysis_years,
             tiles_used=np.array(tiles_used),
             n_region_pixels_tiles=np.array(list(n_region_pixels.keys())),
             n_region_pixels_values=np.array(list(n_region_pixels.values())),
             ecoregion=ecoregion,
             QC_option=QC_option,
             LC_threshold_high=LC_threshold_high if mask_LC_high else np.nan,
             LC_threshold_low=LC_threshold_low if mask_LC_low else np.nan)

    print(f'Saved difference-from-median histograms to {output_file}')
    print(f'elapsed: {(time.time() - start_total)/60:.1f} min')
    
if __name__ == "__main__":
    main()
# %%
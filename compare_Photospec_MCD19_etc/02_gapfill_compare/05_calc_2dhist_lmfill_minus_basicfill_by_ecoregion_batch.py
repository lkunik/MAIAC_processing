#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# For one ecoregion: accumulate fine-resolution histograms of lm-fill minus basic-fill (NDVI and CCI) at gap-filled
# pixels (gapfill_{VI} == 1) for every composite period, pooled over all tiles/pixels in the ecoregion (with land
# cover masking) and all years 2001-2024.
# Input: per-tile, per-year gap-fill diagnostics files (QC_gapfill, from 09_MCD19A1_GapFill_QC_diagnostics_batch.py) on the
#        1 km sinusoidal grid; 2D lat/lon from the annual OutlierFilterQC files of the same tile
# Output: one small .npz per ecoregion in summary_data, used by 05a_plot_lmfill_minus_basicfill_2dhist_by_ecoregion.py
# Same structure as 03_calc_2dhist_diff_from_median_by_ecoregion_batch.py.
#
# Usage: python 05_calc_2dhist_lmfill_minus_basicfill_by_ecoregion_batch.py <ECOREGION (underscores for spaces)> <QC_option> [LC_threshold_high] [LC_threshold_low]
#

#########################################
# Load packages
#########################################
#%%
# File system packages
import os  # operating system library
import glob
from pathlib import Path
import sys
import gc

# runtime packages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# numerical/data management packages
import numpy as np
import xarray as xr  # for multi dimensional data

# shapefile/geospatial packages
import geopandas as gpd
import shapely  # vectorized point-in-polygon test on the curvilinear grid (requires shapely >= 2.0)

# time and date packages
import time

#%%
#########################################
# Define Global Filepaths
#########################################
mask_LC_high = True
mask_LC_low = False

base_dir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/1km_curv/'

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

# settings for the lm-fill minus basic-fill histograms
analysis_years = np.arange(2001, 2025)      # 2000 and 2025 excluded
indices = ['CCI', 'NDVI']                   # axis 0 of the counts array
fine_edges = np.round(np.arange(-2.0, 2.0 + 0.00025, 0.0005), 6)   # fine bin edges for lm-fill minus basic-fill (0.0005 wide)
n_fine = len(fine_edges) + 1                # number of slots per histogram: underflow (< -2) + fine bins + overflow (>= 2)
                                            # slot k (1..len(fine_edges)-1) covers [fine_edges[k-1], fine_edges[k])

composite_periods = np.arange(0,23)


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


def tile_latlon(MODIS_tile, QC_option, gapfill_file):
    # 2D pixel-center lat/lon for the tile: from any annual OutlierFilterQC file of the tile, or from the
    # gap-fill diagnostics file itself if it carries lat/lon coordinates
    QC_files = sorted(glob.glob(os.path.join(base_dir, f'QC_OutlierFilter/QCfilt_{QC_option}/{MODIS_tile}/',
                                             f'MCD19A1.{MODIS_tile}.061_1km_CV-MVC_QCfilt_{QC_option}_annual_OutlierFilterQC_*.nc')))
    if len(QC_files) > 0:
        with xr.open_dataset(QC_files[0]) as ds:
            return ds['lat'].load(), ds['lon'].load()
    with xr.open_dataset(gapfill_file) as ds:
        if ('lat' in ds.variables) and ('lon' in ds.variables):
            return ds['lat'].load(), ds['lon'].load()
    raise FileNotFoundError(f'No 2D lat/lon found for tile {MODIS_tile}')


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

    # per-tile, per-year gap-fill diagnostics files on the 1 km sinusoidal grid
    gapfill_basedir = os.path.join(base_dir, f'QC_gapfill/QCfilt_{QC_option}/')

    print(f'Processing ecoregion: {ecoregion}...')

    tiles_to_process = ecoregions_tiles_list[ecoregion]

    ecoregions_Eastern_gdf = gpd.read_file(EPA_ecoregion_Eastern_file)
    ecoregions_boreal_gdf = gpd.read_file(EPA_ecoregion_boreal_file)
    ecoregions_WUS_gdf = gpd.read_file(EPA_ecoregion_WUS_file)
    ecoregions_US_Mex_gdf = gpd.read_file(EPA_ecoregion_US_Mex_file)


    if ecoregion in ecoregions_l1_list:
        ecoregions_gdf = gpd.read_file(EPA_ecoregion_L1_file)
        ecoregions_gdf_this_eco = ecoregions_gdf[ecoregions_gdf.NA_L1NAME == ecoregion]
        ecoregion_gdf = ecoregions_gdf_this_eco[ecoregions_gdf_this_eco['Shape_Area'] > 1e+9] # filter out small areas
        tiles_to_process = ecoregions_tiles_list[ecoregion]
    elif ecoregion == 'Boreal':
        ecoregions_gdf = ecoregions_boreal_gdf[ecoregions_boreal_gdf['Shape_Area'] > 1e+9]
        ecoregion_gdf = dissolve_ecoregion_shapefile(ecoregions_gdf)
    elif ecoregion == 'Eastern':
        ecoregions_gdf = ecoregions_Eastern_gdf[ecoregions_Eastern_gdf['Shape_Area'] > 1e+9]
        ecoregion_gdf = dissolve_ecoregion_shapefile(ecoregions_gdf)
    elif ecoregion == 'Western':
        ecoregions_gdf = ecoregions_WUS_gdf[ecoregions_WUS_gdf['Shape_Area'] > 1e+9]
        ecoregion_gdf = dissolve_ecoregion_shapefile(ecoregions_gdf)
    elif ecoregion == 'Southwest US Mexico':
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

    # single geometry for point-in-polygon tests (L1 ecoregions have multiple rows)
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
    # accumulate fine-resolution histograms over tiles and years
    #########################################

    # counts[index, composite_period, slot]: number of gap-filled pixel-years in each fine bin
    counts = np.zeros((len(indices), len(composite_periods), n_fine), dtype=np.int64)
    tiles_used = []
    n_region_pixels = {}   # number of pixels in the ecoregion + LC mask, per tile

    for MODIS_tile in tiles_to_process:

        tile_dir = os.path.join(gapfill_basedir, MODIS_tile)
        tile_files = {year: os.path.join(tile_dir, f'MCD19A1.{MODIS_tile}.061_1km_16day_CV-MVC_gapfillQC_QCfilt_{QC_option}_{year}.nc')
                      for year in analysis_years}
        existing_years = [year for year, f in tile_files.items() if os.path.exists(f)]
        missing_years = [year for year in analysis_years if year not in existing_years]
        if len(missing_years) > 0:
            print(f'  Tile {MODIS_tile}: missing gap-fill QC files for years {missing_years}')
        if len(existing_years) == 0:
            print(f'  No gap-fill QC files for tile {MODIS_tile}, skipping...')
            continue

        # tile grid (same for every year) and the combined ecoregion + land cover mask, computed once per tile
        lat2d, lon2d = tile_latlon(MODIS_tile, QC_option, tile_files[existing_years[0]])

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

        for year in existing_years:
            print(f'    Tile {MODIS_tile}, year {year}...')
            with xr.open_dataset(tile_files[year]) as ds:
                assert np.array_equal(ds['composite_period'].values, composite_periods), 'unexpected composite_period coordinate'
                for i_idx, idx in enumerate(indices):
                    diff_vals = ds[f'{idx}_lmfill_minus_basicfill'].values   # (composite_period, y, x)
                    flag_vals = ds[f'gapfill_{idx}'].values                  # (composite_period, y, x); -1 fill decodes to NaN
                    assert diff_vals.shape[1:] == region_mask.shape, f'{idx}: grid {diff_vals.shape[1:]} does not match lat/lon {region_mask.shape}'
                    for i_cp in range(len(composite_periods)):
                        d = diff_vals[i_cp][region_mask]       # 1D: lm-fill minus basic-fill for pixels in the ecoregion + LC mask
                        fl = flag_vals[i_cp][region_mask]
                        ok = (fl == 1) & np.isfinite(d)        # gap-filled pixels only
                        slot = np.searchsorted(fine_edges, d[ok], side='right')   # 0 = underflow, len(fine_edges) = overflow
                        counts[i_idx, i_cp] += np.bincount(slot, minlength=n_fine)
                    del diff_vals, flag_vals
            gc.collect()


    output_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/summary_data'
    Path(output_dir).mkdir(parents=True, exist_ok=True)


    if mask_LC_high:
        output_file = os.path.join(output_dir, f'{ecoregion.replace(" ", "_")}_lmfill_minus_basicfill_hist_LCmask_only.npz')
    elif mask_LC_low:
        output_file = os.path.join(output_dir, f'{ecoregion.replace(" ", "_")}_lmfill_minus_basicfill_hist_LCmask_low.npz')
    else:
        output_file = os.path.join(output_dir, f'{ecoregion.replace(" ", "_")}_lmfill_minus_basicfill_hist.npz')

    np.savez(output_file,
             counts=counts,           # (index, composite_period, n_fine): gap-filled pixel-years per fine bin
             fine_edges=fine_edges,
             indices=np.array(indices),
             composite_periods=composite_periods,
             years=analysis_years,
             tiles_used=np.array(tiles_used),
             n_region_pixels_tiles=np.array(list(n_region_pixels.keys())),
             n_region_pixels_values=np.array(list(n_region_pixels.values())),
             ecoregion=ecoregion,
             QC_option=QC_option,
             LC_threshold_high=LC_threshold_high if mask_LC_high else np.nan,
             LC_threshold_low=LC_threshold_low if mask_LC_low else np.nan)

    print(f'Saved lm-fill minus basic-fill histograms to {output_file}')
    print(f'elapsed: {(time.time() - start_total)/60:.1f} min')

if __name__ == "__main__":
    main()
# %%
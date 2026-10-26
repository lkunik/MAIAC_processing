#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Created on Wed Oct 21 2026
#
# Mosaic the per-tile 0.01° rectilinear 16-day composites from step 11 into a single North American
# grid. Separate files are written for CCI and NDVI (NAm_full/CCI/ and NAm_full/NDVI/), each also holding
# the QA_flag variable (the VI usefulness flag applies to both). One output file per year and variable, with one
# time step per composite (23 per year), where time is the start date of each 16-day composite (DOY 1, 17, 33, ..., 353).
# Each tile file is read once and used for both outputs.
#
# The step 11 tile outputs all sit on the same 0.01° grid (the ESA-CCI North America grid), so each
# tile is pasted into the NAm grid by index - no regridding. The lat/lon bounding boxes of neighbouring
# tiles overlap (sinusoidal tiles), but their valid-data footprints do not, so each tile only fills
# cells that are still empty. Any cell with valid data from more than one tile is counted and reported.
# QA_flag is 0 (no data) inside a tile footprint and NaN outside it, so a tile's flag only replaces a
# cell that is still unfilled or 0. Cells outside every tile footprint get the QA_flag _FillValue (-1).
#
# Usage: python 19_combine_NAm_tiles_01d_rect_CCI_NDVI_annual_batch.py <year> <QC_option>
#    e.g. python 19_combine_NAm_tiles_01d_rect_CCI_NDVI_annual_batch.py 2020 CloudFree_LowAOD_ClearAdj
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

# runtime packages
import gc

# numerical/data management packages
import numpy as np
import xarray as xr  # for multi dimensional data
import netCDF4  # used directly so that composites can be written to the output file one at a time
from pyproj import CRS

# time and date packages
import time
from datetime import datetime as dt  # date time library
from datetime import timedelta

# plotting packages
from matplotlib import pyplot as plt  # primary plotting module
import cartopy.crs as ccrs  #
import cartopy.feature as cfeature


#########################################
# Define Global Variables and constants
#########################################

MODIS_tiles = [
    'h07v06', 'h08v03', 'h08v04', 'h08v05', 'h08v06', 'h08v07',
    'h09v02', 'h09v03', 'h09v04', 'h09v05', 'h09v06', 'h09v07', 'h09v08',
    'h10v02', 'h10v03', 'h10v04', 'h10v05', 'h10v06', 'h10v07', 'h10v08',
    'h11v02', 'h11v03', 'h11v04', 'h11v05', 'h11v06', 'h11v07', 'h11v08',
    'h12v01', 'h12v02', 'h12v03', 'h12v04', 'h12v05',
    'h13v01', 'h13v02', 'h13v03', 'h13v04',
    'h14v01', 'h14v02', 'h14v03', 'h14v04',
    'h15v01', 'h15v02', 'h16v01',
]

var_long_names = {'CCI': 'Chlorophyll/Carotenoid Index',
                  'NDVI': 'Normalized Difference Vegetation Index'}  # one set of output files per variable
n_composites = 23
DOY_composite_starts = np.arange(1, 366, 16)  # composite NN (0-22) starts on DOY_composite_starts[NN]
grid_res = 0.01  # degrees
grid_decimals = 2  # step 11 coordinates carry float noise (e.g. -49.99000000010915), so round to the grid

out_dtype = 'f4'  # float32 is plenty for CCI/NDVI and halves the file size relative to the float64 tile files

# QA_flag (VI usefulness flag) values, as assigned in steps 01, 03 and 08
QA_var_name = 'QA_flag'
QA_flag_values = np.array([0, 1, 2, 3], dtype=np.int8)
QA_flag_meanings = 'no_data original_data gap_filled_after_outlier_removal gap_filled_originally_no_data'
QA_fill_value = np.int8(-1)  # outside every tile footprint
chunk_lat, chunk_lon = 1000, 1000
compress_level = 4

plot_composite = 12  # composite to show in the quick-look map


#########################################
# Define Global functions
#########################################

def format_elapsed(start_time):
    """Elapsed time since start_time (from time.time()) as e.g. '0h 3m 25.4s'."""
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    return f'{int(hours)}h {int(minutes)}m {seconds:.1f}s'


def composite_start_date(year, composite_nn):
    return dt(year, 1, 1) + timedelta(days=int(DOY_composite_starts[composite_nn]) - 1)


def read_tile_QA_flag(ds, f):
    """QA_flag of a step 11 tile as int8 (fill value outside the tile footprint). Step 11 files made before
    QA_flag was regridded by nearest neighbour hold bilinearly interpolated (fractional) flags - refuse those."""
    QA = ds[QA_var_name].transpose('lat', 'lon').sortby(['lat', 'lon']).values
    QA_valid = ~np.isnan(QA)
    if np.any(QA[QA_valid] != np.round(QA[QA_valid])) or not np.all(np.isin(QA[QA_valid], QA_flag_values)):
        raise ValueError(f'{os.path.basename(f)} has non-integer or unexpected {QA_var_name} values - it was likely regridded '
                         f'bilinearly by an older version of step 11; rerun step 11 for this file')
    return np.where(QA_valid, QA, QA_fill_value).astype(np.int8)


def tile_file(dat_basedir, MODIS_tile, QC_option, year, composite_nn):
    # Sample filename: MCD19A1.h09v04.061_01d_16day_CV-MVC_QCfilt_CloudFree_LowAOD_ClearAdj_2020-00_rect.nc
    return os.path.join(dat_basedir, MODIS_tile,
                        f'MCD19A1.{MODIS_tile}.061_01d_16day_CV-MVC_QCfilt_{QC_option}_{year}-{composite_nn:02d}_rect.nc')


def grid_index(coord_vals, coord_min):
    """Integer index of each coordinate value on the NAm grid, starting from coord_min."""
    idx = np.rint((coord_vals - coord_min) / grid_res).astype(int)
    if np.any(np.diff(idx) != 1):
        raise ValueError('tile coordinates are not contiguous on the 0.01° grid')
    return idx


def get_NAm_extent(tile_files):
    """Union of the lat/lon extents of all tile files (they share one 0.01° grid)."""
    lon_min, lon_max, lat_min, lat_max = np.inf, -np.inf, np.inf, -np.inf
    for f in tile_files:
        with xr.open_dataset(f) as ds:
            lon_min = min(lon_min, float(ds.lon.min()))
            lon_max = max(lon_max, float(ds.lon.max()))
            lat_min = min(lat_min, float(ds.lat.min()))
            lat_max = max(lat_max, float(ds.lat.max()))
    return [round(v, grid_decimals) for v in (lon_min, lon_max, lat_min, lat_max)]


def plot_quicklook(dat_2d, lon, lat, var_name, plot_title, plot_filename, coarsen = 10):
    """Coarsened quick-look map of the full NAm mosaic, to check tile placement and seams."""
    dat_plot = dat_2d[::coarsen, ::coarsen]
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    pcm = ax.pcolormesh(lon[::coarsen], lat[::coarsen], dat_plot, transform=ccrs.PlateCarree(),
                        cmap='RdYlGn', vmin=np.nanpercentile(dat_plot, 2), vmax=np.nanpercentile(dat_plot, 98))
    plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.02, shrink=0.75, label=var_name)
    plt.title(plot_title, fontsize=18)
    Path(os.path.dirname(plot_filename)).mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_filename, format='png', bbox_inches='tight', dpi=100)
    plt.close('all')


def create_output_file(out_file_tmp, var_name, composite_year, QC_option, lat, lon, MODIS_tiles_used):
    """Create the yearly NAm output file for var_name (plus QA_flag) with all coordinates and metadata written.
    Returns the open netCDF4 Dataset and its data and QA_flag variables, to be filled one composite at a time."""
    n_lat, n_lon = len(lat), len(lon)
    nc = netCDF4.Dataset(out_file_tmp, 'w', format='NETCDF4')
    nc.createDimension('time', n_composites)
    nc.createDimension('lat', n_lat)
    nc.createDimension('lon', n_lon)
    nc.createDimension('nv', 2)

    # Composite NN covers DOY_composite_starts[NN] up to the next composite start; the last one ends at
    # the end of the year (step 01 composites daily files within a calendar year)
    days_in_year = (dt(composite_year + 1, 1, 1) - dt(composite_year, 1, 1)).days
    time_bnds = np.column_stack([DOY_composite_starts - 1, np.append(DOY_composite_starts[1:] - 1, days_in_year)])

    time_var = nc.createVariable('time', 'i4', ('time',))
    time_var.units = f'days since {composite_year}-01-01 00:00:00'
    time_var.calendar = 'proleptic_gregorian'
    time_var.standard_name = 'time'
    time_var.long_name = 'start date of 16-day composite'
    time_var.axis = 'T'
    time_var.bounds = 'time_bnds'
    time_var[:] = DOY_composite_starts - 1

    time_bnds_var = nc.createVariable('time_bnds', 'i4', ('time', 'nv'))
    time_bnds_var[:] = time_bnds

    composite_var = nc.createVariable('composite', 'i2', ('time',))
    composite_var.long_name = 'composite number within the year (0-22), as in the step 11 filenames'
    composite_var[:] = np.arange(n_composites)

    lat_var = nc.createVariable('lat', 'f8', ('lat',))
    lat_var.units = 'degrees_north'
    lat_var.standard_name = 'latitude'
    lat_var.axis = 'Y'
    lat_var[:] = lat

    lon_var = nc.createVariable('lon', 'f8', ('lon',))
    lon_var.units = 'degrees_east'
    lon_var.standard_name = 'longitude'
    lon_var.axis = 'X'
    lon_var[:] = lon

    crs_var = nc.createVariable('crs', 'i4')
    crs_var.grid_mapping_name = 'latitude_longitude'
    crs_var.semi_major_axis = 6378137.0
    crs_var.inverse_flattening = 298.257223563
    crs_var.crs_wkt = CRS.from_epsg(4326).to_wkt()
    crs_var.epsg_code = 'EPSG:4326'

    chunksizes = (1, min(chunk_lat, n_lat), min(chunk_lon, n_lon))
    dat_var = nc.createVariable(var_name, out_dtype, ('time', 'lat', 'lon'), zlib=True, complevel=compress_level, shuffle=True,
                                chunksizes=chunksizes, fill_value=np.float32(np.nan))
    dat_var.long_name = var_long_names[var_name]
    dat_var.units = '1'
    dat_var.grid_mapping = 'crs'
    dat_var.coordinates = 'composite'

    QA_var = nc.createVariable(QA_var_name, 'i1', ('time', 'lat', 'lon'), zlib=True, complevel=compress_level, shuffle=True,
                               chunksizes=chunksizes, fill_value=QA_fill_value)
    QA_var.long_name = 'VI usefulness flag'
    QA_var.flag_values = QA_flag_values
    QA_var.flag_meanings = QA_flag_meanings
    QA_var.comment = ('0 = no data; 1 = highest quality, original data; '
                      '2 = gap-filled value (originally passed QC, but removed by the outlier filter); '
                      '3 = gap-filled value (originally no data). Regridded to 0.01 degree by nearest neighbour')  # ASCII only, so it is stored as a char attribute
    QA_var.grid_mapping = 'crs'
    QA_var.coordinates = 'composite'

    nc.Conventions = 'CF-1.8'
    nc.title = f'MODIS MCD19A1.061 MAIAC {var_name} - 16-day composite using Constrained View angle - Maximum Value Composite (CV-MVC) algorithm - {composite_year}'
    nc.Comment1 = 'Original files downloaded from https://e4ftl01.cr.usgs.gov/MOTA/MCD19A1.061/'
    nc.Comment2 = 'hdf files converted to netcdf4 using HDF-EOS conversion tool (https://www.hdfeos.org/software/h4toh5.php)'
    nc.Comment3 = 'gap filled using historical fill scaled with ecoregion-specific regression of elevation & fractional tree cover, crop cover and other vegetation cover'
    nc.Regridding = 'Regridded from MODIS sinusoidal 1km grid to 0.01 degree rectilinear grid using xESMF bilinear regridder (QA_flag by nearest neighbour), per tile'
    nc.Time_descr = 'time is the start date of each 16-day composite (DOY 1, 17, ..., 353)'
    nc.QC_option = QC_option
    nc.MODIS_tiles = ', '.join(MODIS_tiles_used)
    nc.Spatial_extent_descr = 'North America'
    nc.references = 'Lyapustin, A., Wang, Y. (2022). MODIS/Terra+Aqua Land Surface BRF Daily L2G Global 500m and 1km SIN Grid V061 [Data set]. NASA EOSDIS Land Processes Distributed Active Archive Center. Accessed from https://doi.org/10.5067/MODIS/MCD19A1.061'
    nc.Processed_by = 'Lewis Kunik, lewis.kunik@utah.edu'
    nc.date_created = str(dt.today().date())

    return nc, dat_var, QA_var


#########################################
# Begin main
#########################################
#%%

def main():

    try:
        composite_year = int(sys.argv[1])  # e.g. 2020
        QC_option = sys.argv[2]  # e.g. 'CloudFree_LowAOD_ClearAdj'

        dat_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/CV-MVC/01d_rect/QCfilt_{QC_option}'
        plot_dir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/maps/MAIAC_CV-MVC/01d_rect/NAm_full/QCfilt_{QC_option}'

        # one output file per variable; only make the ones that don't exist yet
        out_files = {var_name: os.path.join(dat_basedir, 'NAm_full', var_name, f'MCD19A1.061_01d_16day_CV-MVC_NAm_{var_name}_{composite_year}.nc')
                     for var_name in var_long_names}
        out_vars = []
        for var_name, out_file in out_files.items():
            if os.path.exists(out_file):
                print(f'Output file already exists: {out_file}, skipping {var_name}')
            else:
                out_vars.append(var_name)
        if len(out_vars) == 0:
            print('All output files already exist, exiting...')
            sys.exit(0)

        # List the available tile files for each composite of this year
        files_by_composite = {}
        for composite_nn in range(n_composites):
            files_by_composite[composite_nn] = [(MODIS_tile, tile_file(dat_basedir, MODIS_tile, QC_option, composite_year, composite_nn))
                                                for MODIS_tile in MODIS_tiles
                                                if os.path.exists(tile_file(dat_basedir, MODIS_tile, QC_option, composite_year, composite_nn))]
        all_files = [f for files in files_by_composite.values() for _, f in files]
        if len(all_files) == 0:
            print(f'No tile files found for {composite_year} in {dat_basedir}, exiting with error code...')
            sys.exit(1)

        for MODIS_tile in MODIS_tiles:
            n_tile = sum(MODIS_tile == t for files in files_by_composite.values() for t, _ in files)
            if n_tile < n_composites:
                print(f'tile {MODIS_tile}: {n_tile} of {n_composites} composites found for {composite_year}')

        # Establish the NAm grid from the union of the tile extents. Extent comes from one file per tile,
        # since every composite of a tile has the same grid
        first_file_per_tile = {}
        for files in files_by_composite.values():
            for MODIS_tile, f in files:
                first_file_per_tile.setdefault(MODIS_tile, f)
        lon_min, lon_max, lat_min, lat_max = get_NAm_extent(first_file_per_tile.values())
        n_lon = int(round((lon_max - lon_min) / grid_res)) + 1
        n_lat = int(round((lat_max - lat_min) / grid_res)) + 1
        lon = np.round(lon_min + np.arange(n_lon) * grid_res, grid_decimals)
        lat = np.round(lat_min + np.arange(n_lat) * grid_res, grid_decimals)
        print(f'NAm grid: lon {lon_min} to {lon_max} ({n_lon}), lat {lat_min} to {lat_max} ({n_lat})', flush=True)

        # Create the output files and write each composite to them as it is mosaicked, so only one
        # composite of the NAm grid is held in memory at a time
        out_files_tmp, nc_out, dat_vars, QA_vars = {}, {}, {}, {}
        for var_name in out_vars:
            Path(os.path.dirname(out_files[var_name])).mkdir(parents=True, exist_ok=True)
            out_files_tmp[var_name] = out_files[var_name] + '.tmp'  # renamed when complete, so a failed run does not leave a partial file that looks finished
            nc_out[var_name], dat_vars[var_name], QA_vars[var_name] = create_output_file(out_files_tmp[var_name], var_name, composite_year, QC_option,
                                                                                         lat, lon, first_file_per_tile.keys())

        t_start_all = time.time()
        for composite_nn in range(n_composites):
            t_start = time.time()
            composite_date = composite_start_date(composite_year, composite_nn)
            mosaics = {var_name: np.full((n_lat, n_lon), np.nan, dtype=np.float32) for var_name in out_vars}
            QA_mosaic = np.full((n_lat, n_lon), QA_fill_value, dtype=np.int8)
            n_conflict = {var_name: 0 for var_name in out_vars}
            n_QA_conflict = 0

            for MODIS_tile, f in files_by_composite[composite_nn]:
                with xr.open_dataset(f) as ds:
                    file_date = ds.time.values.astype('datetime64[D]')
                    if file_date != np.datetime64(composite_date.date()):
                        print(f'WARNING: {os.path.basename(f)} time {file_date} does not match composite start date {composite_date.date()}')
                    ds = ds[out_vars + [QA_var_name]].transpose(..., 'lat', 'lon').sortby(['lat', 'lon'])
                    i_lat = grid_index(ds.lat.values, lat_min)
                    i_lon = grid_index(ds.lon.values, lon_min)
                    tile_dats = {var_name: ds[var_name].values.astype(np.float32) for var_name in out_vars}
                    tile_QA = read_tile_QA_flag(ds, f)

                lat_slice = slice(i_lat[0], i_lat[-1] + 1)
                lon_slice = slice(i_lon[0], i_lon[-1] + 1)
                for var_name in out_vars:
                    mosaic_sub = mosaics[var_name][lat_slice, lon_slice]  # view into mosaic
                    tile_dat = tile_dats[var_name]
                    tile_valid = ~np.isnan(tile_dat)
                    n_conflict[var_name] += int(np.sum(tile_valid & ~np.isnan(mosaic_sub)))
                    fill = tile_valid & np.isnan(mosaic_sub)
                    mosaic_sub[fill] = tile_dat[fill]

                QA_mosaic_sub = QA_mosaic[lat_slice, lon_slice]  # view into QA_mosaic
                n_QA_conflict += int(np.sum((tile_QA > 0) & (QA_mosaic_sub > 0)))
                QA_fill = (tile_QA != QA_fill_value) & ((QA_mosaic_sub == QA_fill_value) | ((QA_mosaic_sub == 0) & (tile_QA > 0)))
                QA_mosaic_sub[QA_fill] = tile_QA[QA_fill]

            for var_name in out_vars:
                dat_vars[var_name][composite_nn, :, :] = mosaics[var_name]
                QA_vars[var_name][composite_nn, :, :] = QA_mosaic
            valid_counts = ', '.join(f'{var_name} {int(np.sum(~np.isnan(mosaics[var_name])))}' for var_name in out_vars)
            print(f'composite {composite_nn:02d} ({composite_date.date()}): {len(files_by_composite[composite_nn])} tiles, '
                  f'valid cells: {valid_counts}, done in {format_elapsed(t_start)}', flush=True)
            for var_name in out_vars:
                if n_conflict[var_name] > 0:
                    print(f'WARNING: composite {composite_nn:02d} has {n_conflict[var_name]} cells with valid {var_name} in more than one tile (kept the first tile)')
            if n_QA_conflict > 0:
                print(f'WARNING: composite {composite_nn:02d} has {n_QA_conflict} cells with {QA_var_name} > 0 in more than one tile (kept the first tile)')

            if composite_nn == plot_composite:
                for var_name in out_vars:
                    plot_file = os.path.join(plot_dir, f'MCD19A1_NAm_01d_rect_{var_name}_{composite_year}-{composite_nn:02d}.png')
                    plot_quicklook(mosaics[var_name], lon, lat, var_name, plot_title=f'{var_name} NAm mosaic, {composite_year}-{composite_nn:02d} ({composite_date.date()})',
                                   plot_filename=plot_file)

            del mosaics, QA_mosaic
            gc.collect()

        for var_name in out_vars:
            nc_out[var_name].close()
            os.replace(out_files_tmp[var_name], out_files[var_name])
            print(f'Saved {out_files[var_name]} ({os.path.getsize(out_files[var_name]) / 1e9:.2f} GB)')
        print(f'All done in {format_elapsed(t_start_all)}')

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

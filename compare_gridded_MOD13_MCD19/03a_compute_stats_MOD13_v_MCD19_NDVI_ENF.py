#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Step 1 of 2 for the MCD19-vs-MOD13 2D histogram figure (ESSD manuscript, Sec 3.2 /
# planned Fig. 3.2): load the paired dataset, filter, bin it into a 2D histogram, and
# compute OLS regression/agreement statistics - all the slow, data-volume-dependent work.
# Nothing is plotted here. Output is a small .npz summary file that the companion plotting
# script (08b_plot_mcd19_mod13_histogram.py) loads to iterate on plot aesthetics quickly,
# without re-reading the Parquet dataset or re-binning millions of points each time.
#
# Decisions carried over from prior conversation with Lewis (2026-09-03), unchanged by this
# split - see 08b_plot_mcd19_mod13_histogram.py header for the plotting-side decisions:
#   - Axis convention: MOD13 NDVI on x (reference/established product), MCD19 NDVI on y
#     (product being evaluated). Slope/intercept describe how MCD19 deviates from MOD13.
#   - Bias = mean signed difference = mean(MCD19 - MOD13) across all paired pixels
#     (NOT the regression intercept - those are reported separately).
#   - All years (2000-2025) are pooled into one comparison per land-cover class.
#   - Regression is ordinary least squares (simple linear regression treating MOD13/x as
#     the independent variable) - see prior note re: Deming/RMA as a possible substitute.
#   - RMSE and MAE are computed relative to the 1:1 line (i.e. agreement with MOD13 as
#     reference), which is distinct from the regression fit's own residuals.
#   - A pixel-pair is dropped if EITHER product's NDVI is below NDVI_MIN (default 0.1) -
#     intended to exclude snow/non-vegetated low-NDVI contamination from both sides.
#

#########################################
# Load packages
#########################################
#%%
import os
from pathlib import Path

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from scipy import stats as spstats

import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.compute as pc

#%%
#########################################
# Config
#########################################

OUTPUT_ROOT = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/MCD19_MOD13_comparison'
PAIRED_PARQUET_DIR = os.path.join(OUTPUT_ROOT, 'mcd19_mod13_ndvi_paired')

PAIRED_PARTITIONING = pads.partitioning(
    pa.schema([("land_cover", pa.string())]),
    flavor="hive",
)

# Select land cover class to summarize for this run (matches the labels used in
# 07a_extract_MCD19_ndvi_by_landcover.py: 'ENF', 'DBF_MixedForest', 'Grassland', 'Shrubland')
land_cover = 'ENF'

# Minimum NDVI threshold applied to BOTH products (pairs where either MCD19 or MOD13 NDVI
# falls below this are dropped) - primarily to exclude snow/non-vegetated low-NDVI
# contamination, consistent with the site-level NDVI thresholding already used for PhotoSpec
# filtering elsewhere in this project. Set to None to disable.
NDVI_MIN = 0.1

N_BINS = 200  # 2D histogram resolution along each axis

SUMMARY_DIR = os.path.join(OUTPUT_ROOT, 'histogram_summaries')
Path(SUMMARY_DIR).mkdir(parents=True, exist_ok=True)

#########################################
# Helper functions
#########################################


def load_paired_data(land_cover):
    """Read the paired MCD19/MOD13 dataset for one land-cover class (all years pooled)."""
    dataset = pads.dataset(PAIRED_PARQUET_DIR, format='parquet', partitioning=PAIRED_PARTITIONING)
    filt = pc.field('land_cover') == pa.scalar(land_cover, type=pa.string())
    table = dataset.to_table(filter=filt)
    df = table.to_pandas()
    return df


def compute_regression_stats(x, y):
    """
    OLS linear regression of y (MCD19) on x (MOD13), plus agreement metrics relative
    to the 1:1 line. x and y must be 1D arrays of equal length with no NaNs.
    """
    result = spstats.linregress(x, y)
    slope, intercept, rvalue, pvalue, stderr = result[:5]

    diff = y - x  # MCD19 - MOD13
    bias = np.mean(diff)          # mean signed difference (Lewis's chosen bias definition)
    rmse = np.sqrt(np.mean(diff**2))  # relative to 1:1 line
    mae = np.mean(np.abs(diff))       # relative to 1:1 line

    return {
        'n': x.size,
        'slope': slope,
        'intercept': intercept,
        'r2': rvalue**2,
        'p_value': pvalue,
        'slope_stderr': stderr,
        'bias': bias,
        'rmse': rmse,
        'mae': mae,
    }


#########################################
# Begin Main
#########################################
#%%

print(f"Loading paired MCD19/MOD13 dataset for land_cover = {land_cover}...")
df = load_paired_data(land_cover)
print(f"  Loaded {len(df)} paired records.")

df = df.dropna(subset=['mcd19_ndvi', 'mod13_ndvi'])
print(f"  {len(df)} records remain after dropping any NaNs.")

if NDVI_MIN is not None:
    n_before = len(df)
    df = df[(df['mcd19_ndvi'] >= NDVI_MIN) & (df['mod13_ndvi'] >= NDVI_MIN)]
    print(f"  Dropped {n_before - len(df)} records with MCD19 or MOD13 NDVI < {NDVI_MIN}; {len(df)} remain.")

x = df['mod13_ndvi'].to_numpy()
y = df['mcd19_ndvi'].to_numpy()

#%%
print("Computing regression statistics...")
reg_stats = compute_regression_stats(x, y)
for k, v in reg_stats.items():
    print(f"  {k}: {v}")

#%%
print("Binning into 2D histogram...")
lo = np.floor(min(np.nanmin(x), np.nanmin(y)) * 20) / 20
hi = np.ceil(max(np.nanmax(x), np.nanmax(y)) * 20) / 20
axis_range = [[lo, hi], [lo, hi]]

H, xedges, yedges = np.histogram2d(x, y, bins=N_BINS, range=axis_range)
print(f"  Histogram shape: {H.shape}, total binned count: {H.sum():,.0f}")

#%%
print("Saving summary (histogram + stats) to disk...")
summary_file = os.path.join(SUMMARY_DIR, f'{land_cover}_ndvi_hist2d_summary.npz')

np.savez(
    summary_file,
    H=H,
    xedges=xedges,
    yedges=yedges,
    lo=lo,
    hi=hi,
    land_cover=np.array(land_cover),
    n_bins=N_BINS,
    ndvi_min=np.nan if NDVI_MIN is None else NDVI_MIN,
    n=reg_stats['n'],
    slope=reg_stats['slope'],
    intercept=reg_stats['intercept'],
    r2=reg_stats['r2'],
    p_value=reg_stats['p_value'],
    slope_stderr=reg_stats['slope_stderr'],
    bias=reg_stats['bias'],
    rmse=reg_stats['rmse'],
    mae=reg_stats['mae'],
)

print(f"Saved summary to: {summary_file}")
print("Done. Run 08b_plot_mcd19_mod13_histogram.py to plot this summary.")

# %%
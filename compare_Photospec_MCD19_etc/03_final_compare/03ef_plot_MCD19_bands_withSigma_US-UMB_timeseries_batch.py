#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Plots MCD19 composite surface reflectance at a ground-truth site:
#   Band1 panel: Sur_refl1 +/- Sigma_BRFn1 (shaded), plus Sur_refl11 on a
#                second y-axis (dark green, no shading)
#   Band2 panel: Sur_refl2 +/- Sigma_BRFn2 (shaded)
# Saves <panel_dir>/<site_name>_Band1_sigma_panel.png and _Band2_sigma_panel.png
# for assembly by 03_plot_final_bands_panels_withSigma.py.
#
# The ground-based (PhotoSpec) record is only used to set the time window and
# x-limits, so the panels line up with the 02_ NDVI/CCI versions.

#%%
import os
import numpy as np
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from matplotlib.ticker import FuncFormatter as FFmt

#########################################
# Site settings
#########################################

site_name = 'US-UMB'
plot_title = 'US-UMB'
panel_letters = ('e', 'f')          # (Band1, Band2)
figsize = (7, 3)
month_ticks = [1, 4, 7, 10]
NDVI_MIN = 0.6                      # PhotoSpec NDVI floor used to trim the time window (None = no filter)

winter_periods = [
    # Ground snow cover periods
    (pd.Timestamp('2017-11-04'), pd.Timestamp('2018-05-01')),
    (pd.Timestamp('2018-11-09'), pd.Timestamp('2019-04-23')),
    (pd.Timestamp('2019-11-06'), pd.Timestamp('2020-04-24')),
]

#########################################
# Filepaths
#########################################

QC_descr = "CloudFree_LowAOD_ClearAdj"

dat_pt_basedir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/QC_PhotoSpec_sites_backup/'
dat_pt_basedir_QC = os.path.join(dat_pt_basedir, f'CV-MVC/lm-fill/QCfilt_{QC_descr}')
MCD19_QC_file = os.path.join(dat_pt_basedir_QC, f'MCD19_QCfilt_{QC_descr}_{site_name}.nc')

Photospec_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/PhotoSpec/US-UMB/'
Photospec_2018_file = os.path.join(Photospec_dir, 'PhotoSpecM1_2018_v20260903.nc')
Photospec_2019_file = os.path.join(Photospec_dir, 'PhotoSpecM1_2019_v20260903.nc')

# MCD19 mean overpass time at US-UMB is 12:21 local mean solar time (LMST) = 13:00 EST (PhotoSpec clock).
# PhotoSpec data are 90-min averages (Hour_of_Day = interval start); HOD index 8 (12:00-13:30 EST)
# is the period that best matches overpass +/- 1 hr
Photospec_HOD_idx = 8
Photospec_HOD_start = pd.Timedelta(hours=12)
overpass_window = ('12:00', '13:30')
overpass_window_times = tuple(pd.Timestamp(t).time() for t in overpass_window)

plot_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/CVMVC_QC_regression_stats/final/'
panel_dir = os.path.join(plot_dir, 'panels')
os.makedirs(panel_dir, exist_ok=True)

#########################################
# Plot constants
#########################################

MCD19_QC_color = '#BF2665'
BAND11_color = '#1B5E20'   # dark green

# Multiply reflectance and sigma by this. Leave at 1.0 if the point file already
# holds physical reflectance (0-1); set to 0.0001 if it holds raw MAIAC int16 values.
REFL_SCALE = 1.0

#########################################
# Functions
#########################################

def load_reference_times():
    """Return (all times, overpass-window times) of the ground-based record."""
    dfs = []
    for year, f in [(2018, Photospec_2018_file), (2019, Photospec_2019_file)]:
        df = xr.open_dataset(f).sel(HOD=Photospec_HOD_idx).to_dataframe().reset_index()
        df['Time'] = (
            pd.Timestamp(f'{year}-01-01')
            + pd.to_timedelta(df['DOY'].astype(int), unit='D')
            + Photospec_HOD_start
        )
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    if NDVI_MIN is not None:
        df = df[df['NDVI'] >= NDVI_MIN].reset_index(drop=True)
    times = pd.to_datetime(df['Time'])
    mask = times.dt.time.between(*overpass_window_times, inclusive='left')
    return times, times[mask].reset_index(drop=True)


def match_axis_extent(ylim1, ref1, ref2):
    """
    Second-axis limits that place ref2's min/max at the same vertical positions
    that ref1's min/max occupy on the first axis. So ref2 gets the same relative
    padding that the uncertainty shading adds around ref1.
    """
    y1_lo, y1_hi = ylim1
    span1 = y1_hi - y1_lo
    r1_min, r1_max = np.nanmin(ref1), np.nanmax(ref1)
    r2_min, r2_max = np.nanmin(ref2), np.nanmax(ref2)
    f_lo = (r1_min - y1_lo) / span1
    f_hi = (r1_max - y1_lo) / span1
    if not (r2_max > r2_min) or not (f_hi > f_lo):
        pad = 0.05 * max(abs(r2_max), 1e-3)
        return r2_min - pad, r2_max + pad
    span2 = (r2_max - r2_min) / (f_hi - f_lo)
    lo2 = r2_min - f_lo * span2
    return lo2, lo2 + span2


def plot_band_panel(times, refl, sigma, ylabel, title, xlim, outfile,
                    refl2=None, refl2_label=None):
    lo = refl - sigma
    hi = refl + sigma

    fig, ax = plt.subplots(figsize=figsize)

    for start, end in winter_periods:
        ax.axvspan(start, end, color='lightblue', alpha=0.5, zorder=0)

    ax.fill_between(times, lo, hi, color=MCD19_QC_color, alpha=0.25, linewidth=0, zorder=1.5)
    ax.plot(times, refl, color=MCD19_QC_color, marker='o', markersize=8, linewidth=3.5,
            alpha=1, markeredgecolor='black', markeredgewidth=0.5, zorder=3)

    y_min = np.nanmin(lo)
    y_max = np.nanmax(hi)
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    ax.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.2f}'))

    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(MonthLocator(bymonth=month_ticks))
    ax.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))
    ax.set_title(title, loc='center')
    ax.set_ylabel(ylabel)

    if refl2 is not None:
        ax2 = ax.twinx()
        ax2.plot(times, refl2, color=BAND11_color, marker='o', markersize=6, linewidth=2.5,
                 alpha=1, markeredgecolor='black', markeredgewidth=0.5)
        ax2.set_ylim(*match_axis_extent(ax.get_ylim(), refl, refl2))
        ax2.set_ylabel(refl2_label, color=BAND11_color)
        ax2.tick_params(axis='y', colors=BAND11_color)
        ax2.yaxis.set_major_formatter(FFmt(lambda y, _: f'{y:.2f}'))

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches='tight')

#########################################
# Main
#########################################
#%%
def main():

    MCD19_QC_pt = xr.open_dataset(MCD19_QC_file)

    ref_times, ref_times_filtered = load_reference_times()

    MCD19_QC_pt = MCD19_QC_pt.sel(time=slice(
        ref_times.iloc[0] - pd.Timedelta(days=16),
        ref_times.iloc[-1] + pd.Timedelta(days=16),
    ))
    MCD19_QC_times = np.array([np.datetime64(t) for t in MCD19_QC_pt['time'].values])

    b1 = MCD19_QC_pt['Sur_refl1'].values * REFL_SCALE
    b2 = MCD19_QC_pt['Sur_refl2'].values * REFL_SCALE
    b11 = MCD19_QC_pt['Sur_refl11'].values * REFL_SCALE
    sig1 = MCD19_QC_pt['Sigma_BRFn1'].values * REFL_SCALE
    sig2 = MCD19_QC_pt['Sigma_BRFn2'].values * REFL_SCALE

    # Quick check that values are in physical reflectance units
    for name, arr in [('Sur_refl1', b1), ('Sur_refl2', b2), ('Sur_refl11', b11),
                      ('Sigma_BRFn1', sig1), ('Sigma_BRFn2', sig2)]:
        print(f'{site_name} {name}: min={np.nanmin(arr):.4f}, max={np.nanmax(arr):.4f}')

    xlim = (ref_times_filtered.min() - pd.Timedelta(days=10),
            ref_times_filtered.max() + pd.Timedelta(days=10))

    plt.rcParams.update({
        'axes.labelsize': 18,
        'axes.titlesize': 21,
        'xtick.labelsize': 17,
        'ytick.labelsize': 17,
        'legend.fontsize': 13
    })

    plot_band_panel(MCD19_QC_times, b1, sig1, 'Band 1 BRF',
                    f'{panel_letters[0]})  {plot_title}', xlim,
                    os.path.join(panel_dir, f'{site_name}_Band1_sigma_panel.png'),
                    refl2=b11, refl2_label='Band 11 BRF')

    plot_band_panel(MCD19_QC_times, b2, sig2, 'Band 2 BRF',
                    f'{panel_letters[1]})  {plot_title}', xlim,
                    os.path.join(panel_dir, f'{site_name}_Band2_sigma_panel.png'))


if __name__ == "__main__":
    main()

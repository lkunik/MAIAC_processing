#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Multi-panel (3 x 2) version of 06c_plot_MCD19_seasonality_by_ecoregion.py:
# plots CCI and NDVI seasonality for six ecoregions in a single figure.
#

#########################################
# Load packages
#########################################
#%%
# File system packages
import os  # operating system library
from pathlib import Path
import sys

# runtime packages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# numerical/data management packages
import numpy as np

# time and date packages
from datetime import datetime as dt  # date time library
from datetime import timedelta

# plotting packages
import pickle
from matplotlib import pyplot as plt  # primary plotting module
from matplotlib.ticker import MultipleLocator

#%%
#########################################
# Define Global Filepaths
#########################################

mask_LC_high = True
mask_LC_low = False

# directories
QC_option = 'CloudFree_LowAOD_ClearAdj'

dat_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/data/summary_data'
plot_basedir = f'/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/seasonality/QCfilt_{QC_option}/'
Path(plot_basedir).mkdir(parents=True, exist_ok=True)


#########################################
# Define Global Variables and constants
#########################################
#%%

# panels in row-major order: (ecoregion file key, panel title, CCI tick spacing, NDVI tick spacing)
panels = [
    ('Boreal',                 'Boreal Forests (ENF)',        0.1,  0.1),
    ('Western',                'Western Cordillera (ENF)',    0.1,  0.1),
    ('GREAT_PLAINS',           'Great Plains (Grassland)',    0.1,  0.1),
    ('Eastern',                'Eastern Forests (Mixed/DBF)', 0.1,  0.1),
    ('NORTH_AMERICAN_DESERTS', 'Deserts (Shrubland)',         0.05, 0.1),
    ('Southwest_US_Mexico',    'Temperate Sierras (ENF)',     0.05, 0.1),
]
panel_letters = 'abcdefghijklmnopqrstuvwxyz'

DOY_composite_starts = np.arange(1, 366, 16)
composite_dates_2020 = [dt(2020, 1, 1) + timedelta(days=int(DOY) - 1) for DOY in DOY_composite_starts]

#plotting vars
months_begin = np.array([1, 32, 61, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365])
months_doy = (months_begin[1:] + months_begin[:-1]) / 2
months_labs = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

CCI_color = "#A1993D"
NDVI_color = "#006B30"

plt.rcParams.update({
'axes.labelsize': 16,
'axes.titlesize': 17,
'xtick.labelsize': 15,
'ytick.labelsize': 15,
'legend.fontsize': 12,
})


#########################################
# Define Global Functions
#########################################

def load_ecoregion_dict(ecoregion):
    if mask_LC_high:
        input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_seasonality_dict_LCmask_only.pkl')
    elif mask_LC_low:
        input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_seasonality_dict_LCmask_low.pkl')
    else:
        input_file = os.path.join(dat_dir, f'{ecoregion.replace(" ", "_")}_seasonality_dict.pkl')

    with open(input_file, 'rb') as f:
        ecoregion_dat_dict = pickle.load(f)

    print(f'Loaded ecoregion_seasonality_dict from {input_file}')
    return ecoregion_dat_dict


def plot_seasonality_panel(ax, ecoregion_dat_dict, title, CCI_tick, NDVI_tick):
    ax.grid(color = "gray", linestyle = "--", linewidth = 0.4, alpha = 0.4) # add grid lines

    CCI_mean = np.array(ecoregion_dat_dict['CCI_mean'])
    CCI_std = np.array(ecoregion_dat_dict['CCI_std'])
    NDVI_mean = np.array(ecoregion_dat_dict['NDVI_mean'])
    NDVI_std = np.array(ecoregion_dat_dict['NDVI_std'])

    # Find first and last non-nan indices and shade out the missing-data periods
    non_nan_indices = np.where(~np.isnan(CCI_mean))[0]
    if len(non_nan_indices) > 0:
        first_non_nan_idx = non_nan_indices[0]
        last_non_nan_idx = non_nan_indices[-1]
        if first_non_nan_idx > 0:
            ax.axvspan(-10, DOY_composite_starts[first_non_nan_idx], color='gray', alpha=0.2, hatch='///')
        if last_non_nan_idx < len(DOY_composite_starts) - 1:
            ax.axvspan(DOY_composite_starts[last_non_nan_idx], 380, color='gray', alpha=0.2, hatch='///')

    ax2 = ax.twinx()
    ax2.plot(DOY_composite_starts, NDVI_mean, color=NDVI_color, lw=2)
    ax2.plot(DOY_composite_starts, NDVI_mean - NDVI_std, color=NDVI_color, linestyle='--', lw=1.5)
    ax2.plot(DOY_composite_starts, NDVI_mean + NDVI_std, color=NDVI_color, linestyle='--', lw=1.5)

    ax.plot(DOY_composite_starts, CCI_mean, color=CCI_color, lw = 2.5)
    ax.plot(DOY_composite_starts, CCI_mean - CCI_std, color=CCI_color, linestyle='--', lw=1.5)
    ax.plot(DOY_composite_starts, CCI_mean + CCI_std, color=CCI_color, linestyle='--', lw=1.5)

    # Plot shaded region for mean ± representative std
    ax.fill_between(DOY_composite_starts, CCI_mean - CCI_std, CCI_mean + CCI_std,
                    color=CCI_color, alpha=0.35, label='±1 std')

    # set xticks to Month abbreviations
    ax.set_xticks(ticks=months_doy)
    ax.set_xticklabels(months_labs)
    ax.set_xlim(0, 353)

    ax.yaxis.set_major_locator(MultipleLocator(CCI_tick))
    ax2.yaxis.set_major_locator(MultipleLocator(NDVI_tick))

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y)))

    ax.yaxis.label.set_color(CCI_color)
    ax.tick_params(axis='y', colors=CCI_color)
    ax.spines['left'].set_color(CCI_color)

    ax2.yaxis.label.set_color(NDVI_color)
    ax2.tick_params(axis='y', colors=NDVI_color)
    ax2.spines['right'].set_color(NDVI_color)
    ax2.spines['left'].set_color(CCI_color)

    ax.set_ylabel('CCI')
    ax2.set_ylabel('NDVI')
    ax.set_title(title)

    return ax2


#########################################
# Begin Main
#########################################

#%%
fig, axes = plt.subplots(3, 2, figsize=(12, 10.5))

for i, (ax, (ecoregion, title, CCI_tick, NDVI_tick)) in enumerate(zip(axes.flat, panels)):
    ecoregion_dat_dict = load_ecoregion_dict(ecoregion)
    plot_seasonality_panel(ax, ecoregion_dat_dict, f'{panel_letters[i]}) {title}', CCI_tick, NDVI_tick)

fig.tight_layout(h_pad=1.5, w_pad=2.0)

if mask_LC_high:
    plot_filename = 'ecoregions_seasonality_multipanel_LCmask.png'
elif mask_LC_low:
    plot_filename = 'ecoregions_seasonality_multipanel_LCmask_low.png'
else:
    plot_filename = 'ecoregions_seasonality_multipanel.png'

plt.savefig(os.path.join(plot_basedir, plot_filename), dpi=300, bbox_inches='tight')
print(f'Saved {os.path.join(plot_basedir, plot_filename)}')
# plt.close()
plt.show()

# %%

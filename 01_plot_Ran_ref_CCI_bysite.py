#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Lewis Kunik - University of Utah
# Contact: lewis.kunik@utah.edu
#
# Created on Tue Oct 24 2023
#
#

#%%


# File system packages
import os  # operating system library
import sys
import glob
from pathlib import Path
import copy
import gc
import fnmatch

# Custom packages
utils_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/pyutil'
if utils_dir not in sys.path:
   sys.path.append(utils_dir)
import data_utils
import plot_utils

# runtime packages
import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

# numerical/data management packages
import math
import numpy as np
import xarray as xr  # for multi dimensional data
import pandas as pd
import xesmf as xe
# import random
# import h5py

# Statistical packages
# from scipy.stats import gaussian_kde
# from scipy.stats import linregress
import scipy.io

# shapefile/geospatial packages
import geopandas as gpd
from shapely.geometry import mapping

# time and date packages
import time
from datetime import datetime as dt  # date time library
from datetime import timedelta


# plotting packages
from matplotlib import pyplot as plt  # primary plotting module
from matplotlib.ticker import FuncFormatter as FFmt  # need this for formatting plot ticks
from matplotlib.lines import Line2D # for custom legends
from matplotlib.colors import to_hex
from matplotlib.cm import inferno

import cartopy.crs as ccrs  # cartopy is the best spatial plotting library i know
from cartopy.io import img_tiles  # cartopy's implementation of webtiles
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature



#########################################
# Define Global Filepaths
#########################################
#%%
dat_dir = '/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/from_Ran_by_fluxsite'

plot_dir = '/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/plots/Ran_reference'

#########################################
# Define Global Variables and constants
#########################################

#########################################
# Define Global functions
#########################################

def floor(value, decimal_digits):
  scale = (10^decimal_digits)
  return  math.floor(value * scale)/scale

def ceil(value, decimal_digits):
  scale = (10^decimal_digits)
  return  math.ceil(value * scale)/scale

#def


#########################################
# Initialize map data
#########################################



#%%

#########################################
# Begin main
#########################################

files = sorted(os.listdir(dat_dir))

file = files[0]
# for file in files:
    file_path = os.path.join(dat_dir, file)
    df = pd.read_csv(file_path)
# %%

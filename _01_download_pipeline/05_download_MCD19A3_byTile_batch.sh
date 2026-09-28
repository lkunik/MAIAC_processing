#!/bin/bash

# Directory to save downloaded files (there is currently some space on lingroup23 but must
# be careful to only download a few files at first to see how much space your files will take up)
DL_DIR="/scratch/general/nfs1/u6017162/MAIAC/pre_proc/hdf/"
mkdir -p $DL_DIR # create download dir if it doesn't exist

TILE_CODE=$1 # this is the tile of interest, must match the tile code in the files list
YYYY=$2 # this is the year of interest, must match the year in the files list
# This is the list of files to download from earlier steps 01a and 01b
FILES_LIST="/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A3/pre_proc/ind_years/${TILE_CODE}/MCD19A3_files_${TILE_CODE}_${YYYY}.txt"

# The DAAC Data Download script was directly retrieved from NASA Earthaccess hub - don't modify it
DAAC_DL_SCRIPT="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/include/LPDAAC-Data-Resources/python/scripts/daac_data_download_python/DAACDataDownload.py"

# The download script takes a destination directory as the -dir argument
# and a text file with a list of files to download as the -f argument
python $DAAC_DL_SCRIPT -dir $DL_DIR -f $FILES_LIST

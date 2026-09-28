#!/bin/bash

HDF_DIR="/scratch/general/nfs1/u6017162/MAIAC/pre_proc/hdf/"
NC_DIR="/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/pre_proc/raw_nc/"
CUR_DIR=$(pwd)

echo "changing dir to ${HDF_DIR}"
cd ${HDF_DIR}

# change the MODIS tile as needed
MODIS_TILE=$1

# Command line argument - The year of files to be converted
YYYY=$2

TEST_FILE="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/log/hdf2nc/${MODIS_TILE}/TEST_hdf2nc_${MODIS_TILE}_${YYYY}.txt"
# Get list of files to convert
HDF_FILES=$(ls MCD19A1.A${YYYY}*${MODIS_TILE}*.hdf)

FILE_COUNT=0
# Loop through all files
for HDF_FILE in ${HDF_FILES}
do
    # Trim off the ".hdf" file extension by removing last 4 chars
    HDF_NC=${HDF_FILE::-4}.nc

    if [ -f "${NC_DIR}${HDF_NC}" ]; then
        echo "File ${NC_DIR}${HDF_NC} already exists, skipping."  #>> ${TEST_FILE}
        if [ -f "${HDF_FILE}" ]; then
            rm ${HDF_FILE} # remove the original HDF file to save space
        fi
        continue
    fi
    FILE_COUNT=$((FILE_COUNT + 1))

    echo "creating file ${HDF_NC}" #>> ${TEST_FILE}

    #e.g. $/usr/local/bin/h4toh5 -eos -nc4strict AR_RnGd.hdf AR_RnGd.nc
    # echo "/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/Snowcover/data/bin/h4toh5 -eos -nc4strict ${HDF_FILE} ${HDF_NC}" >> ${TEST_FILE}
    # echo "" >> ${TEST_FILE}
    /uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/Snowcover/data/bin/h4toh5 -eos -nc4strict ${HDF_FILE} ${HDF_NC}
    # echo "mv ${HDF_NC} ${NC_DIR}" >> ${TEST_FILE}
    # echo "" >> ${TEST_FILE}
    mv ${HDF_NC} ${NC_DIR}
    rm ${HDF_FILE} # remove the original HDF file to save space
done

echo "Converted ${FILE_COUNT} files for year ${YYYY}"

echo "changing dir back to ${CUR_DIR}"
cd ${CUR_DIR}



#!/bin/bash

NC_DIR="/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/pre_proc/raw_nc/"
IND_DIR="/uufs/chpc.utah.edu/common/home/lin-group28/ltk/MODIS/MAIAC_MCD19A1/pre_proc/ind_years"


MODIS_TILES=(
# "h07v06" 
# "h08v03"
# "h08v04"
# "h08v05"
# "h08v06"
# "h08v07"
# "h09v02"
# "h09v03" 
# "h09v04" # US-NR1
# "h09v05" # ecoregions related to US-NR1
# "h09v06"
# "h09v07"
# "h09v08" 
# "h10v02"
# "h10v03"
# "h10v04" # US-Ne3
# "h10v05"
# "h10v06" # OSBS
# "h10v07"
# "h10v08"
# "h11v02" # DEJU
# "h11v03" # Ca-Obs
# "h11v04"
# "h11v05"
# "h11v06"
# "h11v07"
# "h11v08"
# "h12v01"
# "h12v02"
# "h12v03"
# "h12v04" # US-UMB
# "h12v05"
# "h13v01"
# "h13v02"
# "h13v03"
# "h13v04" 
# "h14v01"
# "h14v02"
# "h14v03"
# "h14v04"
# "h15v01"
# "h15v02"
# "h16v01"
)




# === Configure these ===
TILE="h12v04"
YEAR_START=2018
YEAR_END=2019
# =======================

echo "Checking for missing NC files in ${NC_DIR}"
echo "Tile: ${TILE} | Years: ${YEAR_START}-${YEAR_END}"
echo ""

TOTAL_MISSING=0

for YEAR in $(seq $YEAR_START $YEAR_END); do

    TXT_FILE="${IND_DIR}/${TILE}/MCD19A1_files_${TILE}_${YEAR}.txt"

    if [ ! -f "$TXT_FILE" ]; then
        echo "WARNING: index file not found for ${YEAR}: ${TXT_FILE}"
        continue
    fi

    # Extract DOYs available on NASA's end for this tile/year
    # Filename fragment looks like: MCD19A1.A2001066 -> extract chars at positions for year (5-8) and doy (9-11)
    AVAILABLE_DOYS=$(grep -oP '(?<=MCD19A1\.A\d{4})\d{3}' "$TXT_FILE" | sort -u)

    YEAR_MISSING=0
    while IFS= read -r DOY_PAD; do
        PATTERN="${NC_DIR}MCD19A1.A${YEAR}${DOY_PAD}.${TILE}.*.nc"
        if ! ls ${PATTERN} > /dev/null 2>&1; then
            echo "MISSING: ${YEAR} DOY ${DOY_PAD}"
            ((YEAR_MISSING++))
            ((TOTAL_MISSING++))
        fi
    done <<< "$AVAILABLE_DOYS"

    echo "  Tile: ${TILE} | Year ${YEAR}: ${YEAR_MISSING} missing"
    echo ""

done

echo "Tile: ${TILE} | Total missing: ${TOTAL_MISSING}"
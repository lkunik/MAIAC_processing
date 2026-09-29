#!/bin/bash

BASEDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI"

# YEARS=$(seq 2016 2016)
YEARS=$(seq 2000 2025)

MODIS_TILES=(
"h07v06" 
"h08v03"
# "h08v04"
# "h08v05"
# "h08v06"
# "h08v07"
"h09v02"
"h09v03" 
# "h09v04" # US-NR1
# "h09v05" # ecoregions related to US-NR1
# "h09v06"
# "h09v07"
# "h09v08" 
# "h10v02"
# "h10v03"
# "h10v04" # ecoregions related to US-NR1
# "h10v05"
# "h10v06" # OSBS
# "h10v07"
# "h10v08"
# "h11v02" # DEJU
# "h11v03" # Ca-Obs
# "h11v04"
# "h11v05"
"h11v06"
# "h11v07"
"h11v08"
# "h12v01"
# "h12v02"
# "h12v03"
# "h12v04"
# "h12v05"
# "h13v01"
# "h13v02"
# "h13v03"
# "h13v04" 
# "h14v01"
# "h14v02"
# "h14v03"
"h14v04"
# "h15v01"
# "h15v02"
# "h16v01"
)



njobs=0

for MODIS_TILE in "${MODIS_TILES[@]}"; do
    for YEAR in ${YEARS}; do

        echo "Submitting download pipeline - ${MODIS_TILE} year ${YEAR}"
        LOGDIR="${BASEDIR}/log/pipeline/${MODIS_TILE}/Download"

        # archive past log files
        mkdir -p ${LOGDIR}/archive/ # create log dir and an archive dir to put old files
        CUR_JOBBASE_FILES=$(ls ${LOGDIR}) # list existing log files
        if [ ! -z "${CUR_JOBBASE_FILES}" ] # if there are any existing...
        then
            find ${LOGDIR} -maxdepth 1 -type f -exec mv {} ${LOGDIR}/archive/ \;
        fi

        JOB_ID=$(sbatch --parsable \
        -D ${LOGDIR} \
        --output="${LOGDIR}/out-%x-%j-%N.out" \
        --error="${LOGDIR}/err-%x-%j-%N.err" \
        --job-name="dl_${MODIS_TILE}_t${YEAR}" \
        --export=LOGDIR=${LOGDIR},MODIS_TILE=${MODIS_TILE},INPUT_YEAR=${YEAR} \
        ${BASEDIR}/code/_01_download_pipeline/00~download_pipeline.sbatch)

        njobs=$((njobs+1))

        sleep 0.1 # add a short sleep to avoid overwhelming the scheduler with too many rapid submissions
    done
done

echo "Submitted ${njobs} tile-periods"
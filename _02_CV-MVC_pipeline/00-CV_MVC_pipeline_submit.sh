#!/bin/bash

BASEDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI"



COMPOSITES=$(seq 0 22)
# COMPOSITES=$(seq 15 22)

MODIS_TILES=(
# "h07v06" 
# "h08v03"
# "h08v04"
# "h08v05"
# "h08v06"
"h08v07"
# "h09v02"
# "h09v03" 
# "h09v04" # US-NR1
# "h09v05" # ecoregions related to US-NR1
"h09v06"
"h09v07"
"h09v08" 
# "h10v02"
# "h10v03"
# "h10v04" # ecoregions related to US-NR1
# "h10v05"
# "h10v06" # OSBS
"h10v07"
"h10v08"
# "h11v02" # DEJU
# "h11v03" # Ca-Obs
# "h11v04"
# "h11v05"
# "h11v06"
"h11v07"
# "h11v08"
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
# "h14v04"
# "h15v01"
# "h15v02"
# "h16v01"
)


QC_FLAG="CloudFree_LowAOD_ClearAdj" 

VZA_MIN=0
VZA_MAX=90


# Compare AOD and adjacency masks:
# 1: CloudFree_AllAOD_AllAdj
# 2: CloudFree_LowAOD_AllAdj

# 3: CloudFree_AllAOD_NotSurround
# 4: CloudFree_LowAOD_NotSurround

# 5: CloudFree_AllAOD_ClearAdj
# 6: CloudFree_LowAOD_ClearAdj


njobs=0

for MODIS_TILE in "${MODIS_TILES[@]}"; do
    for COMPOSITE in ${COMPOSITES}; do

        echo "Submitting pipeline with ${QC_FLAG} QC flag - ${MODIS_TILE} composite period ${COMPOSITE}"
        COMPOSITE_2D=$(printf "%02d" ${COMPOSITE})
        LOGDIR="${BASEDIR}/log/pipeline/${MODIS_TILE}/${COMPOSITE_2D}_${QC_FLAG}"

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
        --job-name="CV-MVC_${MODIS_TILE}_t${COMPOSITE_2D}_${QC_FLAG}" \
        --export=LOGDIR=${LOGDIR},MODIS_TILE=${MODIS_TILE},COMPOSITE=${COMPOSITE},QC_FLAG=${QC_FLAG},VZA_MIN=${VZA_MIN},VZA_MAX=${VZA_MAX} \
        ${BASEDIR}/code/_02_CV-MVC_pipeline/00~CV-MVC_pipeline.sbatch)

        njobs=$((njobs+1))

        sleep 0.1 # add a short sleep to avoid overwhelming the scheduler with too many rapid submissions
    done
done

echo "Submitted ${njobs} tile-periods"
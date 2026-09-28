#!/bin/bash

# MODIFY BELOW - CHANGE THE BASE JOB NAME AND INPUT YEARS TO SUBMIT
# JOB NAME WILL BE "JOBBASE_YYYY", e.g. "pyjob_2021"
# UTILDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/bashutil"
BASEDIR="//uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/code/compare_Photospec_MCD19_etc/00_save_rasters_as_point"
LOG_BASEDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/log/QC/extract_point_data"
JOBBASE="extractPoints"

QC_FLAG="CloudFree_LowAOD_ClearAdj" 
SITEs=(
    "US-NR1"
    "OSBS"
    "DEJU"
    "Ca-Obs"
    "US-UMB"
    "US-Ne3"
)


njobs=0

for SITE in "${SITEs[@]}"
do
    LOGDIR="${LOG_BASEDIR}/${SITE}"
    mkdir -p ${LOGDIR}/archive/ # create log dir and an archive dir to put old files


    # Define the job name
    JOBNAME="${JOBBASE}_${SITE}"

    
    # archive past log files tagged with this JOBBASE
    
    CUR_JOBBASE_FILES=$(find ${LOGDIR} -maxdepth 1 -type f -name "*${JOBNAME}*") # list existing log files tagged with this JOBBASE
    if [ ! -z "${CUR_JOBBASE_FILES}" ] # if there are any existing...
    then
        mv ${LOGDIR}/*${JOBNAME}* ${LOGDIR}/archive/ # move to the 'archive' dir
    fi

    # find other files older than 2 weeks
    OLD_FILES=$(find ${LOGDIR} -maxdepth 1 -type f -mtime +14) 
    if [ ! -z "${OLD_FILES}" ] # there are any old files....
    then
        for OLD_FILE in ${OLD_FILES}
        do
        mv ${OLD_FILE} ${LOGDIR}/archive/ # move old file to the 'archive' dir
        done
    fi

    # find old status report files and delete them
    rm -f ${LOGDIR}/RUN_STATUS*.txt

    # submit job to CHPC for this polygon
    sbatch -D $LOGDIR --job-name=$JOBNAME --export=LOGDIR=${LOGDIR},SITE=${SITE},QC_FLAG=${QC_FLAG},JOBNAME=${JOBNAME} $BASEDIR/01~pyjob.sbatch
    njobs=$((njobs+1))

done

echo "Submitted ${njobs} sbatch jobs"

#!/bin/bash

# MODIFY BELOW - CHANGE THE BASE JOB NAME AND INPUT YEARS TO SUBMIT
# JOB NAME WILL BE "JOBBASE_YYYY", e.g. "pyjob_2021"
# UTILDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/bashutil"
BASEDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/code/MOD09_proc"
LOG_BASEDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/log/MOD09_CV-MVC"
JOBBASE="CVMVC_MOD09"

QC_FLAG="CloudFree_LowAOD_ClearAdj"

YEARS=$(seq 2018 2019)
COMPOSITES=$(seq 0 22)

SITE_NAMES=(
# "US-NR1"
# "OSBS"
# "Ca-Obs"
# "DEJU"
"US-UMB"
"US-Ne3"
)


njobs=0

for SITE_NAME in "${SITE_NAMES[@]}"; do
    LOGDIR="${LOG_BASEDIR}/${QC_FLAG}/${SITE_NAME}"
    mkdir -p ${LOGDIR}/archive/ # create log dir and an archive dir to put old files

    for YEAR in ${YEARS}; do    
        for COMPOSITE in ${COMPOSITES}; do
            # archive past log files tagged with this JOBBASE
            COMPOSITE_2D=$(printf "%02d" ${COMPOSITE})

            # Define the job name
            JOBNAME="${JOBBASE}_${SITE_NAME}_${YEAR}-${COMPOSITE_2D}"

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
            sbatch -D $LOGDIR --job-name=$JOBNAME --export=LOGDIR=${LOGDIR},SITE_NAME=${SITE_NAME},COMPOSITE=${COMPOSITE},YEAR=${YEAR},QC_FLAG=${QC_FLAG},JOBNAME=${JOBNAME} $BASEDIR/02~pyjob.sbatch
            njobs=$((njobs+1))
        done
    done
done

echo "Submitted ${njobs} sbatch jobs"

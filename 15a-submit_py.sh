#!/bin/bash

# MODIFY BELOW - CHANGE THE BASE JOB NAME AND INPUT YEARS TO SUBMIT
# JOB NAME WILL BE "JOBBASE_YYYY", e.g. "pyjob_2021"
# UTILDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/bashutil"
BASEDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI"
LOG_BASEDIR="${BASEDIR}/log/maps_seasonality"
QC_FLAG="CloudFree"
JOBBASE="Seasonality"
COMPOSITES=$(seq 0 22)
YEARS=$(seq 2000 2025)
njobs=0

LOGDIR="${LOG_BASEDIR}"
mkdir -p ${LOGDIR}/archive/ # create log dir and an archive dir to put old files

for YEAR in ${YEARS}
do
    for COMPOSITE in ${COMPOSITES}
    do


        # Define the job name
        COMPOSITE_2D=$(printf "%02d" ${COMPOSITE})
        JOBNAME="${JOBBASE}_${YEAR}_${COMPOSITE_2D}"

        
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
        sbatch -D $LOGDIR --job-name=$JOBNAME --export=LOGDIR=${LOGDIR},YEAR=${YEAR},COMPOSITE=${COMPOSITE},QC_FLAG=${QC_FLAG},JOBNAME=${JOBNAME} $BASEDIR/code/05d~pyjob.sbatch
        njobs=$((njobs+1))

    done
done
echo "Submitted ${njobs} sbatch jobs"

#!/bin/bash

# MODIFY BELOW - CHANGE THE BASE JOB NAME AND INPUT YEARS TO SUBMIT
# JOB NAME WILL BE "JOBBASE_YYYY", e.g. "Mosaic_CCI_2020"
BASEDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/code/_02_CV-MVC_pipeline"
LOG_BASEDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/log/combine_CCI_NAm"
JOBBASE="Combine_CCI"

QC_FLAG="CloudFree_LowAOD_ClearAdj"
YEARS=( 2023 )
# YEARS=( $(seq 2000 2025) )


njobs=0

LOGDIR="${LOG_BASEDIR}"
mkdir -p ${LOGDIR}/archive/ # create log dir and an archive dir to put old files

for YYYY in "${YEARS[@]}"
do
    # Define the job name
    JOBNAME="${JOBBASE}_${YYYY}"

    # archive past log files tagged with this JOBNAME
    CUR_JOBBASE_FILES=$(find ${LOGDIR} -maxdepth 1 -type f -name "*${JOBNAME}*") # list existing log files tagged with this JOBNAME
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

    # submit job to CHPC for this year
    sbatch -D $LOGDIR --job-name=$JOBNAME --export=LOGDIR=${LOGDIR},YYYY=${YYYY},QC_FLAG=${QC_FLAG},JOBNAME=${JOBNAME} $BASEDIR/19~pyjob.sbatch
    njobs=$((njobs+1))
done

echo "Submitted ${njobs} sbatch jobs"

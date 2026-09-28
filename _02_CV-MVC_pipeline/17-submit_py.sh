#!/bin/bash

# MODIFY BELOW - CHANGE THE BASE JOB NAME AND INPUT YEARS TO SUBMIT
# JOB NAME WILL BE "JOBBASE_YYYY", e.g. "pyjob_2021"
# UTILDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/bashutil"
BASEDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI/code/_02_CV-MVC_pipeline"
LOG_BASEDIR="${BASEDIR}/log/annualize_std_anom_summer_NAm"
JOBBASE="Annualize_StdAnom"


AVG_WINDOWS=( "JJA" "JAS" )
QC_FLAG="CloudFree"

MODIS_TILES=(
# "h07v06" 
"h08v04"
# "h08v05"
# "h08v06"
# "h08v07"
# "h09v02"
# "h09v03" 
"h09v04"
# "h09v05"
# "h09v06"
# "h09v07"
# "h09v08" 
# "h10v02"
# "h10v03"
# "h10v04"
"h10v05"
"h10v06"
# "h10v07"
# "h10v08"
# "h11v02"
# "h11v03"
# "h11v04"
"h11v05"
# "h11v06"
# "h11v07"
# "h12v01"
# "h12v02"
# "h12v03"
# "h12v04"
# "h12v05"
# "h13v01"
# "h13v02"
# "h13v03"
"h13v04"
# "h14v01"
# "h14v02"
# "h14v03"
# "h14v04"
# "h15v01"
# "h15v02"
)


njobs=0

for MODIS_TILE in "${MODIS_TILES[@]}"
do
    LOGDIR="${LOG_BASEDIR}/${MODIS_TILE}"
    mkdir -p ${LOGDIR}/archive/ # create log dir and an archive dir to put old files

    for AVG_WINDOW in "${AVG_WINDOWS[@]}"
    do
        # Define the job name
        JOBNAME="${JOBBASE}_${MODIS_TILE}_${AVG_WINDOW}"

        
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
        sbatch -D $LOGDIR --job-name=$JOBNAME --export=LOGDIR=${LOGDIR},MODIS_TILE=${MODIS_TILE},AVG_WINDOW_ABBR=${AVG_WINDOW},QC_FLAG=${QC_FLAG},JOBNAME=${JOBNAME} $BASEDIR/17~pyjob.sbatch
        njobs=$((njobs+1))
    done
done

echo "Submitted ${njobs} sbatch jobs"

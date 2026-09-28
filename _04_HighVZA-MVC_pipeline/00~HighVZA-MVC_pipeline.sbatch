#!/bin/bash

#SBATCH --requeue # on guest nodes, opt to requeue any job that is booted out by a higher priority job 
#SBATCH --time=02:00:00 # walltime, abbreviated by -t
#SBATCH --nodes=1      # number of cluster nodes, abbreviated by -N
#SBATCH --ntasks=1     # number of tasks to run
#SBATCH --cpus-per-task=8 # number of CPUs to use per task, abbreviated by -c
# COMMENT #SBATCH -o out-%x-%j-%N.out # name of the stdout, using the job name (%x), job number (%j) and the first node (%N)
# COMMENT #SBATCH -e err-%x-%j-%N.err # name of the stderr, using the job name (%x), job number (%j) and the first node (%N)
#SBATCH --account=owner-guest  # account - abbreviated by -A
#SBATCH --partition=notchpeak-guest  # partition, abbreviated by -p
#SBATCH --mem=150G

# For lin-group kingspeak nodes, use account=lin-kp, partition=lin-kp
# For lin-group notchpeak nodes, use account=lin-np, partition=lin-np
# For guest kingspeak nodes, use account=owner-guest, partition=kingspeak-guest
# For guest notchpeak nodes, use account=owner-guest, partition=notchpeak-guest

source /uufs/chpc.utah.edu/sys/installdir/lmod/8.6-r8/init/bash
source /uufs/chpc.utah.edu/common/home/u6017162/software/pkg/miniforge3/etc/profile.d/conda.sh
conda activate main  # activate main conda environment


# load appropriate modules
module use $HOME/MyModules
module load intel
module load gcc/8.5.0
module load intel-oneapi-mpi/2021.4.0
module load gdal/3.3.3
module load mvapich2
module load miniconda3/latest


printf -v COMPOSITE_2D "%02d" ${COMPOSITE}
BASEDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI"
INPUT_YEARS=$(seq 2000 2025 | tr '\n' '_' | sed 's/_$//')  # Removes trailing underscore
nYEARS=$(echo ${INPUT_YEARS} | tr '_' ' ' | wc -w)

echo "Starting pipeline for ${MODIS_TILE} composite ${COMPOSITE_2D}"
echo "QC Flag: ${QC_FLAG}"


export TIMESTAMP_SCRIPT=/uufs/chpc.utah.edu/common/home/lin-group19/ltk/bashutil/10_print_datetime_info.sh
export STATUS_STR="${MODIS_TILE}_${COMPOSITE_2D}_${QC_FLAG}"
export LOGFILE=${LOGDIR}/RUN_STATUS.txt
export ALL_SUCCESS=0

# Argument $1 = EXIT STATUS
# Argument $2 = EXECUTION SCRIPT
# Argument $3 = previous state of global exit statuses
PRINT_EXIT_STATUS_TO_FILE () {
  
   # Check the exit status and write to the log file
   if [ $1 -eq 0 ]; then
      echo "code ran to completion: $2" >> ${LOGFILE}
      return $3
   else
      echo "code failed, returned with exit code $1: $2" >> ${LOGFILE}
      return 1
   fi

}





# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Step 1: Process this composite for ALL years
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# RUN_SCRIPT=${BASEDIR}/code/_04_HighVZA-MVC_pipeline/01-HighVZA_MVC_pre_fill.sh
# ${RUN_SCRIPT} ${LOGDIR} ${BASEDIR} ${MODIS_TILE} ${COMPOSITE} ${QC_FLAG} ${VZA_MIN} ${VZA_MAX} ${INPUT_YEARS}
# EXIT_STATUS=$?



# PRINT_EXIT_STATUS_TO_FILE ${EXIT_STATUS} ${EXECUTION_SCRIPT} ${ALL_SUCCESS}
# ALL_SUCCESS=$? # get the exit status code from the function

# # At the end of the batch run, change the filename of the "RUN_STATUS" file to either SUCCESS_ or FAIL_
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# if [ $ALL_SUCCESS -eq 0 ]; then
#     mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_SUCCESS_01_${STATUS_STR}_${TIMESTAMP}.txt
# else
#     mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_FAIL_01_${STATUS_STR}_${TIMESTAMP}.txt
#     exit 1 # exit the pipeline if the step fails, since subsequent steps depend on it
# fi


# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Step 2: Historical average for THIS composite (runs after all years complete)
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# RUN_SCRIPT=${BASEDIR}/code/_04_HighVZA-MVC_pipeline/02-HighVZA_MVC_calc_hist.sh
# ${RUN_SCRIPT} ${LOGDIR} ${BASEDIR} ${MODIS_TILE} ${COMPOSITE} ${QC_FLAG}
# EXIT_STATUS=$?


# PRINT_EXIT_STATUS_TO_FILE ${EXIT_STATUS} ${EXECUTION_SCRIPT} ${ALL_SUCCESS}
# ALL_SUCCESS=$? # get the exit status code from the function

# # At the end of the batch run, change the filename of the "RUN_STATUS" file to either SUCCESS_ or FAIL_
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# if [ $ALL_SUCCESS -eq 0 ]; then
#     mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_SUCCESS_02_${STATUS_STR}_${TIMESTAMP}.txt
# else
#     mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_FAIL_02_${STATUS_STR}_${TIMESTAMP}.txt
#     exit 1 # exit the pipeline if the step fails, since subsequent steps depend on it
# fi

# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Step 3: Save scaling factors for each year (depends on hist avg)
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RUN_SCRIPT=${BASEDIR}/code/_04_HighVZA-MVC_pipeline/03-HighVZA_MVC_save_scale.sh
# ${RUN_SCRIPT} ${LOGDIR} ${BASEDIR} ${MODIS_TILE} ${COMPOSITE} ${QC_FLAG} ${INPUT_YEARS}
# EXIT_STATUS=$?


# PRINT_EXIT_STATUS_TO_FILE ${EXIT_STATUS} ${EXECUTION_SCRIPT} ${ALL_SUCCESS}
# ALL_SUCCESS=$? # get the exit status code from the function

# # At the end of the batch run, change the filename of the "RUN_STATUS" file to either SUCCESS_ or FAIL_
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# if [ $ALL_SUCCESS -eq 0 ]; then
#     mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_SUCCESS_03_${STATUS_STR}_${TIMESTAMP}.txt
# else
#     mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_FAIL_03_${STATUS_STR}_${TIMESTAMP}.txt
#     exit 1 # exit the pipeline if the step fails, since subsequent steps depend on it
# fi

# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Step 4: Apply gap-fill for each year (depends on scaling for that year)
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# RUN_SCRIPT=${BASEDIR}/code/_04_HighVZA-MVC_pipeline/04-HighVZA_MVC_apply_scale.sh
# ${RUN_SCRIPT} ${LOGDIR} ${BASEDIR} ${MODIS_TILE} ${COMPOSITE} ${QC_FLAG} ${INPUT_YEARS}
# EXIT_STATUS=$?


# PRINT_EXIT_STATUS_TO_FILE ${EXIT_STATUS} ${EXECUTION_SCRIPT} ${ALL_SUCCESS}
# ALL_SUCCESS=$? # get the exit status code from the function

# # At the end of the batch run, change the filename of the "RUN_STATUS" file to either SUCCESS_ or FAIL_
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# if [ $ALL_SUCCESS -eq 0 ]; then
#     mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_SUCCESS_04_${STATUS_STR}_${TIMESTAMP}.txt
# else
#     mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_FAIL_04_${STATUS_STR}_${TIMESTAMP}.txt
#     exit 1 # exit the pipeline if the step fails, since subsequent steps depend on it
# fi

# # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Step 5: Regrid each year (depends on gap-fill for that year)
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RUN_SCRIPT=${BASEDIR}/code/_04_HighVZA-MVC_pipeline/05-HighVZA_MVC_regrid.sh
# ${RUN_SCRIPT} ${LOGDIR} ${BASEDIR} ${MODIS_TILE} ${COMPOSITE} ${QC_FLAG} ${INPUT_YEARS}
# EXIT_STATUS=$?



# PRINT_EXIT_STATUS_TO_FILE ${EXIT_STATUS} ${EXECUTION_SCRIPT} ${ALL_SUCCESS}
# ALL_SUCCESS=$? # get the exit status code from the function

# # At the end of the batch run, change the filename of the "RUN_STATUS" file to either SUCCESS_ or FAIL_
# if [ $ALL_SUCCESS -eq 0 ]; then
#     mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_SUCCESS_05_${STATUS_STR}_${TIMESTAMP}.txt
# else
#     mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_FAIL_05_${STATUS_STR}_${TIMESTAMP}.txt
# fi


# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Step 6: Calculate Historical Averages for Gap-Fill Composites (1km curvilinear)
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RUN_SCRIPT=${BASEDIR}/code/_04_HighVZA-MVC_pipeline/06-HighVZA_MVC_calc_hist_gapfill_1km_curv.sh
# ${RUN_SCRIPT} ${LOGDIR} ${BASEDIR} ${MODIS_TILE} ${COMPOSITE} ${QC_FLAG}
# EXIT_STATUS=$?



# PRINT_EXIT_STATUS_TO_FILE ${EXIT_STATUS} ${EXECUTION_SCRIPT} ${ALL_SUCCESS}
# ALL_SUCCESS=$? # get the exit status code from the function

# # At the end of the batch run, change the filename of the "RUN_STATUS" file to either SUCCESS_ or FAIL_
# if [ $ALL_SUCCESS -eq 0 ]; then
#     mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_SUCCESS_06_${STATUS_STR}_${TIMESTAMP}.txt
# else
#     mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_FAIL_06_${STATUS_STR}_${TIMESTAMP}.txt
# fi


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Step 7: Calculate Historical Averages for Gap-Fill Composites - 0.01° regridded
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RUN_SCRIPT=${BASEDIR}/code/_04_HighVZA-MVC_pipeline/07-HighVZA_MVC_calc_hist_gapfill_01d_rect.sh
${RUN_SCRIPT} ${LOGDIR} ${BASEDIR} ${MODIS_TILE} ${COMPOSITE} ${QC_FLAG}
EXIT_STATUS=$?


PRINT_EXIT_STATUS_TO_FILE ${EXIT_STATUS} ${EXECUTION_SCRIPT} ${ALL_SUCCESS}
ALL_SUCCESS=$? # get the exit status code from the function

# At the end of the batch run, change the filename of the "RUN_STATUS" file to either SUCCESS_ or FAIL_
if [ $ALL_SUCCESS -eq 0 ]; then
    mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_SUCCESS_07_${STATUS_STR}_${TIMESTAMP}.txt
else
    mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_FAIL_07_${STATUS_STR}_${TIMESTAMP}.txt
fi


# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Step 8: Compute summertime averages from gap-filled composites
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# echo "Step 8: Compute summertime averages from gap-filled composites"
# echo "*** User must run separately after all composite periods have finished***"


# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Step 9: Calculate Standardized Anomalies from Gap-Fill Composites and Historical Averages
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RUN_SCRIPT=${BASEDIR}/code/_04_HighVZA-MVC_pipeline/09-HighVZA_MVC_calc_std_anom_1km_curv.sh
# ${RUN_SCRIPT} ${LOGDIR} ${BASEDIR} ${MODIS_TILE} ${COMPOSITE} ${QC_FLAG} ${INPUT_YEARS}
# EXIT_STATUS=$?



# PRINT_EXIT_STATUS_TO_FILE ${EXIT_STATUS} ${EXECUTION_SCRIPT} ${ALL_SUCCESS}
# ALL_SUCCESS=$? # get the exit status code from the function

# # At the end of the batch run, change the filename of the "RUN_STATUS" file to either SUCCESS_ or FAIL_
# if [ $ALL_SUCCESS -eq 0 ]; then
#     mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_SUCCESS_09_${STATUS_STR}_${TIMESTAMP}.txt
# else
#     mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_FAIL_09_${STATUS_STR}_${TIMESTAMP}.txt
# fi

# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Step 10: Regrid Standardized Anomalies to 0.01° Rectilinear Grid
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RUN_SCRIPT=${BASEDIR}/code/_04_HighVZA-MVC_pipeline/10-HighVZA_MVC_regrid_std_anom.sh
# ${RUN_SCRIPT} ${LOGDIR} ${BASEDIR} ${MODIS_TILE} ${COMPOSITE} ${QC_FLAG} ${INPUT_YEARS}
# EXIT_STATUS=$?

# PRINT_EXIT_STATUS_TO_FILE ${EXIT_STATUS} ${EXECUTION_SCRIPT} ${ALL_SUCCESS}
# ALL_SUCCESS=$? # get the exit status code from the function

# # At the end of the batch run, change the filename of the "RUN_STATUS" file to either SUCCESS_ or FAIL_
# if [ $ALL_SUCCESS -eq 0 ]; then
#     mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_SUCCESS_10_${STATUS_STR}_${TIMESTAMP}.txt
# else
#     mv ${LOGFILE} ${LOGDIR}/RUN_STATUS_FAIL_10_${STATUS_STR}_${TIMESTAMP}.txt
# fi


# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Step 11: Compute summertime averages from gap-filled composites
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# echo "Step 11: Compute summertime averages from Standardized Anomaly composites"
# echo "*** User must run separately after all composite periods have finished***"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# FINAL STEPS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Save job IDs to file for monitoring
cat > ${LOGDIR}/pipeline_jobs.txt <<EOF
Pipeline for ${MODIS_TILE} composite ${COMPOSITE_2D}
Started: $(date)
QC Flag: ${QC_FLAG}
Years: 2000-2025 (${nYEARS} years)



Monitor: squeue -u \$USER | grep ${MODIS_TILE}
EOF


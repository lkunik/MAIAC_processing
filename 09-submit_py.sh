#!/bin/bash

# MODIFY BELOW - CHANGE THE BASE JOB NAME AND INPUT YEARS TO SUBMIT
# JOB NAME WILL BE "JOBBASE_YYYY", e.g. "pyjob_2021"
# UTILDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/bashutil"
BASEDIR="/uufs/chpc.utah.edu/common/home/lin-group19/ltk/MODIS/CCI"
LOGDIR="${BASEDIR}/log/ecoregion_clim_veg_2dhist"
CLIM_NAMES=( "SPEI01" "SPEI12" )
VEG_NAMES=( "CCI_std_anomaly" "NDVI_std_anomaly" "NIRv_std_anomaly" "NDMI_std_anomaly" )
JOBBASE="2dhist"
AVG_WINDOW="JJA"


ECOREGIONS=(
# "Thompson-Okanogan_Plateau"
# "Central_Basin_and_Range"
# "Colorado_Plateaus"
# "Arizona~New_Mexico_Mountains"
# "Sierra_Madre_Occidental_with_Conifer_Oak_and_Mixed_Forests"
# "Interior_Forested_Lowlands_and_Uplands"
# "Interior_Bottomlands"
# "Yukon_Flats"
# "Ogilvie_Mountains"
# "Mackenzie_and_Selwyn_Mountains"
# "Peel_River_and_Nahanni_Plateaus"
# "Great_Bear_Plains"
# "Hay_and_Slave_River_Lowlands"
# "Kazan_River_and_Selwyn_Lake_Uplands"
# "La_Grande_Hills_and_New_Quebec_Central_Plateau"
# "Smallwood_Uplands"
# "Ungava_Bay_Basin_and_George_Plateau"
# "Coppermine_River_and_Tazin_Lake_Uplands"
# "Hudson_Bay_and_James_Bay_Lowlands"
"Athabasca_Plain_and_Churchill_River_Upland"
# "Lake_Nipigon_and_Lac_Seul_Upland"
# "Central_Laurentians_and_Mecatina_Plateau"
# "Hayes_River_Upland_and_Big_Trout_Lake"
# "Abitibi_Plains_and_Riviere_Rupert_Plateau"
# "Algonquin~Southern_Laurentians"
# "Northern_Appalachian_and_Atlantic_Maritime_Highlands"
# "Mid-Boreal_Uplands_and_Peace-Wabaska_Lowlands"
# "Clear_Hills_and_Western_Alberta_Upland"
# "Mid-Boreal_Lowland_and_Interlake_Plain"
# "Interior_Highlands_and_Klondike_Plateau"
# "Copper_Plateau"
# "Watson_Highlands"
# "Yukon-Stikine_Highlands~Boreal_Mountains_and_Plateaus"
# "Skeena-Omineca-Central_Canadian_Rocky_Mountains"
# "Middle_Rockies"
# "Klamath_Mountains"
# "Sierra_Nevada"
# "Wasatch_and_Uinta_Mountains"
# "Southern_Rockies"
# "Idaho_Batholith"
# "Chilcotin_Ranges_and_Fraser_Plateau"
# "Columbia_Mountains~Northern_Rockies"
# "Canadian_Rockies"
# "North_Cascades"
# "Blue_Mountains"
# "Coastal_Western_Hemlock-Sitka_Spruce_Forests"
# "Pacific_and_Nass_Ranges"
# "Coast_Range"
# "Southeastern_Plains"
# "South_Central_Plains"
# "Southern_Coastal_Plain"
# "Cascades"
# "Eastern_Cascades_Slopes_and_Foothills"
)


njobs=0
mkdir -p ${LOGDIR}/archive/ # create log dir and an archive dir to put old files

for ECOREGION in "${ECOREGIONS[@]}"
do
    for CLIM_NAME in "${CLIM_NAMES[@]}"
    do
        for VEG_NAME in "${VEG_NAMES[@]}"
        do
            # Define the job name
            JOBNAME="${JOBBASE}_${CLIM_NAME}_${VEG_NAME}_${AVG_WINDOW}_${ECOREGION}"

            
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
            sbatch -D $LOGDIR --job-name=$JOBNAME --export=LOGDIR=${LOGDIR},ECOREGION=${ECOREGION},CLIM_NAME=${CLIM_NAME},VEG_NAME=${VEG_NAME},AVG_WINDOW=${AVG_WINDOW},JOBNAME=${JOBNAME} $BASEDIR/code/09~pyjob.sbatch
            njobs=$((njobs+1))
        done
    done
done
echo "Submitted ${njobs} sbatch jobs"

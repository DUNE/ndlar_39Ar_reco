#!/usr/bin/env bash

DET="module-1"
if [ "${DET}" = "module-0" ]; then
	OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module0_reco/39Ar_reco
    ADC_1_NAME=0a7a314c
    ADC_2_NAME=0a7b54bd
    PACKET_FILE_NAME_START=datalog_
elif [ "${DET}" = "module-1" ]; then
	OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module1_reco/39Ar_reco_09132023
    ADC_1_NAME=0cd913fb
    ADC_2_NAME=0cd93db0
    PACKET_FILE_NAME_START=packet_
elif [ "${DET}" = "module-2" ]; then
        OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module2_reco/39Ar_reco
        ADC_1_NAME=0cd8d631
        ADC_2_NAME=0cd913fa
	PACKET_FILE_NAME_START=self_trigger_tpc12_run2-packet-
elif [ "${DET}" = "module-X" ]; then
        OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/ModuleX_reco/nominalHV_beforeElevatedHV
else
	echo "Exiting as $DET is not a recognized run name"
	exit 0
fi

TIMESTAMP_CHARGE=(
            "2022_02_08_01_47_59_CET"
            "2022_02_08_02_08_00_CET"
            "2022_02_08_02_28_01_CET"
            "2022_02_08_03_54_19_CET"
            "2022_02_08_04_14_21_CET"
            "2022_02_08_04_34_22_CET"
            "2022_02_08_04_54_23_CET"
            "2022_02_08_05_14_25_CET"
            "2022_02_08_08_54_32_CET"
            "2022_02_08_09_14_33_CET"
            "2022_02_08_09_34_35_CET"
            "2022_02_08_12_48_18_CET"
            "2022_02_08_13_08_19_CET"
            "2022_02_08_13_57_39_CET"
            "2022_02_08_14_17_41_CET"
            )
TIMESTAMP_LIGHT=(
            "20220208_014759"
            "20220208_020800"
            "20220208_022802"
            "20220208_035419"
            "20220208_041421"
            "20220208_043422"
            "20220208_045424"
            "20220208_051425"
            "20220208_085432"
            "20220208_091433"
            "20220208_093435"
            "20220208_124818"
            "20220208_130819"
            "20220208_135739"
            "20220208_141741"
            )


#TIMESTAMP_CHARGE=(
#	    "2021_04_04_11_38_43_CEST"
#            "2021_04_04_12_18_48_CEST"
#            "2021_04_04_13_18_54_CEST"
#            "2021_04_04_13_38_56_CEST"
#            "2021_04_04_14_19_01_CEST"
#            "2021_04_04_14_39_03_CEST"
#            "2021_04_04_15_19_07_CEST"
#            "2021_04_04_16_59_17_CEST"
#            "2021_04_04_17_19_19_CEST"
#            "2021_04_04_17_39_22_CEST"
#            "2021_04_04_17_59_24_CEST"
#            )
#TIMESTAMP_LIGHT=(
#            "20210404_113847"
#            "20210404_121849"
#            "20210404_131854"
#            "20210404_133856"
#            "20210404_141859"
#            "20210404_143903"
#            "20210404_151909"
#            "20210404_165920"
#	    "20210404_171920"
#            "202210404_173921"
#            "20210404_175925"
#            )

#TIMESTAMP_CHARGE=(
#            "2022_12_02_00_07_CET"
#            "2022_12_02_01_38_CET"
#            "2022_12_02_01_48_CET"
#            "2022_12_02_01_58_CET"
#            "2022_12_02_02_08_CET"
#	    "2022_12_02_02_18_CET"
#	    "2022_12_02_02_28_CET"
#	    "2022_12_02_02_38_CET"
#	    "2022_12_02_02_48_CET"
#	    #"2022_12_02_02_58_CET"
#	    #"2022_12_02_03_08_CET"
#	    #"2022_12_02_03_18_CET"
#	    #"2022_12_02_03_28_CET"
#           )  
#TIMESTAMP_LIGHT=(
#        "20221202_000743"
#	"20221202_013841"
#	"20221202_014842"
#	"20221202_015844"
#	"20221202_020845"
#	"20221202_021847"
#	"20221202_022848"
#	"20221202_023849"
#	"20221202_024851"
#	)
list_length=${#TIMESTAMP_CHARGE[@]}

cd ..
#rm -rf convert.venv
#python3 -m venv convert.venv
source convert.venv/bin/activate
#pip install --upgrade pip
#pip install -r requirements.txt

cd charge_reco
for ((i = 0; i < list_length; i++)); do
    timestamp_charge="${TIMESTAMP_CHARGE[i]}"
    timestamp_light="${TIMESTAMP_LIGHT[i]}"
    INFILENAME_CLUSTERS=${OUTDIR}/${PACKET_FILE_NAME_START}${timestamp_charge}_clusters.h5
    INFILENAME_LIGHT_1=${OUTDIR}/${ADC_1_NAME}_${timestamp_light}.h5
    INFILENAME_LIGHT_2=${OUTDIR}/${ADC_2_NAME}_${timestamp_light}.h5
    OUTFILENAME=${OUTDIR}/charge-light-matched-clusters_${timestamp_charge}.h5

    python3 match_light_to_clusters.py ${INFILENAME_CLUSTERS} ${OUTFILENAME} ${INFILENAME_LIGHT_1} ${INFILENAME_LIGHT_2} --input_config_name ${DET}
done


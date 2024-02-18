#!/usr/bin/env bash

DET="module0_run2"
if [ "${DET}" = "module0_run1" ]; then
    data_folder=/global/cfs/cdirs/dune/www/data/Module0/TPC1+2/dataRuns/packetData
    OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module0_run1_reco/39Ar_reco
    file_name_start=datalog_
elif [ "${DET}" = "module0_run2" ]; then
    data_folder=/global/cfs/cdirs/dune/www/data/Module0-Run2/TPC1+2/dataRuns/packetData
    OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module0_run2_reco/39Ar_reco
    file_name_start=packet_
elif [ "${DET}" = "module1" ]; then
    data_folder=/global/cfs/cdirs/dune/www/data/Module1/TPC12/dataRuns/packetData
    OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module1_reco/39Ar_reco_09132023
    file_name_start=packet_
elif [ "${DET}" = "module2" ]; then
    data_folder=/global/cfs/cdirs/dune/www/data/Module2/packetized/TPC12_run2
    OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module2_reco/39Ar_reco
    file_name_start=self_trigger_tpc12_run2-packet-
elif [ "${DET}" = "moduleX" ]; then
    data_folder=/global/cfs/cdirs/dune/users/sfogarty/ModuleX_reco
    OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/ModuleX_reco
elif [ "${DET}" = "SingleCube" ]; then
    #data_folder=/global/cfs/cdirs/dune/www/data/ModuleX/commission
    data_folder=/global/cfs/cdirs/dune/users/sfogarty/Bern_SC_reco
    OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Bern_SC_reco
else
    echo "Exiting as $DET is not a recognized run name"
    exit 0
fi

current_dir=${pwd}
# module 0 run1
#TIMESTAMP_CHARGE=(
#            "2021_04_04_09_38_27_CEST"
#            "2021_04_04_09_58_31_CEST"
#            "2021_04_04_10_38_37_CEST"
#            "2021_04_04_11_18_41_CEST"
#            "2021_04_04_11_38_43_CEST"
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

# module 0 run2
TIMESTAMP_CHARGE=(
             #"2021_06_23_12_42_01_CEST"
	     #"2021_06_23_13_02_02_CEST"
	     #"2021_06_23_13_22_03_CEST"
	     #"2021_06_23_13_42_04_CEST"
	     #"2021_06_23_14_02_05_CEST"
	     #"2021_06_23_14_22_06_CEST"
	     #"2021_06_23_14_42_07_CEST"
	     #"2021_06_23_15_02_08_CEST"
	     #"2021_06_23_15_22_09_CEST"
	     "2021_06_23_15_42_10_CEST"
	     "2021_06_23_16_02_11_CEST"
	     "2021_06_23_16_22_12_CEST"
	     "2021_06_23_16_42_13_CEST"
	     "2021_06_23_17_02_14_CEST"
	     "2021_06_23_17_22_15_CEST"
	)
list_length=${#TIMESTAMP_CHARGE[@]}
cd ..
#rm -rf convert.venv
#python3 -m venv convert.venv
#source convert.venv/bin/activate
#pip install --upgrade pip
#pip install -r requirements.txt
cd charge_reco
module load python # nersc specific
for ((i = 0; i < list_length; i++)); do
    timestamp="${TIMESTAMP_CHARGE[i]}"
    PKT_FILENAME=${file_name_start}${timestamp}
    INFILENAME=${data_folder}/${PKT_FILENAME}.h5
    OUTFILENAME=${OUTDIR}/${PKT_FILENAME}_clusters.h5
    python3 charge_clustering.py ${DET} ${INFILENAME} ${OUTFILENAME} --save_hits=True --match_to_ext_trig=False
done



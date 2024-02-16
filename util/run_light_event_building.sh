#!/usr/bin/env bash

DET="module-2"
if [ "${DET}" = "module-0" ]; then
    OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module0_reco/39Ar_reco
    data_folder=/global/cfs/cdirs/dune/www/data/Module0/LRS
    ADC_name_1=0a7a314c
    ADC_name_2=0a7b54bd
elif [ "${DET}" = "module-1" ]; then
    OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module1_reco/39Ar_reco_09132023
    data_folder=/global/cfs/cdirs/dune/www/data/Module1/LRS/SingleModule_Jan22
    ADC_name_1=0cd913fb
    ADC_name_2=0cd93db0
elif [ "${DET}" = "module-2" ]; then
    OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module2_reco/39Ar_reco
    data_folder=/global/cfs/cdirs/dune/www/data/Module2/LRS.2/cosmic
    ADC_name_1=0cd8d631
    ADC_name_2=0cd913fa
elif [ "${DET}" = "module-X" ]; then
    OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/ModuleX_reco/nominalHV_lastDataTaken
    data_folder=/global/cfs/cdirs/dune/www/data/ModuleX/LRS/cosmic
    ADC_name_1=0cd8d631
    ADC_name_2=0cd913fa
else
    echo "Exiting as $DET is not a recognized run name"
    exit 0
fi

current_dir=${pwd}

TIMESTAMP_LIGHT=(
            #"20221201_233739"
            #"20221201_234740"
            #"20221201_235741"
            #"20221202_000743"
            #"20221202_013841"
            #"20221202_014842"
            #"20221202_015844"
            #"20221202_020845"
	    "20221202_020845"
	    "20221202_021847"
	    "20221202_022848"
	    "20221202_023849"
	    "20221202_024851"
            )


list_length=${#TIMESTAMP_LIGHT[@]}
cd ..
rm -rf convert.venv
python3 -m venv convert.venv
source convert.venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cd util
for ((i = 0; i < list_length; i++)); do
    timestamp_light="${TIMESTAMP_LIGHT[i]}"
    LRS_file_1=${ADC_name_1}_${timestamp_light}
    LRS_file_2=${ADC_name_2}_${timestamp_light}

    python3 ./adc64format/adc64_to_hdf5.py ${data_folder}/${LRS_file_1}.data ${OUTDIR}/${LRS_file_1}.h5
    python3 ./adc64format/adc64_to_hdf5.py ${data_folder}/${LRS_file_2}.data ${OUTDIR}/${LRS_file_2}.h5
done

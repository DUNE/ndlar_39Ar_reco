#!/usr/bin/env bash

DET="module-X"
if [ "${DET}" = "module-0" ]; then
    OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module0_reco
elif [ "${DET}" = "module-1" ]; then
    OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module1_reco/39Ar_reco_09132023
    data_folder=/global/cfs/cdirs/dune/www/data/Module1/LRS/SingleModule_Jan22
    ADC_name_1=0cd913fb
    ADC_name_2=0cd93db0
    file_timestamp=20220208_141741
elif [ "${DET}" = "module-2" ]; then
    OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module2_reco
    data_folder=/global/cfs/cdirs/dune/www/data/Module2/LRS.1/cosmic
    ADC_name_1=0cd8d631
    ADC_name_2=0cd913fa
    file_timestamp=20221119_043121
elif [ "${DET}" = "module-X" ]; then
    OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/ModuleX_reco
    data_folder=/global/cfs/cdirs/dune/www/data/ModuleX/LRS/cosmic
    ADC_name_1=0cd8d631
    ADC_name_2=0cd913fa
    file_timestamp=20231004_111624
else
    echo "Exiting as $DET is not a recognized run name"
    exit 0
fi

current_dir=${pwd}
LRS_file_1=${ADC_name_1}_${file_timestamp}
LRS_file_2=${ADC_name_2}_${file_timestamp}
output_file=${OUTDIR}/light_events_${file_timestamp}.h5

shifter --image=mjkramer/sim2x2:genie_edep.3_04_00.20230620 --module=None -- /bin/bash << EOF
set +o posix
source /environment

rm -rf convert.venv
python3 -m venv convert.venv
source convert.venv/bin/activate
pip install --upgrade pip
pip install -r ../requirements.txt

python3 ./adc64format/adc64_to_hdf5.py ${data_folder}/${LRS_file_1}.data ${OUTDIR}/${LRS_file_1}.h5
python3 ./adc64format/adc64_to_hdf5.py ${data_folder}/${LRS_file_2}.data ${OUTDIR}/${LRS_file_2}.h5
EOF




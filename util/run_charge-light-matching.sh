#!/usr/bin/env bash

DET="module-X"
if [ "${DET}" = "module-0" ]; then
	OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module0_reco
elif [ "${DET}" = "module-1" ]; then
	OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module1_reco/39Ar_reco_09132023
elif [ "${DET}" = "module-X" ]; then
        OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/ModuleX_reco
else
	echo "Exiting as $DET is not a recognized run name"
	exit 0
fi

#INFILENAME_CLUSTERS=${OUTDIR}/clusters_selection_2022_02_08_01_47_59_CET.h5
INFILENAME_CLUSTERS=${OUTDIR}/clusters_selection_2023_10_04_11_16_CEST.h5
INFILENAME_LIGHT_1=${OUTDIR}/0cd8d631_20231004_111624.h5
INFILENAME_LIGHT_2=${OUTDIR}/0cd913fa_20231004_111624.h5
OUTFILENAME=${OUTDIR}/charge-light-matched-clusters_2023_10_04_11_16_CEST.h5

shifter --image=mjkramer/sim2x2:genie_edep.3_04_00.20230620 --module=None -- /bin/bash << EOF
set +o posix
source /environment

cd ..
rm -rf convert.venv
python3 -m venv convert.venv
source convert.venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cd charge_reco
python3 match_light_to_ext_triggers.py ${INFILENAME_CLUSTERS} ${OUTFILENAME} ${INFILENAME_LIGHT_1} ${INFILENAME_LIGHT_2} --input_config_name ${DET}
EOF




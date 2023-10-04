#!/usr/bin/env bash

DET="Module2"
if [ "${DET}" = "Module0" ]; then
	data_folder=/global/cfs/cdirs/dune/www/data/Module0/TPC1+2/dataRuns/packetData
	OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module0_reco
elif [ "${DET}" = "Module1" ]; then
	data_folder=/global/cfs/cdirs/dune/www/data/Module1/TPC12/dataRuns/packetData
	OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module1_reco/39Ar_reco_09132023
elif [ "${DET}" = "Module2" ]; then
	data_folder=/global/cfs/cdirs/dune/www/data/Module2/packetized/TPC12
	OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module2_reco
else
	echo "Exiting as $DET is not a recognized run name"
	exit 0
fi

#PKT_FILENAME=packet_2022_02_08_12_48_18_CET
PKT_FILENAME=selftrigger-binary-2022_11_19_04_31_CET.packet
INFILENAME=${data_folder}/${PKT_FILENAME}.h5
OUTFILENAME=${OUTDIR}/${PKT_FILENAME}_clusters_test.h5

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
python3 charge_clustering.py module-2 ${INFILENAME} ${OUTFILENAME}
EOF




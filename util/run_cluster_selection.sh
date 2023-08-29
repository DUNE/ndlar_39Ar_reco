#!/usr/bin/env bash

DET="Module1"
if [ "${DET}" = "Module0" ]; then
	OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module0_reco
elif [ "${DET}" = "Module1" ]; then
	OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module1_reco
else
	echo "Exiting as $DET is not a recognized run name"
	exit 0
fi

INFILENAME_CLUSTERS=${OUTDIR}/packet_2022_02_08_01_47_59_CET_clusters.h5
OUTFILENAME=${OUTDIR}/clusters_selection_2022_02_08_01_47_59_CET.h5

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
python3 charge_cluster_selections.py ${INFILENAME_CLUSTERS} ${OUTFILENAME}
EOF




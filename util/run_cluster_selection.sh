#!/usr/bin/env bash

DET="ModuleX"
if [ "${DET}" = "Module0" ]; then
	INDIR=/global/cfs/cdirs/dune/users/sfogarty/Module0_reco
	OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module0_reco
elif [ "${DET}" = "Module1" ]; then
	INDIR=/global/cfs/cdirs/dune/users/sfogarty/Module1_reco/39Ar_reco_09132023
	OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module1_reco/39Ar_reco_09132023
elif [ "${DET}" = "Module2" ]; then
	INDIR=/global/cfs/cdirs/dune/users/sfogarty/Module2_reco
	OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module2_reco
elif [ "${DET}" = "ModuleX" ]; then
	INDIR=/global/cfs/cdirs/dune/users/sfogarty/ModuleX_reco
	OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/ModuleX_reco/60Co_source
else
	echo "Exiting as $DET is not a recognized run name"
	exit 0
fi

INFILENAME_CLUSTERS=${INDIR}/packets-2023_10_06_04_40_CEST_clusters.h5
OUTFILENAME=${OUTDIR}/clusters_selection_2023_10_06_04_40_CEST.h5

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




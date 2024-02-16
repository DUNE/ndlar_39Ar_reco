#!/usr/bin/env bash

DET="module-2"
if [ "${DET}" = "module-0" ]; then
	data_folder=/global/cfs/cdirs/dune/www/data/Module0/TPC1+2/dataRuns/packetData
	OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module0_reco/39Ar_reco
elif [ "${DET}" = "module-1" ]; then
	data_folder=/global/cfs/cdirs/dune/www/data/Module1/TPC12/dataRuns/packetData
	OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module1_reco/39Ar_reco_09132023
elif [ "${DET}" = "module-2" ]; then
	data_folder=/global/cfs/cdirs/dune/www/data/Module2/packetized/TPC12_run2
        #PEDESTAL_FOLDER=/global/cfs/cdirs/dune/www/data/Module2/packetized/TPC12_run2
	PEDESTAL_FOLDER=/global/cfs/cdirs/dune/users/sfogarty/Module2_reco/39Ar_reco
	OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module2_reco/39Ar_reco
elif [ "${DET}" = "module-X" ]; then
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

PKT_FILENAME=self_trigger_tpc12_run2_tile3-packet-2022_12_03_13_52_CET
#PKT_FILENAME=datalog_2021_04_04_10_38_37_CEST
INFILENAME=${data_folder}/${PKT_FILENAME}.h5
OUTFILENAME=${OUTDIR}/${PKT_FILENAME}_clusters.h5
PEDESTAL_FILE=${PEDESTAL_FOLDER}/pedestal_run2_tpc12_prc_4096_HV_off-binary-2022_12_02_23_34_CET_packets.h5

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
python3 charge_clustering.py ${DET} ${INFILENAME} ${OUTFILENAME} --save_hits=True --match_to_ext_trig=False

EOF
#--pedestal_file=${PEDESTAL_FILE} --vcm_dac=71 --vref_dac=217


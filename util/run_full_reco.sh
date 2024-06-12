#!/usr/bin/env bash

DET="module0_run1"
if [ "${DET}" = "module0_run1" ]; then
	OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module0_run1_reco/39Ar_reco
    ADC_1_NAME=0a7a314c
    ADC_2_NAME=0a7b54bd
    PACKET_FILE_NAME_START=datalog_
    data_folder_charge=/global/cfs/cdirs/dune/www/data/Module0/TPC1+2/dataRuns/packetData
    data_folder_light=/global/cfs/cdirs/dune/www/data/Module0/LRS
elif [ "${DET}" = "module0_run2" ]; then
	OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module0_run2_reco/39Ar_reco
    ADC_1_NAME=0a7a314c
    ADC_2_NAME=0a7b54bd
    PACKET_FILE_NAME_START=packet_
    data_folder_charge=/global/cfs/cdirs/dune/www/data/Module0-Run2/TPC1+2/dataRuns/packetData
	data_folder_light=/global/cfs/cdirs/dune/www/data/Module0-Run2/LRS
elif [ "${DET}" = "module1" ]; then
	OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module1_reco/39Ar_reco_09132023
    ADC_1_NAME=0cd913fb
    ADC_2_NAME=0cd93db0
    PACKET_FILE_NAME_START=packet_
    data_folder_charge=/global/cfs/cdirs/dune/www/data/Module1/TPC12/dataRuns/packetData
    data_folder_light=/global/cfs/cdirs/dune/www/data/Module1/LRS/SingleModule_Jan22
elif [ "${DET}" = "module2" ]; then
    OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module2_reco/39Ar_reco
    ADC_2_NAME=0cd8d631 # LCM
    ADC_1_NAME=0cd913fa # ACL
    #PACKET_FILE_NAME_START=self_trigger_tpc12_run2-packet-
    #PACKET_FILE_NAME_START=self_trigger_tpc12_run2_tile3-packet-
    PACKET_FILE_NAME_START=selftrigger-run2-packet-
    data_folder_charge=/global/cfs/cdirs/dune/www/data/Module2/packetized/TPC12_run2
    data_folder_light=/global/cfs/cdirs/dune/www/data/Module2/LRS.2/cosmic/incoming
    #data_folder=/global/cfs/cdirs/dune/www/data/Module2/LRS.2/cosmic
elif [ "${DET}" = "module3" ]; then
    OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module3_reco/39Ar_reco
    ADC_2_NAME=0cd8d631 # LCM
    ADC_1_NAME=0cd913fa # ACL
    #PACKET_FILE_NAME_START=self_trigger_tpc12_run2-packet-
    #PACKET_FILE_NAME_START=self_trigger_tpc12_run2_tile3-packet-
    PACKET_FILE_NAME_START=tpc12-packet-
    data_folder_charge=/global/cfs/cdirs/dune/www/data/Module3/run3/packet/tpc12
    data_folder_light=/global/cfs/cdirs/dune/www/data/Module3/run3/LRS/cosmic/
else
	echo "Exiting as $DET is not a recognized run name"
	exit 0
fi

# module 0 run 1
TIMESTAMP_CHARGE=(
            "2021_04_04_09_38_27_CEST")
            #"2021_04_04_09_58_31_CEST"
            #"2021_04_04_10_38_37_CEST"
            #"2021_04_04_11_18_41_CEST"
            #"2021_04_04_11_38_43_CEST"
            #"2021_04_04_12_18_48_CEST"
            #"2021_04_04_13_18_54_CEST"
            #"2021_04_04_13_38_56_CEST"
            #"2021_04_04_14_19_01_CEST"
            #"2021_04_04_14_39_03_CEST"
            #"2021_04_04_15_19_07_CEST"
#            "2021_04_04_16_59_17_CEST"
#            "2021_04_04_17_19_19_CEST"
#            "2021_04_04_17_39_22_CEST"
#            "2021_04_04_17_59_24_CEST"
#            )
TIMESTAMP_LIGHT=(
            "20210404_093828")
            #"20210404_095833"
            #"20210404_103840"
            #"20210404_111841"
            #"20210404_113847"
            #"20210404_121849"
            #"20210404_131854"
            #"20210404_133856"
            #"20210404_141859"
            #"20210404_143903"
            #"20210404_151909"
#            "20210404_165920"
#            "20210404_171920"
#            "20210404_173921"
#            "20210404_175925"
#            )

# module 0 run 2
#TIMESTAMP_CHARGE=(
#             "2021_06_23_12_42_01_CEST"
	     #"2021_06_23_13_02_02_CEST"
	     #"2021_06_23_13_22_03_CEST"
	     #"2021_06_23_13_42_04_CEST"
	     #"2021_06_23_14_02_05_CEST"
	     #"2021_06_23_14_22_06_CEST"
	     #"2021_06_23_14_42_07_CEST"
	     #"2021_06_23_15_02_08_CEST"
	     #"2021_06_23_15_22_09_CEST"
	     #"2021_06_23_15_42_10_CEST"
	     #"2021_06_23_16_02_11_CEST"
	     #"2021_06_23_16_22_12_CEST"
	     #"2021_06_23_16_42_13_CEST"
	     #"2021_06_23_17_02_14_CEST"
	     #"2021_06_23_17_22_15_CEST"
#	)

#TIMESTAMP_LIGHT=(
#	"20210623_124201"
	##"20210623_130207"
	#"20210623_132205"
	#"20210623_134209"
	#"20210623_140208"
	#"20210623_142208"
	#"20210623_144208"
	#"20210623_150209"
	#"20210623_152217"
        #"20210623_154232"
	#"20210623_160212"
	#"20210623_162213"
	#"20210623_164213"
	#"20210623_170215"
	#"20210623_172216"
#	)

# module 1
#TIMESTAMP_CHARGE=(
#            "2022_02_08_01_47_59_CET"
#            "2022_02_08_02_08_00_CET"
#            "2022_02_08_02_28_01_CET"
#            "2022_02_08_03_54_19_CET"
#            "2022_02_08_04_14_21_CET"
#            "2022_02_08_04_34_22_CET"
#            "2022_02_08_04_54_23_CET"
#            "2022_02_08_05_14_25_CET"
#            "2022_02_08_08_54_32_CET"
#            "2022_02_08_09_14_33_CET"
#            "2022_02_08_09_34_35_CET"
#            "2022_02_08_12_48_18_CET"
#            "2022_02_08_13_08_19_CET"
#            "2022_02_08_13_57_39_CET"
#            "2022_02_08_14_17_41_CET"
#            )
#TIMESTAMP_LIGHT=(
#	    "20220208_014759"
#            "20220208_020800"
#            "20220208_022802"
#            "20220208_035419"
#            "20220208_041421"
#            "20220208_043422"
#            "20220208_045424"
#            "20220208_051425"
#            "20220208_085432"
#            "20220208_091433"
#	    "20220208_093435"
#            "20220208_124818"
#            "20220208_130819"
#            "20220208_135739"
#            "20220208_141741"
#            )

# module 2
#TIMESTAMP_CHARGE=(
#        "2022_12_02_14_00_CET"
#        "2022_12_02_14_10_CET"
#        "2022_12_02_14_20_CET"
#        "2022_12_02_14_30_CET"
#        "2022_12_02_14_40_CET"
#        "2022_12_02_14_50_CET"
# 	"2022_12_02_15_00_CET"
#        "2022_12_02_15_10_CET"
#	"2022_12_02_15_20_CET"
#	"2022_12_02_15_30_CET"
#        "2022_12_02_15_50_CET"	
#	)  
#TIMESTAMP_LIGHT=(
#	"20221202_140028"
#	"20221202_141029"
#	"20221202_142031"
#	"20221202_143032"
#	"20221202_144034"
# 	"20221202_145035"
#	"20221202_150036"
#	"20221202_151038"
#	"20221202_152039"
#	"20221202_153041"
#	"20221202_155044"
#	)

# module 2 3000 threshold
#TIMESTAMP_CHARGE=(
#	"2022_11_30_04_57_CET"
#	"2022_11_30_05_17_CET"
#	"2022_11_30_05_37_CET"
#	"2022_11_30_05_57_CET"
#	"2022_11_30_06_37_CET"
 #       "2022_11_30_06_57_CET"
#	"2022_11_30_07_17_CET"
#	"2022_11_30_07_37_CET"
#	"2022_11_30_07_57_CET"
#	"2022_11_30_08_17_CET"
#	"2022_11_30_08_37_CET"
#	"2022_11_30_08_57_CET"
#	"2022_11_30_09_17_CET"
#	"2022_11_30_09_37_CET"
#	"2022_11_30_09_57_CET"
#	"2022_11_30_10_17_CET"
#	"2022_11_30_10_37_CET"
#	"2022_11_30_10_58_CET"
#	"2022_11_30_11_18_CET"
#	"2022_11_30_11_38_CET"
#	)
#TIMESTAMP_LIGHT=(
#	"20221130_045727"
#	"20221130_051728"
#	"20221130_053730"
#	"20221130_055731"
#	"20221130_063742"
#	"20221130_065744"
#	"20221130_071745"
#	"20221130_073747"
#	"20221130_075748"
#	"20221130_081749"
#	"20221130_083751"
#	"20221130_085752"
#	"20221130_091754"
#	"20221130_093755"
#	"20221130_095756"
#	"20221130_101758"
#	"20221130_103759"
#	"20221130_105800"
#	"20221130_111802"
#	"20221130_113803"
#	)

# module 3
#TIMESTAMP_CHARGE=(
#	"2023_03_14_09_34_CET"
#	"2023_03_14_09_44_CET"
#	"2023_03_14_09_54_CET"
#	"2023_03_14_10_04_CET"
#	"2023_03_14_10_14_CET"
#	"2023_03_14_10_24_CET"
#	"2023_03_14_10_34_CET"
#	"2023_03_14_10_44_CET"
#	"2023_03_14_10_54_CET"
#	"2023_03_14_11_04_CET"
#	"2023_03_14_11_15_CET"
#	"2023_03_14_11_25_CET"
#	"2023_03_14_11_35_CET"
#	"2023_03_14_11_45_CET"
#	"2023_03_14_11_55_CET"
#	"2023_03_14_12_05_CET"
#	"2023_03_14_12_15_CET"
#	"2023_03_14_12_25_CET"
#	"2023_03_14_12_35_CET"
#	"2023_03_14_12_45_CET"
#	"2023_03_14_12_55_CET"
#	"2023_03_14_13_05_CET"
#	"2023_03_14_13_15_CET"
#	"2023_03_14_13_25_CET"
#	"2023_03_14_13_35_CET"
#	"2023_03_14_13_45_CET"
#	"2023_03_14_13_55_CET"
#	)
#TIMESTAMP_LIGHT=(
#	"20230314_093427"
#	"20230314_094447"
#	"20230314_095434"
#	"20230314_100438"
#	"20230314_101441"
#	"20230314_102445"
#	"20230314_103448"
#	"20230314_104452"
#	"20230314_105455"
#	"20230314_110459"
#	"20230314_111502"
#	"20230314_112506"
#	"20230314_113509"
#	"20230314_114513"
#	"20230314_115516"
#	"20230314_120520"
#	"20230314_121523"
#	"20230314_122527"
#	"20230314_123530"
#	"20230314_124534"
#	"20230314_125537"
#	"20230314_130540"
#	"20230314_131544"
#	"20230314_132547"
#	"20230314_133551"
#	"20230314_134554"
#	"20230314_135558"
#	)

list_length=${#TIMESTAMP_CHARGE[@]}

cd ..
#rm -rf convert.venv
#python3 -m venv convert.venv
source convert.venv/bin/activate
#pip install --upgrade pip
#pip install -r requirements.txt

cd reco
for ((i = 0; i < list_length; i++)); do
    timestamp_charge="${TIMESTAMP_CHARGE[i]}"
    timestamp_light="${TIMESTAMP_LIGHT[i]}"
    INFILENAME_PACKETS=${data_folder_charge}/${PACKET_FILE_NAME_START}${timestamp_charge}.h5
    INFILENAME_LIGHT_1=${data_folder_light}/${ADC_1_NAME}_${timestamp_light}.data
    INFILENAME_LIGHT_2=${data_folder_light}/${ADC_2_NAME}_${timestamp_light}.data
    OUTFILENAME=${OUTDIR}/charge-light-matched-clusters_test_${timestamp_charge}.h5

    python3 full_reco.py ${INFILENAME_PACKETS} ${OUTFILENAME} ${INFILENAME_LIGHT_1} ${INFILENAME_LIGHT_2} --input_config_name ${DET}
done


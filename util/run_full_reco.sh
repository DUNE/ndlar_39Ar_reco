#!/usr/bin/env bash

DET="module2"
TEST=false
if [ "${DET}" = "module0_run1" ]; then
    OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module0_run1_reco/39Ar_reco
    ADC_1_NAME=0a7a314c
    ADC_2_NAME=0a7b54bd
    PACKET_FILE_NAME_START=datalog_
    data_folder_charge=/global/cfs/cdirs/dune/www/data/Module0/TPC1+2/dataRuns/packetData
    pedestal_folder=/global/cfs/cdirs/dune/www/data/Module0/TPC1+2/pedestalRuns/packetData/ldsTriggering
    data_folder_light=/global/cfs/cdirs/dune/www/data/Module0/LRS
    
    PEDESTAL_FILE="datalog_2021_04_04_09_31_43_CEST.h5"
    if [ ${TEST} = true ]; then
        TIMESTAMP_CHARGE=(
                "2021_04_04_09_38_27_CEST")
        TIMESTAMP_LIGHT=(
                "20210404_093828")
    else
        TIMESTAMP_CHARGE=(
                    "2021_04_04_09_38_27_CEST"
                    "2021_04_04_09_58_31_CEST"
                    "2021_04_04_10_38_37_CEST"
                    "2021_04_04_11_18_41_CEST"
                    "2021_04_04_11_38_43_CEST"
                    "2021_04_04_12_18_48_CEST"
                    "2021_04_04_13_18_54_CEST"
                    "2021_04_04_13_38_56_CEST"
                    "2021_04_04_14_19_01_CEST"
                    "2021_04_04_14_39_03_CEST"
                    "2021_04_04_15_19_07_CEST"
                    "2021_04_04_16_59_17_CEST"
                    "2021_04_04_17_19_19_CEST"
                    "2021_04_04_17_39_22_CEST"
                    "2021_04_04_17_59_24_CEST"
                    )
        TIMESTAMP_LIGHT=(
                    "20210404_093828"
                    "20210404_095833"
                    "20210404_103840"
                    "20210404_111841"
                    "20210404_113847"
                    "20210404_121849"
                    "20210404_131854"
                    "20210404_133856"
                    "20210404_141859"
                    "20210404_143903"
                    "20210404_151909"
                    "20210404_165920"
                    "20210404_171920"
                    "20210404_173921"
                    "20210404_175925"
                    )
    fi
elif [ "${DET}" = "module1" ]; then
    OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module1_reco/39Ar_reco_09132023
    ADC_1_NAME=0cd913fb
    ADC_2_NAME=0cd93db0
    PACKET_FILE_NAME_START=packet_
    data_folder_charge=/global/cfs/cdirs/dune/www/data/Module1/TPC12/dataRuns/packetData
    pedestal_folder=/global/cfs/cdirs/dune/www/data/Module1/TPC12/dataRuns/packetData
    data_folder_light=/global/cfs/cdirs/dune/www/data/Module1/LRS/SingleModule_Jan22
    
    PEDESTAL_FILE="packet_2022_02_08_01_40_31_CET.h5"
    if [ ${TEST} = true ]; then
        TIMESTAMP_CHARGE=(
             "2022_02_08_01_47_59_CET")
        TIMESTAMP_LIGHT=(
             "20220208_014759")
    else
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
    fi
elif [ "${DET}" = "module2" ]; then
    OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module2_reco/39Ar_reco
    ADC_2_NAME=0cd8d631 # LCM
    ADC_1_NAME=0cd913fa # ACL
    PACKET_FILE_NAME_START=self_trigger_tpc12_run2-packet-
    #PACKET_FILE_NAME_START=self_trigger_tpc12_run2_tile3-packet-
    #PACKET_FILE_NAME_START=selftrigger-run2-packet-
    data_folder_charge=/global/cfs/cdirs/dune/www/data/Module2/packetized/TPC12_run2
    pedestal_folder=/global/cfs/cdirs/dune/www/data/Module2/TPC12_run2
    data_folder_light=/global/cfs/cdirs/dune/www/data/Module2/LRS.2/cosmic
    
    PEDESTAL_FILE="pedestal_run2_tpc12_prc_6400_everything_on-packet-2022_12_02_16_57_CET.h5"
    if [ ${TEST} = true ]; then
        TIMESTAMP_CHARGE=(
            "2022_12_02_14_00_CET")
        TIMESTAMP_LIGHT=(
            "20221202_140028")
    else
        TIMESTAMP_CHARGE=(
            "2022_12_02_14_00_CET"
            "2022_12_02_14_10_CET"
            "2022_12_02_14_20_CET"
            "2022_12_02_14_30_CET"
            "2022_12_02_14_40_CET"
            "2022_12_02_14_50_CET"
            "2022_12_02_15_00_CET"
            "2022_12_02_15_10_CET"
            "2022_12_02_15_20_CET"
            "2022_12_02_15_30_CET"
            "2022_12_02_15_50_CET"
        )  
        TIMESTAMP_LIGHT=(
                "20221202_140028"
                "20221202_141029"
                "20221202_142031"
                "20221202_143032"
                "20221202_144034"
                "20221202_145035"
                "20221202_150036"
                "20221202_151038"
                "20221202_152039"
                "20221202_153041"
                "20221202_155044"
        )
    fi
elif [ "${DET}" = "module3" ]; then
    OUTDIR=/global/cfs/cdirs/dune/users/sfogarty/Module3_reco/39Ar_reco
    ADC_2_NAME=0cd8d631 # LCM
    ADC_1_NAME=0cd913fa # ACL
    #PACKET_FILE_NAME_START=self_trigger_tpc12_run2-packet-
    #PACKET_FILE_NAME_START=self_trigger_tpc12_run2_tile3-packet-
    PACKET_FILE_NAME_START=tpc12-packet-
    data_folder_charge=/global/cfs/cdirs/dune/www/data/Module3/run3/packet/tpc12
    pedestal_folder=/global/cfs/cdirs/dune/www/data/Module3/run3/packet/tpc12
    data_folder_light=/global/cfs/cdirs/dune/www/data/Module3/run3/LRS/cosmic/
    
    PEDESTAL_FILE="pedestal_HV_on_tpc12-packet-2023_03_15_22_16_CET.h5"
    if [ ${TEST} = true ]; then
        TIMESTAMP_CHARGE=(
            "2023_03_14_09_34_CET")
        TIMESTAMP_LIGHT=(
            "20230314_093427")
    else
        TIMESTAMP_CHARGE=(
            "2023_03_14_09_34_CET"
            "2023_03_14_09_44_CET"
            "2023_03_14_09_54_CET"
            "2023_03_14_10_04_CET"
            "2023_03_14_10_14_CET"
            "2023_03_14_10_24_CET"
            "2023_03_14_10_34_CET"
            "2023_03_14_10_44_CET"
            "2023_03_14_10_54_CET"
            "2023_03_14_11_04_CET"
            "2023_03_14_11_15_CET"
            "2023_03_14_11_25_CET"
            "2023_03_14_11_35_CET"
            "2023_03_14_11_45_CET"
            "2023_03_14_11_55_CET"
            "2023_03_14_12_05_CET"
            "2023_03_14_12_15_CET"
            "2023_03_14_12_25_CET"
            "2023_03_14_12_35_CET"
            "2023_03_14_12_45_CET"
            "2023_03_14_12_55_CET"
            "2023_03_14_13_05_CET"
            "2023_03_14_13_15_CET"
            "2023_03_14_13_25_CET"
            "2023_03_14_13_35_CET"
            "2023_03_14_13_45_CET"
            "2023_03_14_13_55_CET"
        )
        TIMESTAMP_LIGHT=(
            "20230314_093427"
            "20230314_094447"
            "20230314_095434"
            "20230314_100438"
            "20230314_101441"
            "20230314_102445"
            "20230314_103448"
            "20230314_104452"
            "20230314_105455"
            "20230314_110459"
            "20230314_111502"
            "20230314_112506"
            "20230314_113509"
            "20230314_114513"
            "20230314_115516"
            "20230314_120520"
            "20230314_121523"
            "20230314_122527"
            "20230314_123530"
            "20230314_124534"
            "20230314_125537"
            "20230314_130540"
            "20230314_131544"
            "20230314_132547"
            "20230314_133551"
            "20230314_134554"
            "20230314_135558"
        )
        fi
else
    echo "Exiting as $DET is not a recognized run name"
    exit 0
fi

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
    INFILENAME_PEDESTAL=${pedestal_folder}/${PEDESTAL_FILE}
    INFILENAME_LIGHT_1=${data_folder_light}/${ADC_1_NAME}_${timestamp_light}.data
    INFILENAME_LIGHT_2=${data_folder_light}/${ADC_2_NAME}_${timestamp_light}.data
    OUTFILENAME=${OUTDIR}/charge-light-matched-clusters_test_${timestamp_charge}.h5

    python3 full_reco.py ${INFILENAME_PACKETS} ${OUTFILENAME} ${INFILENAME_PEDESTAL} ${INFILENAME_LIGHT_1} ${INFILENAME_LIGHT_2} --input_config_name ${DET} 
done


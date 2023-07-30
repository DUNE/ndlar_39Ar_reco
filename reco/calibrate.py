import numpy as np
import json
from collections import defaultdict
from consts import *

def calibrations(packets, mc_assn, module):
    # unique id for each pixel, not to be confused with larnd-sim's pixel id
    unique_ids = ((((packets['io_group'].astype(int)) * 256 \
        + packets['io_channel'].astype(int)) * 256 \
        + packets['chip_id'].astype(int)) * 64 \
        + packets['channel_id'].astype(int)).astype(str)
    v_ped, v_cm, v_ref, gain = pedestal_and_config(unique_ids, mc_assn, module)
    return v_ped, v_cm, v_ref, gain, unique_ids

def adcs_to_ke(adcs, v_ref, v_cm, v_ped, gain):
    ### converts adc counts to charge in ke-
    # Inputs:
    #   adcs: array of packet ADC counts
    #   v_ref, v_cm, v_ped, gain: array of pixel calibration parameters
    # Outputs:
    #   array of charge in ke- 
    charge = (adcs.astype('float64')/float(ADC_COUNTS)*(v_ref - v_cm)+v_cm-v_ped)/gain * 1e-3
    return charge

def PACMAN_drift(packets, module):
    # only supports module-0
    ts = packets['timestamp'].astype('i8')
    mask_io1 = packets['io_group'] == 1
    mask_io2 = packets['io_group'] == 2
    ts[mask_io1] = (packets[mask_io1]['timestamp'].astype('i8') - module.PACMAN_clock_correction1[0]) / (1. + module.PACMAN_clock_correction1[1])
    ts[mask_io2] = (packets[mask_io2]['timestamp'].astype('i8') - module.PACMAN_clock_correction2[0]) / (1. + module.PACMAN_clock_correction2[1])
    return ts
    
def timestamp_corrector(packets, mc_assn, unix, module):
    # Corrects larpix clock timestamps due to slightly different PACMAN clock frequencies 
    # (from module0_flow timestamp_corrector.py)
    ts = packets['timestamp'].astype('i8')
    packet_type_0 = packets['packet_type'] == 0
    ts = ts[packet_type_0]
    packets = packets[packet_type_0]
    if mc_assn is not None:
        mc_assn = mc_assn[packet_type_0]
    if mc_assn is None and module.timestamp_cut:
        # cut needed due to noisy packets too close to PPS pulse
        # (unless this problem has been fixed in the hardware)
        timestamps = packets['timestamp']
        timestamp_data_cut = np.invert((timestamps > 2e7) | (timestamps < 1e6))
        ts = ts[timestamp_data_cut]
        packets = packets[timestamp_data_cut]
        unix = unix[timestamp_data_cut]
    if mc_assn is None and module.PACMAN_clock_correction:
        ts = PACMAN_drift(packets, module).astype('i8')
    
    return ts, packets, mc_assn, unix

def pedestal_and_config(unique_ids, mc_assn, module):
    # function to open the pedestal and configuration files to get the dictionaries
    #   for the values of v_ped, v_cm, and v_ref for individual pixels. 
    #   Values are fixed in simulation but vary in data depending on pixel.
    # Inputs:
    #   unique_ids: 
    #       note: array containing a unique id for each pixel
    #       size: same as packets dataset (after selections)
    #   mc_assn:
    #       note: mc_truth information for simulation (None for data)
    # Returns:
    #   v_ped, v_cm, v_ref, gain arrays; size of packets dataset

    config_dict = defaultdict(lambda: dict(
        vref_mv=1300,
        vcm_mv=288
    ))
    pedestal_dict = defaultdict(lambda: dict(
        pedestal_mv=580
    ))
    if module.use_ped_config_files:
        pedestal_file = module.pedestal_file
        config_file = module.config_file
        # reading the data from the file
        with open(pedestal_file,'r') as infile:
            for key, value in json.load(infile).items():
                pedestal_dict[key] = value

        with open(config_file, 'r') as infile:
            for key, value in json.load(infile).items():
                config_dict[key] = value
  
    v_ped,v_cm,v_ref,gain = np.zeros_like(unique_ids,dtype='float64'),np.zeros_like(unique_ids,dtype='float64'),np.zeros_like(unique_ids,dtype='float64'),np.ones_like(unique_ids,dtype='float64')

    # make arrays with values for v_ped,v_cm,v_ref, and gain for ADC to ke- conversion 
    # FIXME: Can we make this without the for loop?
    for i,id in enumerate(unique_ids):
        if mc_assn is None:
            v_ped[i] = pedestal_dict[id]['pedestal_mv']
            v_cm[i] = config_dict[id]['vcm_mv']
            v_ref[i] = config_dict[id]['vref_mv']
            gain[i] = gain_data
        else:
            v_ped[i] = v_pedestal_sim
            v_cm[i] = v_cm_sim
            v_ref[i] = v_ref_sim
            gain[i] = gain_sim
    
    return v_ped, v_cm, v_ref, gain

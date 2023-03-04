import numpy as np
import json
from collections import defaultdict
from consts import *

def calibrations(packets, mc_assn, detector):
    # unique id for each pixel, not to be confused with larnd-sim's pixel id
    unique_ids = ((((packets['io_group'].astype(int)) * 256
        + packets['io_channel'].astype(int)) * 256
        + packets['chip_id'].astype(int)) * 64 \
        + packets['channel_id'].astype(int)).astype(str)
    v_ped, v_cm, v_ref, gain = pedestal_and_config(unique_ids, mc_assn, detector)
    return v_ped, v_cm, v_ref, gain

def adcs_to_ke(adcs, v_ref, v_cm, v_ped, gain):
    ### converts adc counts to charge in ke-
    # Inputs:
    #   adcs: array of packet ADC counts
    #   v_ref, v_cm, v_ped, gain: array of pixel calibration parameters
    # Outputs:
    #   array of charge in ke- 
    charge = (adcs.astype('float64')/float(ADC_COUNTS)*(v_ref - v_cm)+v_cm-v_ped)/gain * 1e-3
    return charge

def PACMAN_drift(packets, detector):
    # only supports module-0
    ts = packets['timestamp'].astype('i8')
    if detector == 'module-0':
        correction1 = [-9.597, 4.0021e-6]
        correction2 = [-9.329, 1.1770e-6]
    elif detector == 'module-3':
        # assuming correction[0] is 0, certainly isn't exactly true
        correction1 = [0., 3.267e-6]
        correction2 = [0., -8.9467e-7]
    mask_io1 = packets['io_group'] == 1
    mask_io2 = packets['io_group'] == 2
    ts[mask_io1] = (packets[mask_io1]['timestamp'].astype('i8') - correction1[0]) / (1. + correction1[1])
    ts[mask_io2] = (packets[mask_io2]['timestamp'].astype('i8') - correction2[0]) / (1. + correction2[1])
    return ts
    
def timestamp_corrector(packets, mc_assn, unix, detector):
    # Corrects larpix clock timestamps due to slightly different PACMAN clock frequencies 
    # (from module0_flow timestamp_corrector.py)
    ts = packets['timestamp'].astype('i8')
    packet_type_0 = packets['packet_type'] == 0
    ts = ts[packet_type_0]
    packets = packets[packet_type_0]
    if mc_assn != None:
        mc_assn = mc_assn[packet_type_0]
    
    if mc_assn == None and timestamp_cut:
        # cut needed for module0 data, due to noisy packets too close to PPS pulse
        timestamp_data_cut = np.invert((packets['timestamp'] > 2e7) | (packets['timestamp'] < 1e6))
        ts = ts[timestamp_data_cut]
        packets = packets[timestamp_data_cut]
        unix = unix[timestamp_data_cut]
    
    if mc_assn == None and PACMAN_clock_correction:
        ts = PACMAN_drift(packets, detector).astype('i8')
    
    return ts, packets, mc_assn, unix

def pedestal_and_config(unique_ids, mc_assn, detector):
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
    if detector == 'module-0':
        pedestal_file = 'pedestal/module-0/datalog_2021_04_02_19_00_46_CESTevd_ped.json'
        config_file = 'config/module-0/evd_config_21-03-31_12-36-13.json'
    elif detector == 'module-3':
        pedestal_file = 'pedestal/module-3/pedestal-diagnostic-packet-2023_01_28_22_33_CETevd_ped.json'
        config_file = 'config/module-3/evd_config_23-01-29_11-12-16.json'

    config_dict = defaultdict(lambda: dict(
        vref_mv=1300,
        vcm_mv=288
    ))
    pedestal_dict = defaultdict(lambda: dict(
        pedestal_mv=580
    ))

    # reading the data from the file
    with open(pedestal_file,'r') as infile:
        for key, value in json.load(infile).items():
            pedestal_dict[key] = value

    with open(config_file, 'r') as infile:
        for key, value in json.load(infile).items():
            config_dict[key] = value
  
    v_ped,v_cm,v_ref,gain = np.zeros_like(unique_ids,dtype='float64'),np.zeros_like(unique_ids,dtype='float64'),np.zeros_like(unique_ids,dtype='float64'),np.ones_like(unique_ids,dtype='float64')

    # make arrays with values for v_ped,v_cm,v_ref, and gain for ADC to ke- conversion 
    for i,id in enumerate(unique_ids):
        if not mc_assn:
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

import numpy as np
import json
from collections import defaultdict
from consts import *

def calibrations(packets, mc_assn):
    # unique id for each pixel, not to be confused with larnd-sim's pixel id
    unique_ids = ((((packets['io_group'].astype(int)) * 256
        + packets['io_channel'].astype(int)) * 256
        + packets['chip_id'].astype(int)) * 64 \
        + packets['channel_id'].astype(int)).astype(str)
    v_ped, v_cm, v_ref, gain = pedestal_and_config(unique_ids, mc_assn)
    return v_ped, v_cm, v_ref, gain

def adcs_to_ke(adcs, v_ref, v_cm, v_ped, gain):
    ### converts adc counts to charge in ke-
    # Inputs:
    #   adcs: array of packet ADC counts
    #   v_ref, v_cm, v_ped, gain: array of pixel calibration parameters
    #   indices: array of indices
    # Outputs:
    #   array of charge in ke- 
    charge = (adcs.astype('float64')/float(ADC_COUNTS)*(v_ref - v_cm)+v_cm-v_ped)/gain * 1e-3
    return charge

def timestamp_corrector(packets, mc_assn):
    # Corrects larpix clock timestamps due to slightly different PACMAN clock frequencies 
    # (from module0_flow timestamp_corrector.py)
    ts = packets['timestamp'].astype('f8')

    if mc_assn == None:
        timestamp_cut = (packets['timestamp'] > 2e7) | (packets['timestamp'] < 1e6)
        ts = ts[np.invert(timestamp_cut)]
        packets = packets[np.invert(timestamp_cut)]

    if mc_assn == None:
        # only supports module-0
        correction1 = [-9.597, 4.0021e-6]
        correction2 = [-9.329, 1.1770e-6]
        mask_io1 = packets['io_group'] == 1
        mask_io2 = packets['io_group'] == 2
        ts[mask_io1] = (packets[mask_io1]['timestamp'].astype('f8') - correction1[0]) / (1. + correction1[1])
        ts[mask_io2] = (packets[mask_io2]['timestamp'].astype('f8') - correction2[0]) / (1. + correction2[1])

    # correct for timestamp rollovers (PPS)
    rollover_ticks = 1e7
    rollover_io1 = np.zeros(packets.shape[0], dtype = int)
    rollover_io2 = np.zeros(packets.shape[0], dtype = int)

    #Check for rollovers
    rollover_io1[(packets['io_group'] == 1) & (packets['packet_type'] == 6) & (packets['trigger_type'] == 83)] = rollover_ticks
    rollover_io2[(packets['io_group'] == 2) & (packets['packet_type'] == 6) & (packets['trigger_type'] == 83)] = rollover_ticks

    #Reset the rollover arrays
    rollover_io1 = np.cumsum(rollover_io1)
    rollover_io2 = np.cumsum(rollover_io2)
    
    #Apply the rollovers to ts
    ts[(packets['io_group'] == 1) & (packets['packet_type'] == 0) & (packets['receipt_timestamp'].astype(int) - packets['timestamp'].astype(int) < 0)] += rollover_io1[(packets['io_group'] == 1) & (packets['packet_type'] == 0) & (packets['receipt_timestamp'].astype(int) - packets['timestamp'].astype(int) < 0)] - rollover_ticks
    ts[(packets['io_group'] == 1) & (packets['packet_type'] == 0) & (packets['receipt_timestamp'].astype(int) - packets['timestamp'].astype(int) > 0)] += rollover_io1[(packets['io_group'] == 1) & (packets['packet_type'] == 0) & (packets['receipt_timestamp'].astype(int) - packets['timestamp'].astype(int) > 0)]
    ts[(packets['io_group'] == 2) & (packets['packet_type'] == 0) & (packets['receipt_timestamp'].astype(int) - packets['timestamp'].astype(int) < 0)] += rollover_io2[(packets['io_group'] == 2) & (packets['packet_type'] == 0) & (packets['receipt_timestamp'].astype(int) - packets['timestamp'].astype(int) < 0)] - rollover_ticks
    ts[(packets['io_group'] == 2) & (packets['packet_type'] == 0) & (packets['receipt_timestamp'].astype(int) - packets['timestamp'].astype(int) > 0)] += rollover_io2[(packets['io_group'] == 2) & (packets['packet_type'] == 0) & (packets['receipt_timestamp'].astype(int) - packets['timestamp'].astype(int) > 0)]
    
    packet_type_0 = packets['packet_type'] == 0
    ts = ts[packet_type_0]
    packets = packets[packet_type_0]
    
    sorted_idcs = np.argsort(ts)
    ts_corr_sorted = ts[sorted_idcs]
    packets_sorted = packets[sorted_idcs]
    return ts_corr_sorted, packets_sorted

def pedestal_and_config(unique_ids, mc_assn):
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

    pedestal_file = 'datalog_2021_04_02_19_00_46_CESTevd_ped.json'
    config_file = 'evd_config_21-03-31_12-36-13.json'

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

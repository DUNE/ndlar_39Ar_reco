import numpy as np
import json
from consts import *

def adcs_to_mV(adcs, v_ref, v_cm, v_ped):
    ### converts adc counts to charge in mV
    # Inputs:
    #   adcs: array of packet ADC counts
    #   v_ref, v_cm, v_ped, gain: array of pixel calibration parameters
    # Outputs:
    #   array of charge in mV
    charge = (adcs.astype('float64')/float(ADC_COUNTS)*(v_ref - v_cm)+v_cm-v_ped)
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
    packet_type_0 = (packets['packet_type'] == 0)# & (packets['valid_parity'] == 1)
    ts = ts[packet_type_0]
    packets = packets[packet_type_0]
    if mc_assn is not None:
        mc_assn = mc_assn[packet_type_0]
    if mc_assn is None and module.timestamp_cut:
        # cut needed due to noisy packets too close to PPS pulse
        # (unless this problem has been fixed in the hardware)
        timestamp_data_cut = np.invert((ts > 2e7) | (ts < 1e6))
        ts = ts[timestamp_data_cut]
        packets = packets[timestamp_data_cut]
        unix = unix[timestamp_data_cut]
    if mc_assn is None and module.PACMAN_clock_correction:
        ts = PACMAN_drift(packets, module).astype('i8')
    
    return ts, packets, mc_assn, unix

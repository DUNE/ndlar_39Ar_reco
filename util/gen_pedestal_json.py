import argparse
import time
import json
import h5py
import numpy as np
from tqdm import tqdm
import os

import larpix

_default_vdda = 1800
_default_vref_dac = 217
_default_vcm_dac = 71
_default_mean_trunc = 3

def adc2mv(adc, ref, cm, bits=8):
    return (ref-cm) * adc/(2**bits) + cm

def dac2mv(dac, max, bits=8):
    return max * dac/(2**bits)

def main(infile, vdda=_default_vdda, vref_dac=_default_vref_dac,
    vcm_dac=_default_vcm_dac, mean_trunc=_default_mean_trunc, **kwargs):

    f = h5py.File(infile,'r')
    packets = np.array(f['packets'])
    good_data_mask = (packets['valid_parity'] == 1) & (packets['packet_type'] == 0)
    packets = packets[good_data_mask]
    unique_id = ((packets['io_group'].astype(int)*256 \
        + packets['io_channel'].astype(int))*256 \
        + packets['chip_id'].astype(int))*64 \
        + packets['channel_id'].astype(int)
    unique_id_sort = np.argsort(unique_id)
    packets[:] = packets[unique_id_sort]
    unique_id = unique_id[unique_id_sort]
    
    # find start and stop indices for each occurrance of a unique id
    unique_id_set, start_indices = np.unique(unique_id, return_index=True)
    end_indices = np.roll(start_indices, shift=-1)
    end_indices[-1] = len(packets) - 1
    
    unique_id_chunk_indices = {}
    for val, start_idx, end_idx in zip(unique_id_set, start_indices, end_indices):
        unique_id_chunk_indices[val] = (start_idx, end_idx)

    config_dict = dict()
    dataword = packets['dataword']

    vref_mv = dac2mv(vref_dac,vdda)
    vcm_mv = dac2mv(vcm_dac,vdda)
    
    for unique in tqdm(unique_id_set, desc=' Channels'):
        start_index, stop_index = unique_id_chunk_indices[unique]
        adcs = dataword[start_index:stop_index]
        if len(adcs) < 1:
            continue
        vals,bins = np.histogram(adcs,bins=np.arange(257))
        peak_bin = np.argmax(vals)
        min_idx,max_idx = max(peak_bin-mean_trunc,0), min(peak_bin+mean_trunc,len(vals))
        ped_adc = np.average(bins[min_idx:max_idx]+0.5, weights=vals[min_idx:max_idx])

        config_dict[str(unique)] = dict(
            pedestal_mv=adc2mv(ped_adc,vref_mv,vcm_mv)
            )
    
    with open(os.path.basename(infile).strip('.h5')+'evd_ped.json','w') as fo:
        json.dump(config_dict, fo, sort_keys=True, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
        A script for generating pedestal configurations used by the to_evd_file.py
        script. To use, specify the ``controller_config`` that was used for the
        pedestal run, the path to the pedestal datafile, and the settings used
        for the pedestal run (vdda,vref,vcm). This script will then take the
        truncated mean for each channel's adc values and store them in pedestal
        config file.'''
        )
    parser.add_argument('--infile','-i',required=True,type=str)
    parser.add_argument('--vdda',default=_default_vdda,type=float,help='''default=%(default)s mV''')
    parser.add_argument('--vref_dac',default=_default_vref_dac,type=int,help='''default=%(default)s''')
    parser.add_argument('--vcm_dac',default=_default_vcm_dac,type=int,help='''default=%(default)s''')
    parser.add_argument('--mean_trunc',default=_default_mean_trunc,type=int,help='''
        default=%(default)s''')
    args = parser.parse_args()
    main(**vars(args))

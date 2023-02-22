#!/usr/bin/env python
"""
Command-line interface to LArNDLE
"""
from build_events import *
from preclustering import *
from light import *
import h5py
import fire
import time
import os
from tqdm import tqdm
from adc64format import dtypes, ADC64Reader

def run_reconstruction(input_packets_filename, output_events_filename, nSec_start, nSec_end):
    
    packets_filename_base = input_packets_filename.split('.h5')[0]
    if os.path.exists(output_events_filename):
        raise Exception('Output file '+ str(output_events_filename) + ' already exists.')
    if nSec_start <= 0 or nSec_start - int(nSec_start) or nSec_end <= 0 or nSec_end - int(nSec_end) > 0:
        raise ValueError('nSec_start and nSec_end must be greater than zero and be an integer.')
    if input_packets_filename.split('.')[-1] != 'h5':
        raise Exception('Input file must be an h5 file.')
        
    dict_path = 'layout/multi_tile_layout-2.3.16.pkl'
    pixel_xy = load_geom_dict(dict_path)
    print('Using pixel layout dictionary: ', dict_path)
    
    print('Opening packets file: ', input_packets_filename)
    f_packets = h5py.File(input_packets_filename)
    try:
        f_packets['packets']
    except: 
        raise KeyError('Packets not found in ' + input_packets_filename)
    
    analysis_start = time.time()
    # outputs: nhit, charge, time since file start, x (mm) of hits, y (mm) of hits, z (mm) of hits; for events
    mc_assn=None
    try:
        mc_assn = f_packets['mc_packets_assn']
    except:
        mc_assn=None
        print("No 'mc_packets_assn' dataset found, processing as real data.")
    
    packets = f_packets['packets']
    PPS_indices = np.where((packets['packet_type'] == 6) & (packets['trigger_type'] == 83))[0]

    print('Processing the first '+ str(nSec_end - nSec_start) + ' seconds of data, starting at '+\
         str(nSec_start) + ' seconds and stopping at ', str(nSec_end) + ' ...')
    for sec in tqdm(range(nSec_start,int(nSec_end)+1),desc=" Seconds Processed: "):
        # grab 1s at a time to analyze, plus the next 1s 
        if sec == 1:
            packets_1sec = packets[0:PPS_indices[sec-1]]
            packets_nextPPS = packets[PPS_indices[sec-1]:PPS_indices[sec]]
            if mc_assn != None:
                mc_assn_1sec = mc_assn[0:PPS_indices[sec-1]]
                mc_assn_nextPPS = mc_assn[PPS_indices[sec-1]:PPS_indices[sec]]
        else:
            packets_1sec = packets[PPS_indices[sec-2]:PPS_indices[sec-1]]
            packets_nextPPS = packets[PPS_indices[sec-1]:PPS_indices[sec]]
            if mc_assn != None:
                mc_assn_1sec = mc_assn[PPS_indices[sec-2]:PPS_indices[sec-1]]
                mc_assn_nextPPS = mc_assn[PPS_indices[sec-1]:PPS_indices[sec]]
        
        # remove packets from the 1sec that belongs in the previous second
        packets_1sec_receipt_diff_mask = (packets_1sec['receipt_timestamp'].astype(int) - packets_1sec['timestamp'].astype(int) < 0)\
                & (packets_1sec['packet_type'] == 0)
        packets_1sec = packets_1sec[np.invert(packets_1sec_receipt_diff_mask)]
        
        # move packets from nextPPS to 1sec that belong 1sec earlier
        packets_nextPPS_receipt_diff_mask = (packets_nextPPS['receipt_timestamp'].astype(int) - packets_nextPPS['timestamp'].astype(int) < 0) \
                & (packets_nextPPS['packet_type'] == 0)
        # move those packets from nextPPS to 1sec. Now we will only work on packets_1sec
        packets_1sec = np.concatenate((packets_1sec, packets_nextPPS[packets_nextPPS_receipt_diff_mask]))
        if mc_assn != None:
            mc_assn_1sec = mc_assn_1sec[np.invert(packets_1sec_receipt_diff_mask)]
            mc_assn_1sec = np.concatenate((mc_assn_1sec, mc_assn_nextPPS[packets_nextPPS_receipt_diff_mask]))
        else:
            mc_assn_1sec = None
        if sec == nSec_start:
            results_small_clusters, results_large_clusters, unix_pt7, PPS_pt7 = analysis(packets_1sec, pixel_xy, mc_assn_1sec)
        elif sec > nSec_start:
            results_small_clusters_temp, results_large_clusters_temp, unix_pt7_temp, PPS_pt7_temp = analysis(packets_1sec, pixel_xy, mc_assn_1sec)
            results_small_clusters = np.concatenate((results_small_clusters, results_small_clusters_temp))
            results_large_clusters = np.concatenate((results_large_clusters, results_large_clusters_temp))
            unix_pt7 = np.concatenate((unix_pt7, unix_pt7_temp))
            PPS_pt7 = np.concatenate((PPS_pt7, PPS_pt7_temp))
    
    # loop through the light files and only select triggers within the second ranges specified
    # note that not all these light events will have a match within the packets data selection
    print('Loading light files with a batch size of ', batch_size, ' ...')
    events_314C_all, events_54BD_all = read_light_files(0, nSec_end)
    # match light triggers to packet type 7's
    matched_triggers_314C, indices_314C = match(events_314C_all, PPS_pt7, unix_pt7, '314C')
    matched_triggers_54BD, indices_54BD = match(events_54BD_all, PPS_pt7, unix_pt7, '54BD')
    # indices of light events to keep
    light_event_indices_314C = np.where(indices_314C != -1)[0]
    light_event_indices_54BD = np.where(indices_54BD != -1)[0]
    # indices of pkt type 7's for each light event
    indices_314C = indices_314C[indices_314C != -1]
    indices_54BD = indices_54BD[indices_54BD != -1]
    #print('index in pkt 7 array 314C = ', indices_314C[0:20])
    #print('index in pkt 7 array 54BD = ', indices_54BD[0:20])
    # exclude events where one ADC is not matched to charge data (probably won't happen?)
    charge_indices = np.intersect1d(indices_314C, indices_54BD) 
    light_event_indices = np.intersect1d(light_event_indices_314C, light_event_indices_54BD)
    print(unix_pt7)

    print('Saving events to ', output_events_filename)
    with h5py.File(output_events_filename, 'w') as f:
        dset_small_clusters = f.create_dataset('small_clusters', data=results_small_clusters, dtype=results_small_clusters.dtype)
        dset_large_clusters = f.create_dataset('large_clusters', data=results_large_clusters, dtype=results_large_clusters.dtype)
    
    analysis_end = time.time()
    print('Time to do full analysis = ', analysis_end-analysis_start, ' seconds')
    print('Total small clusters = ', len(results_small_clusters), ' with a rate of ', len(results_small_clusters)/(nSec_end-nSec_start), ' Hz')
    print('Total large clusters = ', len(results_large_clusters), ' with a rate of ', len(results_large_clusters)/(nSec_end - nSec_start), ' Hz')

if __name__ == "__main__":
    fire.Fire(run_reconstruction)
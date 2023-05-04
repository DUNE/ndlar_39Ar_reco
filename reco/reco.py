#!/usr/bin/env python
"""
Command-line interface to LArNDLE
"""
from build_events import *
from preclustering import *
import charge_event_cuts
import matching
from light import *
import h5py
import fire
import time
import os
from tqdm import tqdm
from adc64format import dtypes, ADC64Reader
import importlib.util

def reco_loop(nSec_start, nSec_end, PPS_indices, packets,\
              mc_assn, tracks, pixel_xy, detector, hits_clusters_start_cindex):
    ## loop through seconds of data and do charge reconstruction
    for sec in tqdm(range(nSec_start,int(nSec_end)+1),desc=" Seconds Processed: "):
        # Grab 1s at a time to analyze, plus the next 1s.
        # Each 1s is determined by getting the packets between PPS pulses (packet type 6).
        if sec == 1:
            packets_1sec = packets[0:PPS_indices[sec-1]]
            packets_nextPPS = packets[PPS_indices[sec-1]:PPS_indices[sec]]
        elif sec >= nSec_start and sec <= nSec_end:
            packets_1sec = packets[PPS_indices[sec-2]:PPS_indices[sec-1]]
            packets_nextPPS = packets[PPS_indices[sec-1]:PPS_indices[sec]]
        
        # remove packets from the 1sec that belongs in the previous second
        packets_1sec_receipt_diff_mask = (packets_1sec['receipt_timestamp'].astype(int) - packets_1sec['timestamp'].astype(int) < 0)\
                & (packets_1sec['packet_type'] == 0)
        packets_1sec = packets_1sec[np.invert(packets_1sec_receipt_diff_mask)]
        
        # move packets from nextPPS to 1sec that belong 1sec earlier
        packets_nextPPS_receipt_diff_mask = (packets_nextPPS['receipt_timestamp'].astype(int) - packets_nextPPS['timestamp'].astype(int) < 0) \
                & (packets_nextPPS['packet_type'] == 0)
        # move those packets from nextPPS to 1sec. Now we will only work on packets_1sec
        packets_1sec = np.concatenate((packets_1sec, packets_nextPPS[packets_nextPPS_receipt_diff_mask]))
        # run reconstruction on selected packets.
        # this block is be run first and thus defines all the arrays for concatenation later.
        if sec == nSec_start:
            results_clusters, unix_pt7, PPS_pt7,\
                hits_clusters = analysis(packets_1sec, pixel_xy,\
                mc_assn, tracks, detector, hits_clusters_start_cindex,sec)
        elif sec > nSec_start:
            # making sure to continously increment cluster_index as we go onto the next PPS
            if np.size(hits_clusters['cluster_index']) > 0:
                hits_clusters_max_cindex = np.max(hits_clusters['cluster_index'])+1
            else:
                hits_clusters_max_cindex = 0
            # run reconstruction and save temporary arrays of results
            results_clusters_temp, unix_pt7_temp, PPS_pt7_temp,\
                hits_clusters_temp\
                = analysis(packets_1sec, pixel_xy, mc_assn, tracks, detector,\
                                        hits_clusters_max_cindex,sec)
            # concatenate temp arrays to main arrays
            results_clusters = np.concatenate((results_clusters, results_clusters_temp))
            hits_clusters = np.concatenate((hits_clusters, hits_clusters_temp))
            unix_pt7 = np.concatenate((unix_pt7, unix_pt7_temp))
            PPS_pt7 = np.concatenate((PPS_pt7, PPS_pt7_temp))
    hits_clusters_max_cindex = np.max(hits_clusters['cluster_index'])+1
    return results_clusters,unix_pt7,PPS_pt7, hits_clusters, hits_clusters_max_cindex

def reco_MC(packets, mc_assn, tracks, pixel_xy, detector):
    results_clusters, unix_pt7, PPS_pt7,\
        hits_clusters = analysis(packets, pixel_xy, mc_assn, tracks, detector, 0, 0)
    return results_clusters, unix_pt7,PPS_pt7, hits_clusters

def run_reconstruction(input_config_filename):
    ## main function
    
    # import input variables. Get variables with module.<variable>
    input_config_filepath = 'input_config/' + input_config_filename
    module_name = "detector_module"
    spec = importlib.util.spec_from_file_location(module_name, input_config_filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # set variables from config file
    detector = module.detector
    input_packets_filename = module.input_packets_filename
    input_light_filename_1 = module.input_light_filename_1
    input_light_filename_2 = module.input_light_filename_2
    output_events_filename = module.output_events_filename
    nSec_start = module.nSec_start_packets
    nSec_end = module.nSec_end_packets
    nSec_start_light = module.nSec_start_light
    nSec_end_light = module.nSec_end_light
    sync_filename = module.sync_filename
    light_time_steps = module.light_time_steps
    nchannels_adc1 = module.nchannels_adc1
    nchannels_adc2 = module.nchannels_adc2
    matching_tolerance_unix = module.matching_tolerance_unix
    matching_tolerance_PPS = module.matching_tolerance_PPS
    
    input_packets_filepath = input_packets_filename
    output_events_filepath = output_events_filename
    
    if detector not in ['module-0','module-1', 'module-3']:
        raise Exception("Possible values of 'detector' are only 'module-0', 'module-1', and 'module-3' (without quotes).")
    if os.path.exists(output_events_filepath):
        raise Exception('Output file '+ str(output_events_filepath) + ' already exists.')
    if nSec_start <= 0 or nSec_start - int(nSec_start) or nSec_end < -1 or nSec_end - int(nSec_end) > 0:
        raise ValueError('nSec_start and nSec_end must be greater than zero and be an integer.')
    if input_packets_filename.split('.')[-1] != 'h5':
        raise Exception('Input file must be an h5 file.')
    
    # load dictionary for calculating hit position on pixel plane. Made using larpix-readout-parser.
    if detector == 'module-0':
        dict_path = 'layout/module-0/multi_tile_layout-2.3.16.pkl'
    elif detector == 'module-1':
        dict_path = 'layout/module-1/module1_layout-2.3.16.pkl'
    elif detector == 'module-3':
        dict_path = 'layout/module-3/multi_tile_layout-module3.pkl'
    pixel_xy = load_geom_dict(dict_path)
    print('Using pixel layout dictionary: ', dict_path)
    
    # open packets file
    print('Opening packets file: ', input_packets_filepath)
    f_packets = h5py.File(input_packets_filepath)
    try:
        f_packets['packets']
    except: 
        raise KeyError('Packets not found in ' + input_packets_filepath)
    
    analysis_start = time.time()
    
    # open mc_assn dataset for MC
    mc_assn=None
    tracks=None
    try:
        mc_assn = f_packets['mc_packets_assn']
        tracks = f_packets['tracks']
    except:
        mc_assn=None
        print("No 'mc_packets_assn' dataset found, processing as real data.")
    
    # get packets and indices of PPS pulses
    packets = f_packets['packets']
    
    if nSec_end == -1 and mc_assn is None:
        print('nSec_end was set to -1, so processing entire file.')
    
    # run reconstruction
    if mc_assn is None:
        hits_clusters_start_cindex = 0
        io_groups = np.unique(packets['io_group'])
        for io in tqdm(io_groups, desc = 'Processing io_groups: '):
            packets_io = packets[packets['io_group'] == io]
            PPS_indices = np.where((packets_io['packet_type'] == 6) & (packets_io['trigger_type'] == 83))[0]
            if nSec_end == -1:
                nSec_end = len(PPS_indices)-1
                nSec_end_light = nSec_end
            elif nSec_end > len(PPS_indices)-1:
                nSec_end = len(PPS_indices)-1
                nSec_end_light = nSec_end
                print('Note: nSec_end is set greater than the total seconds in file, ', nSec_end, ', so processing entire file.')
            if nSec_start > len(PPS_indices)-1:
                raise ValueError('nSec_start is greater than possible values of seconds. Set nSec_start to be smaller.')
            if io == io_groups[0]:
                results_clusters, unix_pt7, \
                PPS_pt7, hits_clusters,\
                hits_clusters_start_cindex = \
                reco_loop(nSec_start, nSec_end, PPS_indices, packets_io, mc_assn, tracks, pixel_xy, detector,\
                    hits_clusters_start_cindex)
            else:
                results_clusters_temp, unix_pt7_temp, \
                    PPS_pt7_temp, hits_clusters_temp,\
                    hits_clusters_start_cindex = \
                    reco_loop(nSec_start, nSec_end, PPS_indices, packets_io, mc_assn, tracks, pixel_xy, detector,\
                        hits_clusters_start_cindex)
                results_clusters = np.concatenate((results_clusters, results_clusters_temp))
                unix_pt7 = np.concatenate((unix_pt7, unix_pt7_temp))
                PPS_pt7 = np.concatenate((PPS_pt7, PPS_pt7_temp))
                hits_clusters = np.concatenate((hits_clusters, hits_clusters_temp))
    else:
        results_clusters, unix_pt7, PPS_pt7, hits_clusters = \
            reco_MC(packets, mc_assn, tracks, pixel_xy, detector)
    # do cuts on charge events. See toggles for cuts in consts.py.
    # if all toggles are False then this command simply returns `results` unchanged.
    #results_small_clusters = charge_event_cuts.all_charge_event_cuts(results_small_clusters)
    
    if do_match_of_charge_to_light and mc_assn is None:
        # loop through the light files and only select triggers within the second ranges specified
        # note that not all these light events will have a match within the packets data selection
        input_light_filepath_1 = adc_folder + detector + '/' + input_light_filename_1
        input_light_filepath_2 = adc_folder + detector + '/' +  input_light_filename_2
        adc_sn_1 = input_light_filename_1.split('_')[0]
        adc_sn_2 = input_light_filename_2.split('_')[0]
        print('Using light file ', input_light_filepath_1)
        print('Using light file ', input_light_filepath_2)
        print('Loading light files with a batch size of ', batch_size, ' ...')
        light_events_all = read_light_files(input_light_filepath_1, input_light_filepath_2, nSec_start_light, nSec_end_light, detector, light_time_steps, nchannels_adc1, nchannels_adc2,adc_sn_1, adc_sn_2)
    
        # match light events to ext triggers in packets (packet type 7)
        light_events_all = light_events_all[light_events_all['unix'] != 0]
        print('Matching light triggers to external triggers in packets...')
        indices_in_ext_triggers = matching.match_light_to_ext_trigger(light_events_all, PPS_pt7, unix_pt7, \
            matching_tolerance_unix, matching_tolerance_PPS) # length of light_events_all
        evt_mask = indices_in_ext_triggers != -1
        indices_in_ext_triggers = indices_in_ext_triggers[evt_mask]
        light_events_all = light_events_all[evt_mask]
        PPS_pt7_light = PPS_pt7[indices_in_ext_triggers]
        unix_pt7_light = unix_pt7[indices_in_ext_triggers]
        
        # match ext triggers / light events to charge events for the large clusters
        print('Performing charge-light matching for the charge clusters...')
        results_clusters, results_clusters_light_events = matching.match_light_to_charge(light_events_all, results_clusters, PPS_pt7_light, unix_pt7_light)
    else:
        print('Skipping getting light events and doing charge-light matching...')
    
    print('Saving events to ', output_events_filepath)
    with h5py.File(output_events_filepath, 'w') as f:
        dset_hits_clusters = f.create_dataset('clusters_hits', data=hits_clusters, dtype=hits_clusters.dtype)
        if do_match_of_charge_to_light:
            dset_light_events = f.create_dataset('clusters_matched_light', data=results_clusters_light_events, dtype=results_clusters_light_events.dtype)
        dset_clusters = f.create_dataset('clusters', data=results_clusters, dtype=results_clusters.dtype)
    
    analysis_end = time.time()
    print('Time to do full analysis = ', analysis_end-analysis_start, ' seconds')

if __name__ == "__main__":
    fire.Fire(run_reconstruction)

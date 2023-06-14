#!/usr/bin/env python
"""
Command-line interface to LArNDLE
"""
from build_events import *
from preclustering import *
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
              mc_assn, tracks, pixel_xy, module, hits_clusters_start_cindex):
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
            results_clusters, unix_pt7, PPS_pt7, io_group_pt7_temp,\
                hits_clusters = analysis(packets_1sec, pixel_xy,\
                mc_assn, tracks, module, hits_clusters_start_cindex,sec)
        elif sec > nSec_start:
            # making sure to continously increment cluster_index as we go onto the next PPS
            if np.size(hits_clusters['cluster_index']) > 0:
                hits_clusters_max_cindex = np.max(hits_clusters['cluster_index'])+1
            else:
                hits_clusters_max_cindex = 0
            # run reconstruction and save temporary arrays of results
            results_clusters_temp, unix_pt7_temp, PPS_pt7_temp, io_group_pt7_temp,\
                hits_clusters_temp\
                = analysis(packets_1sec, pixel_xy, mc_assn, tracks, module,\
                                        hits_clusters_max_cindex,sec)
            # concatenate temp arrays to main arrays
            results_clusters = np.concatenate((results_clusters, results_clusters_temp))
            hits_clusters = np.concatenate((hits_clusters, hits_clusters_temp))
            unix_pt7 = np.concatenate((unix_pt7, unix_pt7_temp))
            PPS_pt7 = np.concatenate((PPS_pt7, PPS_pt7_temp))
            io_group_pt7 = np.concatenate((io_group_pt7, io_group_pt7_temp))
    hits_clusters_max_cindex = np.max(hits_clusters['cluster_index'])+1
    return results_clusters, unix_pt7, PPS_pt7, io_group_pt7, hits_clusters, hits_clusters_max_cindex

def reco_MC(packets, mc_assn, tracks, pixel_xy, module):
    results_clusters, unix_pt7, PPS_pt7, io_group_pt7,\
        hits_clusters = analysis(packets, pixel_xy, mc_assn, tracks, module, 0, 0)
    return results_clusters, unix_pt7, PPS_pt7, io_group_pt7, hits_clusters

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
    data_type = module.data_type
    
    if data_type == 'data':
        nSec_start = module.nSec_start_packets
        nSec_end = module.nSec_end_packets
        nSec_start_light = module.nSec_start_light
        nSec_end_light = module.nSec_end_light
    elif data_type == 'MC':
        pass
    else:
        raise ValueError(f'Data type {data_type} not supported. Choose data or MC.')
        
    # do various file / parameter checks
    if os.path.exists(module.output_events_filename):
        raise Exception('Output file '+ str(module.output_events_filename) + ' already exists.')
    if not os.path.exists(module.detector_dict_path):
        raise Exception(f'Dictionary file {module.detector_dict_path} does not exist.')
    else:
        print('Using pixel layout dictionary: ', module.detector_dict_path)
    if not os.path.exists(module.input_packets_filename):
        raise Exception(f'Packets file {module.input_packets_filename} does not exist.')
    else:
        print('Opening packets file: ', module.input_packets_filename)
    if module.input_packets_filename.split('.')[-1] != 'h5':
        raise Exception('Input file must be an h5 file.')
    if data_type == 'data':
        if nSec_start <= 0 or nSec_start - int(nSec_start) or nSec_end < -1 or nSec_end - int(nSec_end) > 0:
            raise ValueError('nSec_start and nSec_end must be greater than zero and be an integer.')
        if module.use_disabled_channels_list:
            if not os.path.exists(module.disabled_channels_list):
                raise Exception(f'Disabled channels file {module.disabled_channels_list} does not exist.')
            elif os.path.exists(module.disabled_channels_list) and module.use_disabled_channels_list:
                print('Using disabled channels list: ', module.disabled_channels_list)
        else:
            pass
    if module.do_match_of_charge_to_light and data_type == 'data':
        if not os.path.exists(module.adc_folder + module.input_light_filename_1):
            raise Exception(f'Input light file {module.adc_folder + module.input_light_filename_1} does not exist.')
        if not os.path.exists(module.adc_folder + module.input_light_filename_2):
            raise Exception(f'Input light file {module.adc_folder + module.input_light_filename_2} does not exist.')
    # detector dict file must be pkl file made with larpix_readout_parser
    pixel_xy = load_geom_dict(module.detector_dict_path)
    
    # open packets file
    f = h5py.File(module.input_packets_filename)
    try:
        f['packets']
    except: 
        raise KeyError('Packets not found in ' + module.input_packets_filename)
    
    analysis_start = time.time()
    
    # open mc_assn dataset for MC
    mc_assn=None
    tracks=None
    try:
        mc_assn = f['mc_packets_assn']
        tracks = f['tracks']
    except:
        print("No 'mc_packets_assn' dataset found, processing as real data.")
    
    # get packets and indices of PPS pulses
    packets = f['packets']
    
    if data_type == 'data':
        if nSec_end == -1:
            print('nSec_end was set to -1, so processing entire file.')
        
        if module.use_disabled_channels_list:
            disabled_channels = np.load(module.disabled_channels_list)
            keys = disabled_channels['keys']

            unique_ids_packets = ((((packets['io_group'].astype(int)) * 256 \
                + packets['io_channel'].astype(int)) * 256 \
                + packets['chip_id'].astype(int)) * 64 \
                + packets['channel_id'].astype(int)).astype(int)

            nonType0_mask = packets['packet_type'] != 0
            unique_ids_packets[nonType0_mask] = -1 # just to make sure we don't remove non data packets

            print('Removing noisy packets...')
            packets_to_keep_mask = np.isin(unique_ids_packets, keys, invert=True)
            packets = packets[packets_to_keep_mask]
            print('Finished. Removed ', 100 - np.sum(packets['packet_type'] == 0)/np.sum(np.invert(nonType0_mask)) * 100, ' % of data packets.')
    
    # run reconstruction
    io_groups = np.unique(packets['io_group'])
    if mc_assn is None or data_type == 'data':
        hits_clusters_start_cindex = 0
        
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
                print('len(PPS_indices)-1:',len(PPS_indices)-1)
                raise ValueError('nSec_start is greater than possible values of seconds. Set nSec_start to be smaller.')
            if io == io_groups[0]:
                results_clusters, unix_pt7, \
                PPS_pt7, io_group_pt7, hits_clusters,\
                hits_clusters_start_cindex = \
                reco_loop(nSec_start, nSec_end, PPS_indices, packets_io, mc_assn, tracks, pixel_xy, module,\
                    hits_clusters_start_cindex)
            else:
                results_clusters_temp, unix_pt7_temp, \
                    PPS_pt7_temp, io_group_pt7_temp, hits_clusters_temp,\
                    hits_clusters_start_cindex = \
                    reco_loop(nSec_start, nSec_end, PPS_indices, packets_io, mc_assn, tracks, pixel_xy, module,\
                        hits_clusters_start_cindex)
                results_clusters = np.concatenate((results_clusters, results_clusters_temp))
                unix_pt7 = np.concatenate((unix_pt7, unix_pt7_temp))
                PPS_pt7 = np.concatenate((PPS_pt7, PPS_pt7_temp))
                io_group_pt7 = np.concatenate((io_group_pt7, io_group_pt7_temp))
                hits_clusters = np.concatenate((hits_clusters, hits_clusters_temp))
    else:
        results_clusters, unix_pt7, PPS_pt7, io_group_pt7, hits_clusters = \
            reco_MC(packets, mc_assn, tracks, pixel_xy, module)
    
    if module.do_match_of_charge_to_light:
        # loop through the light files and only select triggers within the second ranges specified
        # note that not all these light events will have a match within the packets data selection
        
        if mc_assn is None:
            light_events_all = read_light_files(module)
        else:
            # get light triggers from larnd-sim file
            wvfms = f['light_wvfm']
            wvfms_shape = wvfms.shape
            light_MC_dtype = np.dtype([('op_channel', '<i4', (wvfms_shape[1],)), ('unix', '<f8'), ('tai_ns', '<u8'),\
                                       ('waveforms','<i2',(wvfms_shape[1], wvfms_shape[2]))])
            light_events_all = np.zeros((0,), dtype=light_MC_dtype)
            for i in range(len(f['light_trig'])):
                light_event = np.zeros((1,), dtype=light_MC_dtype)
                light_event['op_channel'] = f['light_trig']['op_channel'][i]
                light_event['unix'] = f['light_trig']['ts_s'][i]
                light_event['tai_ns'] = f['light_trig']['ts_sync'][i]
                light_event['waveforms'] = f['light_wvfm'][i]
                light_events_all = np.concatenate((light_events_all, light_event))
            
        # match light events to ext triggers in packets (packet type 7)
        light_events_all = light_events_all[light_events_all['unix'] != 0]
        with open(module.detprop_path) as df:
            detprop = yaml.load(df, Loader=yaml.FullLoader)
        module_to_io_group = detprop['module_to_io_groups']
        #io_group_to_module = {value: key for key, values in module_to_io_group.items() for value in values}
        starting_light_index = 0
        for k in tqdm(range(int(len(io_groups)/2)), desc = ' Performing matching for each module: '):
            i,j = 0,0
            tpc_to_op_channel = detprop['tpc_to_op_channel']
            # get relationship between module to op channel
            module_to_op_channel = np.zeros((int(len(tpc_to_op_channel)/2), len(tpc_to_op_channel[0]*2)))
            while i < len(tpc_to_op_channel):
                module_to_op_channel[j] = np.concatenate((tpc_to_op_channel[i], tpc_to_op_channel[i+1]))
                j += 1
                i += 2
            
            # get light trigger corresponding to the module in question at this point in the loop
            light_dtype = light_events_all.dtype
            indices = np.searchsorted(module_to_op_channel[:,0], light_events_all['op_channel'][:,0]) == k
            #this_module_light_events = np.array(np.argsort(light_events_all)[indices], dtype=light_dtype)
            print("light_events_all['unix'][indices]: ", light_events_all['unix'][indices])
            print("light_events_all['tai_ns'][indices]: ", light_events_all['tai_ns'][indices])
            this_module_light_events = light_events_all[indices]
            print("this_module_light_events['unix']: ", this_module_light_events['unix'])
            print("this_module_light_events['tai_ns']: ", this_module_light_events['tai_ns'])
            #print('this_module_light_events: ', this_module_light_events[0])
            # get PPS_pt7 and unix_pt7 values corresponding to module in question
            list_of_io = module_to_io_group[k+1]
            this_module_io_mask = (io_group_pt7 == list_of_io[0]) | (io_group_pt7 == list_of_io[1])
            this_module_PPS_pt7 = PPS_pt7[this_module_io_mask]
            this_module_unix_pt7 = unix_pt7[this_module_io_mask]
            
            # find charge clusters corresponding to module in question
            results_clusters_mask = (results_clusters['io_group'] == list_of_io[0]) & (results_clusters['io_group'] == list_of_io[1])
            
            print('Matching light triggers to external triggers in packets...')
            indices_in_ext_triggers = matching.match_light_to_ext_trigger(this_module_light_events, this_module_PPS_pt7, this_module_unix_pt7, module)
            evt_mask = indices_in_ext_triggers != -1
            indices_in_ext_triggers = indices_in_ext_triggers[evt_mask]
            light_events_all = this_module_light_events[evt_mask]
            this_module_PPS_pt7 = this_module_PPS_pt7[indices_in_ext_triggers]
            this_module_unix_pt7 = this_module_unix_pt7[indices_in_ext_triggers]

            # match ext triggers / light events to charge events for the large clusters
            print(f'Performing charge-light matching in module #{k+1}...')
            this_module_light_events = matching.match_light_to_charge(this_module_light_events, results_clusters[results_clusters_mask], this_module_PPS_pt7, this_module_unix_pt7, module, starting_light_index)
            if k == 0:
                results_clusters_light_events = this_module_light_events
            else:
                results_clusters_light_events = np.concatenate((results_clusters_light_events, this_module_light_events))
            starting_light_index = len(this_module_light_events)
            
    else:
        print('Skipping getting light events and doing charge-light matching...')
    
    print('Saving clusters and light data to ', module.output_events_filename)
    with h5py.File(module.output_events_filename, 'w') as f_out:
        dset_hits_clusters = f_out.create_dataset('clusters_hits', data=hits_clusters, dtype=hits_clusters.dtype)
        if module.do_match_of_charge_to_light:
            dset_light_events = f_out.create_dataset('clusters_matched_light', data=results_clusters_light_events, dtype=results_clusters_light_events.dtype)
        dset_clusters = f_out.create_dataset('clusters', data=results_clusters, dtype=results_clusters.dtype)
    
    analysis_end = time.time()
    print('Time to do full analysis = ', analysis_end-analysis_start, ' seconds')

if __name__ == "__main__":
    fire.Fire(run_reconstruction)

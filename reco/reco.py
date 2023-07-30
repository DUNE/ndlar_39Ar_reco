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
import math

def run_reconstruction(input_config_filename, input_filepath=None, output_filepath=None):
    ## main function
    
    # Import input variables file. Get variables with module.<variable>
    input_config_filepath = 'input_config/' + input_config_filename
    module_name = "detector_module"
    spec = importlib.util.spec_from_file_location(module_name, input_config_filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # set some variables from config file
    detector = module.detector
    data_type = module.data_type
    
    if input_filepath is not None:
        input_packets_filename = input_filepath
    else:
        input_packets_filename = module.input_packets_filename

    if output_filepath is not None:
        output_events_filename = output_filepath
    else:
        output_events_filename = module.output_events_filename
    
    # do various file / parameter checks
    if os.path.exists(output_events_filename):
        raise Exception('Output file '+ str(output_events_filename) + ' already exists.')
    if not os.path.exists(module.detector_dict_path):
        raise Exception(f'Dictionary file {module.detector_dict_path} does not exist.')
    else:
        print('Using pixel layout dictionary: ', module.detector_dict_path)
    if not os.path.exists(input_packets_filename):
        raise Exception(f'Packets file {input_packets_filename} does not exist.')
    else:
        print('Opening packets file: ', input_packets_filename)
    if input_packets_filename.split('.')[-1] != 'h5':
        raise Exception('Input file must be an h5 file.')
    if data_type == 'data':
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
    
    # detector dictionary file must be pkl file made with larpix_readout_parser
    pixel_xy = load_geom_dict(module.detector_dict_path)
    
    # open packets file
    f = h5py.File(input_packets_filename)
    try:
        # get packets
        packets = f['packets']
    except: 
        raise KeyError('Packets not found in ' + input_packets_filename)
    
    # open mc_assn dataset for MC
    mc_assn, tracks = None, None
    try:
        mc_assn = f['mc_packets_assn']
        tracks = f['segments']
    except:
        print("No 'mc_packets_assn' dataset found, processing as real data.")
    
    analysis_start = time.time()
    # load disabled channels npz file (for e.g. excluding noisy channels)
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
    
    nBatches = 200
    # run reconstruction
    hits_max_cindex = 0
    batch_size = math.ceil(len(packets)/nBatches)
    index_start = 0
    index_end = batch_size

    for i in tqdm(range(nBatches), desc = 'Processing batches...'):
        packets_batch = packets[index_start:index_end]
        if mc_assn is not None:
            mc_assn = mc_assn[index:index_end]
        clusters, unix_pt7, PPS_pt7, hits = \
            analysis(packets_batch, pixel_xy, mc_assn, tracks, module, hits_max_cindex)
        # making sure to continously increment cluster_index as we go onto the next batch
        if np.size(hits['cluster_index']) > 0:
            hits_max_cindex = np.max(hits['cluster_index'])+1
        if i == 0:
            # create the hdf5 datasets with initial results
            with h5py.File(output_events_filename, 'a') as output_file:
                output_file.create_dataset('clusters', data=clusters, maxshape=(None,))
                output_file.create_dataset('hits', data=hits, maxshape=(None,))
                output_file.create_dataset('ext_trig_unix', data=unix_pt7, maxshape=(None,))
                output_file.create_dataset('ext_trig_PPS', data=PPS_pt7, maxshape=(None,))
        else:
            # add new results to hdf5 file
            with h5py.File(output_events_filename, 'a') as f:
                f['clusters'].resize((f['clusters'].shape[0] + clusters.shape[0]), axis=0)
                f['clusters'][-clusters.shape[0]:] = clusters
                f['hits'].resize((f['hits'].shape[0] + hits.shape[0]), axis=0)
                f['hits'][-hits.shape[0]:] = hits
                f['ext_trig_unix'].resize((f['ext_trig_unix'].shape[0] + unix_pt7.shape[0]), axis=0)
                f['ext_trig_unix'][-unix_pt7.shape[0]:] = unix_pt7
                f['ext_trig_PPS'].resize((f['ext_trig_PPS'].shape[0] + PPS_pt7.shape[0]), axis=0)
                f['ext_trig_PPS'][-PPS_pt7.shape[0]:] = PPS_pt7
    
    if module.do_match_of_charge_to_light and mc_assn is None:
        # loop through the light files and only select triggers within the second ranges specified
        # note that not all these light events will have a match within the packets data selection
        print('Loading light files with a batch size of ', batch_size, ' ...')
        light_events_all = read_light_files(module)
    
        # match light events to ext triggers in packets (packet type 7)
        light_events_all = light_events_all[light_events_all['unix'] != 0]
        print('Matching light triggers to external triggers in packets...')
        indices_in_ext_triggers = matching.match_light_to_ext_trigger(light_events_all, PPS_pt7, unix_pt7, module) # length of light_events_all
        evt_mask = indices_in_ext_triggers != -1
        indices_in_ext_triggers = indices_in_ext_triggers[evt_mask]
        light_events_all = light_events_all[evt_mask]
        PPS_pt7_light = PPS_pt7[indices_in_ext_triggers]
        unix_pt7_light = unix_pt7[indices_in_ext_triggers]
        
        # match ext triggers / light events to charge events for the large clusters
        print('Performing charge-light matching for the charge clusters...')
        results_clusters, results_clusters_light_events = matching.match_light_to_charge(light_events_all, results_clusters, PPS_pt7_light, unix_pt7_light, module)
    else:
        print('Skipping getting light events and doing charge-light matching...')
    
    print('Saving reconstruction results to ', output_events_filename)
    analysis_end = time.time()
    print('Time to do full analysis = ', analysis_end-analysis_start, ' seconds')

if __name__ == "__main__":
    fire.Fire(run_reconstruction)

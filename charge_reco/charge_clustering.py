#!/usr/bin/env python
"""
Command-line interface to the charge clustering and matching to external triggers
"""
from build_events import *
import h5py
import fire
import time
import os
from tqdm import tqdm
import importlib.util
from math import ceil
import consts
import loading

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
    match_charge_to_ext_trig = True
    
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
    # detector dictionary file must be pkl file made with larpix_readout_parser
    pixel_xy = loading.load_geom_dict(module.detector_dict_path)
    pedestal_dict, config_dict = loading.load_pedestal_and_config(module)
    detprop = loading.load_detector_properties(module)
    disabled_channel_IDs = loading.load_disabled_channels_list(module)
    
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
        print("No 'mc_packets_assn' or 'segments' dataset found, processing as real data.")
    
    analysis_start = time.time()
    nBatches = module.nBatches
    batches_limit = module.batches_limit
    # run reconstruction
    hits_max_cindex = 0
    ext_trig_max_index = 0
    batch_size = ceil(len(packets)/nBatches)
    index_start = 0
    index_end = batch_size

    lower_PPS_window = module.charge_light_matching_lower_PPS_window
    upper_PPS_window = module.charge_light_matching_upper_PPS_window
    unix_window = module.charge_light_matching_unix_window
    z_drift_factor = 10*consts.v_drift/1e3
    
    for i in tqdm(range(batches_limit), desc = ' Processing batches...'):
        batch_start_time = time.time()
        
        packets_batch = np.array(packets[index_start:index_end])
        if mc_assn is not None:
            mc_assn = np.array(mc_assn[index_start:index_end])
        
        analysis_start_time = time.time()
        clusters, ext_trig, hits = \
            analysis(packets_batch, pixel_xy, mc_assn, tracks, module, hits_max_cindex, disabled_channel_IDs, \
                     detprop, pedestal_dict, config_dict)
        analysis_end_time = time.time()
        
        list_of_trigs = []
        # match clusters to external triggers
        matching_start_time = time.time()
        if match_charge_to_ext_trig:
            for j, trig in enumerate(ext_trig):
                # match clusters to ext triggers
                matched_clusters_mask = (clusters['t_min'] > trig['t'] - lower_PPS_window) & \
                                        (clusters['t_max'] < trig['t'] + upper_PPS_window) & \
                                        (trig['unix'] == clusters['unix']) #& \
                                        #(clusters['io_group'] == trig['io_group'])
                matched_clusters_indices = np.where(matched_clusters_mask)[0]
                np.put(clusters['ext_trig_index'], matched_clusters_indices, j+ext_trig_max_index)
                np.put(clusters['t0'], matched_clusters_indices, trig['t'])
                
                # loop through hits in clusters to calculate drift position
                for cluster_index in matched_clusters_indices:
                    hits_this_cluster_mask = hits['cluster_index'] == cluster_index + hits_max_cindex
                    hits_this_cluster = np.copy(hits[hits_this_cluster_mask])
                    z_drift_shift = hits_this_cluster['z_drift']*(hits_this_cluster['t'] - clusters[cluster_index]['t0']).astype('f8')*z_drift_factor
                    z_drift = hits_this_cluster['z_anode'] + z_drift_shift
                    np.put(hits['z_drift'], np.where(hits_this_cluster_mask)[0], z_drift)
                    np.put(clusters['z_drift_mid'], cluster_index, np.mean(z_drift))
                    np.put(clusters['z_drift_min'], cluster_index, np.min(z_drift))
                    np.put(clusters['z_drift_max'], cluster_index, np.max(z_drift))
        matching_end_time = time.time()
               
        # making sure to continously increment cluster_index as we go onto the next batch
        hits_max_cindex = np.max(hits['cluster_index'])+1
        ext_trig_max_index += len(ext_trig)
        
        if i == 0:
            # create the hdf5 datasets with initial results
            with h5py.File(output_events_filename, 'a') as output_file:
                output_file.create_dataset('clusters', data=clusters, maxshape=(None,))
                output_file.create_dataset('hits', data=hits, maxshape=(None,))
                output_file.create_dataset('ext_trig', data=ext_trig, maxshape=(None,))
        else:
            # add new results to hdf5 file
            with h5py.File(output_events_filename, 'a') as f:
                f['clusters'].resize((f['clusters'].shape[0] + clusters.shape[0]), axis=0)
                f['clusters'][-clusters.shape[0]:] = clusters
                f['hits'].resize((f['hits'].shape[0] + hits.shape[0]), axis=0)
                f['hits'][-hits.shape[0]:] = hits
                f['ext_trig'].resize((f['ext_trig'].shape[0] + ext_trig.shape[0]), axis=0)
                f['ext_trig'][-ext_trig.shape[0]:] = ext_trig
                
        index_start += batch_size
        index_end += batch_size
        batch_end_time = time.time()
        if consts.time_the_reconstruction:
            batch_total_time = batch_end_time-batch_start_time
            analysis_total_time = analysis_end_time-analysis_start_time
            print(f"Batch {i} took {(batch_total_time):.3f} seconds")
            print(f"Analysis function took {analysis_total_time:.3f} seconds, and {(analysis_total_time/batch_total_time * 100):.3f}% of the total time.")
            if match_charge_to_ext_trig:
                matching_total_time = matching_end_time - matching_start_time
                print(f"Ext trigger matching took {matching_total_time:.3f} seconds, and {(matching_total_time/batch_total_time  * 100):.3f}% of the total time.")
    print('Saving reconstruction results to ', output_events_filename)
    analysis_end = time.time()
    print(f'Time to do full analysis = {((analysis_end-analysis_start)/60):.3f} minutes')

if __name__ == "__main__":
    fire.Fire(run_reconstruction)

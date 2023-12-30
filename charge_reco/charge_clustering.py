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
from math import ceil
import consts
import loading
from input_config import ModuleConfig

def run_reconstruction(input_config_name, input_filepath, output_filepath, save_hits=0, match_to_ext_trig=True, pedestal_file=None, vcm_dac=None, vref_dac=None):
    ## main function
    if save_hits:
        save_hits = True
    else:
        save_hits = False
    
    # Get input variables. Get variables with module.<variable>
    module = ModuleConfig(input_config_name)
    
    detector = module.detector
    data_type = module.data_type
    
    input_packets_filename = input_filepath
    output_events_filename = output_filepath
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
    pixel_xy = loading.load_geom_dict(module)
    pedestal_dict, config_dict = loading.load_pedestal_and_config(module)
    if pedestal_file is not None:
        if vcm_dac is None and vref_dac is None:
            raise Exception('Specify vcm_dac and vref_dac at commandline with --vcm_dac and --vref_dac flags.')
        print(f'Loading pedestals from {pedestal_file}')
        pedestal_dict = loading.load_pedestals(pedestal_file, vref_dac, vcm_dac)
    
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
    max_cluster_index = 0
    ext_trig_max_index = 0
    batch_size = ceil(len(packets)/nBatches)
    index_start = 0
    index_end = batch_size

    lower_PPS_window = module.charge_light_matching_lower_PPS_window
    upper_PPS_window = module.charge_light_matching_upper_PPS_window
    unix_window = module.charge_light_matching_unix_window
    z_drift_factor = 10*consts.v_drift/1e3
    dbscan = DBSCAN(min_samples=min_samples, eps=eps)
    
    nClusters = 0
    nClusters_Matched = 0
    
    for i in tqdm(range(batches_limit), desc = ' Processing batches...'):
        batch_start_time = time.time()
        
        packets_batch = np.array(packets[index_start:index_end])
        if mc_assn is not None:
            mc_assn_batch = np.array(mc_assn[index_start:index_end])
        else:
            mc_assn_batch = None
        
        analysis_start_time = time.time()
        results = \
            analysis(packets_batch, pixel_xy, mc_assn_batch, tracks, module, max_cluster_index, disabled_channel_IDs, \
                     detprop, pedestal_dict, config_dict, dbscan, save_hits)
        if save_hits:
            clusters, ext_trig, hits, benchmarks = results
        else:
            clusters, ext_trig, benchmarks = results
        #clusters, ext_trig, hits, benchmarks
        analysis_end_time = time.time()
        
        list_of_trigs = []
        # match clusters to external triggers
        matching_start_time = time.time()
        if match_to_ext_trig:
            for j, trig in enumerate(ext_trig):
                # match clusters to ext triggers
                matched_clusters_mask = (clusters['t_min'] > trig['t'] - lower_PPS_window) & \
                                        (clusters['t_max'] < trig['t'] + upper_PPS_window) & \
                                        (trig['unix'] == clusters['unix'])
                matched_clusters_indices = np.where(matched_clusters_mask)[0]
                np.put(clusters['ext_trig_index'], matched_clusters_indices, j+ext_trig_max_index)
                np.put(clusters['t0'], matched_clusters_indices, trig['t'])
                
                # loop through hits in clusters to calculate drift position
                for cluster_index in matched_clusters_indices:
                    if save_hits:
                        hits_this_cluster_mask = hits['cluster_index'] == cluster_index + max_cluster_index
                        hits_this_cluster = np.copy(hits[hits_this_cluster_mask])
                        z_drift_shift = hits_this_cluster['z_drift']*(hits_this_cluster['t'] - clusters[cluster_index]['t0']).astype('f8')*z_drift_factor
                        z_drift = hits_this_cluster['z_anode'] + z_drift_shift
                        np.put(hits['z_drift'], np.where(hits_this_cluster_mask)[0], z_drift)
                    #print(f"z_anode = {clusters[cluster_index]['z_anode']}; direction = {clusters[cluster_index]['z_drift_min']}; delta z = {clusters[cluster_index]['z_drift_min']*(clusters[cluster_index]['t_min'] - clusters[cluster_index]['t0']).astype('f8')*z_drift_factor}")
                    # calculate drift coordinate for clusters
                    z_drift_min = clusters[cluster_index]['z_anode'] + clusters[cluster_index]['z_drift_min']*(clusters[cluster_index]['t_min'] - clusters[cluster_index]['t0']).astype('f8')*z_drift_factor
                    z_drift_mid = clusters[cluster_index]['z_anode'] + clusters[cluster_index]['z_drift_mid']*(clusters[cluster_index]['t_mid'] - clusters[cluster_index]['t0']).astype('f8')*z_drift_factor
                    z_drift_max = clusters[cluster_index]['z_anode'] + clusters[cluster_index]['z_drift_max']*(clusters[cluster_index]['t_max'] - clusters[cluster_index]['t0']).astype('f8')*z_drift_factor
                    np.put(clusters['z_drift_mid'], cluster_index, z_drift_min)
                    np.put(clusters['z_drift_min'], cluster_index, z_drift_mid)
                    np.put(clusters['z_drift_max'], cluster_index, z_drift_max)
        matching_end_time = time.time()
        
        ext_trig_max_index += len(ext_trig)
        
        if i == 0:
            # create the hdf5 datasets with initial results
            with h5py.File(output_events_filename, 'a') as output_file:
                output_file.create_dataset('clusters', data=clusters, maxshape=(None,))
                # making sure to continously increment cluster_index as we go onto the next batch
                nClusters += len(clusters)
                nClusters_Matched += np.sum(clusters['ext_trig_index'] != -1)
                print(f'Fraction of clusters matched to ext trigger for this batch: {nClusters_Matched/nClusters}')
                fracMatch = len(np.unique(clusters['ext_trig_index']))/len(ext_trig)
                print(f"Fraction of ext triggers with matched clusters: {fracMatch}")
                max_cluster_index += len(clusters)-1
                if save_hits:
                    output_file.create_dataset('hits', data=hits, maxshape=(None,))
                if match_to_ext_trig:
                    output_file.create_dataset('ext_trig', data=ext_trig, maxshape=(None,))
        else:
            # add new results to hdf5 file
            with h5py.File(output_events_filename, 'a') as f:
                nClusters += len(clusters)
                nClusters_Matched += np.sum(clusters['ext_trig_index'] != -1)
                print(f'Fraction of clusters matched to ext trigger for this batch: {nClusters_Matched/nClusters}')
                fracMatch = len(np.unique(clusters['ext_trig_index']))/len(ext_trig)
                print(f"Fraction of ext triggers with matched clusters: {fracMatch}")
                f['clusters'].resize((f['clusters'].shape[0] + clusters.shape[0]), axis=0)
                f['clusters'][-clusters.shape[0]:] = clusters
                max_cluster_index += len(clusters)-1
                if save_hits:
                    f['hits'].resize((f['hits'].shape[0] + hits.shape[0]), axis=0)
                    f['hits'][-hits.shape[0]:] = hits
                if len(ext_trig) > 0 and match_to_ext_trig:
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
            for benchmark_key in benchmarks.keys():
                print(f"{benchmark_key} step took {benchmarks[benchmark_key]:.3f} seconds.")
            if module.match_charge_to_ext_trig:
                matching_total_time = matching_end_time - matching_start_time
                print(f"Ext trigger matching took {matching_total_time:.3f} seconds, and {(matching_total_time/batch_total_time  * 100):.3f}% of the total time.")
            print(' ')
    if module.match_charge_to_ext_trig:
        print(f'Fraction of clusters matched to ext trigger: {nClusters_Matched/nClusters}')
    print('Saving reconstruction results to ', output_events_filename)
    analysis_end = time.time()
    print(f'Time to do full analysis = {((analysis_end-analysis_start)/60):.3f} minutes')

if __name__ == "__main__":
    fire.Fire(run_reconstruction)

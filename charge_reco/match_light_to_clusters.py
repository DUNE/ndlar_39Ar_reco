#!/usr/bin/env python
"""
Command-line interface to the matching between clusters and light triggers.
"""
import fire
import numpy as np
from tqdm import tqdm 
import h5py
import os
from input_config import ModuleConfig
import consts

def main(input_clusters_file, output_filename, *input_light_files, input_config_name):
    """
    # Args:
          input_clusters_file (str): path to file that contains charge clusters 
                and external triggers processed with charge_clustering.py
          output_filename (str): path to hdf5 output file
          input_light_files (str): paths to files that contain hdf5 files containing light data processed with adc64format
          input_config_name (str): name of detector (e.g. module-1)
    """
    module = ModuleConfig(input_config_name)
    if os.path.exists(output_filename):
        raise Exception('Output file '+ str(output_filename) + ' already exists.')
    # get clusters
    f_charge = h5py.File(input_clusters_file, 'r')
    clusters = np.array(f_charge['clusters'])
    
    sorted_indices = np.argsort(clusters['unix'])
    clusters[:] = clusters[sorted_indices]
    
    # find start and stop indices for each occurrance of a unix second value
    unique_unix, start_indices = np.unique(clusters['unix'], return_index=True)
    end_indices = np.roll(start_indices, shift=-1)
    end_indices[-1] = len(clusters) - 1
    
    unix_chunk_indices = {}
    for unix_val, start_idx, end_idx in zip(unique_unix, start_indices, end_indices):
        unix_chunk_indices[unix_val] = (start_idx, end_idx)
    
    # create new clusters array from old dataset
    indices_array_size = 5
    clusters_dtype = clusters.dtype
    new_dtype = np.dtype(clusters_dtype.descr + [('light_trig_index', '<i4', (indices_array_size,))])
    default_light_trig_indices = np.full(indices_array_size, -1, dtype='<i4')
    clusters_new = np.empty(clusters.shape, dtype=new_dtype)
    for field in clusters_dtype.names:
        clusters_new[field] = clusters[field]
    clusters_new['light_trig_index'] = default_light_trig_indices
    
    # get light events for each ADC
    f_adc1 = h5py.File(input_light_files[0], 'r')
    f_adc2 = h5py.File(input_light_files[1], 'r')
    
    # get timestamps for matching
    if module.detector == 'module-0':
        clock_correction_factor = 0.625
        tai_ns_adc1 = np.array(f_adc1['time']['tai_ns']*clock_correction_factor + f_adc1['time']['tai_s']*1e9)
        tai_ns_adc2 = np.array(f_adc2['time']['tai_ns']*clock_correction_factor + f_adc2['time']['tai_s']*1e9)
    else:
        clock_correction_factor = 1
        tai_ns_adc1 = np.array(f_adc1['time']['tai_ns'])*clock_correction_factor
        tai_ns_adc2 = np.array(f_adc2['time']['tai_ns'])*clock_correction_factor

    unix_adc1 = (np.array(f_adc1['header']['unix']) * 1e-3).astype('i8')
    unix_adc2 = (np.array(f_adc2['header']['unix']) * 1e-3).astype('i8')
    tai_ns_tolerance = 1000
    unix_tolerance = 1
    
    ts_window = 2000 # nsec
    light_events_matched = []
    light_wvfms_matched = []
    light_index = 0
    light_event_indices = np.zeros(len(clusters), dtype='i8')
    
    samples = module.samples
    nchannels = module.nchannels
    light_events_dtype = np.dtype([('id', '<i4'), ('tai_ns', '<i8'), \
                                   ('unix', '<i8'), ('channels_adc1', 'u1', (nchannels,)), \
                                   ('channels_adc2', 'u1', (nchannels,)), \
                                    ('voltage_adc1', 'i4', (nchannels, samples)), ('voltage_adc2', 'i4', (nchannels, samples))])
    light_events_all = np.zeros((0,), dtype=light_events_dtype)
    
    batch_size = 25
    batch_index = 0
    first_batch = True
    
    lower_PPS_window = module.charge_light_matching_lower_PPS_window
    upper_PPS_window = module.charge_light_matching_upper_PPS_window
    light_trig_index = 0
    
    # for selection
    max_hits = 10 # maximum hits per cluster
    max_clusters = 5 # maximum clusters per event
    clusters_keep = 0
    
    nMatches_Total = 0
    nMatches_Selection = 0
    clock_correction_factor = 0.625
    z_drift_factor = 10*consts.v_drift/1e3

    # loop through light triggers
    #for i in tqdm(range(10000), desc=' Matching clusters to light events: '):
    for i in tqdm(range(len(tai_ns_adc1)), desc=' Matching clusters to light events: '):
        light_tai_ns_mask = (tai_ns_adc2 > tai_ns_adc1[i] - tai_ns_tolerance) & ((tai_ns_adc2 < tai_ns_adc1[i] + tai_ns_tolerance))
        light_unix_mask = (unix_adc2 > unix_adc1[i] - unix_tolerance) & ((unix_adc2 < unix_adc1[i] + unix_tolerance))
        light_match_mask = light_tai_ns_mask & light_unix_mask
        
        # only match to ext triggers if there is an event in each ADC
        if np.sum(light_match_mask) == 1:
            nMatches_Total += 1
            
            light_event = np.zeros((1,), dtype=light_events_dtype)
            if module.detector == 'module-0':
                light_tai_ns = (float(tai_ns_adc1[i]) + float(f_adc2['time'][light_match_mask]['tai_ns'][0] + f_adc2['time'][light_match_mask]['tai_s'][0]*1e9 ) * clock_correction_factor)/2 
            else:
                light_tai_ns = (float(tai_ns_adc1[i]) + float(f_adc2['time'][light_match_mask]['tai_ns'][0]))/2 
            light_unix_s = unix_adc1[i]
            
            # only match light trig to clusters of same unix second
            try:
                start_index, stop_index = unix_chunk_indices[int(light_unix_s)]
            except:
                continue
            clusters_chunk = clusters[start_index:stop_index]
            #print(f"unique unix = {np.unique(clusters_chunk['unix'], return_counts=True)}")
            #print(" ")
            # match light trig to clusters
            matched_clusters_mask = (clusters_chunk['t_min'] > light_tai_ns - lower_PPS_window) & \
                                        (clusters_chunk['t_max'] < light_tai_ns + upper_PPS_window)
            indices_of_clusters = np.where(matched_clusters_mask)[0]
            
            # keep only matched light events, and keep track of indices for associations
            clusters_nhit = []
            if len(indices_of_clusters) > 0:
                indices_of_clusters = np.array(indices_of_clusters) + start_index
                # log index of light trig in each cluster
                for index in indices_of_clusters:
                    # replace the -1's from the left
                    for I in range(indices_array_size):
                        if clusters_new[index]['light_trig_index'][I] == -1:
                            clusters_new[index]['light_trig_index'][I] = light_trig_index
                            break
                    clusters_nhit.append(clusters_new[index]['nhit'])
                nClustersLimit = len(indices_of_clusters) <= max_clusters
                # require limit on number of hits per cluster in match
                nHitsLimit = np.all(np.array(clusters_nhit) <= max_hits)
                
                if light_trig_index == 0 and nClustersLimit and nHitsLimit:
                    clusters_keep = clusters_new[indices_of_clusters]
                elif light_trig_index > 0 and nClustersLimit and nHitsLimit:
                    clusters_keep = np.concatenate((clusters_keep, clusters_new[indices_of_clusters]))
                if nClustersLimit and nHitsLimit:
                    nMatches_Selection += 1
                    # get data for event
                    data_adc1 = f_adc1['data'][f_adc1['ref'][i]['start']:f_adc1['ref'][i]['stop']]
                    data_adc2 = f_adc2['data'][f_adc2['ref'][light_match_mask][0]['start']:f_adc2['ref'][light_match_mask][0]['stop']]
                    
                    # get channels and voltages
                    channels_adc1 = data_adc1['channel']
                    voltage_adc1 = data_adc1['voltage']
                    channels_adc2 = data_adc2['channel']
                    voltage_adc2 = data_adc2['voltage']

                    # save light event data to array
                    light_event['id'] = light_trig_index
                    light_event['channels_adc1'] = channels_adc1
                    light_event['channels_adc2'] = channels_adc2
                    light_event['voltage_adc1'] = voltage_adc1
                    light_event['voltage_adc2'] = voltage_adc2
                    light_event['tai_ns'] = light_tai_ns
                    light_event['unix'] = light_unix_s
                    light_events_all = np.concatenate((light_events_all, light_event))

                    light_trig_index += 1
                    #print('batch_index=', batch_index)
                    if batch_index > batch_size and first_batch:
                        #print('Saving first batch:')
                        first_batch = False
                        with h5py.File(output_filename, 'a') as output_file:
                            output_file.create_dataset('light_events', data=light_events_all, maxshape=(None,))
                        light_events_all = np.zeros((0,), dtype=light_events_dtype)
                        batch_index = 0
                    elif batch_index > batch_size and not first_batch:
                        #print('Saving batch:')
                        with h5py.File(output_filename, 'a') as output_file:
                            output_file['light_events'].resize((output_file['light_events'].shape[0] + light_events_all.shape[0]), axis=0)
                            output_file['light_events'][-light_events_all.shape[0]:] = light_events_all
                        light_events_all = np.zeros((0,), dtype=light_events_dtype)
                        batch_index = 0
                    else:
                        batch_index += 1
    # make sure to save last batch no matter what
    if len(light_events_all) > 0:
        with h5py.File(output_filename, 'a') as output_file:
            output_file['light_events'].resize((output_file['light_events'].shape[0] + light_events_all.shape[0]), axis=0)
            output_file['light_events'][-light_events_all.shape[0]:] = light_events_all

    with h5py.File(output_filename, 'a') as output_file:
        tai_ns_all = np.array(output_file['light_events']['tai_ns'])
        one_match_mask = (clusters_keep['light_trig_index'] != -1).sum(axis=1) == 1
        
        # for clusters with one light trig match, get array of associated light trig indices
        cluster_light_trig_indices_one_match = np.array(clusters_keep[one_match_mask]['light_trig_index'][:,0])
        z_anode_one_match = clusters_keep[one_match_mask]['z_anode']
        tai_ns_one_match = tai_ns_all[cluster_light_trig_indices_one_match]
        
        # calculate z_drift values and place in clusters array
        sign = np.zeros(len(z_anode_one_match), dtype=int)
        sign[z_anode_one_match < 0] = 1
        sign[z_anode_one_match > 0] = -1
        clusters_keep['z_drift_min'][one_match_mask] = z_anode_one_match + sign*(clusters_keep[one_match_mask]['t_min'] - tai_ns_one_match).astype('f8')*z_drift_factor
        clusters_keep['z_drift_mid'][one_match_mask] = z_anode_one_match + sign*(clusters_keep[one_match_mask]['t_mid'] - tai_ns_one_match).astype('f8')*z_drift_factor
        clusters_keep['z_drift_max'][one_match_mask] = z_anode_one_match + sign*(clusters_keep[one_match_mask]['t_max'] - tai_ns_one_match).astype('f8')*z_drift_factor
        clusters_keep['t0'][one_match_mask] = tai_ns_one_match
        
    with h5py.File(output_filename, 'a') as output_file:
        output_file.create_dataset('clusters', data=clusters_keep)
        
    print(f'Fraction of light triggers with matches to clusters = {nMatches_Total/len(unix_adc1)}')
    print(f'Fraction of light trigger matches left after selection = {nMatches_Selection/len(unix_adc1)}')
    print(f'Total events = {nMatches_Selection}')
    print(f'Saved output to {output_filename}')
    
if __name__ == "__main__":
    fire.Fire(main)

#!/usr/bin/env python
"""
Command-line interface to the matching between external trigger and light triggers.
"""
import fire
import numpy as np
from tqdm import tqdm 
import h5py
import os
from input_config import ModuleConfig

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
    ext_trig_indices = clusters['ext_trig_index']
    
    # get external triggers
    ext_trig_unix = np.array(f_charge['ext_trig']['unix'])
    ext_trig_t = np.array(f_charge['ext_trig']['t'])
    
    # get light events for each ADC
    f_adc1 = h5py.File(input_light_files[0], 'r')
    f_adc2 = h5py.File(input_light_files[1], 'r')
    
    # get timestamps for matching
    tai_ns_adc1 = np.array(f_adc1['time']['tai_ns'])
    unix_adc1 = (np.array(f_adc1['header']['unix']) * 1e-3).astype('i8')
    tai_ns_adc2 = np.array(f_adc2['time']['tai_ns'])
    unix_adc2 = (np.array(f_adc2['header']['unix']) * 1e-3).astype('i8')
    tai_ns_tolerance = 1000
    unix_tolerance = 1
    
    ts_window = 2000 # nsec
    unix_mask = np.zeros(len(ext_trig_unix), dtype=bool)
    tai_ns_mask = np.zeros(len(ext_trig_unix), dtype=bool)
    isMatch_mask = np.zeros(len(ext_trig_unix), dtype=bool)

    #num_light_events = int(len(light_events)/2) # len(light_events)
    #num_light_events = 10000
    light_events_matched = []
    light_wvfms_matched = []
    light_index = 0
    light_event_indices = np.zeros(len(clusters), dtype='i8')
    
    samples = module.samples
    light_events_dtype = np.dtype([('id', '<i4'), ('tai_ns', '<i8'), \
                                   ('unix', '<i8'), ('channels_adc1', 'u1', (24,)), \
                                   ('channels_adc2', 'u1', (24,)), \
                                    ('voltage_adc1', 'i4', (24, samples)), ('voltage_adc2', 'i4', (24, samples))])
    light_events_all = np.zeros((0,), dtype=light_events_dtype)
    
    batch_size = 500
    batch_index = 0
    first_batch = True
    # match between external triggers and light triggers
    for i in tqdm(range(len(tai_ns_adc1)), desc=' Matching external triggers to light events: '):
        light_tai_ns_mask = (tai_ns_adc2 > tai_ns_adc1[i] - tai_ns_tolerance) & ((tai_ns_adc2 < tai_ns_adc1[i] + tai_ns_tolerance))
        light_unix_mask = (unix_adc2 > unix_adc1[i] - unix_tolerance) & ((unix_adc2 < unix_adc1[i] + unix_tolerance))
        light_match_mask = light_tai_ns_mask & light_unix_mask
        
        # only match to ext triggers if there is an event in each ADC
        if np.sum(light_match_mask) == 1:
            light_event = np.zeros((1,), dtype=light_events_dtype)
            light_tai_ns = (float(tai_ns_adc1[i]) + float(f_adc2['time'][light_match_mask]['tai_ns'][0]))/2 
            light_unix_s = unix_adc1[i]
            
            # match light trig to external triggers
            isUnixMatch = ext_trig_unix == light_unix_s
            isPPSMatch = (ext_trig_t > light_tai_ns - ts_window) & \
                (ext_trig_t < light_tai_ns + ts_window)
            unix_mask += isUnixMatch
            tai_ns_mask += isPPSMatch
            isMatch = isUnixMatch & isPPSMatch
            isMatch_mask += isMatch
            
            # keep only matched light events, and keep track of indices for associations
            if np.sum(isMatch) == 1:
                # get data for event
                data_adc1 = f_adc1['data'][f_adc1['ref'][i]['start']:f_adc1['ref'][i]['stop']]
                data_adc2 = f_adc2['data'][f_adc2['ref'][light_match_mask][0]['start']:f_adc2['ref'][light_match_mask][0]['stop']]

                # get channels and voltages
                channels_adc1 = data_adc1['channel']
                voltage_adc1 = data_adc1['voltage']
                channels_adc2 = data_adc2['channel']
                voltage_adc2 = data_adc2['voltage']

                # save light event data to array
                light_event['id'] = light_index
                light_event['channels_adc1'] = channels_adc1
                light_event['channels_adc2'] = channels_adc2
                light_event['voltage_adc1'] = voltage_adc1
                light_event['voltage_adc2'] = voltage_adc2
                light_event['tai_ns'] = light_tai_ns
                light_event['unix'] = light_unix_s
                light_events_all = np.concatenate((light_events_all, light_event))
                
                ext_trig_index = np.where(isMatch)[0]
                np.put(light_event_indices, np.where(clusters['ext_trig_index'] == ext_trig_index)[0], light_index)
                light_index += 1
                
                if batch_index > batch_size and first_batch:
                    first_batch = False
                    with h5py.File(output_filename, 'a') as output_file:
                        output_file.create_dataset('light_events', data=light_events_all, maxshape=(None,))
                    light_events_all = np.zeros((0,), dtype=light_events_dtype)
                    batch_index = 0
                elif batch_index > batch_size and not first_batch:
                    with h5py.File(output_filename, 'a') as output_file:
                        output_file['light_events'].resize((output_file['light_events'].shape[0] + light_events_all.shape[0]), axis=0)
                        output_file['light_events'][-light_events_all.shape[0]:] = light_events_all
                    light_events_all = np.zeros((0,), dtype=light_events_dtype)
                    batch_index = 0
                else:
                    batch_index += 1
        # make sure to save last batch no matter what
        if i == len(tai_ns_adc1)-1 and len(light_events_all) > 0:
            with h5py.File(output_filename, 'a') as output_file:
                output_file['light_events'].resize((output_file['light_events'].shape[0] + light_events_all.shape[0]), axis=0)
                output_file['light_events'][-light_events_all.shape[0]:] = light_events_all
    
    # get matched clusters
    ext_trig_mask = isMatch_mask
    total_matches = np.sum(ext_trig_mask)
    #print(f'Efficiency of light event matches = {total_matches/num_light_events}')
    #print(f'Efficiency of unix matches = {np.sum(unix_mask)/num_light_events}')
    #print(f'Efficiency of PPS matches = {np.sum(tai_ns_mask)/num_light_events}')
    #print(' ')
    print(f'Efficiency of matching between ext triggers and light events = {total_matches/len(ext_trig_mask)}')
    print(f'Efficiency of unix matches = {np.sum(unix_mask)/len(ext_trig_mask)}')
    print(f'Efficiency of PPS matches = {np.sum(tai_ns_mask)/len(ext_trig_mask)}')

    ext_trig_indices_matched = np.where(ext_trig_mask)[0]
    clusters_matched_mask = np.isin(ext_trig_indices, ext_trig_indices_matched)
    clusters_matched = clusters[clusters_matched_mask]
    light_event_indices = light_event_indices[clusters_matched_mask]
    # replace ext_trig_index with light event indices
    clusters_new = np.zeros_like(clusters_matched)
    for name in clusters_matched.dtype.names:
        if name != 'ext_trig_index':
            clusters_new[name] = clusters_matched[name]
        else:
            clusters_new['ext_trig_index'] = light_event_indices
    
    clusters_matched = clusters_new
    with h5py.File(output_filename, 'a') as output_file:
        output_file.create_dataset('clusters', data=clusters_matched)
    print(f'Saved output to {output_filename}')
    
if __name__ == "__main__":
    fire.Fire(main)

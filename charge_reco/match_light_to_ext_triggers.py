#!/usr/bin/env python
"""
Command-line interface to the matching between external trigger and light triggers.
"""
import fire
import numpy as np
from tqdm import tqdm 
import h5py
import os

def main(input_clusters_file, input_light_file, output_filename):
    """
    # Args:
          input_clusters_file (str): path to file that contains charge clusters 
                and external triggers processed with charge_clustering.py
          input_light_file (str): path to file that contains ndlar_flow-processed 
                light data
    """
    if os.path.exists(output_filename):
        raise Exception('Output file '+ str(output_filename) + ' already exists.')
    # get clusters
    f_charge = h5py.File(input_clusters_file, 'r')
    clusters = np.array(f_charge['clusters'])
    ext_trig_indices = clusters['ext_trig_index']
    
    # get external triggers
    ext_trig_unix = np.array(f_charge['ext_trig']['unix'])
    ext_trig_t = np.array(f_charge['ext_trig']['t'])
    
    # get light events
    f_light = h5py.File(input_light_file, 'r')
    light_events = f_light['light/events/data']

    ts_window = 1600 # nsec
    unix_mask = np.zeros(len(ext_trig_unix), dtype=bool)
    tai_ns_mask = np.zeros(len(ext_trig_unix), dtype=bool)
    light_events_matched = np.zeros((0,), dtype=light_events.dtype)
   
    num_light_events = 2500 # len(light_events)
    light_events_matched = []
    light_index = 0
    light_event_indices = np.zeros(len(clusters))
    # match between external triggers and light triggers
    for i in tqdm(range(num_light_events), desc=' Matching external triggers to light events: '):
        light_unix_s = int(np.unique(light_events[i]['utime_ms']*1e-3)[1])
        light_tai_ns = int((np.unique(light_events[i]['tai_ns'])[1]/0.625) % 1e9)
        isUnixMatch = ext_trig_unix == light_unix_s
        isPPSMatch = (ext_trig_t > light_tai_ns - ts_window) & \
            (ext_trig_t < light_tai_ns + ts_window)
        unix_mask += isUnixMatch
        tai_ns_mask += isPPSMatch
        isMatch = isUnixMatch & isPPSMatch
        
        # keep only matched light events, and keep track of indices for associations
        if np.sum(isMatch) == 1:
            light_events_matched.append(light_events[i])
            #light_events_matched = np.concatenate((light_events_matched, np.array(light_events[i], dtype=light_events.dtype)))
            ext_trig_index = np.where(isMatch)[0]
            light_event_indices[clusters['ext_trig_index'] == ext_trig_index] \
                        = light_index
            light_index += 1
    light_events_matched = np.array(light_events_matched, dtype=light_events.dtype)
    # get matched clusters
    ext_trig_mask = unix_mask & tai_ns_mask
    total_matches = np.sum(ext_trig_mask)
    print(f'Efficiency of ext trigger matches = {total_matches/num_light_events}')
    print(f'Efficiency of unix matches = {np.sum(unix_mask)/num_light_events}')
    print(f'Efficiency of PPS matches = {np.sum(tai_ns_mask)/num_light_events}')

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
        output_file.create_dataset('light_events', data=light_events_matched)
    print(f'Saved output to {output_filename}')
if __name__ == "__main__":
    fire.Fire(main)

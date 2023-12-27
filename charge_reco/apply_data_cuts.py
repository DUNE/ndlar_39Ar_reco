#!/usr/bin/env python
"""
Script for applying a cut to only include small clusters near the photon detectors.
As input requires a charge-light matched file with charge clusters and light waveforms.
The goal of these cuts are to create a high purity sample.

Output: 
     - file with the same format as the input but only includes events after cuts. 
     - various plots
"""

import fire
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import loading
from input_config import ModuleConfig
from typing import List, Dict, Union
import time
import sys
from tqdm import tqdm

def is_point_inside_ellipse(x, y, h, k, a, b):
    """
    Check if a point (x, y) is inside an ellipse centered at (h, k) with semi-axes a and b.
    This is used to determine if a cluster is within an ellipse relative to the middle 
    of a photon detector tile. 

    Parameters:
        x (float): x-coordinate of the point to check.
        y (float): y-coordinate of the point to check.
        h (float): x-coordinate of the ellipse's center.
        k (float): y-coordinate of the ellipse's center.
        a (float): Length of the semi-major axis of the ellipse.
        b (float): Length of the semi-minor axis of the ellipse.

    Returns:
        bool: True if the point is inside the ellipse, False otherwise.
    """
    return ((x - h)**2 / a**2) + ((y - k)**2 / b**2) <= 1

def get_detector_position(adc: int, channel: int, geometry_data: Dict) -> Union[List[float], str]:
    # Extract relevant data from the geometry data
    tpc_center = geometry_data['tpc_center']
    det_center = geometry_data['det_center']
    det_adc_all = geometry_data['det_adc']  # This now corresponds to all TPCs
    det_chan_all = geometry_data['det_chan']  # This also corresponds to all TPCs
    
    # Initialize variables to hold detector and tpc numbers
    detector_number = None
    tpc_number = None
    
    # Loop through all TPCs to find the one corresponding to the given channel and ADC
    for tpc in range(len(det_adc_all)):
        det_adc = det_adc_all[tpc]
        det_chan = det_chan_all[tpc]
        for det_num, adc_num in det_adc.items():
            if adc_num != adc:  # Skip if the ADC number doesn't match
                continue
            if channel in det_chan[det_num]:
                detector_number = det_num
                tpc_number = tpc
                break
                
    # If detector_number and tpc_number are still None, the channel was not found
    if detector_number is None or tpc_number is None:
        return None
    
    # Calculate 3D position
    x, y, _ = det_center[int(detector_number)]
    _, _, z = tpc_center[int(tpc_number)]
    
    return [x, y, z]

def sum_waveforms(wvfms, channels, plot_to_adc_channel_dict, adc_channel_to_position, light_id):
    # Sum the waveforms in a particular tile in a particular event
    position = np.array([0.0, 0.0, 0.0])
    for j, adc_ch in enumerate(plot_to_adc_channel_dict):
            position += np.array(adc_channel_to_position[adc_ch])
            if j==0:
                wvfm_sum = np.array(wvfms[adc_ch[0]][light_id])[channels[adc_ch[0]][light_id] == adc_ch[1]]
                if np.size(wvfm_sum) > 0:
                    wvfm_sum -= np.mean(wvfm_sum[0][600:1000]).astype('int16')
            else:
                wvfm = np.array(wvfms[adc_ch[0]][light_id])[channels[adc_ch[0]][light_id] == adc_ch[1]]
                if np.size(wvfm) > 0:
                    wvfm_sum = wvfm_sum + wvfm[0] - np.mean(wvfm[0][600:1000]).astype('int16')
    position = position / 6
    return wvfm_sum, position

def apply_data_cuts(input_config_name, *input_filepath):
    
    use_disabled_channels_cut=True
    use_light_proximity_cut=True
    use_fiducial_cut=True
    
    input_filepaths = sys.argv[2:]
    output_filepaths = []
    for input_filepath in input_filepaths:
        start_time = time.time()
        # check that the input file exists
        if not os.path.exists(input_filepath):
            raise Exception(f'File {input_filepath} does not exist.')
        else:
            print('Opening file: ', input_filepath)
        # open input file 
        f_in = h5py.File(input_filepath, 'r')

        # check that the relevant datasets are in the input file, if not raise error
        try:
            f_in['clusters']
        except:
            raise Exception("'clusters' dataset not found in input file.")
        try:
            f_in['light_events']
        except:
            raise Exception("'light_events' dataset not found in input file.")
        N_files = len(list(input_filepaths))
        data_directory = os.path.dirname(input_filepath)
        output_filepath = input_filepath.split('.')[0] + '_with-cuts' + '.h5'
        
        if N_files > 1:
            combined_output_filepath = f"tagged_clusters_with_cuts_{input_config_name}.h5"
            if os.path.exists(combined_output_filepath):
                raise Exception('Output file '+ str(combined_output_filepath) + ' already exists.')
        if os.path.exists(output_filepath):
            os.remove(output_filepath)
        output_filepaths.append(output_filepath)
        # Get path for light geometry
        module = ModuleConfig(input_config_name)
        detector = module.detector
        light_geometry_path = module.light_det_geom_path
        light_geometry = loading.load_light_geometry(light_geometry_path)

        # make dictionaries of (adc_num, channel_num) keys with positions
        io0_dict_left = {} 
        io0_dict_right = {} 
        io1_dict_left = {}
        io1_dict_right = {}
        for adc_id in range(0,2):
            for channel_id in range(4, 64):
                position = get_detector_position(adc_id, channel_id, light_geometry)
                if position is not None:
                    if position[2] < 0 and position[0] < 0:
                        io0_dict_left[(adc_id, channel_id)] = position
                    elif position[2] < 0 and position[0] > 0:
                        io0_dict_right[(adc_id, channel_id)] = position
                    elif position[2] > 0 and position[0] < 0:
                        io1_dict_left[(adc_id, channel_id)] = position
                    elif position[2] > 0 and position[0] > 0:
                        io1_dict_right[(adc_id, channel_id)] = position

        # plot index to list of (adc, channel) combos that correspond to a full PD tile
        io0_left_y_plot_dict = {0: [(1, 15),(1, 14),(1, 13),(1, 12),(1, 11),(1, 10)], \
                               1: [(0, 15),(0, 14),(0, 13),(0, 12),(0, 11),(0, 10)], \
                               2: [(1, 9),(1, 8),(1, 7),(1, 6),(1, 5),(1, 4)], \
                               3: [(0, 9),(0, 8),(0, 7),(0, 6),(0, 5),(0, 4)]}

        io0_right_y_plot_dict = {0: [(1, 31),(1, 30),(1, 29),(1, 28),(1, 27),(1, 26)], \
                               1: [(0, 31),(0, 30),(0, 29),(0, 28),(0, 27),(0, 26)], \
                               2: [(1, 25),(1, 24),(1, 23),(1, 22),(1, 21),(1, 20)], \
                               3: [(0, 25),(0, 24),(0, 23),(0, 22),(0, 21),(0, 20)]}

        io1_left_y_plot_dict = {0: [(1, 63),(1, 62),(1, 61),(1, 60),(1, 59),(1, 58)], \
                               1: [(0, 63),(0, 62),(0, 61),(0, 60),(0, 59),(0, 58)], \
                               2: [(1, 57),(1, 56),(1, 55),(1, 54),(1, 53),(1, 52)], \
                               3: [(0, 57),(0, 56),(0, 55),(0, 54),(0, 53),(0, 52)]}

        io1_right_y_plot_dict = {0: [(1, 47),(1, 46),(1, 45),(1, 44),(1, 43),(1, 42)], \
                               1: [(0, 47),(0, 46),(0, 45),(0, 44),(0, 43),(0, 42)], \
                               2: [(1, 41),(1, 40),(1, 39),(1, 38),(1, 37),(1, 36)], \
                               3: [(0, 41),(0, 40),(0, 39),(0, 38),(0, 37),(0, 36)]}

        # parameters for cuts
        N = 5 # clusters allowed in drift window
        x = 10 # max hits per cluster allowed
        d = 150 # mm, max distance of cluster from light hit
        hit_threshold = 1000
        hit_upper_bound = 1e9
        rate_threshold = 0.5 # channel rate (Hz) threshold for disabled channels cut
        opt_cut_shape = 'ellipse' # proximity cut type. Other option: 'circle'.
        ellipse_b = 175 # mm
        clusters = np.array(f_in['clusters'])
        clusters_groups_selection = []

        # Remove single hit clusters that come from "noisy" channels
        if use_disabled_channels_cut:
            print(f'Applying disabled channels cut with {rate_threshold} Hz threshold ...')
            timestamp = input_filepath.split('charge-light-matched-clusters_')[1].split('.')[0]
            try:
                clusters_file = h5py.File(data_directory+'/packet_'+timestamp+'_clusters.h5','r')
            except:
                clusters_file = h5py.File(data_directory+'/packets-'+timestamp+'_clusters.h5','r')
            single_hit_clusters = clusters_file['clusters'][clusters_file['clusters']['nhit'] == 1]
            combined_dstack = np.dstack((single_hit_clusters['x_mid'], single_hit_clusters['y_mid'], single_hit_clusters['z_anode']))[0]

            from collections import Counter
            # count occurrences of each unique tuple of x,y, and z
            count_dict = Counter([tuple(row) for row in combined_dstack])
            tuples_to_remove = np.array(list(count_dict.keys()))[np.array(list(count_dict.values()))/(20*60) > rate_threshold]
            print(f'{len(tuples_to_remove)} channels disabled.')
            print(f'{(100*len(tuples_to_remove)/len(list(count_dict.keys()))):.4f} percentage of channels disabled.')

            print('Culling clusters from noisy channels...')
            # remove 1-hit clusters that come from channel to remove
            
            for key in tuples_to_remove:
                mask = (single_hit_clusters['x_mid'] == key[0]) & (single_hit_clusters['y_mid'] == key[1]) & (single_hit_clusters['z_anode'] == key[2])
                clusters = clusters[~np.isin(clusters['id'], single_hit_clusters[mask]['id'])]
            # clear memory before moving on
            combined_dstack=0
            single_hit_clusters=0
            mask=0
            count_dict=0
        print(' ')
        print('Grouping clusters by light event ...')
        # group clusters by light event
        light_ids = np.array(clusters['ext_trig_index'])
        sorted_indices = np.argsort(light_ids)
        light_ids = light_ids[sorted_indices]
        clusters = clusters[sorted_indices]
        light_trig_indices = np.concatenate(([0], np.flatnonzero(light_ids[:-1] != light_ids[1:])+1, [len(light_ids)]))
        grouped_clusters = np.split(clusters, light_trig_indices[1:-1])
        light_ids = np.unique(light_ids)

        # make event selection based on number of clusters allowed in event and number of hits allowed per cluster
        print(f'Selecting events with < {N+1} clusters in event and < {x+1} hits per cluster ...')
        total_clusters = 0
        light_ids_mask = np.zeros(len(light_ids), dtype=bool)
        i = 0
        for group in tqdm(grouped_clusters):
            if len(group) <= N and np.all(group['nhit'] <= x): 
                total_clusters += len(group)
                clusters_groups_selection.append(group)
                light_ids_mask[i] = True
            i += 1
        light_ids = light_ids[light_ids_mask]

        if use_light_proximity_cut:
            print(f'Applying optical proximity cut with {hit_threshold} ADC hit-threshold ...')
            wvfms = [f_in['light_events']['voltage_adc1'], f_in['light_events']['voltage_adc2']]
            channels = [f_in['light_events']['channels_adc1'], f_in['light_events']['channels_adc2']]
            plot_to_adc_channel_dict = [io0_left_y_plot_dict, io0_right_y_plot_dict, io1_left_y_plot_dict, io1_right_y_plot_dict]
            adc_channel_to_position = [io0_dict_left, io0_dict_right, io1_dict_left, io1_dict_right]

            n_light_hits = 0
            light_ids_new = []
            clusters_groups_new = []

            # loop through light_ids to do light hit proximity cut
            index = 0
            for light_id in tqdm(light_ids, desc=' Looping through events: '):
                #tai_ns = f_in['light_events'][light_id]['tai_ns']
                #unix = f_in['light_events'][light_id]['unix']
                clusters_group = []
                for i in range(4):
                    for j in range(4):
                        plot_to_adc_channel = list(plot_to_adc_channel_dict[i].values())[j]

                        # this is a summed waveform for one PD tile (sum of 6 SiPMs)
                        wvfm_sum, tile_position = sum_waveforms(wvfms, channels, plot_to_adc_channel, adc_channel_to_position[i], light_id)
                        if np.size(wvfm_sum) > 0:
                            wvfm_max = np.max(wvfm_sum)
                        else:
                            wvfm_max = 0

                        # only keep events with a summed waveform above the threshold
                        if wvfm_max > hit_threshold and wvfm_max < hit_upper_bound:
                            #print(f'light_id = {light_id}; tile position = {tile_position}')
                            n_light_hits += 1
                            clusters_event = clusters_groups_selection[index]
                            if tile_position[2] < 0:
                                clusters_event = clusters_event[(clusters_event['io_group'] == 1)]
                            else:
                                clusters_event = clusters_event[(clusters_event['io_group'] == 2)]
                            hit_to_clusters_dist = np.sqrt((clusters_event['x_mid'] - tile_position[0])**2 + (clusters_event['y_mid'] - tile_position[1])**2)

                            if opt_cut_shape == 'circle':
                                clusters_mask = (hit_to_clusters_dist < d) & (np.abs(clusters_event['x_mid']) < 315) & (np.abs(clusters_event['y_mid']) < 630)
                            elif opt_cut_shape == 'ellipse':
                                clusters_mask = is_point_inside_ellipse(clusters_event['x_mid'], clusters_event['y_mid'], tile_position[0], tile_position[1], d, ellipse_b)
                            else:
                                raise ValueError('shape not supported')

                            if np.sum(clusters_mask) > 0:
                                group = clusters_event[clusters_mask]
                                if np.any(~(((group['y_mid'] >= 304.31) & (group['y_mid'] <= 600)) 
                                            | ((group['y_mid'] <= 0) & (group['y_mid'] > -295)))):
                                    # if near ACLs or corners
                                    continue
                                for cluster in clusters_event[clusters_mask]:                                
                                    clusters_group.append(cluster)
                index += 1                    
                if len(clusters_group) > 0:
                    clusters_groups_new.append(clusters_group)
                    light_ids_new.append(light_id)
            clusters_groups_selection = clusters_groups_new
            light_ids = light_ids_new

            # Loop through remaining clusters groups and add to new output file.
            # Note that we are not saving the light data again. Use `ext_trig_index` to refer
            # to the light data in the charge-light-matched files.
            print(f'Looping through remaining events and saving to new file {output_filepath} ...')
            i = 0
            for clusters_group in tqdm(clusters_groups_selection):
                clusters_group = np.array(clusters_group)
                if i == 0:
                    # create the hdf5 datasets
                    with h5py.File(output_filepath, 'a') as f_out:
                        f_out.create_dataset('clusters', data=clusters_group, maxshape=(None,))
                else:
                    # add new results to hdf5 file
                    with h5py.File(output_filepath, 'a') as f_out:
                        f_out['clusters'].resize((f_out['clusters'].shape[0] + clusters_group.shape[0]), axis=0)
                        f_out['clusters'][-clusters_group.shape[0]:] = clusters_group
                i += 1
        f_in.close()
        end_time = time.time()
        print(f'Total elapsed time for this file = {((end_time-start_time)/60):.3f} minutes')
        
    # only create combined file when there is more than 1 input file
    if N_files > 1:
        # Initialize an empty list to store clusters data
        clusters_data = []
        timestamps = []

        # Loop through the files
        for file_path in output_filepaths:
            # Extract the timestamp from the file name
            file_timestamp = file_path.split('charge-light-matched-clusters_')[1].split('_with-cuts')[0]

            # Open the HDF5 file
            with h5py.File(file_path, 'r') as file:
                # Get the clusters dataset from the file
                clusters_dataset = file['clusters']

                # Add the clusters data to the list
                clusters_data.append(np.array(clusters_dataset))
                timestamps.extend([file_timestamp] * len(clusters_dataset))

        # Concatenate all the clusters data
        combined_clusters_data = np.concatenate(clusters_data, axis=0)

        # Define the new dtype with the 'file_timestamp' field
        new_dtype = combined_clusters_data.dtype.descr + [('file_timestamp', 'S23')]
        combined_clusters_data_with_timestamps = np.empty(combined_clusters_data.shape, dtype=new_dtype)

        # Copy the data from the original clusters dataset
        for field in combined_clusters_data.dtype.names:
            combined_clusters_data_with_timestamps[field] = combined_clusters_data[field]

        # Add the 'file_timestamp' data
        combined_clusters_data_with_timestamps['file_timestamp'] = np.array(timestamps, dtype='S23')

        # Create a new HDF5 file to store the combined data
        with h5py.File(data_directory + '/' + combined_output_filepath, 'w') as output_file:
            # Create a new dataset for the combined clusters data
            clusters_dataset = output_file.create_dataset('clusters', data=combined_clusters_data_with_timestamps)

        print(f"Combined clusters with timestamps saved to {combined_output_filepath}")
if __name__ == "__main__":
    fire.Fire(apply_data_cuts)


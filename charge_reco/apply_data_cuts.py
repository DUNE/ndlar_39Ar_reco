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
from numpy.lib.recfunctions import append_fields
from math import ceil

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
    det_adc_all = geometry_data['det_adc']  
    det_chan_all = geometry_data['det_chan'] 
    
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

def sum_waveforms(light_event, batch_start, plot_to_adc_channel_dict, adc_channel_to_position, light_id, pedestal_range):
    # Sum the waveforms in a particular tile in a particular event
    position = np.array([0.0, 0.0, 0.0])
    positions = []
    wvfms_det = [] # all individual SiPM wvfms
    adc_channels = []
    
    for j, adc_ch in enumerate(plot_to_adc_channel_dict):
            position += np.array(adc_channel_to_position[adc_ch])
            positions.append(np.array(adc_channel_to_position[adc_ch]))
            adc_channels.append(adc_ch)
            
            if adc_ch[0] == 0:
                dtype_names = ['voltage_adc1', 'channels_adc1']
            else:
                dtype_names = ['voltage_adc2', 'channels_adc2']
            if j==0:
                wvfm_sum = np.array(light_event[dtype_names[0]])[light_event[dtype_names[1]] == adc_ch[1]]
                wvfms_det.append(wvfm_sum)
                if np.size(wvfm_sum) > 0:
                    wvfm_sum -= np.mean(wvfm_sum[0][pedestal_range[0]:pedestal_range[1]]).astype('int16')
            else:
                wvfm = np.array(light_event[dtype_names[0]])[light_event[dtype_names[1]] == adc_ch[1]]
                wvfms_det.append(wvfm)
                if np.size(wvfm) > 0:
                    wvfm_sum = wvfm_sum + wvfm[0] - np.mean(wvfm[0][pedestal_range[0]:pedestal_range[1]]).astype('int16')
    position = position / 6
    return wvfm_sum, position, wvfms_det, positions, adc_channels

def add_dtype_to_array(array, dtype_name, dtype_format, new_data):
    # add a dtype and corresponding data to an already existing array/dataset.
    # INPUTS: array: data to add to (arr)
    #         dtype_name: name of dtype to add to array (str)
    #         dtype_format: format of data to add (str, e.g. '<i4', 'f8', 'S10')
    #         new_data: data to add to array (arr)
    # OUTPUTS: new array with added data
    new_dtype = array.dtype.descr + [(dtype_name, dtype_format)]
    array_new = np.empty(array.shape, dtype=new_dtype)
    for field in array.dtype.names:
        array_new[field] = array[field]
    array_new[dtype_name] = np.array(new_data, dtype=dtype_format)
    return array_new

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
            combined_output_filepath_justClusters = f"tagged_clusters_with_cuts_{input_config_name}_justClusters.h5"
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

        # plot index to list of (adc, channel) combos that correspond to a full PD tile
        # these dictionaries can be made by referring to the light detector geometry yaml
        if input_config_name == 'module0_run1' or input_config_name == 'module0_run2':
            # plot index to list of (adc, channel) combos that correspond to a full PD tile
            io0_left_y_plot_dict = {0: [(0, 30),(0, 29),(0, 28),(0, 27),(0, 26),(0, 25)], \
                                   1: [(0, 23),(0, 22),(0, 21),(0, 20),(0, 19),(0, 18)], \
                                   2: [(0, 14),(0, 13),(0, 12),(0, 11),(0, 10),(0, 9)], \
                                   3: [(0, 7),(0, 6),(0, 5),(0, 4),(0, 3),(0, 2)]}

            io0_right_y_plot_dict = {0: [(1, 62),(1, 61),(1, 60),(1, 59),(1, 58),(1, 57)], \
                                   1: [(1, 55),(1, 54),(1, 53),(1, 52),(1, 51),(1, 50)], \
                                   2: [(1, 46),(1, 45),(1, 44),(1, 43),(1, 42),(1, 41)], \
                                   3: [(1, 39),(1, 38),(1, 37),(1, 36),(1, 35),(1, 34)]}

            io1_left_y_plot_dict = {0: [(1, 30),(1, 29),(1, 28),(1, 27),(1, 26),(1, 25)], \
                                   1: [(1, 23),(1, 22),(1, 21),(1, 20),(1, 19),(1, 18)], \
                                   2: [(1, 14),(1, 13),(1, 12),(1, 11),(1, 10),(1, 9)], \
                                   3: [(1, 7),(1, 6),(1, 5),(1, 4),(1, 3),(1, 2)]}

            io1_right_y_plot_dict = {0: [(0, 62),(0, 61),(0, 60),(0, 59),(0, 58),(0, 57)], \
                                   1: [(0, 55),(0, 54),(0, 53),(0, 52),(0, 51),(0, 50)], \
                                   2: [(0, 46),(0, 45),(0, 44),(0, 43),(0, 42),(0, 41)], \
                                   3: [(0, 39),(0, 38),(0, 37),(0, 36),(0, 35),(0, 34)]}
        else:
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
        
        if input_config_name == 'module0_run1':
            rows_to_use = [0,2]
            row_column_to_remove = [] #[(2,0), (0,3)]
            pedestal_range = (0, 80)
            channel_range = (1, 63)
        elif input_config_name == 'module0_run2':
            rows_to_use = [0,1,2,3]
            row_column_to_remove = []
            pedestal_range = (0, 200)
            channel_range = (1, 63)
        elif input_config_name == 'module1':
            rows_to_use = [0,1,2,3]
            row_column_to_remove = []
            pedestal_range = (0, 200)
            channel_range = (4, 64)
        elif input_config_name == 'module2':
            rows_to_use = [0,1,2,3]
            row_column_to_remove = []
            pedestal_range = (0, 200)
            channel_range = (4, 64)
        elif input_config_name == 'module3':
            rows_to_use = [0,1,2,3]
            row_column_to_remove = []
            pedestal_range = (0, 200)
            channel_range = (4, 64)
        else:
            raise ValueError(f'Input config {input_config_name} not recognized.')
        # make dictionaries of (adc_num, channel_num) keys with positions
        io0_dict_left = {}
        io0_dict_right = {}
        io1_dict_left = {}
        io1_dict_right = {}
        for adc_id in range(0,2):
            for channel_id in range(channel_range[0], channel_range[1]):
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

        # parameters for cuts
        use_old_method = True
        d_LCM = 50 # mm, max distance of cluster from light hit, for 'rect' or 'circle' cuts
        d_ACL = 50
        hit_threshold_LCM = 4000
        hit_threshold_ACL = 4000
        hit_upper_bound = 1e9
        rate_threshold = 0.5 # channel rate (Hz) threshold for disabled channels cut
        opt_cut_shape = 'rect' # proximity cut type. Options: 'ellipse', 'circle', 'rect'.
        ellipse_b = 150 # ellipse semi-minor axis in mm
        clusters = np.array(f_in['clusters'])

        # Remove single hit clusters that come from noisy channels
        if use_disabled_channels_cut:
            print(f'Applying disabled channels cut with {rate_threshold} Hz threshold ...')
            timestamp = input_filepath.split('charge-light-matched-clusters_')[1].split('.')[0]
            # The first word of the clusters file may differ, so might run into an error here. So just make sure
            # you include a case here for your files. This is meant to catch some different cases.
            try:
                clusters_file = h5py.File(data_directory+'/packet_'+timestamp+'_clusters.h5','r')
            except:
                try:
                    clusters_file = h5py.File(data_directory+'/packets-'+timestamp+'_clusters.h5','r')
                except:
                    try:
                        clusters_file = h5py.File(data_directory+'/datalog_'+timestamp+'_clusters.h5','r')
                    except:
                        clusters_file = h5py.File(data_directory+'/self_trigger_tpc12_run2-packet-'+timestamp+'_clusters.h5', 'r')
            # Most of the noise is single hits, so we calculate the rate of single hits from each channel, then
            # apply a cut on channels with a rate above a threshold. This should remove most if not all of the 
            # noisy channels with the optimal cut. Obviously be careful not to set the threshold too low otherwise
            # you run the risk of cutting out well-behaving channels.
            
            if not use_old_method:
                single_hit_clusters = clusters_file['clusters'][clusters_file['clusters']['nhit'] == 1]
                combined_dstack = np.dstack((single_hit_clusters['x_mid'], single_hit_clusters['y_mid'], \
                                             single_hit_clusters['z_anode']))[0]

                from collections import Counter
                runtime_seconds = abs(np.max(single_hit_clusters['unix']) - np.min(single_hit_clusters['unix']))
                # count occurrences of each unique tuple of x,y, and z
                count_dict = Counter([tuple(row) for row in combined_dstack])
                tuples_to_remove = np.array(list(count_dict.keys()))[np.array(list(count_dict.values()))/(runtime_seconds) > rate_threshold]
                print(f'File runtime = {(runtime_seconds/60):.3f} minutes')
                print(f'{len(tuples_to_remove)} channels disabled.')
                print(f'{(100*len(tuples_to_remove)/len(list(count_dict.keys()))):.4f} percentage of channels disabled.')

                print('Removing clusters from noisy channels...')
                # remove 1-hit clusters that come from channel to remove
                for key in tuples_to_remove:
                    mask = (single_hit_clusters['x_mid'] == key[0]) & (single_hit_clusters['y_mid'] == key[1]) \
                            & (single_hit_clusters['z_anode'] == key[2])
                    clusters = clusters[~np.isin(clusters['id'], single_hit_clusters[mask]['id'])]
                # clear memory before moving on
                combined_dstack=0
                single_hit_clusters=0
                mask=0
                count_dict=0
            else:
                # find hit count per channel
                clusters_all = clusters_file['clusters'][clusters_file['clusters']['nhit'] <= 10]
                hits = clusters_file['hits'][np.isin(clusters_file['hits']['cluster_index'], clusters_all['id'])]
                hit_ids = hits['unique_id']
                hits_channel_count = np.bincount(hit_ids)
                hits_channel_indices = np.arange(0, len(hits_channel_count), 1)
                hits_channel_count = hits_channel_count[np.min(hit_ids):np.max(hit_ids)]
                hits_channel_indices = hits_channel_indices[np.min(hit_ids):np.max(hit_ids)]
                hits_channel_indices = hits_channel_indices[hits_channel_count != 0]
                hits_channel_count = hits_channel_count[hits_channel_count != 0]
                
                # calculate hit rate per channel
                total_time_seconds = np.max(hits['unix']) - np.min(hits['unix'])
                hits_channel_rate = hits_channel_count/total_time_seconds
                #print('Rate of hits in detector = ',len(hits)/total_time_seconds, ' Hz')
                
                hit_rate_cut = rate_threshold
                rate_cut_mask = hits_channel_rate < hit_rate_cut
                hits_channel_indices_keep = hits_channel_indices[rate_cut_mask]
                hits_channel_indices_cut = hits_channel_indices[np.invert(rate_cut_mask)]
                
                # find hits that have hit-rate less than hit rate cut
                # note we only need to loop through the channels we want to disable
                hit_mask_all = np.zeros(len(hits), dtype=bool)
                for i in tqdm(range(len(hits_channel_indices_cut))):
                    hit_mask = hits_channel_indices_cut[i] == hits['unique_id']
                    hit_mask_all += hit_mask
                hits_rate_cut_keep = hits[np.invert(hit_mask_all)]
                hits_rate_cut_remove = hits[hit_mask_all]
                
                cluster_indices_rate_cut = np.unique(hits_rate_cut_keep['cluster_index'])
                total_clusters = len(clusters)
                print(f'len(cluster_indices_rate_cut) = {len(cluster_indices_rate_cut)}')
                clusters = clusters[np.isin(clusters['id'], cluster_indices_rate_cut)]
                print('Percentage of clusters removed: ', 1 - (len(clusters) / total_clusters * 100))
                clusters_all=0
                hits=0
        print(' ')
        
        light_ids = np.array(clusters['light_trig_index'])
        
        # select only clusters with one light trig match for this selection
        light_trig_mask = (light_ids != -1).sum(axis=1) == 1
        light_ids = light_ids[light_trig_mask]
        clusters = clusters[light_trig_mask]
        
        # sort by light trig index
        #clusters_indices_sorted = np.argsort(clusters['light_trig_index'][:,0])
        #light_ids = light_ids[clusters_indices_sorted]
        #clusters = clusters[clusters_indices_sorted]
        
        # sort by unix second, find start and stop indices for each occurrance of a unix second value
        sorted_indices = np.argsort(clusters['unix'])
        clusters[:] = clusters[sorted_indices]

        # find start and stop indices for each occurrance of a unix second value
        unique_unix, start_indices = np.unique(clusters['unix'], return_index=True)
        end_indices = np.roll(start_indices, shift=-1)
        end_indices[-1] = len(clusters) - 1

        unix_chunk_indices = {}
        for unix_val, start_idx, end_idx in zip(unique_unix, start_indices, end_indices):
            unix_chunk_indices[unix_val] = (start_idx, end_idx)
        
        clusters_mask = np.zeros(len(clusters), dtype=bool)
        clusters_light_hit_index = np.zeros(len(clusters))
        light_ids_so_far = []
        
        batch_size = 25
        batch_index = 0
        firstBatch = True
        clustersSaved = False
        
        if use_light_proximity_cut:
            print(f'Applying optical proximity cut with {hit_threshold_LCM} ADC hit-threshold for LCM, {hit_threshold_ACL} ADC hit-threshold for ACL ...')
            #light_events = f_in['light_events']
            #wvfms_adc1 = f_in['light_events']['voltage_adc1']
            #wvfms_adc2 = f_in['light_events']['voltage_adc2']
            #channels = [f_in['light_events']['channels_adc1'], f_in['light_events']['channels_adc2']]
            plot_to_adc_channel_dict = [io0_left_y_plot_dict, io0_right_y_plot_dict, \
                                        io1_left_y_plot_dict, io1_right_y_plot_dict]
            adc_channel_to_position = [io0_dict_left, io0_dict_right, io1_dict_left, io1_dict_right]

            nsamples = module.samples
            light_hits_summed_dtype = np.dtype([('light_trig_index', '<i4'), ('light_hit_index', '<i4'), ('tai_ns', '<i8'), \
                ('unix', '<i8'), ('samples', 'i4', (nsamples)), ('io_group', '<i4'), \
                ('rowID', '<i4'), ('columnID', '<i4'), ('det_type', 'S3')])
            light_hits_summed = np.zeros((0,), dtype=light_hits_summed_dtype)
            light_hits_SiPM_dtype = np.dtype([('light_trig_index', '<i4'), ('light_hit_index', '<i4'), \
                ('channelID', '<i4', (2,)), ('position', '<f4', (3,)), ('tai_ns', '<i8'), \
                ('unix', '<i8'), ('samples', 'i4', (nsamples,)), ('io_group', '<i4'), \
                ('rowID', '<i4'), ('columnID', '<i4'), ('det_type', 'S3')])
            light_hits_SiPM = np.zeros((0,), dtype=light_hits_SiPM_dtype)
            
            # loop through light_ids to do light hit proximity cut
            index = 0
            cluster_light_hit_indices = []
            light_hit_index_local = 0
            light_id_last = 0
            
            # load in a chunk of the light data at a time into a numpy array to speed up code
            light_batch_size = 25
            unique_light_ids = np.unique(light_ids[:,0])
            light_nBatches = len(unique_light_ids)/light_batch_size
            light_batch_index = 0
            batch_start = 0
            batch_end = light_batch_size

            for lb in tqdm(range(ceil(light_nBatches)), desc='Processing light event batches: '):
                cluster_keep = 0
                
                if lb < ceil(light_nBatches):
                    light_events = np.array(f_in['light_events'][batch_start:batch_end])
                    batch_start += light_batch_size
                    batch_end += light_batch_size
                else:
                    batch_end = len(f_in['light_events'])
                    light_events = np.array(f_in['light_events'][batch_start:batch_end])
                for light_event in light_events:
                    # only match light trig to clusters of same unix second
                    light_unix_s = light_event['unix']
                    light_tai_ns = light_event['tai_ns']
                    try:
                        start_index, stop_index = unix_chunk_indices[int(light_unix_s)]
                    except:
                        continue
                    clusters_chunk = clusters[start_index:stop_index]
                
                    light_hit_index_local = 0
                    # loop through columns of p.detector tiles
                    for i in range(4):
                        # loop through rows of p.detector tiles
                        for j in range(4):
                            # optionally skip some rows, like for module-0 ACLs
                            if j in rows_to_use and (j,i) not in row_column_to_remove:
                                if j in [0,2]:
                                    hit_threshold = hit_threshold_LCM
                                    d = d_LCM
                                else:
                                    hit_threshold = hit_threshold_ACL
                                    d = d_ACL
                                #print(f'(i,j) = {(i,j)}')
                                plot_to_adc_channel = list(plot_to_adc_channel_dict[i].values())[j]

                                # this is a summed waveform for one PD tile (sum of 6 SiPMs)
                                wvfm_sum, tile_position, wvfms_det, positions, adc_channels = \
                                        sum_waveforms(light_event, batch_start, plot_to_adc_channel, \
                                        adc_channel_to_position[i], light_batch_index, pedestal_range)
                                if np.size(wvfm_sum) > 0:
                                    wvfm_max = np.max(wvfm_sum)
                                else:
                                    wvfm_max = 0

                                # only keep events with a summed waveform above the threshold
                                if wvfm_max > hit_threshold:
                                    if tile_position[2] < 0:
                                        tpc_id = 1
                                    else:
                                        tpc_id = 2
                                    hit_to_clusters_dist = np.sqrt((clusters_chunk['x_mid'] - tile_position[0])**2 + \
                                                                   (clusters_chunk['y_mid'] - tile_position[1])**2)
                                    temporal_proximity_mask = (clusters_chunk['t_mid'] > light_tai_ns - 400000) \
                                        & (clusters_chunk['t_mid'] < light_tai_ns + 400000) & (clusters_chunk['unix'] == light_unix_s)
                                    # can add additional optical cut shapes here as addition elif statements
                                    if opt_cut_shape == 'circle':
                                        cluster_in_shape = (hit_to_clusters_dist < d) \
                                        & (np.abs(clusters_chunk['x_mid']) < 315) \
                                        & (np.abs(clusters_chunk['y_mid']) < 630) \
                                        & (clusters_chunk['io_group'] == tpc_id)
                                    elif opt_cut_shape == 'ellipse':
                                        cluster_in_shape = is_point_inside_ellipse(clusters_chunk['x_mid'], clusters_chunk['y_mid'],tile_position[0], tile_position[1], d, ellipse_b)
                                    elif opt_cut_shape == 'rect':
                                        if tile_position[0] < 0:
                                            cluster_in_shape = (clusters_chunk['x_mid'] < tile_position[0]+d) \
                                                & (clusters_chunk['y_mid'] > tile_position[1]-304/2) \
                                                & (clusters_chunk['y_mid'] < tile_position[1]+304/2)
                                        elif tile_position[0] > 0:
                                            cluster_in_shape = (clusters_chunk['x_mid'] > tile_position[0]-d) \
                                                & (clusters_chunk['y_mid'] > tile_position[1]-304/2) \
                                                & (clusters_chunk['y_mid'] < tile_position[1]+304/2)
                                    else:
                                        raise ValueError('shape not supported')
                                    # Only save a cluster, and corresponding light waveforms, if it occurs 
                                    # within the confines of the shape relative to the center of the p.detector tile
                                    corner_mask  = ~(((clusters_chunk['y_mid'] >= 304.31) & (clusters_chunk['y_mid'] <= 600)) \
                                                | ((clusters_chunk['y_mid'] <= 0) & (clusters_chunk['y_mid'] > -295)))
                                    tpc_mask = clusters_chunk['io_group'] == tpc_id
                                    #corner_mask = np.abs(clusters_chunk['z_drift_mid']) < 290
                                    cluster_in_shape = cluster_in_shape & temporal_proximity_mask & ~corner_mask & tpc_mask
                                    clusters_mask[start_index:stop_index] += cluster_in_shape
                                    clusters_light_hit_index[start_index:stop_index] = int(light_hit_index_local)
                                    
                                    hit_summed = np.zeros((1,), dtype=light_hits_summed_dtype)
                                    hit_summed['light_trig_index'] = light_event['id']
                                    hit_summed['light_hit_index'] = int(light_hit_index_local)
                                    hit_summed['tai_ns'] = light_event['tai_ns']
                                    hit_summed['unix'] = light_event['unix']
                                    hit_summed['samples'] = wvfm_sum
                                    if j in [0,1]:
                                        hit_summed['io_group'] = 1
                                    else:
                                        hit_summed['io_group'] = 2
                                    hit_summed['rowID'] = i
                                    hit_summed['columnID'] = j
                                    if i in [0,2]:
                                        hit_summed['det_type'] = 'LCM'
                                    else:
                                        hit_summed['det_type'] = 'ACL'
                                    light_hits_summed = np.concatenate((light_hits_summed, hit_summed))

                                    # save individual SiPM waveform hits
                                    for k, wvfm_SiPM in enumerate(wvfms_det):
                                        hit_SiPM = np.zeros((1,), dtype=light_hits_SiPM_dtype)
                                        hit_SiPM['light_trig_index'] = light_event['id']
                                        hit_SiPM['light_hit_index'] = int(light_hit_index_local)
                                        hit_SiPM['channelID'] = np.array(list(adc_channels[k]),dtype='i4')
                                        hit_SiPM['position'] = positions[k]
                                        hit_SiPM['tai_ns'] = light_event['tai_ns']
                                        hit_SiPM['unix'] = light_event['unix']
                                        hit_SiPM['samples'] = wvfm_sum
                                        if j in [0,1]:
                                            hit_SiPM['io_group'] = 1
                                        else:
                                            hit_SiPM['io_group'] = 2
                                        hit_SiPM['rowID'] = i
                                        hit_SiPM['columnID'] = j
                                        if i in [0,2]:
                                            hit_SiPM['det_type'] = 'LCM'
                                        else:
                                            hit_SiPM['det_type'] = 'ACL'
                                        light_hits_SiPM = np.concatenate((light_hits_SiPM, hit_SiPM))
                                    #light_ids_so_far.append((light_event['id'], light_hit_index_local))
                                    light_hit_index_local += 1
                                    #light_id_last = light_id

                    # Save the light waveforms in batches. This is important to do
                    # otherwise the code will get agonizingly slow due to concatenating large arrays.
                    if firstBatch and batch_index == batch_size:
                        with h5py.File(output_filepath, 'a') as f_out:
                            f_out.create_dataset('light_hits_summed', data=light_hits_summed, maxshape=(None,))
                            f_out.create_dataset('light_hits_SiPM', data=light_hits_SiPM, maxshape=(None,))
                            light_hits_SiPM = np.zeros((0,), dtype=light_hits_SiPM_dtype)
                            light_hits_summed = np.zeros((0,), dtype=light_hits_summed_dtype)
                            batch_index = 0
                            firstBatch = False
                    elif not firstBatch and batch_index == batch_size:
                        with h5py.File(output_filepath, 'a') as f_out:
                            f_out['light_hits_summed'].resize((f_out['light_hits_summed'].shape[0] + light_hits_summed.shape[0]), axis=0)
                            f_out['light_hits_summed'][-light_hits_summed.shape[0]:] = light_hits_summed
                            f_out['light_hits_SiPM'].resize((f_out['light_hits_SiPM'].shape[0] + light_hits_SiPM.shape[0]), axis=0)
                            f_out['light_hits_SiPM'][-light_hits_SiPM.shape[0]:] = light_hits_SiPM
                            light_hits_SiPM = np.zeros((0,), dtype=light_hits_SiPM_dtype)
                            light_hits_summed = np.zeros((0,), dtype=light_hits_summed_dtype)
                            batch_index = 0
                    else:
                        batch_index += 1

                    index += 1
                
            clusters = clusters[clusters_mask]
            clusters_light_hit_index = clusters_light_hit_index[clusters_mask]
            clusters = add_dtype_to_array(clusters, 'light_hit_index', '<i4', clusters_light_hit_index)
            with h5py.File(output_filepath, 'a') as f_out:
                f_out.create_dataset('clusters', data=clusters)
            clustersSaved = True
            # save last batch no matter what
            if firstBatch and batch_index < batch_size:
                with h5py.File(output_filepath, 'a') as f_out:
                    f_out.create_dataset('light_hits_summed', data=light_hits_summed)
                    f_out.create_dataset('light_hits_SiPM', data=light_hits_SiPM)
            elif not firstBatch and batch_index < batch_size:
                with h5py.File(output_filepath, 'a') as f_out:
                    f_out['light_hits_summed'].resize((f_out['light_hits_summed'].shape[0] + light_hits_summed.shape[0]), axis=0)
                    f_out['light_hits_summed'][-light_hits_summed.shape[0]:] = light_hits_summed
                    f_out['light_hits_SiPM'].resize((f_out['light_hits_SiPM'].shape[0] + light_hits_SiPM.shape[0]), axis=0)
                    f_out['light_hits_SiPM'][-light_hits_SiPM.shape[0]:] = light_hits_SiPM
            
        # Note that one can use the `light_trig_index` to refer to the light data in the charge-light-matched files.
        if not clustersSaved:
            with h5py.File(output_filepath, 'a') as f_out:
                f_out.create_dataset('clusters', data=clusters)
        f_in.close()
        end_time = time.time()
        print(f'Total elapsed time for processing this file = {((end_time-start_time)/60):.3f} minutes')
        print(f'File saved to {output_filepath}')
        
    # only create combined file when there is more than 1 input file
    if N_files > 1:
        # Initialize an empty list to store clusters data
        clusters_data, light_hits_summed_data, light_hits_SiPM_data = [], [], []
        timestamps = []

        # Loop through the files
        max_light_trig_index_last = 0
        for i, file_path in enumerate(output_filepaths):
            # Extract the timestamp from the file name
            file_timestamp = file_path.split('charge-light-matched-clusters_')[1].split('_with-cuts')[0]

            # Open the HDF5 file
            with h5py.File(file_path, 'r') as file:
                # Get the clusters dataset from the file
                clusters_dataset = np.array(file['clusters'])
                light_hits_summed_dataset = np.array(file['light_hits_summed'])
                light_hits_SiPM_dataset = np.array(file['light_hits_SiPM'])
                clusters_dataset['light_trig_index'] += max_light_trig_index_last
                light_hits_summed_dataset['light_trig_index'] += max_light_trig_index_last
                light_hits_SiPM_dataset['light_trig_index'] += max_light_trig_index_last
                max_light_trig_index_last = np.max(light_hits_summed_dataset['light_trig_index'])
                # Add the clusters data to the list
                clusters_data.append(clusters_dataset)
                light_hits_summed_data.append(light_hits_summed_dataset)
                light_hits_SiPM_data.append(light_hits_SiPM_dataset)
                timestamps.extend([file_timestamp] * len(clusters_dataset))

        # Concatenate all the clusters data
        combined_clusters_data = np.concatenate(clusters_data, axis=0)
        combined_clusters_data_with_timestamps = add_dtype_to_array(combined_clusters_data, 'file_timestamp', 'S24', timestamps)
        # Create a new HDF5 file to store the combined data
        with h5py.File(data_directory + '/' + combined_output_filepath, 'w') as f_out:
            f_out.create_dataset('clusters', data=combined_clusters_data_with_timestamps)
            for i in range(len(light_hits_summed_data)):
                if i == 0:
                    f_out.create_dataset('light_hits_summed', data=light_hits_summed_data[i], maxshape=(None,))
                    f_out.create_dataset('light_hits_SiPM', data=light_hits_SiPM_data[i], maxshape=(None,))
                else:
                    f_out['light_hits_summed'].resize((f_out['light_hits_summed'].shape[0] + light_hits_summed_data[i].shape[0]), axis=0)
                    f_out['light_hits_summed'][-light_hits_summed_data[i].shape[0]:] = light_hits_summed_data[i]
                    f_out['light_hits_SiPM'].resize((f_out['light_hits_SiPM'].shape[0] + light_hits_SiPM_data[i].shape[0]), axis=0)
                    f_out['light_hits_SiPM'][-light_hits_SiPM_data[i].shape[0]:] = light_hits_SiPM_data[i]
        with h5py.File(data_directory + '/' + combined_output_filepath_justClusters, 'w') as f_out:
            f_out.create_dataset('clusters', data=combined_clusters_data_with_timestamps)
        
        print(f"Combined file saved to {combined_output_filepath}")
if __name__ == "__main__":
    fire.Fire(apply_data_cuts)


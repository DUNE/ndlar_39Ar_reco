#!/usr/bin/env python
"""
Command-line interface to the matching between clusters and light triggers.
"""
import fire
import numpy as np
#from tqdm import TQDM
import tqdm
import h5py
import os
from input_config import ModuleConfig
import consts
from adc64format import adc64format
from collections import defaultdict
from cuts_functions import *
import loading

def main(input_clusters_file, output_filename, *input_light_files, input_config_name):
    """
    # Args:
          input_clusters_file (str): path to file that contains charge clusters 
                and external triggers processed with charge_clustering.py
          output_filename (str): path to hdf5 output file
          input_light_files (str): paths to files that contain hdf5 files containing light data processed with adc64format
          input_config_name (str): name of detector (e.g. module-1)
    """
    
    # constants
    max_hits = 10 # maximum hits per cluster
    max_clusters = 5 # maximum clusters per event
    
    module = ModuleConfig(input_config_name)
    if os.path.exists(output_filename):
        raise Exception('Output file '+ str(output_filename) + ' already exists.')
    # get clusters
    f_charge = h5py.File(input_clusters_file, 'r')
    
    rate_threshold = 0.5 # channel rate (Hz) threshold for disabled channels cut
    clusters_indices_cut = disabled_channel_cut(f_charge, rate_threshold, max_hits)
    
    hit_threshold_LCM = 4800
    hit_threshold_ACL = 1500
    d_LCM = 300 # mm, max distance of cluster from light hit, for 'rect' or 'circle' cuts
    d_ACL = 300
    hit_upper_bound = 1e9
    opt_cut_shape = 'rect' # proximity cut type. Options: 'ellipse', 'circle', 'rect'.
    ellipse_b = 150 # ellipse semi-minor axis in mm
    light_geometry_path = module.light_det_geom_path
    light_geometry = loading.load_light_geometry(light_geometry_path)
    io0_left_y_plot_dict, io0_right_y_plot_dict, io1_left_y_plot_dict, io1_right_y_plot_dict = get_io_channel_map(input_config_name)
    rows_to_use, row_column_to_remove, pedestal_range, channel_range = get_cut_config(input_config_name)
    io0_dict_left, io0_dict_right, io1_dict_left, io1_dict_right = get_adc_channel_map(channel_range, light_geometry)
    plot_to_adc_channel_dict = [io0_left_y_plot_dict, io0_right_y_plot_dict, \
                                    io1_left_y_plot_dict, io1_right_y_plot_dict]
    adc_channel_to_position = [io0_dict_left, io0_dict_right, io1_dict_left, io1_dict_right]

    nsamples = module.samples
    nchannels = module.nchannels
    light_hits_summed_dtype = np.dtype([('light_trig_index', '<i4'), ('tai_ns', '<i8'), \
        ('unix', '<i8'), ('samples', 'i4', (nsamples)), ('io_group', '<i4'), ('tile_x', '<f8'), ('tile_y', '<f8'), ('tile_z', '<f8'), \
        ('rowID', '<i4'), ('columnID', '<i4'), ('det_type', 'S3'), ('wvfm_max', '<i4')])
    light_hits_summed = np.zeros((0,), dtype=light_hits_summed_dtype)
    light_wvfms_dtype = np.dtype([('voltage_adc1', 'i4', (nchannels, nsamples)), ('voltage_adc2', 'i4', (nchannels, nsamples))])
    
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
    default_light_trig_indices = np.full(indices_array_size, -1, dtype='<i4')
    clusters = add_dtype_to_array(clusters, 'light_trig_index', '<i4', default_light_trig_indices, size=(indices_array_size,))
    
    samples = module.samples
    nchannels = module.nchannels
        
    lower_PPS_window = module.charge_light_matching_lower_PPS_window
    upper_PPS_window = module.charge_light_matching_upper_PPS_window
    light_trig_index = 0
    
    clusters_keep = 0
    
    nMatches_Total = 0
    nMatches_Selection = 0
    clock_correction_factor = 0.625
    z_drift_factor = 10*consts.v_drift/1e3
    
    # peak in one file to get number of events for progress bar
    total_events = 0
    with adc64format.ADC64Reader(input_light_files[0]) as reader:
        size = reader.streams[0].seek(0, 2)
        reader.streams[0].seek(0, 0)
        chunk_size = adc64format.chunk_size(reader.streams[0])
        total_events = size // chunk_size
        #print(f'file contains {size // chunk_size} events')
        
    batch_size = 25 # how many events to load on each iteration
    batch_size_save = 25
    firstBatch = True
    batch_index = 0
    saveHeader = True
    nBatches = 0
    with adc64format.ADC64Reader(input_light_files[0], input_light_files[1]) as reader:
        with tqdm(total=int((size//chunk_size)/batch_size), unit=' chunks', smoothing=0) as pbar:
            while True:
                events = reader.next(batch_size)
                nBatches += 1
                if nBatches > int((size//chunk_size)/batch_size):
                    break
                # get matched events between multiple files
                if events is not None:
                    events_file0, events_file1 = events
                else:
                    continue

                # loop through events in this batch and do matching to charge for each event
                for evt_index in range(len(events_file0['header'])):
                    if events_file0['header'][evt_index] is not None and events_file1['header'][evt_index] is not None:
                    
                        # save header once to file
                        if saveHeader:
                            channels_adc1 = events_file0['data'][0]['channel']
                            channels_adc2 = events_file1['data'][0]['channel']
                            
                            # Define the dtype for your structured array
                            header_dtype = np.dtype([
                                ('channels_adc1', channels_adc1.dtype, channels_adc1.shape),
                                ('channels_adc2', channels_adc1.dtype, channels_adc1.shape),
                                ('max_hits', int),
                                ('max_clusters', int),
                                ('rate_threshold', float),
                                ('hit_threshold_LCM', int),
                                ('hit_threshold_ACL', int)
                            ])

                            # Create the structured array
                            header_data = np.empty(1, dtype=header_dtype)
                            header_data['channels_adc1'] = channels_adc1
                            header_data['channels_adc2'] = channels_adc2
                            header_data['max_hits'] = max_hits
                            header_data['max_clusters'] = max_clusters
                            header_data['rate_threshold'] = rate_threshold
                            header_data['hit_threshold_LCM'] = hit_threshold_LCM
                            header_data['hit_threshold_ACL'] = hit_threshold_ACL

                            with h5py.File(output_filename, 'a') as output_file:
                                output_file.create_dataset('header', data=header_data)
                            saveHeader = False
                            
                        tai_ns_adc1 = events_file0['time'][evt_index][0]['tai_ns']
                        tai_ns_adc2 = events_file1['time'][evt_index][0]['tai_ns']

                        # correct timestamps
                        if module.detector == 'module0_run1' or module.detector == 'module0_run2':
                            clock_correction_factor = 0.625
                            tai_s_adc1 = events_file0['time'][evt_index][0]['tai_s']
                            tai_s_adc2 = events_file1['time'][evt_index][0]['tai_s']
                            tai_ns_adc1 = np.array(tai_ns_adc1*clock_correction_factor + tai_s_adc1*1e9)
                            tai_ns_adc2 = np.array(tai_ns_adc2*clock_correction_factor + tai_s_adc2*1e9)
                        else:
                            clock_correction_factor = 1
                            tai_ns_adc1 = np.array(tai_ns_adc1)*clock_correction_factor
                            tai_ns_adc2 = np.array(tai_ns_adc2)*clock_correction_factor

                        unix_adc1 = int(events_file0['header'][evt_index][0]['unix']*1e-3)
                        unix_adc2 = int(events_file1['header'][evt_index][0]['unix']*1e-3)
                        
                        nMatches_Total += 1

                        light_tai_ns = (tai_ns_adc1+tai_ns_adc2)/2
                        light_unix_s = int(unix_adc1)

                        # only match light trig to clusters of same unix second
                        try:
                            start_index, stop_index = unix_chunk_indices[light_unix_s]
                        except:
                            continue
                        clusters_chunk = clusters[start_index:stop_index]

                        # match light trig to clusters
                        matched_clusters_mask = (clusters_chunk['t_min'] > light_tai_ns - lower_PPS_window) & \
                                                    (clusters_chunk['t_max'] < light_tai_ns + upper_PPS_window)
                        indices_of_clusters = np.where(matched_clusters_mask)[0]
                        
                        clusters_nhit = []
                        
                        if len(indices_of_clusters) > 0:
                            for index in indices_of_clusters:
                                clusters_nhit.append(clusters_chunk[index]['nhit'])
                                
                            # require limit on number of clusters per event
                            nClustersLimit = len(indices_of_clusters) <= max_clusters
                            
                            # require limit on number of hits per cluster in match
                            nHitsLimit = np.all(np.array(clusters_nhit) <= max_hits)

                            if nClustersLimit and nHitsLimit:
                                
                                clusters_new = clusters_chunk[indices_of_clusters]
                                #clusters_new = clusters_new[np.isin(clusters_new['id'], clusters_indices_cut)]
                                #indices_of_clusters = np.array(indices_of_clusters) + start_index
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
                                            plot_to_adc_channel = list(plot_to_adc_channel_dict[i].values())[j]

                                            voltage_adc1 = np.array(events_file0['data'][evt_index]['voltage'])
                                            voltage_adc2 = np.array(events_file1['data'][evt_index]['voltage'])
                                            
                                            # this is a summed waveform for one PD tile (sum of 6 SiPMs)
                                            wvfm_sum, tile_position, wvfms_det, positions, adc_channels = \
                                                    sum_waveforms(voltage_adc1, voltage_adc2, plot_to_adc_channel, \
                                                    adc_channel_to_position[i], pedestal_range,\
                                                    channels_adc1, channels_adc2)
                                            
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
                                                
                                                tpc_mask = clusters_new['io_group'] == tpc_id
                                                
                                                clusters_new = clusters_new[tpc_mask]
                                                #print(np.sum(clusters_new['io_group'] == 2))
                                                if np.size(clusters_new) > 0:
                                                    for index in range(len(clusters_new)):
                                                        # replace the -1's from the left
                                                        for I in range(indices_array_size):
                                                            if clusters_new[index]['light_trig_index'][I] == -1:
                                                                clusters_new[index]['light_trig_index'][I] = light_trig_index
                                                                break
                                                        clusters_new[index]['t0'] = light_tai_ns
                                                
                                                if np.size(clusters_new) > 0:          
                                                    nMatches_Selection+=1
                                                    hit_summed = np.zeros((1,), dtype=light_hits_summed_dtype)
                                                    hit_summed['light_trig_index'] = light_trig_index
                                                    hit_summed['tai_ns'] = light_tai_ns
                                                    hit_summed['unix'] = light_unix_s
                                                    hit_summed['samples'] = wvfm_sum
                                                    hit_summed['io_group'] = tpc_id
                                                    hit_summed['tile_x'] = tile_position[0]
                                                    hit_summed['tile_y'] = tile_position[1]
                                                    hit_summed['tile_z'] = tile_position[2]
                                                    hit_summed['wvfm_max'] = wvfm_max
                                                    hit_summed['rowID'] = i
                                                    hit_summed['columnID'] = j
                                                    if i in [0,2]:
                                                        hit_summed['det_type'] = 'LCM'
                                                    else:
                                                        hit_summed['det_type'] = 'ACL'
                                                    if light_trig_index == 0 or batch_index == 0:
                                                        hits_summed_all = hit_summed
                                                        clusters_keep = clusters_new
                                                    else:
                                                        hits_summed_all = np.concatenate((hits_summed_all, hit_summed))
                                                        clusters_keep = np.concatenate((clusters_keep, clusters_new))
                                                    if batch_index == batch_size_save:
                                                        clusters_keep = clusters_keep[np.isin(clusters_keep['id'], clusters_indices_cut)]
                                                        one_match_mask = (clusters_keep['light_trig_index'] != -1).sum(axis=1) == 1
                                                        # calculate z_drift values and place in clusters array
                                                        sign = np.zeros(np.sum(one_match_mask), dtype=int)
                                                        z_anode = clusters_keep['z_anode'][one_match_mask]
                                                        sign[z_anode < 0] = 1
                                                        sign[z_anode > 0] = -1
                                                        clusters_keep['z_drift_min'][one_match_mask] = \
                                                                z_anode + \
                                                                sign*(clusters_keep[one_match_mask]['t_min'] - \
                                                                clusters_keep['t0'][one_match_mask]).astype('f8')*z_drift_factor
                                                        clusters_keep['z_drift_mid'][one_match_mask] = \
                                                                z_anode + \
                                                                sign*(clusters_keep[one_match_mask]['t_mid'] - \
                                                                clusters_keep['t0'][one_match_mask]).astype('f8')*z_drift_factor
                                                        clusters_keep['z_drift_max'][one_match_mask] = \
                                                                z_anode + \
                                                                sign*(clusters_keep[one_match_mask]['t_max'] - \
                                                                clusters_keep['t0'][one_match_mask]).astype('f8')*z_drift_factor
                                                        
                                                        if firstBatch:
                                                            firstBatch = False
                                                            with h5py.File(output_filename, 'a') as f_out:
                                                                f_out.create_dataset('light_hits_summed', data=hits_summed_all, maxshape=(None,))
                                                                f_out.create_dataset('clusters', data=clusters_keep, maxshape=(None,))
                                                            batch_index = 0
                                                        elif not firstBatch:
                                                            with h5py.File(output_filename, 'a') as f_out: 
                                                                f_out['light_hits_summed'].resize((f_out['light_hits_summed'].shape[0] + hits_summed_all.shape[0]), axis=0)
                                                                f_out['light_hits_summed'][-hits_summed_all.shape[0]:] = hits_summed_all
                                                                f_out['clusters'].resize((f_out['clusters'].shape[0] + clusters_keep.shape[0]), axis=0)
                                                                f_out['clusters'][-clusters_keep.shape[0]:] = clusters_keep
                                                            batch_index = 0
                                                    else:
                                                        batch_index+=1
                                                    light_trig_index += 1
                pbar.update()
                if len(events_file0['header']) < batch_size:
                    break
    print(f'Fraction of light events with matches to clusters = {nMatches_Total/total_events}')
    print(f'Fraction of light event matches left after selection = {nMatches_Selection/total_events}')
    print(f'Total events = {nMatches_Selection}')
    print(f'Saved output to {output_filename}')
    
if __name__ == "__main__":
    fire.Fire(main)

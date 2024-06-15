import numpy as np
from typing import List, Dict, Union
from tqdm import tqdm

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

def sum_waveforms(voltage_adc1, voltage_adc2, plot_to_adc_channel_dict, adc_channel_to_position, pedestal_range, channels_adc1, channels_adc2, isMod2):
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
                voltage = voltage_adc1
                channels = channels_adc1
            else:
                voltage = voltage_adc2
                channels = channels_adc2
            if j==0:
                wvfm_sum = voltage[channels == adc_ch[1]]
                wvfms_det.append(wvfm_sum)
                if np.size(wvfm_sum) > 0:
                    wvfm_sum -= np.mean(wvfm_sum[0][pedestal_range[0]:pedestal_range[1]]).astype('int16')
            else:
                wvfm = voltage[channels == adc_ch[1]]
                wvfms_det.append(wvfm)
                if np.size(wvfm) > 0:
                    wvfm_sum = wvfm_sum + wvfm[0] - np.mean(wvfm[0][pedestal_range[0]:pedestal_range[1]]).astype('int16')
    position = position / 6
    isTPC1 = position[2] < 0
    if isMod2 and not (((position[1] < 310 and position[1] > 0) or (position[1] > -620 and position[1] < -310)) and isTPC1):
        position[0] = position[0]*-1
    if isMod2 and ((position[1] > 310 and position[1] < 620) and isTPC1 and position[0] > 0):
        position[1] = 155.0975
    elif isMod2 and ((position[1] > 0 and position[1] < 310) and isTPC1 and position[0] > 0):
        position[1] = 465.2925
    return wvfm_sum, position, wvfms_det, positions, adc_channels

def add_dtype_to_array(array, dtype_name, dtype_format, new_data, size=None):
    # add a dtype and corresponding data to an already existing array/dataset.
    # INPUTS: array: data to add to (arr)
    #         dtype_name: name of dtype to add to array (str)
    #         dtype_format: format of data to add (str, e.g. '<i4', 'f8', 'S10')
    #         new_data: data to add to array (arr)
    # OUTPUTS: new array with added data
    if size is None:
        new_dtype = array.dtype.descr + [(dtype_name, dtype_format)]
    else:
        new_dtype = array.dtype.descr + [(dtype_name, dtype_format, size)]
    array_new = np.empty(array.shape, dtype=new_dtype)
    for field in array.dtype.names:
        array_new[field] = array[field]
    array_new[dtype_name] = np.array(new_data, dtype=dtype_format)
    return array_new

def get_io_channel_map(input_config_name):
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
    return io0_left_y_plot_dict, io0_right_y_plot_dict, io1_left_y_plot_dict, io1_right_y_plot_dict

def get_cut_config(input_config_name):
    if input_config_name == 'module0_run1':
        rows_to_use = [0,2]
        row_column_to_remove = [] #[(2,0), (0,3)]
        pedestal_range = (0, 80)
        channel_range = (1, 63)
    elif input_config_name == 'module0_run2':
        rows_to_use = [0,1,2,3]
        row_column_to_remove = []
        pedestal_range = (0, 80)
        channel_range = (1, 63)
    elif input_config_name == 'module1':
        rows_to_use = [0,1,2,3]
        row_column_to_remove = []
        pedestal_range = (0, 50)
        channel_range = (4, 64)
    elif input_config_name == 'module2':
        rows_to_use = [0,1,2,3]
        row_column_to_remove = []
        pedestal_range = (0, 20)
        channel_range = (4, 64)
    elif input_config_name == 'module3':
        rows_to_use = [0,1,2,3]
        row_column_to_remove = []
        pedestal_range = (0, 50)
        channel_range = (4, 64)
    else:
        raise ValueError(f'Input config {input_config_name} not recognized.')
    return rows_to_use, row_column_to_remove, pedestal_range, channel_range

def get_adc_channel_map(channel_range, light_geometry):
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
    return io0_dict_left, io0_dict_right, io1_dict_left, io1_dict_right

def disabled_channel_cut(clusters_file, rate_threshold, nhit_cut):
    
    # find hit count per channel
    clusters_all = clusters_file['clusters'][clusters_file['clusters']['nhit'] <= nhit_cut]
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

    rate_cut_mask = hits_channel_rate < rate_threshold
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
    total_clusters = len(clusters_all)
    print('Percentage of clusters to be removed: ', (1 - (len(cluster_indices_rate_cut) / total_clusters))*100)
    clusters_all=0
    hits=0
    hits_rate_cut_keep=0
    hits_rate_cut_remove=0
    return cluster_indices_rate_cut
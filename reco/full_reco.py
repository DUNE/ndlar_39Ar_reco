#!/usr/bin/env python
"""
Command-line interface to the matching between clusters and light triggers.
"""
import fire
import numpy as np
import tqdm
import h5py
import os
from input_config import ModuleConfig
import consts
from adc64format import adc64format
from cuts_functions import *
import loading
from sklearn.cluster import DBSCAN

def get_pixel_positions(packets, ts, pixel_xy):
    xyz_values, ts_inmm, unique_ids = [], [], []
    io_channels_all = np.array([[1+i*4,2+i*4,3+i*4,4+i*4] for i in range(0, 8)])
    packets_keep_mask = np.zeros(len(packets), dtype=bool)
    total_key_errors = 0
    for i in range(len(packets)):
        if packets[i]['io_group'] % 2 == 1:
            io_group = 1
        else:
            io_group = 2
        io_channel, chip_id, channel_id = packets[i]['io_channel'], packets[i]['chip_id'], packets[i]['channel_id']
        unique_id = ((io_group * 256 + io_channel) * 256 + chip_id)*64 + channel_id
        dict_values = pixel_xy.get((io_group, io_channel, chip_id, channel_id))
        
        if dict_values is not None:
            xyz_values.append([dict_values[0], dict_values[1], dict_values[2], dict_values[3]])
            ts_inmm.append(consts.v_drift*1e1*ts[i]*1e-3)
            packets_keep_mask[i] = True
            unique_ids.append(unique_id)
        else:
            io_channel_indices = np.where(io_channels_all == io_channel)
            io_channels_tile = io_channels_all[io_channel_indices[0], :]
            other_io_channels = io_channels_tile[io_channels_tile != io_channel]
            for other_io in other_io_channels:
                dict_values = pixel_xy.get((io_group, other_io, chip_id, channel_id))
                if dict_values is not None:
                    xyz_values.append([dict_values[0], dict_values[1], dict_values[2], dict_values[3]])
                    ts_inmm.append(consts.v_drift*1e1*ts[i]*1e-3)
                    packets_keep_mask[i] = True
                    unique_ids.append(unique_id)     
                    break
            if dict_values is None:
                total_key_errors += 1
    if len(xyz_values) > 0:
        txyz = np.hstack((np.array(ts_inmm)[:, np.newaxis], np.array(xyz_values)))
        #print(f'number of keyerrors = {total_keyerrors}')
    else:
        txyz=None
    return txyz, packets[packets_keep_mask]

def corner_cut(event, tolerance, special_cases=None):
    # inputs: clusters; clusters array to apply corner cut on
    #         tolerance; distance in mm away from corner to cut clusters in any direction
    #         special_cases (optional); extra points to apply corner cut on, for instance when there are disabled tiles and
    #                       need to cut extra cosmic clippers. This is a dictionary where the key is either 'xy', 'xz', or 'zy',
    #                       and the value is a list or tuple of two values being the point to apply cut on in the corresponding
    #                       view. 
    # outputs: mask to apply to clusters to remove clusters near corners.
    pos_names = {'xy': ['x', 'y'], 'xz': ['x', 'z'], 'zy': ['z', 'y']}
    x_edge, y_edge, z_edge = 305, 615, 305
    signs = [(-1,-1), (-1,1), (1,1), (1,-1)]
    overall_mask = True
    for dims in ['xy', 'xz', 'zy']:
        pos_name_list = pos_names[dims]
        if dims == 'xy':
            xlim, ylim = x_edge, y_edge
        elif dims == 'xz':
            xlim, ylim = x_edge, z_edge
        elif dims == 'zy':
            xlim, ylim = z_edge, y_edge
        for sign in signs:
            dist = np.sqrt((event[pos_name_list[0]] - sign[0]*xlim)**2 + (event[pos_name_list[1]] - sign[1]*ylim)**2)
            overall_mask = overall_mask & (dist > tolerance)
        if special_cases is not None:
            for dim in special_cases.keys():
                xlim, ylim, side = special_cases[dim][0], special_cases[dim][1], special_cases[dim][2]
                if dim == 'xy':
                    dist = np.sqrt((event['x'] - xlim)**2 + (event['y'] - ylim)**2)
                elif dim == 'xz':
                    dist = np.sqrt((event['x'] - xlim)**2 + (event['z'] - ylim)**2)
                elif dim == 'zy':
                    dist = np.sqrt((event['z'] - xlim)**2 + (event['y'] - ylim)**2)
                if side == 'left' and dim == 'zy':
                    side_mask = event['x'] < 0
                elif side == 'right' and dim == 'zy':
                    side_mask = event['y'] > 0
                if side == 'left' and dim == 'xy':
                    side_mask = event['io_group'] == 2
                elif side == 'right' and dim == 'xy':
                    side_mask = event['io_group'] == 1
                overall_mask = overall_mask & ((dist > tolerance) | side_mask)
    return overall_mask

def main(input_packets_file, output_filename, pedestal_file, *input_light_files, input_config_name):
    """
    # Args:
          input_packets_file (str): path to file that contains packets
          output_filename (str): path to hdf5 output file
          pedestal_file (str): path to pedestal h5 file to use
          input_light_files (str): paths to files that contain hdf5 files containing light data processed with adc64format
          input_config_name (str): name of detector (e.g. module-1)
    """
    module = ModuleConfig(input_config_name)
    # packets requirements
    max_packets = 25 # maximum data packets per event
    # cluster requirements after matching
    max_hits = 10 # maximum hits per cluster
    max_clusters = 5 # maximum clusters per event
    
    data_dtype = np.dtype([('nhit', '<i4'), ('q', '<f8'), ('adcs', '<i4'), ('io_group', '<i4'),\
                        ('t', '<i8'), ('t0', '<i8'), ('x', '<f8'),('y', '<f8'),('z', '<f8'), ('z_anode', '<f8'), \
                        ('unix', '<i8'), ('light_trig_index', '<i4'), ('light_hit_index', '<i4'), ('samples', 'i4', (module.samples)), \
                        ('tile_position', 'f8', (3,)), ('rowID', '<i4'), ('columnID', '<i4'), ('det_type', 'S3'), ('amplitude', '<i8')])
    
    module = ModuleConfig(input_config_name)
    if os.path.exists(output_filename):
        raise Exception('Output file '+ str(output_filename) + ' already exists.')
    if not os.path.exists(pedestal_file):
        raise Exception('Input pedestal file '+ str(pedestal_file) + ' does not exist.')
    
    # load charge configuration parameters
    vref = loading.dac2mv(module.vref_dac, consts.vdda)
    vcm = loading.dac2mv(module.vcm_dac, consts.vdda)
    print(f'Loading pedestals from {pedestal_file} using vref = {vref:.5f} and vcm_dac = {vcm:.5f}')
    pedestal_dict = loading.load_pedestals(pedestal_file, vref, vcm)
    
    # get packets
    f_charge = h5py.File(input_packets_file, 'r')
    packets = np.array(f_charge['packets'])
    f_charge.close()
    pixel_xy = loading.load_geom_dict(module)
    #pedestal_dict, config_dict = loading.load_pedestal_and_config(module)
    
    # parameters
    rate_threshold = 150 # channel rate (Hz) threshold for disabled channels cut
    hit_threshold_LCM = module.hit_threshold_LCM
    hit_threshold_ACL = module.hit_threshold_ACL
    d_LCM = 150 # mm, max distance of cluster from light hit, for 'rect' or 'circle' cuts
    d_ACL = 150
    use_proximity_cut = True
    use_corner_cut = True
    corner_tolerance = 25
    
    # get light geometry information
    light_geometry = loading.load_light_geometry(module.light_det_geom_path)
    io0_left_y_plot_dict, io0_right_y_plot_dict, io1_left_y_plot_dict, io1_right_y_plot_dict = get_io_channel_map(input_config_name)
    rows_to_use, row_column_to_remove, pedestal_range, channel_range = get_cut_config(input_config_name)
    io0_dict_left, io0_dict_right, io1_dict_left, io1_dict_right = get_adc_channel_map(channel_range, light_geometry)
    plot_to_adc_channel_dict = [io0_left_y_plot_dict, io0_right_y_plot_dict, \
                                    io1_left_y_plot_dict, io1_right_y_plot_dict]
    adc_channel_to_position = [io0_dict_left, io0_dict_right, io1_dict_left, io1_dict_right]

    print('Getting packet timestamps and sorting by unix timestamp...')    
    
    pkt_0_mask = (packets['packet_type'] == 0) & (packets['valid_parity'] == 1)
    
    # apply high rate channel cut
    print(f'Disabling channels with rate > {rate_threshold} Hz')
    nPackets_before = len(packets)
    packet_unique_id = (((packets['io_group'] * 256 + packets['io_channel']) * 256 + packets['chip_id'])*64 + packets['channel_id'])
    unique_unique_ids, unique_ids_counts = np.unique(packet_unique_id[pkt_0_mask], return_counts=True)
    pkt_4_timestamps = packets['timestamp'][packets['packet_type'] == 4]
    total_time = np.max(pkt_4_timestamps) - np.min(pkt_4_timestamps)
    unique_ids_rate = unique_ids_counts/total_time
    unique_ids_remove = unique_unique_ids[unique_ids_rate > rate_threshold]
    
    packets = packets[~np.isin(packet_unique_id, unique_ids_remove) | ~pkt_0_mask]
    print(f'Removed {len(unique_ids_remove)} channels due to high rate.')
    nPackets_after = len(packets)
    print(f'{nPackets_before-nPackets_after} packets removed, {((1-nPackets_after/nPackets_before) * 100):.3f}% packets removed')
        
    unix_timestamps = np.copy(packets['timestamp']).astype('i8')
    unix_timestamps[packets['packet_type'] != 4] = 0
    unix_timestamps = np.interp(np.arange(len(unix_timestamps)), np.nonzero(unix_timestamps)[0], unix_timestamps[np.nonzero(unix_timestamps)[0]])
    pps_timestamps = (packets['timestamp']*0.1*1e3).astype('i8')
    
    pkt_0_mask = (packets['packet_type'] == 0) & (packets['valid_parity'] == 1)
    
    # get timestamps and packets for packet type 0 only, apply PPS noise cut
    ts_mask = pkt_0_mask \
            & np.invert((pps_timestamps > 2e7*0.1*1e3) | (pps_timestamps < 1e6*0.1*1e3))
    unix_timestamps = unix_timestamps[ts_mask]
    pps_timestamps = pps_timestamps[ts_mask]
    packets = packets[ts_mask]
    
    # find start and stop indices for each occurrance of a unix second value
    unique_unix, start_indices = np.unique(unix_timestamps, return_index=True)
    end_indices = np.roll(start_indices, shift=-1)
    end_indices[-1] = len(unix_timestamps) - 1
    
    unix_chunk_indices = {}
    for unix_val, start_idx, end_idx in zip(unique_unix, start_indices, end_indices):
        unix_chunk_indices[unix_val] = (start_idx, end_idx)
    
    print('Finished with packets.')
    
    lower_PPS_window, upper_PPS_window = module.charge_light_matching_lower_PPS_window, module.charge_light_matching_upper_PPS_window
    light_trig_index, light_hit_index = 0, 0
        
    # peak in one file to get number of events for progress bar
    total_events = 0
    with adc64format.ADC64Reader(input_light_files[0]) as reader:
        size = reader.streams[0].seek(0, 2)
        reader.streams[0].seek(0, 0)
        chunk_size = adc64format.chunk_size(reader.streams[0])
        total_events = size // chunk_size
        #print(f'file contains {size // chunk_size} events')

    saveHeader = True
    isMod2 = input_config_name == 'module2'
    batch_size = 1
    with adc64format.ADC64Reader(input_light_files[0], input_light_files[1]) as reader:
        with tqdm(total=int(total_events/batch_size), unit=' events', smoothing=0) as pbar:
            event_index = 0
            while True:
                if event_index > total_events:
                    break
                else:
                    event_index += batch_size
                events = reader.next(batch_size)
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
                            
                            channels_adc1 = events_file0['data'][evt_index]['channel']
                            channels_adc2 = events_file1['data'][evt_index]['channel']
                            
                            # Define the dtype for your structured array
                            header_dtype = np.dtype([
                                ('channels_adc1', channels_adc1.dtype, channels_adc1.shape),
                                ('channels_adc2', channels_adc1.dtype, channels_adc1.shape),
                                ('max_packets', int),
                                ('max_hits', int),
                                ('max_clusters', int),
                                ('rate_threshold', float),
                                ('hit_threshold_LCM', int),
                                ('hit_threshold_ACL', int)
                            ])

                            # Create the structured array
                            header_data = np.empty(1, dtype=header_dtype)
                            header_data['channels_adc1'], header_data['channels_adc2'] = channels_adc1, channels_adc2
                            header_data['max_packets'] = max_packets
                            header_data['max_hits'] = max_hits
                            header_data['max_clusters'] = max_clusters
                            header_data['rate_threshold'] = rate_threshold
                            header_data['hit_threshold_LCM'], header_data['hit_threshold_ACL'] = hit_threshold_LCM, hit_threshold_ACL

                            with h5py.File(output_filename, 'a') as output_file:
                                output_file.create_dataset('header', data=header_data)
                                output_file.create_dataset('events', data=np.zeros((0,), dtype=data_dtype), maxshape=(None,))
                            saveHeader = False
                            
                        tai_ns_adc1 = events_file0['time'][evt_index][0]['tai_ns']
                        tai_ns_adc2 = events_file1['time'][evt_index][0]['tai_ns']

                        # correct timestamps
                        if module.detector == 'module0_run1' or module.detector == 'module0_run2':
                            tai_s_adc1 = events_file0['time'][evt_index][0]['tai_s']
                            tai_s_adc2 = events_file1['time'][evt_index][0]['tai_s']
                            tai_ns_adc1 = tai_ns_adc1*0.625 + tai_s_adc1*1e9
                            tai_ns_adc2 = tai_ns_adc2*0.625 + tai_s_adc2*1e9
                        
                        unix_adc1 = int(events_file0['header'][evt_index][0]['unix']*1e-3)
                        unix_adc2 = int(events_file1['header'][evt_index][0]['unix']*1e-3)
                        
                        # using these timestamps for matching
                        light_tai_ns = (tai_ns_adc1+tai_ns_adc2)/2
                        light_unix_s = int(unix_adc1)
  
                        # only match light trig to packets with same unix second
                        try:
                            start_index, stop_index = unix_chunk_indices[light_unix_s]
                        except:
                            continue
                        pps_timestamps_chunk = pps_timestamps[start_index:stop_index]
                        
                        # match light trig to packets
                        matched_packets_mask = (pps_timestamps_chunk < light_tai_ns + upper_PPS_window) & \
                                                (pps_timestamps_chunk > light_tai_ns - lower_PPS_window)
                        if np.sum(matched_packets_mask) > max_packets or np.sum(matched_packets_mask) == 0:
                            continue
                        else:
                            packets_chunk = packets[start_index:stop_index][matched_packets_mask]
                            pps_timestamps_chunk = pps_timestamps_chunk[matched_packets_mask]
                            txyz, packets_chunk = get_pixel_positions(packets_chunk, pps_timestamps_chunk, pixel_xy)
                            if txyz is None:
                                continue
                            
                            # cluster packets
                            db = DBSCAN(eps=consts.eps, min_samples=consts.min_samples).fit(txyz) 
                            labels = np.array(db.labels_)
                            unique_labels, labels_counts = np.unique(labels, return_counts=True)
                            
                            # apply max clusters and max hits cut
                            if len(unique_labels) > max_clusters or np.any(labels_counts > max_hits):
                                continue
                            else:
                                unique_io = np.unique(packets_chunk['io_group'])
                                # loop through columns of p.detector tiles
                                iterate_light_trig_index = False
                                for i in range(4):
                                    # loop through rows of p.detector tiles
                                    for j in range(4):
                                        # optionally skip some rows, like for module-0 ACLs
                                        if j in rows_to_use and (j,i) not in row_column_to_remove:
                                            if j in [0,2]:
                                                hit_threshold = hit_threshold_LCM
                                                proximity_distance = d_LCM
                                            else:
                                                hit_threshold = hit_threshold_ACL
                                                proximity_distance = d_ACL
                                            plot_to_adc_channel = list(plot_to_adc_channel_dict[i].values())[j]

                                            voltage_adc1 = np.array(events_file0['data'][evt_index]['voltage'])
                                            voltage_adc2 = np.array(events_file1['data'][evt_index]['voltage'])

                                            # this is a summed waveform for one PD tile (sum of 6 SiPMs)
                                            wvfm_sum, tile_position, wvfms_det, positions, summed_channels = \
                                                    sum_waveforms(voltage_adc1, voltage_adc2, plot_to_adc_channel, \
                                                    adc_channel_to_position[i], pedestal_range,\
                                                    channels_adc1, channels_adc2, isMod2)

                                            if np.size(wvfm_sum) > 0:
                                                wvfm_max = np.max(wvfm_sum[0][pedestal_range[1]:pedestal_range[1]+100])
                                            else:
                                                wvfm_max = 0

                                            # only keep events with a summed waveform above the threshold
                                            if wvfm_max > hit_threshold:
                                                if tile_position[2] < 0:
                                                    tpc_id = 1
                                                else:
                                                    tpc_id = 2

                                                # skip if no clusters in the same tpc as waveform
                                                if tpc_id not in unique_io:
                                                    continue
                                                else:
                                                    tpc_mask = packets_chunk['io_group'] == tpc_id
                                                    if use_proximity_cut:
                                                        if tile_position[0] < 0:
                                                            prox_mask = (txyz[:,1] < tile_position[0]+proximity_distance) \
                                                                & (txyz[:,2] > tile_position[1]-304/2) \
                                                                & (txyz[:,2] < tile_position[1]+304/2)
                                                        elif tile_position[0] > 0:
                                                            prox_mask = (txyz[:,1] > tile_position[0]-proximity_distance) \
                                                                & (txyz[:,2] > tile_position[1]-304/2) \
                                                                & (txyz[:,2] < tile_position[1]+304/2)
                                                        overall_mask = tpc_mask & prox_mask
                                                    else:
                                                        overall_mask = tpc_mask
                                                    labels_sel = labels[overall_mask]
                                                    packets_chunk_sel = packets_chunk[overall_mask]
                                                    txyz_sel = txyz[overall_mask]
                                                    if len(packets_chunk_sel) == 0:
                                                        continue
                                                    # get quantities for clusters
                                                    label_indices = np.concatenate(([0], np.flatnonzero(labels_sel[:-1] != labels_sel[1:])+1, [len(labels_sel)]))[1:-1]
                                                    label_timestamps = np.split(txyz_sel[:,0]*consts.mm_to_ns, label_indices)
                                                    label_x = np.split(txyz_sel[:,1], label_indices)
                                                    label_y = np.split(txyz_sel[:,2], label_indices)
                                                    label_z = np.split(txyz_sel[:,3], label_indices)
                                                    label_direction = np.split(txyz_sel[:,4], label_indices)
                                                    
                                                    min_timestamps = np.array(list(map(np.min, label_timestamps)), dtype='i8')
                                                    max_timestamps = np.array(list(map(np.max, label_timestamps)), dtype='i8')
                                                    x_min, x_max = np.array(list(map(min, label_x))), np.array(list(map(max, label_x)))
                                                    y_min, y_max = np.array(list(map(min, label_y))), np.array(list(map(max, label_y)))
                                                    z_anode = np.array(list(map(min, label_z)))
                                                    z_direction = np.array(list(map(min, label_direction)))
                                                    
                                                    iterate_light_trig_index = True
                                                    
                                                    label_ids, n_counts = np.unique(labels_sel, return_counts=True)
                                                    corner_cut = False
                                                    for k in range(len(n_counts)):
                                                        data = np.zeros((1,), dtype=data_dtype)
                                                        # write information from light to array
                                                        data['light_trig_index'] = light_trig_index
                                                        data['light_hit_index'] = light_hit_index
                                                        data['t0'] = light_tai_ns
                                                        data['unix'] = light_unix_s
                                                        data['samples'] = wvfm_sum
                                                        data['io_group'] = tpc_id
                                                        data['tile_position'] = (tile_position[0],tile_position[1],tile_position[2])
                                                        data['amplitude'] = wvfm_max
                                                        data['rowID'] = j
                                                        data['columnID'] = i
                                                        if j in [0,2]:
                                                            data['det_type'] = 'LCM'
                                                        else:
                                                            data['det_type'] = 'ACL'

                                                        # write information from charge to array
                                                        data['nhit'] = n_counts[k]
                                                        data['x'] = (x_max[k] + x_min[k])/2
                                                        data['y'] = (y_max[k] + y_min[k])/2
                                                        t = (max_timestamps[k] + min_timestamps[k])/2
                                                        data['t'] = t
                                                        data['z'] = z_anode[k] + z_direction[k]*(t - light_tai_ns).astype('f8')*consts.z_drift_factor
                                                        data['z_anode'] = z_anode[k]
                                                        
                                                        # cut out events with activity near corners
                                                        #if use_corner_cut and corner_cut(data, corner_tolerance, special_cases=None):
                                                        #    corner_cut = True
                                                        
                                                        packets_cluster = packets_chunk_sel[labels_sel == label_ids[k]]
                                                        v_ped = []
                                                        for packet in packets_cluster:
                                                            unique_id = (((packet['io_group'] * 256 + packet['io_channel']) * 256 + packet['chip_id'])*64 + packet['channel_id'])
                                                            v_ped.append(pedestal_dict[unique_id]['pedestal_mv'])
                                                        v_ped = np.array(v_ped)
                                                        
                                                        #adcs = np.sum(np.around(packets_cluster['dataword'].astype('float64') - (v_ped - vcm)/(vref - vcm) * consts.ADC_COUNTS)).astype('i4')
                                                        #adcs = np.sum(np.around(packets_cluster['dataword'].astype('float64') - \
                                                        #                    v_ped * consts.ADC_COUNTS / (vref - vcm))).astype('i4')
                                                        q = np.sum((packets_cluster['dataword'].astype('float64')/consts.ADC_COUNTS*(vref - vcm)+vcm-v_ped)*consts.gain * 1e-3)
                                                        adcs = np.sum(np.around(packets_cluster['dataword'].astype('float64') - (v_ped - vcm)/(vref - vcm) * consts.ADC_COUNTS))
                                                        data['adcs'] = adcs
                                                        #q = adcs.astype('f8')*consts.gain * 1e-3 * (vref - vcm)/consts.ADC_COUNTS
                                                        data['q'] = q
                                                        # save new data to hdf5 files
                                                        with h5py.File(output_filename, 'a') as f:
                                                            f['events'].resize((f['events'].shape[0] + data.shape[0]), axis=0)
                                                            f['events'][-data.shape[0]:] = data
                                                    light_hit_index += 1
                                if iterate_light_trig_index:
                                    #print(light_trig_index)
                                    light_trig_index += 1
                pbar.update()
    
    with h5py.File(output_filename, 'a') as f:
        total_matches = len(np.unique(f['events']['light_trig_index']))
    print(f'Total number of light triggers matched to clusters = {total_matches}')
    print(f'Total number of light triggers in original files = {total_events}')
    print(f'Fraction of light events with matches to clusters = {total_matches/total_events}')
    print(f'Saved output to {output_filename}')
    
if __name__ == "__main__":
    fire.Fire(main)

from json import load as load_json
from collections import defaultdict
import yaml
import numpy as np
import h5py
from consts import vdda, mean_trunc
from tqdm import tqdm
import os
import json

def load_disabled_channels_list(module):
    if module.use_disabled_channels_list:
        disabled_channels = np.load(module.disabled_channels_list)
        disabled_channel_IDs = np.array(disabled_channels['keys']).astype('int')
    else:
        disabled_channel_IDs = None
    return disabled_channel_IDs

def load_detector_properties(module):
    with open(module.detprop_path) as df:
        detprop = yaml.load(df, Loader=yaml.FullLoader)
    return detprop

def rotate_pixel(pixel_pos, tile_orientation):
    return pixel_pos[0]*tile_orientation[2], pixel_pos[1]*tile_orientation[1]

def load_geom_dict(module):
    ## load geometry dictionary for pixel positions
    # code copied from larpix-readout-parser
    
    with open(module.detector_dict_path) as f_larpix:
        geometry_yaml = yaml.load(f_larpix, Loader=yaml.FullLoader)
    use_pixel_layout = True
    try:
        geometry_yaml['chips']
        use_pixel_layout = False
    except:
        use_pixel_layout = True
    
    if use_pixel_layout:
        pixel_pitch = geometry_yaml['pixel_pitch']
        chip_channel_to_position = geometry_yaml['chip_channel_to_position']
        tile_orientations = geometry_yaml['tile_orientations']
        tile_positions = geometry_yaml['tile_positions']
        tile_indeces = geometry_yaml['tile_indeces']
        xs = np.array(list(chip_channel_to_position.values()))[:, 0] * pixel_pitch
        ys = np.array(list(chip_channel_to_position.values()))[:, 1] * pixel_pitch
        x_size = max(xs) - min(xs) + pixel_pitch
        y_size = max(ys) - min(ys) + pixel_pitch

        geometry = defaultdict(dict)

        for tile in geometry_yaml['tile_chip_to_io']:
            tile_orientation = tile_orientations[tile]

            for chip_channel in geometry_yaml['chip_channel_to_position']:
                chip = chip_channel // 1000
                channel = chip_channel % 1000
                try:
                    io_group_io_channel = geometry_yaml['tile_chip_to_io'][tile][chip]
                except KeyError:
                    continue

                io_group = io_group_io_channel // 1000 # io_group per module (not the real io_group)
                io_channel = io_group_io_channel % 1000
                x = chip_channel_to_position[chip_channel][0] * \
                    pixel_pitch + pixel_pitch / 2 - x_size / 2
                y = chip_channel_to_position[chip_channel][1] * \
                    pixel_pitch + pixel_pitch / 2 - y_size / 2

                x, y = rotate_pixel((x, y), tile_orientation)

                x += tile_positions[tile][2]
                y += tile_positions[tile][1]
                z = tile_positions[tile][0]
                direction = tile_orientations[tile][0]

                geometry[(io_group, io_channel, chip, channel)] = np.array([x, y, z, direction])
    else:
        chip_pix = dict([(chip_id, pix) for chip_id,pix in geometry_yaml['chips']])
        nonrouted_v2a_channels=[6,7,8,9,22,23,24,25,38,39,40,54,55,56,57]
        routed_v2a_channels=[i for i in range(64) if i not in nonrouted_v2a_channels]
        z = 304.15 # mm
        geometry = defaultdict(dict)
        for chipid in chip_pix.keys():
            for channelid in routed_v2a_channels:
                x = geometry_yaml['pixels'][chip_pix[chipid][channelid]][1]
                y = geometry_yaml['pixels'][chip_pix[chipid][channelid]][2]
                geometry[(chipid, channelid)] = np.array([x, y, z, -1])

    return geometry

def load_pedestal_and_config(module):
    # function to open the pedestal and configuration files to get the dictionaries
    #   for the values of v_ped, v_cm, and v_ref for individual pixels. 
    #   Values are fixed in simulation but vary in data depending on pixel.
    # Inputs:
    #   module: input detector configuration
    # Returns:
    #   pedestal and configuration dictionaries
    config_dict = defaultdict(lambda: dict(
        vref_mv=1300,
        vcm_mv=288
    ))
    pedestal_dict = defaultdict(lambda: dict(
        pedestal_mv=580
    ))
    if module.use_ped_config_files:
        pedestal_file = module.pedestal_file
        config_file = module.config_file
        # reading the data from the file
        with open(pedestal_file,'r') as infile:
            for key, value in load_json(infile).items():
                pedestal_dict[key] = value

        with open(config_file, 'r') as infile:
            for key, value in load_json(infile).items():
                config_dict[key] = value
    return pedestal_dict, config_dict

def load_light_geometry(light_geometry_path):
    # Function to open light detector geometry yaml
    with open(light_geometry_path, 'r') as file:
        geometry_data = yaml.safe_load(file)
    return geometry_data

def adc2mv(adc, ref, cm):
    return (ref-cm) * adc/256 + cm

def dac2mv(dac, max):
    return max * dac/256

def load_pedestals(pedestal_file, vref_mv, vcm_mv):
    ### Calculate channel by channel pedestals from pedestal h5 file
    ### Adapted from larpix-v2-testing-scripts
    json_file = os.path.basename(pedestal_file).strip('.h5')+'evd_ped.json'
    
    f = h5py.File(pedestal_file,'r')
    packets = np.array(f['packets'])
    f.close()
    
    good_data_mask = (packets['valid_parity'] == 1) & (packets['packet_type'] == 0)
    packets = packets[good_data_mask]
    unique_id = ((packets['io_group'].astype(int)*256 \
        + packets['io_channel'].astype(int))*256 \
        + packets['chip_id'].astype(int))*64 \
        + packets['channel_id'].astype(int)
    unique_id_sort = np.argsort(unique_id)
    packets[:] = packets[unique_id_sort]
    unique_id = unique_id[unique_id_sort]
    
    # find start and stop indices for each occurrance of a unique id
    unique_id_set, start_indices = np.unique(unique_id, return_index=True)
    end_indices = np.roll(start_indices, shift=-1)
    end_indices[-1] = len(packets) - 1
    
    unique_id_chunk_indices = {}
    for val, start_idx, end_idx in zip(unique_id_set, start_indices, end_indices):
        unique_id_chunk_indices[val] = (start_idx, end_idx)

    dataword = packets['dataword']
    
    avg_pedestal = 0
    num = 0
    unique_ped = []
    for unique in tqdm(unique_id_set, desc=' Calculating channel pedestals'):
        start_index, stop_index = unique_id_chunk_indices[unique]
        adcs = dataword[start_index:stop_index]
        if len(adcs) < 1:
            continue
        vals,bins = np.histogram(adcs,bins=np.arange(257))
        peak_bin = np.argmax(vals)
        min_idx,max_idx = max(peak_bin-mean_trunc,0), min(peak_bin+mean_trunc,len(vals))
        ped_adc = np.average(bins[min_idx:max_idx]+0.5, weights=vals[min_idx:max_idx])

        ped = adc2mv(ped_adc,vref_mv,vcm_mv)
        num+=1
        avg_pedestal += ped
        
        unique_ped.append((unique, ped))

    # make dictionary with found avg pedestal for default
    avg_pedestal = avg_pedestal/num
    print(f'avg pedestal = {avg_pedestal}')
    pedestal_dict = defaultdict(lambda: dict(
        pedestal_mv=avg_pedestal
    ))
    for i in range(num):
        pedestal_dict[str(unique_ped[i][0])] = dict(
             pedestal_mv=unique_ped[i][1]
        )
    packets=0
    dataword=0
    unique_id=0
    good_data_mask=0
    unique_ped=0
    
    with open(json_file,'w') as fo:
        json.dump(pedestal_dict, fo, sort_keys=True, indent=4)
    return pedestal_dict
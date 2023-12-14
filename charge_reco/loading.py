from json import load as load_json
from collections import defaultdict
import yaml
import numpy as np
import h5py
from consts import vdda, mean_trunc
from tqdm import tqdm

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

def adc2mv(adc, ref, cm, bits=8):
    return (ref-cm) * adc/(2**bits) + cm

def dac2mv(dac, max, bits=8):
    return max * dac/(2**bits)

def load_pedestals(pedestal_file, vref_dac, vcm_dac):
    ### Calculate channel by channel pedestals from pedestal h5 file
    ### Adapted from larpix-v2-testing-scripts
    json_file = 'pedestal/' + pedestal_file.split('.h5')[0] + '_evd_ped.json'
    
    import os
    # if the pedestal json doesn't already exist, then load the h5 file and calculate pedestals
    if not os.path.exists(json_file):
        f = h5py.File(pedestal_file,'r')
        good_data_mask = f['packets']['packet_type'] == 0
        good_data_mask = np.logical_and(f['packets']['valid_parity'] == 1, good_data_mask)

        unique_id = ((f['packets'][good_data_mask]['io_group'].astype(int)*256 \
            + f['packets'][good_data_mask]['io_channel'].astype(int))*256 \
            + f['packets'][good_data_mask]['chip_id'].astype(int))*64 \
            + f['packets'][good_data_mask]['channel_id'].astype(int)
        unique_id_set = np.unique(unique_id)

        pedestal_dict = dict()
        dataword = f['packets'][good_data_mask]['dataword']

        for unique in tqdm(unique_id_set):
            vref_mv = dac2mv(vref_dac,vdda)
            vcm_mv = dac2mv(vcm_dac,vdda)
            channel_mask = unique_id == unique
            adcs = dataword[channel_mask]
            if len(adcs) < 1:
                continue
            vals,bins = np.histogram(adcs,bins=np.arange(257))
            peak_bin = np.argmax(vals)
            min_idx,max_idx = max(peak_bin-mean_trunc,0), min(peak_bin+mean_trunc,len(vals))
            ped_adc = np.average(bins[min_idx:max_idx]+0.5, weights=vals[min_idx:max_idx])

            pedestal_dict[str(unique)] = dict(
                pedestal_mv=adc2mv(ped_adc,vref_mv,vcm_mv)
                )
        avg_pedestal = np.sum(list(pedestal_dict.values()))/len(pedestal_dict)
        pedestal_dict_new = defaultdict(lambda: avg_pedestal, pedestal_dict)

        import os
        if not os.path.exists(json_file):
            import json
            with open(json_file,'w') as fo:
                json.dump(pedestal_dict, fo, sort_keys=True, indent=4)
        
        return pedestal_dict_new
    else:
        # if pedestal json already exists, then just open it.
        with open(json_file,'r') as infile:
            for key, value in load_json(infile).items():
                pedestal_dict[key] = value
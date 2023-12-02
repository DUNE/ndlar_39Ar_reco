from json import load as load_json
from collections import defaultdict
import yaml
import numpy as np

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
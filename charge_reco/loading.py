from pickle import load as load_pickle
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

def load_geom_dict(geom_dict_path):
    ## load geometry dictionary for pixel positions
    with open(geom_dict_path, "rb") as f_geom_dict:
        geom_dict = load_pickle(f_geom_dict)
    return geom_dict

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
  
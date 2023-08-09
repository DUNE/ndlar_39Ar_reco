import pickle
from calibrate import timestamp_corrector
from consts import *
import numpy as np
import yaml

def load_geom_dict(geom_dict_path):
    ## load geometry dictionary for pixel positions
    with open(geom_dict_path, "rb") as f_geom_dict:
        geom_dict = pickle.load(f_geom_dict)
    return geom_dict

def zip_pixel_tyz(packets,ts, pixel_xy, module):
    ## form zipped array using info from dictionary to use in clustering
    ## calculates first relative packet coordinates in each module, then adjust by TPC offsets
    ## some of this code copied from larnd-sim
    with open(module.detprop_path) as df:
        detprop = yaml.load(df, Loader=yaml.FullLoader)
    #ts_inmm = v_drift*1e1*ts*0.1 # timestamp in us * drift speed in mm/us

    TPC_OFFSETS = np.array(detprop['tpc_offsets'])*10
    # Inverting x and z axes
    TPC_OFFSETS[:, [2, 0]] = TPC_OFFSETS[:, [0, 2]]

    module_to_io_groups = detprop['module_to_io_groups']
    io_groups = packets['io_group']
    io_groups_rel = np.copy(io_groups)
    io_groups_rel[io_groups_rel % 2 == 1] = 1
    io_groups_rel[io_groups_rel % 2 == 0] = 2
    io_channels = packets['io_channel']
    chip_ids = packets['chip_id']
    channel_ids = packets['channel_id']
    #keys = np.stack((io_group_rel, io_channel, chip_id, channel_id), axis=-1)

    xyz_values = []
    ts_inmm = []
    packets_keep_mask = np.zeros(len(packets), dtype=bool)
    for i in range(len(io_channels)):
        io_group = io_groups_rel[i]
        io_channel = io_channels[i]
        chip_id = chip_ids[i]
        channel_id = channel_ids[i]
        
        dict_values = pixel_xy.get((io_group, io_channel, chip_id, channel_id))
        if dict_values is not None:
            xyz_values.append([dict_values[0], dict_values[1], dict_values[2], dict_values[3]])
            ts_inmm.append(v_drift*1e1*ts[i]*0.1)
            packets_keep_mask[i] = True
        #else:
            #print(f'KeyError {(io_group, io_channel, chip_id, channel_id)}')
    xyz_values = np.array(xyz_values)
    ts_inmm = np.array(ts_inmm)
    
    # adjust coordinates by TPC offsets, if there's >1 module
    io_group_grouped_by_module = list(module_to_io_groups.values())
    for i in range(len(io_group_grouped_by_module)):
        io_group_group = io_group_grouped_by_module[i]
        if len(np.unique(io_groups)) > 2:
            xyz_values[(io_groups == io_group_group[0]) | (io_groups == io_group_group[1])] += np.concatenate((TPC_OFFSETS[i], np.array([0])))
        else:
            pass
    txyz = np.hstack((ts_inmm[:, np.newaxis], xyz_values))
    
    return txyz, packets_keep_mask


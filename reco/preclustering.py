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
    ts_inmm = v_drift*1e1*ts*0.1 # timestamp in us * drift speed in mm/us

    TPC_OFFSETS = np.array(detprop['tpc_offsets'])*10
    # Inverting x and z axes
    TPC_OFFSETS[:, [2, 0]] = TPC_OFFSETS[:, [0, 2]]

    module_to_io_groups = detprop['module_to_io_groups']
    io_group = packets['io_group']
    io_group_rel = np.copy(io_group)
    io_group_rel[io_group_rel % 2 == 1] = 1
    io_group_rel[io_group_rel % 2 == 0] = 2
    io_channel = packets['io_channel']
    chip_id = packets['chip_id']
    channel_id = packets['channel_id']
    keys = np.stack((io_group_rel, io_channel, chip_id, channel_id), axis=-1)

    xyz_values = np.array([pixel_xy.get(tuple(key), [0.0, 0.0, 0.0,0.0]) for key in keys])

    # adjust coordinates by TPC offsets, if there's >1 module
    io_group_grouped_by_module = list(module_to_io_groups.values())
    for i in range(len(io_group_grouped_by_module)):
        io_group_group = io_group_grouped_by_module[i]
        if len(np.unique(io_group)) > 2:
            xyz_values[(io_group == io_group_group[0]) | (io_group == io_group_group[1])] += np.concatenate((TPC_OFFSETS[i], np.array([0])))
        else:
            pass
    txyz = np.hstack((ts_inmm[:, np.newaxis], xyz_values))
    return txyz


import pickle
from calibrate import timestamp_corrector
from consts import *
import numpy as np

def load_geom_dict(geom_dict_path):
    ## load geometry dictionary for pixel positions
    with open(geom_dict_path, "rb") as f_geom_dict:
        geom_dict = pickle.load(f_geom_dict)
    return geom_dict

def zip_pixel_tyz(packets,ts, pixel_xy):
    ## form zipped array using info from dictionary to use in clustering
    ts_inmm = v_drift*1e1*ts*0.1 # timestamp in us * drift speed in mm/us
    io_group = packets['io_group']
    io_channel = packets['io_channel']
    chip_id = packets['chip_id']
    channel_id = packets['channel_id']
    keys = np.stack((io_group, io_channel, chip_id, channel_id), axis=-1)
    xyz_values = np.array([pixel_xy.get(tuple(key), [0.0, 0.0, 0.0,0.0]) for key in keys])
    txyz = np.hstack((ts_inmm[:, np.newaxis], xyz_values))
    return txyz


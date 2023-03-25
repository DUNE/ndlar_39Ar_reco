import pickle
from calibrate import timestamp_corrector
from consts import *

def load_geom_dict(geom_dict_path):
    ## load geometry dictionary for pixel positions
    with open(geom_dict_path, "rb") as f_geom_dict:
        geom_dict = pickle.load(f_geom_dict)
    return geom_dict

def zip_pixel_tyz(packets,ts, pixel_xy):
    ## form zipped array to use in larnd-sim. Maybe this can be made without a for loop? FIXME?
    ts_inmm = v_drift*1e1*ts*0.1 # timestamp in us * drift speed in mm/us
    txyz = []
    for i in range(len(packets)):
        try:
            xyz = pixel_xy[packets['io_group'][i],packets['io_channel'][i],packets['chip_id'][i],packets['channel_id'][i]]
            txyz.append([ts_inmm[i],xyz[0],xyz[1],xyz[2]])
        except:
            txyz.append([ts_inmm[i],0.0,0.0,0.0])
    return txyz


import pickle
from calibrate import timestamp_corrector

def load_geom_dict(geom_dict_path):
    with open(geom_dict_path, "rb") as f_geom_dict:
        geom_dict = pickle.load(f_geom_dict)
    return geom_dict

def getPackets(file, sel_start, sel_end):
    #packets = packets[packets['valid_parity'] == 1]
    mc_assn=None
    try:
        mc_assn = file['mc_packets_assn']
    except:
        mc_assn=None
    
    # load packets and make selection
    packets = file['packets']
    if sel_end == -1:
        sel_end = len(packets)
    packets = packets[sel_start:sel_end]
    ts, packets = timestamp_corrector(packets, mc_assn)

    return ts, packets, mc_assn

def zip_pixel_tyz(packets,ts, pixel_xy):
    v_drift = 1.6 # mm/us, 500V/cm
    ts_inmm = v_drift*ts*0.1 # timestamp in us * drift speed in mm/us
    txyz = []
    for i in range(len(packets)):
        try:
            xyz = pixel_xy[packets['io_group'][i],packets['io_channel'][i],packets['chip_id'][i],packets['channel_id'][i]]
            txyz.append([ts_inmm[i],xyz[0],xyz[1],xyz[2]])
        except:
            txyz.append([ts_inmm[i],0.0,0.0,0.0])

    return txyz


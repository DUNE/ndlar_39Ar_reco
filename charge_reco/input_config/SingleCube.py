### SingleCube input configuration

# detector name
detector = 'SingleCube'
data_type = 'data'

# required filepaths
input_packets_filename = '/Users/samuelfogarty/Documents/SingleCube/datalog_2023_01_20_04_29_36_MST.h5'
output_events_filename = input_packets_filename.split('.h5')[0] + '_events_SingleCube.h5'
detector_dict_path = 'layout/single_tile_layout-2.0.1.pkl'
detprop_path = 'detector_properties/SingleCube.yaml'

# optional filepaths, set to None to bypass. Can alternatively just toggle them on/off below.
disabled_channels_list = None
pedestal_file = 'pedestal/CSUSingleCube_tile-id-3-pedestal_2022_12_19_14_31_54_MST_evd_ped.json'
config_file = 'config/CSUSingleCube_config_23-06-05_13-59-25.json'

# toggles
use_disabled_channels_list = False
use_ped_config_files = True

# charge
PACMAN_clock_correction1 = [0., 0.]
PACMAN_clock_correction2 = [0., 0.]
PACMAN_clock_correction = True
timestamp_cut = True

# matching
ext_trig_matching_tolerance_unix = 1
ext_trig_matching_tolerance_PPS = 1e3 # ns
from consts import drift_distance, v_drift
charge_light_matching_PPS_window = int(drift_distance / v_drift * 1e3)
charge_light_matching_unix_window = 1

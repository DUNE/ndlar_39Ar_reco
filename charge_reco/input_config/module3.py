### module-3 input configuration

# detector name
detector = 'module-3'
data_type = 'data'

# required filepaths
input_packets_filename = '/Users/samuelfogarty/Desktop/mod0files.nosync/datalog_2021_04_06_00_50_23_CEST.h5'
output_events_filename = input_packets_filename.split('.h5')[0] + '_events.h5'
detector_dict_path = 'layout/module3_multi_tile_layout.pkl'
detprop_path = 'detector_properties/module0.yaml'

# optional filepaths, set to None to bypass. Can alternatively just toggle them on/off below.
disabled_channels_list = None
pedestal_file = 'pedestal/module3_pedestal-diagnostic-packet-2023_01_28_22_33_CETevd_ped.json'
config_file = 'config/module3_evd_config_23-01-29_11-12-16.json'

# toggles
use_disabled_channels_list = False
use_ped_config_files = True

# charge
PACMAN_clock_correction1 = [0., 3.267e-6]
PACMAN_clock_correction2 = [0., -8.9467e-7]
PACMAN_clock_correction = True
timestamp_cut = True

# matching
ext_trig_matching_tolerance_unix = 1
ext_trig_matching_tolerance_PPS = 1.5e3 # ns
from consts import drift_distance, v_drift
charge_light_matching_PPS_window = int(drift_distance / v_drift * 1e3)
charge_light_matching_unix_window = 1
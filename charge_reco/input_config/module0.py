### module-0 input configuration

# detector name
detector = 'module-0'
data_type = 'data'

# required filepaths
input_packets_filename = '/sdf/group/neutrino/sfogarty/ND_prototype_files/charge_data/module-0/radiologicals_study/data/datalog_2021_04_04_20_59_11_CEST.h5'
output_events_filename = input_packets_filename.split('.h5')[0] + '_events_test.h5'
detector_dict_path = 'layout/module0_multi_tile_layout-2.3.16.pkl'
detprop_path = 'detector_properties/module0.yaml'

# optional filepaths, set to None to bypass. Can alternatively just toggle them on/off below.
disabled_channels_list = None
pedestal_file = 'pedestal/module0_datalog_2021_04_02_19_00_46_CESTevd_ped.json'
config_file = 'config/module0_evd_config_21-03-31_12-36-13.json'

# toggles
use_disabled_channels_list = False
use_ped_config_files = True

# charge
PACMAN_clock_correction1 = [-9.597, 3.7453e-06]
PACMAN_clock_correction2 = [-9.329, 9.0283e-07]
PACMAN_clock_correction = True
timestamp_cut = True
nBatches = 400
batches_limit = 50

# matching
ext_trig_matching_tolerance_unix = 1
ext_trig_matching_tolerance_PPS = 1e3 # ns
from consts import drift_distance, v_drift
charge_light_matching_PPS_window = int(drift_distance / v_drift * 1e3)
charge_light_matching_unix_window = 0

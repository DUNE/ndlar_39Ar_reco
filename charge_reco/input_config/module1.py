### module-1 input configuration

# detector name
detector = 'module-1'
data_type = 'data'

# required filepaths
input_packets_filename = '/sdf/group/neutrino/sfogarty/ND_prototype_files/charge_data/module-1/light_study/packet_2022_02_08_01_47_59_CET.h5'
output_events_filename = input_packets_filename.split('.h5')[0] + '_clusters_test_test.h5'
#detector_dict_path = 'layout/multi_tile_layout-2.4.16.pkl'
detector_dict_path = 'layout/module1_layout-2.3.16.pkl'
detprop_path = 'detector_properties/module0.yaml'

# optional filepaths, set to None to bypass. Can alternatively just toggle them on/off below.
disabled_channels_list = None
pedestal_file = 'pedestal/module1_packet_2022_02_08_01_40_31_CETevd_ped.json'
config_file = 'config/module1_config_22-02-08_13-37-39.json'

# toggles
use_disabled_channels_list = False
use_ped_config_files = True

# charge
PACMAN_clock_correction1 = [0., -2.5825e-07]
PACMAN_clock_correction2 = [0., 4.0650e-07]
PACMAN_clock_correction = True
timestamp_cut = True
nBatches = 400
batches_limit = 25
# matching
ext_trig_matching_tolerance_unix = 1
ext_trig_matching_tolerance_PPS = 1.5e3 # ns
from consts import drift_distance, v_drift
full_drift_time = int(drift_distance / v_drift * 1e3)
charge_light_matching_lower_PPS_window = 61000
charge_light_matching_upper_PPS_window = full_drift_time + 61000
charge_light_matching_unix_window = 0
ext_trig_PPS_window = 1000

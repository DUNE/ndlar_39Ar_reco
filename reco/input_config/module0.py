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
adc_folder = '/Users/samuelfogarty/Desktop/mod0files.nosync/'
input_light_filename_1 = '0a7a314c_20210406_005022.data'
input_light_filename_2 = '0a7b54bd_20210406_005022.data'
disabled_channels_list = None
pedestal_file = 'pedestal/module0_datalog_2021_04_02_19_00_46_CESTevd_ped.json'
config_file = 'config/module0_evd_config_21-03-31_12-36-13.json'

# toggles
do_match_of_charge_to_light = False
use_disabled_channels_list = False
use_ped_config_files = True

# charge
PACMAN_clock_correction1 = [-9.597, 3.7453e-06]
PACMAN_clock_correction2 = [-9.329, 9.0283e-07]
PACMAN_clock_correction = True
timestamp_cut = True

# light
light_time_steps = 256
nchannels_adc1 = 58
nchannels_adc2 = 58
clock_correction_factor = 0.625
ADC_drift_slope_0 = -1.18e-7 # sn 175780172 (#314C)
ADC_drift_slope_1 = 1.18e-7 # sn 175854781 (#54BD)

# matching
ext_trig_matching_tolerance_unix = 1
ext_trig_matching_tolerance_PPS = 1e3 # ns
from consts import drift_distance, v_drift
charge_light_matching_PPS_window = int(drift_distance / v_drift * 1e3)
charge_light_matching_unix_window = 1

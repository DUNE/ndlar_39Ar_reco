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
adc_folder = '/Users/samuelfogarty/Desktop/mod0files.nosync/'
input_light_filename_1 = '0a7a314c_20210406_005022.data'
input_light_filename_2 = '0a7b54bd_20210406_005022.data'
disabled_channels_list = None
pedestal_file = 'pedestal/CSUSingleCube_tile-id-3-pedestal_2022_12_19_14_31_54_MST_evd_ped.json'
config_file = 'config/CSUSingleCube_config_23-06-05_13-59-25.json'

# toggles
do_match_of_charge_to_light = False
use_disabled_channels_list = False
use_ped_config_files = True

# data selection
nSec_start_packets = 1
nSec_end_packets = -1
nSec_start_light = nSec_start_packets
nSec_end_light = nSec_end_packets

# charge
PACMAN_clock_correction1 = [-9.597, 4.0021e-6]
PACMAN_clock_correction2 = [-9.329, 1.1770e-6]
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

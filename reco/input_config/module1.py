### module-1 input configuration

# detector name
detector = 'module-1'
data_type = 'data'

# required filepaths
input_packets_filename = '/Users/samuelfogarty/Desktop/mod0files.nosync/datalog_2021_04_06_00_50_23_CEST.h5'
output_events_filename = input_packets_filename.split('.h5')[0] + '_events.h5'
detector_dict_path = 'layout/module1_layout-2.3.16.pkl'

# optional filepaths, set to None to bypass. Can alternatively just toggle them on/off below.
adc_folder = '/Users/samuelfogarty/Desktop/mod0files.nosync/'
input_light_filename_1 = '0a7a314c_20210406_005022.data'
input_light_filename_2 = '0a7b54bd_20210406_005022.data'
disabled_channels_list = None
pedestal_file = 'pedestal/module1_packet_2022_02_08_01_40_31_CETevd_ped.json'
config_file = 'config/module1_config_22-02-08_13-37-39.json'

# toggles
do_match_of_charge_to_light = False
use_disabled_channels_list = False
use_ped_config_files = True

# data selection
nSec_start_packets = 1
nSec_end_packets = 400
nSec_start_light = nSec_start_packets
nSec_end_light = nSec_end_packets

# charge
PACMAN_clock_correction1 = [0., 0.]
PACMAN_clock_correction2 = [0., 0.]
PACMAN_clock_correction = True
timestamp_cut = True

# light
light_time_steps = 1000
nchannels_adc1 = 48
nchannels_adc2 = 48
clock_correction_factor = 0.625
ADC_drift_slope_0 = 0. # sn 175780172 (#314C)
ADC_drift_slope_1 = 0. # sn 175854781 (#54BD)

# matching
ext_trig_matching_tolerance_unix = 1
ext_trig_matching_tolerance_PPS = 1.5e3 # ns
from consts import drift_distance, v_drift
charge_light_matching_PPS_window = int(drift_distance / v_drift * 1e3)
charge_light_matching_unix_window = 1
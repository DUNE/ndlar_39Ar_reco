### 2x2 MC input configuration

# detector name
detector = '2x2'
data_type = 'MC'

# required filepaths
input_packets_filename = '/Users/samuelfogarty/Desktop/mod0files.nosync/2x2_MC/MiniRun3_1E19_RHC.larnd_v2.00000.LARNDSIM.h5'
output_events_filename = input_packets_filename.split('.h5')[0] + '_events.h5'
detector_dict_path = 'layout/2x2_multi_tile_layout-2.3.16.pkl'
detprop_path = 'detector_properties/2x2.yaml'

# toggles
do_match_of_charge_to_light = False
use_disabled_channels_list = False
use_ped_config_files = False

# light
light_time_steps = 256
nchannels_adc1 = 58
nchannels_adc2 = 58
clock_correction_factor = 0.625
ADC_drift_slope_0 = 0. # sn 175780172 (#314C)
ADC_drift_slope_1 = 0. # sn 175854781 (#54BD)

# charge
PACMAN_clock_correction1 = [0., 0.]
PACMAN_clock_correction2 = [0., 0.]

# matching
ext_trig_matching_tolerance_unix = 1
ext_trig_matching_tolerance_PPS = 1e3 # ns
from consts import drift_distance, v_drift
charge_light_matching_PPS_window = int(drift_distance / v_drift * 1e3)
charge_light_matching_unix_window = 1

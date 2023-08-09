### module-0 MC input configuration

# detector name
detector = 'module-0'
data_type = 'MC'

# required filepaths
input_packets_filename = '/sdf/group/neutrino/sfogarty/ND_prototype_files/MC/module-0/radiologicals_afterNESTfix/larnd-sim/larndsim_238U_gammas_10k_2.h5'
output_events_filename = '/sdf/group/neutrino/sfogarty/ND_prototype_files/MC/module-0/radiologicals_afterNESTfix/reco/larndsim_238U_gammas_10k_2_clusters.h5'
detector_dict_path = 'layout/module0_multi_tile_layout-2.3.16.pkl'
detprop_path = 'detector_properties/module0.yaml'

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
nBatches = 1
batches_limit = 1

# matching
ext_trig_matching_tolerance_unix = 1
ext_trig_matching_tolerance_PPS = 1e3 # ns
from consts import drift_distance, v_drift
#charge_light_matching_PPS_window = int(drift_distance / v_drift * 1e3)
#charge_light_matching_unix_window = 1
#charge_light_matching_PPS_window = int(drift_distance / v_drift * 1e3)
charge_light_matching_PPS_window = 400000
charge_light_matching_unix_window = 0

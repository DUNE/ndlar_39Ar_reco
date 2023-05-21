# module-0 input configuration
detector = 'module-0'
input_packets_filename = '/sdf/group/neutrino/sfogarty/ND_prototype_files/charge_data/module-0/light_study/datalog_2021_04_04_18_58_59_CEST.h5'
input_light_filename_1 = '0a7a314c_20210404_185859.data'
input_light_filename_2 = '0a7b54bd_20210404_185859.data'
output_events_filename = input_packets_filename.split('.h5')[0] + '_events_charge_lighttest.h5'
disabled_channels_list = '/sdf/home/s/sfogarty/Desktop/RadDecay/39Ar_reco_sim/LArNDLE/reco/disabled_channels/module0_disabled_channels_noise_cut.npz'
nSec_start_packets = 1
nSec_end_packets = 100
nSec_start_light = nSec_start_packets
nSec_end_light = nSec_end_packets
sync_filename = None # set to None to bypass
light_time_steps = 256
nchannels_adc1 = 58
nchannels_adc2 = 58
matching_tolerance_unix = 1 # s
matching_tolerance_PPS = 1e3 # ns

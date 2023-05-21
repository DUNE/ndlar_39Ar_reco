# module-3 input configuration
detector = 'module-3'
input_packets_filename = '/sdf/group/neutrino/sfogarty/ND_prototype_files/charge_data/module-3/tpc12-packet-2023_03_15_00_14_CET.h5'
input_light_filename_1 = '0cd913fa_20230222_124553.data'
input_light_filename_2 = '0cd9414a_20230222_124553.data'
output_events_filename = input_packets_filename.split('.h5')[0] + '_' + detector + '_events_charge_1.h5'
disabled_channels_list = '/sdf/home/s/sfogarty/Desktop/RadDecay/39Ar_reco_sim/LArNDLE/reco/disabled_channels/module3_disabled_channels_noise_cut.npz'
nSec_start_packets = 1
nSec_end_packets = 200
nSec_start_light = 1
nSec_end_light = nSec_end_packets
sync_filename = None # set to None to bypass
light_time_steps = 512
nchannels_adc1 = 48
nchannels_adc2 = 16
matching_tolerance_unix = 1 # s
matching_tolerance_PPS = 2e3 # ns
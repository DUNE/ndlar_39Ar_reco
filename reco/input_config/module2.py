# module-1 input configuration
detector = 'module-2'
input_packets_filename = '/sdf/group/neutrino/sfogarty/ND_prototype_files/charge_data/module-2/selftriggering-binary-2022_11_18_00_36_CET.packet.h5'
input_light_filename_1 = '0cd913fb_20220208_014759.data'
input_light_filename_2 = '0cd93db0_20220208_014759.data'
output_events_filename = input_packets_filename.split('.h5')[0] + '_events_charge_1.h5'
disabled_channels_list = '/sdf/home/s/sfogarty/Desktop/RadDecay/39Ar_reco_sim/LArNDLE/reco/disabled_channels/module2_disabled_channels_noise_cut.npz'
nSec_start_packets = 1
nSec_end_packets = 200
nSec_start_light = nSec_start_packets
nSec_end_light = nSec_end_packets
sync_filename = None # set to None to bypass
light_time_steps = 1000
nchannels_adc1 = 48
nchannels_adc2 = 48
matching_tolerance_unix = 1 # s
matching_tolerance_PPS = 1.5e3 # ns

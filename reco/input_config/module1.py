# module-1 input configuration
detector = 'module-1'
input_packets_filename = '/sdf/group/neutrino/sfogarty/ND_prototype_files/charge_data/module-1/packet_2022_02_08_01_47_59_CET.h5'
input_light_filename_1 = '0cd913fb_20220208_014759.data'
input_light_filename_2 = '0cd93db0_20220208_014759.data'
output_events_filename = input_packets_filename.split('.h5')[0] + '_events_branch_eps300mm_minSamples1_light_4.h5'
nSec_start_packets = 601
nSec_end_packets = 800
nSec_start_light = nSec_start_packets
nSec_end_light = nSec_end_packets
sync_filename = None # set to None to bypass
light_time_steps = 1000
nchannels_adc1 = 48
nchannels_adc2 = 48
matching_tolerance_unix = 1 # s
matching_tolerance_PPS = 2e3 # ns

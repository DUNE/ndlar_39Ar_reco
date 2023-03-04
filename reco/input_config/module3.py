# module-3 input configuration
detector = 'module-3'
input_packets_filename = 'tpc_12-packet-2023_02_22_12_45_CET.h5'
input_light_filename_1 = '0cd913fa_20230222_124553.data'
input_light_filename_2 = '0cd9414a_20230222_124553.data'
output_events_filename = input_packets_filename.split('.h5')[0] + '_' + detector + '_events.h5'
nSec_start_packets = 1
nSec_end_packets = 10
nSec_start_light = 0
nSec_end_light = 20
sync_filename = 'sync_packet_mask_tpc_12-packet-2023_02_22_12_45_CET.npz' # set to None to bypass
light_time_steps = 512
nchannels_adc1 = 48
nchannels_adc2 = 16
matching_tolerance_unix = 1 # s
matching_tolerance_PPS = 2e3 # ns
# module-0 input configuration
detector = 'module-0'
input_packets_filename = 'datalog_2021_04_04_00_41_40_CEST.h5'
input_light_filename_1 = '0a7a314c_20210404_004206.data'
input_light_filename_2 = '0a7b54bd_20210404_004206.data'
output_events_filename = input_packets_filename.split('.h5')[0] + '_' + detector + '_events_60s_to_360s.h5'
nSec_start_packets = 60
nSec_end_packets = 360
nSec_start_light = 0
nSec_end_light = nSec_end_packets
sync_filename = None # set to None to bypass
light_time_steps = 256
nchannels_adc1 = 58
nchannels_adc2 = 58
matching_tolerance_unix = 1 # s
matching_tolerance_PPS = 1e3 # ns

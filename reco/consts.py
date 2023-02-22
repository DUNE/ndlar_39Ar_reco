# set values for simulation (copy from larnd-sim)
#v_cm_sim = 288 # mV
#v_pedestal_sim = 580 
#v_ref_sim = 1300 

v_cm_sim = 288.28125
v_ref_sim = 1300.78125
v_pedestal_sim = 580 # make sure it is the same as in larnd-sim

v_cm_data = 288.28125
v_ref_data = 1300.78125

#: Number of ADC counts
ADC_COUNTS = 2**8

#gain_sim = 0.004 # mV/e-
gain_sim = 1/221
gain_data = 1/221 # mV/e-

# DBSCAN parameters
eps_tracks = 20 ## mm
min_samples_tracks = 8

eps_noise = 10 ## mm
min_samples_noise = 1

# Toggles for cuts and calibrations
timestamp_cut = True
PACMAN_clock_correction = True

# light
adc_file_1 = '/Users/samuelfogarty/Desktop/mod0files.nosync/0a7a314c_20210404_004206.data'
adc_file_2 = '/Users/samuelfogarty/Desktop/mod0files.nosync/0a7b54bd_20210404_004206.data'
batch_size = 50 # how many events to load on each iteration




### set values calibration values for simulation
# make sure these are the same as in larnd-sim

## probably the default values
#v_cm_sim = 288 # mV
#v_pedestal_sim = 580 
#v_ref_sim = 1300 

# set to match data
v_cm_sim = 288.28125
v_ref_sim = 1300.78125
v_pedestal_sim = 580 

v_cm_data = 288.28125
v_ref_data = 1300.78125

#: Number of ADC counts
ADC_COUNTS = 2**8

#gain_sim = 0.004 # mV/e-
gain_sim = 1/221
gain_data = 1/221 # mV/e-

### charge
charge_data_folder = '/sdf/group/neutrino/sfogarty/ND_prototype_files/charge_data/'

# DBSCAN parameters
eps_tracks = 20 ## mm
min_samples_tracks = 8

eps_noise = 10 ## mm
min_samples_noise = 1

# Toggles for cuts and calibrations
timestamp_cut = True
PACMAN_clock_correction = True
use_charge_event_drift_window_cut = False
use_pixel_plane_cut = False

# light
adc_folder = '/sdf/group/neutrino/sfogarty/ND_prototype_files/light_data/'
batch_size = 50 # how many events to load on each iteration
do_match_of_charge_to_light = True

# matching
v_drift = 0.16 # cm/usec, 500V/cm
drift_distance = 30.27 # cm




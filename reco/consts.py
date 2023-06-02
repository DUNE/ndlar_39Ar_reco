# set ADC parameters for simulation
# make sure these are the same as in the simulation file made by larnd-sim
v_cm_sim = 288.28125
v_ref_sim = 1300.78125
v_pedestal_sim = 580 

# set ADC parameters for data
v_cm_data = 288.28125
v_ref_data = 1300.78125

# Total number of ADC counts
ADC_COUNTS = 2**8

# gain values
gain_sim = 1/221 # mV/e-, make sure to match simulation file
gain_data = 1/221 # mV/e-

# DBSCAN parameters
eps = 20 ## mm
min_samples = 1

# light
batch_size = 50 # how many events to load on each iteration

# matching
v_drift = 0.16 # cm/usec, 500V/cm
drift_distance = 30.27 # cm




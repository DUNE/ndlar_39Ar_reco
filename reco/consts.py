import numpy as np

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

# DBSCAN parameters
eps = 20 ## mm
min_samples = 1

# light
batch_size = 50 # how many events to load on each iteration

# matching
v_drift = 0.16 # cm/usec, 500V/cm
drift_distance = 30.27 # cm
    
hits_dtype = np.dtype([('q', '<f8'),('io_group', '<i4'),('unique_id', 'i4'),\
                        ('t', '<i8'),('x', '<f8'), ('y', '<f8'), ('z', '<f8'),\
                        ('unix', '<i8'), ('cluster_index', '<i4'),('event_id', '<i4')])

clusters_dtype = np.dtype([('nhit', '<i4'), ('q', '<f8'),('io_group', '<i4'),\
                        ('t_max', '<i8'), ('t_min', '<i8'),('x_max', '<f8'), ('x_min', '<f8'),
                        ('y_max', '<f8'), ('y_min', '<f8'),\
                        ('unix', '<i8'), ('matched', '<i4'), ('light_index', '<i4')])
    




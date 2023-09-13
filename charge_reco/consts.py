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
ADC_COUNTS = 256

# DBSCAN parameters
eps = 20 ## mm
min_samples = 1

# matching
v_drift = 0.16 # cm/usec, 500V/cm
drift_distance = 30.27 # cm

mm_to_ns = 1/(v_drift*1e1) * 1e3

hits_dtype = np.dtype([('q', '<f8'),('io_group', '<i4'),('unique_id', 'i4'),\
                        ('t', '<i8'),('x', '<f8'), ('y', '<f8'), ('z_anode', '<f8'), ('z_drift', '<f8'), \
                        ('unix', '<i8'), ('cluster_index', '<i4'),('event_id', '<i4')])

clusters_dtype = np.dtype([('id', '<i4'), ('nhit', '<i4'), ('q', '<f8'),('io_group', '<i4'),\
                        ('t_max', '<i8'), ('t_mid', '<i8'), ('t_min', '<i8'),('t0', '<i8'),('x_max', '<f8'), ('x_mid', '<f8'), ('x_min', '<f8'), 
                        ('y_max', '<f8'),('y_mid', '<f8'), ('y_min', '<f8'),('z_anode', '<f8'), \
                        ('z_drift_max', '<f8'),('z_drift_mid', '<f8'), ('z_drift_min', '<f8'),
                        ('unix', '<i8'), ('ext_trig_index', '<i4')])
    
ext_trig_dtype = np.dtype([('unix', '<i8'), ('t', '<i8')])

EVENT_SEPARATOR='event_id'
time_the_reconstruction = False

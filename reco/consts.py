import numpy as np

# parameters
vdda = 1800
mean_trunc = 3
gain = 221 # e/mV

# Total number of ADC counts
ADC_COUNTS = 256

# DBSCAN parameters
eps = 20 # mm
min_samples = 1

# matching
v_drift = 0.16 # cm/usec, 500V/cm
z_drift_factor = 10*v_drift/1e3
drift_distance = 30.27 # cm

mm_to_ns = 1/(v_drift*1e1) * 1e3

hits_dtype = np.dtype([('q', '<f8'), ('adcs', '<i4'), ('io_group', '<i4'),('unique_id', 'i4'),\
                        ('t', '<i8'),('x', '<f8'), ('y', '<f8'), ('z_anode', '<f8'), ('z_drift', '<f8'), \
                        ('unix', '<i8'), ('cluster_index', '<i4'),('event_id', '<i4')])

clusters_dtype = np.dtype([('id', '<i4'), ('nhit', '<i4'), ('q', '<f8'), ('adcs', '<i4'), ('io_group', '<i4'),\
                        ('t_max', '<i8'), ('t_mid', '<i8'), ('t_min', '<i8'),('t0', '<i8'),('x_max', '<f8'), ('x_mid', '<f8'), ('x_min', '<f8'), 
                        ('y_max', '<f8'),('y_mid', '<f8'), ('y_min', '<f8'),('z_anode', '<f8'), \
                        ('z_drift_max', '<f8'),('z_drift_mid', '<f8'), ('z_drift_min', '<f8'),
                        ('unix', '<i8'), ('ext_trig_index', '<i4')])

EVENT_SEPARATOR='event_id'
time_the_reconstruction = False

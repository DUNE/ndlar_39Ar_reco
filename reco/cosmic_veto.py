# use multiprocessing to speed up the cosmic veto process. This code finds small clusters within a specified window from the minT and maxT of a large cluster. This is done in batches of large clusters and each batch is analyzed with a different CPU process, the number of possible processes that can occur at the same time depends on the number CPUs allocated.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import plotly.graph_objects as go
import h5py
import time
from consts import *
from multiprocessing import Pool, cpu_count
PPS_window = drift_distance/v_drift * 1e3
filepath = '/sdf/group/neutrino/sfogarty/ND_prototype_files/charge_data/module-0/datalog_2021_04_04_01_19_19_CEST_module-0_events.h5'
file = h5py.File(filepath, 'r')
large_clusters = file['large_clusters']
small_clusters = file['small_clusters']
# Define a function to compute the matches for a given chunk of events
def compute_matches(chunk):
    t_min = chunk['t_min']
    t_max = chunk['t_max']
    unix = chunk['unix']
    
    t_small_clusters = small_clusters['t']
    unix_small_clusters = small_clusters['unix']
    
    PPS_min_match = t_min[:, np.newaxis] - PPS_window < t_small_clusters
    PPS_max_match = t_max[:, np.newaxis] + PPS_window > t_small_clusters
    unix_match = unix[:, np.newaxis] == unix_small_clusters
    PPS_unix_match = np.logical_and(PPS_min_match, np.logical_and(PPS_max_match, unix_match))
    return np.sum(PPS_unix_match, axis=0).astype(bool)

def cosmic_veto():
    start = time.time()

    # Split the large_clusters array into smaller chunks
    chunk_size = 25
    chunks = [large_clusters[i:i+chunk_size] for i in range(0, len(large_clusters), chunk_size)]

    print('Determining small_clusters within ', PPS_window, ' ns from the start and end of large clusters...')
    print('Running a multiprocessing pool with a large_clusters chunksize of ', chunk_size, '...')
    # Create a multiprocessing pool with the number of available CPU cores
    with Pool(cpu_count()) as p:
        # Compute the matches for each chunk in parallel
        results = p.map(compute_matches, chunks)

    small_clusters_near_mask = np.zeros(len(small_clusters), dtype=bool)
    for i in range(len(results)):
        small_clusters_near_mask += results[i]

    indices = np.where(small_clusters_near_mask.astype(bool))[0]

    print('number of small_clusters near track = ', len(indices))
    print('number total of small_clusters = ', len(small_clusters))
    end = time.time()
    print('Total time to do veto = ', end-start)
    return small_clusters, results

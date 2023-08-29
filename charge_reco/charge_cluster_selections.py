#!/usr/bin/env python
"""
Command-line interface to applying selection criteria to charge clusters using matched external triggers
"""

import numpy as np
import h5py
import fire
from tqdm import tqdm

def main(input_filename, output_filename):
    ## Make selections on the charge cluster data using external triggers and
    ## output a new file for downstream charge-light matching.
    f = h5py.File(input_filename, 'r')
    clusters = np.array(f['clusters'])
    ext_trig = f['ext_trig']
    
    max_hits = 10 # maximum hits per cluster
    max_clusters = 5 # maximum clusters per event
    
    # get only matched clusters
    clusters = clusters[clusters['ext_trig_index'] != -1]
    
    # get ext trig indices, sort them to enable grouping by light trigger
    light_ids = clusters['ext_trig_index']
    sorted_indices = np.argsort(light_ids)
    light_ids = light_ids[sorted_indices]
    clusters = clusters[sorted_indices]
    
    # group clusters by light event
    light_trig_indices = np.concatenate(([0], np.flatnonzero(light_ids[:-1] != light_ids[1:])+1, [len(light_ids)]))
    grouped_clusters = np.split(clusters, light_trig_indices[1:-1])
    
    numEvents = 0
    numEvents_nonUniqueMatch = 0
    
    clusters_keep = []
    ext_trig_keep = []
    print('Total groups = ', len(grouped_clusters))
    for i in tqdm(range(len(grouped_clusters)), desc=' Finding events according to criteria: '):
        group = grouped_clusters[i]
        unique_ext_trig_indices = np.unique(group['ext_trig_index'])
        # require limit on number of clusters per ext trig
        nClustersLimit = len(group) <= max_clusters
        # require limit on number of hits per cluster in match
        nHitsLimit = np.all(group['nhit'] <= max_hits)
        # require no clusters with multiple matches
        uniqueExtTrigs = len(unique_ext_trig_indices) == 1
        if nClustersLimit and nHitsLimit and uniqueExtTrigs:
            numEvents += 1
            for cluster in group:
                clusters_keep.append(cluster)
            ext_trig_keep.append(ext_trig[unique_ext_trig_indices[0]])
            
        if not uniqueExtTrigs:
            numEvents_nonUniqueMatch += 1
    
    clusters = np.array(clusters_keep)
    ext_trig = np.array(ext_trig_keep)
    print(f'Number of events satifying criteria = {numEvents}; {numEvents/len(grouped_clusters)} fraction of total events.')
    print(f'Number of events with multi-matched clusters = {numEvents_nonUniqueMatch}; {numEvents_nonUniqueMatch/len(grouped_clusters)} fraction of total events.')
    print(f'Total events in file = {len(grouped_clusters)}')
    
    with h5py.File(output_filename, 'a') as output_file:
        output_file.create_dataset('clusters', data=clusters)
        output_file.create_dataset('ext_trig', data=ext_trig)
    print(f'Saved output to {output_filename}')
    
if __name__ == "__main__":
    fire.Fire(main)

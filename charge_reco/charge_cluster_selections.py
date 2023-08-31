#!/usr/bin/env python
"""
Command-line interface to applying selection criteria to charge clusters using matched external triggers
"""

import numpy as np
import h5py
import fire
from tqdm import tqdm
import os

def main(input_filename, output_filename):
    ## Make selections on the charge cluster data using external triggers and
    
    if os.path.exists(output_filename):
        raise Exception('Output file '+ str(output_filename) + ' already exists.')
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
    new_ext_trig_index = 0
    new_ext_trig_indices = []
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
                new_ext_trig_indices.append(new_ext_trig_index)
            ext_trig_keep.append(ext_trig[unique_ext_trig_indices[0]])
            new_ext_trig_index += 1
             
    new_ext_trig_indices = np.array(new_ext_trig_indices)
    clusters = np.array(clusters_keep)
    ext_trig = np.array(ext_trig_keep)
    clusters['ext_trig_index'] = new_ext_trig_indices
    
    print(f'Number of events satifying criteria = {numEvents}; {numEvents/len(grouped_clusters)} fraction of total events.')
    print(f'Total events in file = {len(grouped_clusters)}')
    
    with h5py.File(output_filename, 'a') as output_file:
        output_file.create_dataset('clusters', data=clusters)
        output_file.create_dataset('ext_trig', data=ext_trig)
    print(f'Saved output to {output_filename}')
    
if __name__ == "__main__":
    fire.Fire(main)
